"""Published methods wrapped as external adapters under ``adapters/<name>/``.

Each adapter has its own environment (uv or pixi), because the wrapped code is
GPL, needs PyTorch or Uni-Core, or comes from conda-forge. ``run.py`` speaks the
JSONL protocol of :mod:`.external`. ``setup.sh``, where present, fetches pinned
upstream code and weights into ``adapters/<name>/vendor/`` (not committed).
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import ClassVar

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ...data.manifest import REPO_ROOT
from ..core.geometry import CONTACT_SCALE, contact_map
from ..query import Query
from .base import Capabilities, register
from .external import ExternalMethod

ADAPTERS = REPO_ROOT / "adapters"


def adapter_command(name: str, runner: str = "uv") -> str:
    """Command template that runs ``adapters/<name>/run.py`` in its own environment."""
    root = ADAPTERS / name
    script = shlex.quote(str(root / "run.py"))
    if runner == "uv":
        env = f"uv run --quiet --project {shlex.quote(str(root))}"
    elif runner == "pixi":
        env = f"pixi run --quiet --manifest-path {shlex.quote(str(root / 'pixi.toml'))}"
    else:
        raise ValueError(f"unknown runner {runner!r}")
    return f"{env} python {script} {{input}} {{output}}"


class AdapterMethod(ExternalMethod):
    """An :class:`ExternalMethod` backed by ``adapters/<adapter>/``."""

    adapter: ClassVar[str]
    runner: ClassVar[str] = "uv"
    #: paths under the adapter directory that ``setup.sh`` must have created
    requires: ClassVar[tuple[str, ...]] = ()
    #: params passed to ``run.py`` as ``key=value`` arguments
    forwarded: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if "adapter" in cls.__dict__:
            cls.defaults = {**cls.defaults, "command": adapter_command(cls.adapter, cls.runner)}

    def options(self) -> list[str]:
        return [f"{k}={self.params[k]}" for k in self.forwarded]

    def command(self, input_path: Path, output_path: Path) -> list[str]:
        return [*super().command(input_path, output_path), *self.options()]

    def check_available(self) -> str | None:
        root = ADAPTERS / self.adapter
        if not (root / "run.py").exists():
            return f"adapters/{self.adapter} missing"
        missing = [p for p in self.requires if not (root / p).exists()]
        if missing:
            return f"run adapters/{self.adapter}/setup.sh ({missing[0]} missing)"
        if shutil.which(self.runner) is None:
            return f"{self.runner} not on PATH"
        return None

    def version(self) -> str:
        try:
            out = subprocess.run(
                [*shlex.split(self.params["command"])[:-2], "--version", *self.options()],
                cwd=self.params["cwd"], capture_output=True, text=True, timeout=600,
            )
            return out.stdout.strip() or "unknown"
        except (OSError, subprocess.TimeoutExpired):
            return "unknown"


@register
class OpenBabel(AdapterMethod):
    """Open Babel ConnectTheDots + PerceiveBondOrders (GPL-2.0)."""

    name = "openbabel"
    adapter = "openbabel"
    capabilities = Capabilities(stripped_h=True, charged=True, radicals=False)


#: Elements of the Wang et al. (2006) atomic penalty table (``eval/tables/atom_penalties.csv``).
_APS_ELEMENTS = frozenset({1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53})


@register
class Antechamber(AdapterMethod):
    """AmberTools antechamber ``-j 4`` (bondtype, Wang et al. 2006). Explicit H only; no formal charges."""

    name = "antechamber"
    family = "optimization"
    adapter = "antechamber"
    runner = "pixi"
    capabilities = Capabilities(stripped_h=False, charged=True, radicals=False, elements=_APS_ELEMENTS)


@register
class Mambax(AdapterMethod):
    """mambax random forest (Loschen 2018), pretrained ``clf.p``. Reports ``edge_probs`` (use ``+hybrid``)."""

    name = "mambax"
    family = "ml"
    adapter = "mambax"
    requires = ("vendor/mamba/clf.p",)
    # elements of the training set shipped with the repo (largefile.sdf)
    capabilities = Capabilities(stripped_h=True, charged=True, radicals=False,
                                elements=frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}))


@register
class YuelBond(AdapterMethod):
    """YuelBond edge GNN (Wang & Dokholyan 2026). Heavy atoms only; reports ``edge_probs``.

    The upstream repository has no licence; ``setup.sh`` fetches it, bond-bench does not ship it.
    """

    name = "yuelbond"
    family = "ml"
    adapter = "yuelbond"
    requires = ("vendor/yuel_bond/src/lightning.py", "vendor/models/geom_kekulized_3d.ckpt")
    forwarded = ("checkpoint",)
    defaults = {**AdapterMethod.defaults, "checkpoint": "geom_kekulized_3d"}
    # upstream's atom vocabulary (src/const.py); H is stripped before the model sees it
    capabilities = Capabilities(stripped_h=True, charged=True, radicals=False,
                                elements=frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53}))
    #: upstream stores edge indices as int8
    max_heavy_atoms: ClassVar[int] = 127

    def unsupported_reason(self, query):
        reason = super().unsupported_reason(query)
        if reason is None and int((query.atomic_numbers != 1).sum()) > self.max_heavy_atoms:
            return "too many heavy atoms"
        return reason


@register
class BondNet(AdapterMethod):
    """BondNet PaiNN (Zhang 2026), v4 joint checkpoints: explicit-H or heavy-only by observation.

    Reports ``edge_probs`` (use ``+hybrid``). Trained on neutral molecules; reads no charge.
    """

    name = "bondnet"
    family = "ml"
    adapter = "bondnet"
    requires = ("vendor/BondNet/evaluate.py", "vendor/checkpoints/joint_explicit_h/seed42/best_e2e.pt",
                "vendor/checkpoints/joint_heavy_only/seed42/best_e2e.pt")
    forwarded = ("seed",)
    defaults = {**AdapterMethod.defaults, "seed": 42}
    # training elements (scripts/build_heavy_hcount_cache.py)
    capabilities = Capabilities(stripped_h=True, charged=True, radicals=False,
                                elements=frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53, 80, 83}))


@register
class MPerformer(AdapterMethod):
    """MPerformer SE(3) Transformer (Wang et al. 2023). Heavy atoms only; predicts charges and H counts."""

    name = "mperformer"
    family = "ml"
    adapter = "mperformer"
    requires = ("vendor/MPerformer/dict.txt", "vendor/weight/checkpoint.pt")
    # upstream's token vocabulary (dict.txt)
    capabilities = Capabilities(stripped_h=True, charged=True, radicals=False, elements=frozenset({
        1, 3, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 19, 20, 24, 26, 30, 33, 34, 35, 50, 53, 64, 79, 80}))
    #: upstream crops larger molecules at random
    max_heavy_atoms: ClassVar[int] = 256

    def unsupported_reason(self, query):
        reason = super().unsupported_reason(query)
        if reason is None and int((query.atomic_numbers != 1).sum()) > self.max_heavy_atoms:
            return "too many heavy atoms"
        return reason


class ConnectivityAdapter(AdapterMethod):
    """A method that assigns bond orders to a given graph: it receives the contact map A(X).

    Both wrapped tools handle one connected molecule only, so a query whose contact
    graph has several components is reported as unsupported.
    """

    send_contacts = True
    defaults = {**AdapterMethod.defaults, "contact_scale": CONTACT_SCALE}

    def unsupported_reason(self, query: Query) -> str | None:
        reason = super().unsupported_reason(query)
        if reason is not None:
            return reason
        edges, _ = contact_map(query.atomic_numbers, query.positions, self.params["contact_scale"])
        n = query.n_atoms
        adj = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
        if connected_components(adj, directed=False)[0] > 1:
            return "disconnected contact graph"
        return None


@register
class Indigo(ConnectivityAdapter):
    """indigo-bondorder electron assignment (Welsh & Allison 2019) on the contact map. Explicit H only."""

    name = "indigo"
    family = "optimization"
    adapter = "indigo"
    runner = "pixi"
    requires = ("vendor/indigo-bondorder/data", "vendor/build/src/python")
    forwarded = ("algorithm",)
    defaults = {**ConnectivityAdapter.defaults, "algorithm": "fpt"}
    # elements of the default energy tables (data/*_def2svpd.txt)
    capabilities = Capabilities(stripped_h=False, charged=True, radicals=False,
                                elements=frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35}))


@register
class Ball(ConnectivityAdapter):
    """BALL AssignBondOrderProcessor (Dehof et al. 2011) on the contact map. Explicit H only; no formal charges."""

    name = "ball"
    family = "optimization"
    adapter = "ball"
    runner = "pixi"
    requires = ("vendor/ball/data/bond_lengths/BondOrder.xml", "vendor/build")
    forwarded = ("strategy",)
    defaults = {**ConnectivityAdapter.defaults, "strategy": "fpt"}
    # main-group elements of the BALL penalty table (data/bond_lengths/BondOrder.xml)
    capabilities = Capabilities(stripped_h=False, charged=True, radicals=False, elements=_APS_ELEMENTS)
