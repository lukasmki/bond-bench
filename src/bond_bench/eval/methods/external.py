"""Methods that run as a separate process in their own environment.

Protocol (one JSON object per line, UTF-8):

input  ``{"structure_id", "atomic_numbers", "positions", "total_charge",
          "multiplicity", "observation"[, "contacts"]}``
output ``{"structure_id", "status", "reason", "atomic_numbers", "bonds",
          "bond_orders", "formal_charges", "implicit_hydrogens",
          "radical_electrons", "runtime_s"[, "edge_pairs", "edge_probs"]}``

The command gets the input and output paths through ``{input}`` and
``{output}``. Atom order must be preserved; ``atomic_numbers`` is echoed back
and checked. ``contacts`` (atom-index pairs of the contact map A(X), as used by
``canonical-O``) is sent only to methods that set ``send_contacts``: those that
assign bond orders to a given connectivity rather than perceive it.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np

from ...data.manifest import REPO_ROOT
from ..core.geometry import CONTACT_SCALE, contact_map
from ..core.graph import Graph
from ..prediction import Prediction, Status
from ..query import Query
from .base import Capabilities, Method, register


def query_to_json(q: Query, contact_scale: float | None = None) -> dict:
    """One protocol input line; ``contact_scale`` set adds the contact map."""
    d = {
        "structure_id": q.structure_id,
        "atomic_numbers": q.atomic_numbers.tolist(),
        "positions": np.round(q.positions, 6).tolist(),
        "total_charge": q.total_charge,
        "multiplicity": q.multiplicity,
        "observation": str(q.observation),
    }
    if contact_scale is not None:
        d["contacts"] = contact_map(q.atomic_numbers, q.positions, contact_scale)[0].tolist()
    return d


def prediction_from_json(d: dict, q: Query) -> Prediction:
    status = Status(d.get("status", "ok"))
    if status is not Status.OK:
        return Prediction.fail(status, d.get("reason", ""), runtime_s=float(d.get("runtime_s", 0.0)))
    if list(d.get("atomic_numbers", [])) != q.atomic_numbers.tolist():
        return Prediction.fail(Status.ERROR, "adapter changed atom order or count")
    n = q.n_atoms
    hyd = d.get("implicit_hydrogens") or [0] * n
    if q.hydrogens_observed:
        hyd = [0] * n
    graph = Graph(
        n,
        np.asarray(d["bonds"], dtype=np.int64).reshape(-1, 2),
        d["bond_orders"],
        d.get("formal_charges") or [0] * n,
        hyd,
        d.get("radical_electrons") or [0] * n,
    )
    pred = Prediction.ok(graph, runtime_s=float(d.get("runtime_s", 0.0)), flags=list(d.get("flags", [])))
    if d.get("edge_probs") is not None:
        pred.edge_pairs = np.asarray(d["edge_pairs"], dtype=np.int64).reshape(-1, 2)
        pred.edge_probs = np.asarray(d["edge_probs"], dtype=np.float64).reshape(-1, 4)
    return pred


@register
class ExternalMethod(Method):
    """Generic subprocess method: ``--param command='prog {input} {output}'``."""

    name = "external"
    family = "external"
    capabilities = Capabilities(stripped_h=True, charged=True, radicals=True, elements=None)
    defaults = {"command": "", "batch_timeout": 3600.0, "cwd": str(REPO_ROOT), "label": ""}
    #: send the contact map A(X) with each query (params must then include ``contact_scale``)
    send_contacts: ClassVar[bool] = False

    @property
    def tag(self) -> str:
        return self.params["label"] or super().tag

    def command(self, input_path: Path, output_path: Path) -> list[str]:
        template = self.params["command"]
        if not template:
            raise ValueError(f"{self.name}: no command configured")
        return [part.format(input=input_path, output=output_path) for part in shlex.split(template)]

    def check_available(self) -> str | None:
        """None if runnable, else why not."""
        exe = shlex.split(self.params["command"])[:1]
        if not exe:
            return "no command configured"
        return None if shutil.which(exe[0]) else f"{exe[0]} not on PATH"

    def predict(self, query: Query) -> Prediction:
        return self.predict_batch([query])[0]

    def predict_batch(self, queries: Sequence[Query]) -> list[Prediction]:
        with tempfile.TemporaryDirectory(prefix="bond-bench-") as tmp:
            inp, out = Path(tmp) / "queries.jsonl", Path(tmp) / "predictions.jsonl"
            scale = self.params.get("contact_scale", CONTACT_SCALE) if self.send_contacts else None
            with inp.open("w") as f:
                for q in queries:
                    f.write(json.dumps(query_to_json(q, scale)) + "\n")
            try:
                proc = subprocess.run(
                    self.command(inp, out), cwd=self.params["cwd"], capture_output=True, text=True,
                    timeout=self.params["batch_timeout"],
                )
            except subprocess.TimeoutExpired:
                return [Prediction.fail(Status.TIMEOUT, "batch timeout") for _ in queries]
            if proc.returncode != 0 or not out.exists():
                msg = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
                return [Prediction.fail(Status.ERROR, f"exit {proc.returncode}: {msg[0]}") for _ in queries]
            results = {}
            for line in out.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    results[d["structure_id"]] = d
        preds = []
        for q in queries:
            d = results.get(q.structure_id)
            if d is None:
                preds.append(Prediction.fail(Status.ERROR, "missing from adapter output"))
                continue
            try:
                preds.append(prediction_from_json(d, q))
            except (KeyError, ValueError) as exc:
                preds.append(Prediction.fail(Status.ERROR, f"bad adapter output: {exc}"))
        return preds

