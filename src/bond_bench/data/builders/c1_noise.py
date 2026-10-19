"""C1: Gaussian coordinate-noise ladder derived from computed strata."""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np

from ..sampling import derive_seed, sample
from ..schema import MoleculeRecord
from ..store import StratumReader
from .base import StratumBuilder

log = logging.getLogger(__name__)


def perturb(record: MoleculeRecord, sigma: float, seed: int) -> np.ndarray:
    """Positions with isotropic N(0, sigma^2) noise on every coordinate."""
    if sigma == 0:
        return record.positions.copy()
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=record.positions.shape)
    return (record.positions.astype(np.float64) + noise).astype(np.float32)


class C1Builder(StratumBuilder):
    stratum_id = "C1"
    defaults = {
        "sigmas": [0.0, 0.05, 0.10, 0.20],
        "parents": {"B1": 0.5, "B4": 0.5},
    }

    def fetch(self) -> None:
        """Parents are earlier build outputs; nothing to download."""

    def _parent_quotas(self, n_parents: int) -> dict[str, int]:
        weights = self.params["parents"]
        available = {s: w for s, w in weights.items() if (self.out_dir / f"{s}.h5").exists()}
        missing = sorted(set(weights) - set(available))
        if not available:
            raise FileNotFoundError(
                f"C1 needs at least one parent stratum built in {self.out_dir}: {sorted(weights)}"
            )
        if missing:
            log.warning("[C1] parent strata %s not built; drawing from %s only", missing, sorted(available))
        total = sum(available.values())
        names = sorted(available)
        quotas = {s: int(n_parents * available[s] / total) for s in names}
        quotas[names[0]] += n_parents - sum(quotas.values())
        return quotas

    def records(self) -> list[MoleculeRecord]:
        sigmas = [float(s) for s in self.params["sigmas"]]
        n_parents = -(-self.n // len(sigmas))
        quotas = self._parent_quotas(n_parents)
        self.extra_attrs["parent_strata"] = quotas

        out: list[MoleculeRecord] = []
        for stratum, quota in quotas.items():
            reader = StratumReader(self.out_dir / f"{stratum}.h5")
            ids = sample(reader.ids(), quota, derive_seed(self.seed, "C1", stratum), key=str)
            if len(ids) < quota:
                log.warning("[C1] %s has only %d structures (wanted %d)", stratum, len(ids), quota)
            self.funnel[f"parents_{stratum}"] = len(ids)
            for parent_id in ids:
                parent = reader.get(parent_id)
                for sigma in sigmas:
                    positions = perturb(parent, sigma, derive_seed(self.seed, parent_id, sigma))
                    rmsd = float(np.sqrt(((positions - parent.positions) ** 2).sum(axis=1).mean()))
                    out.append(
                        replace(
                            parent,
                            structure_id=f"C1-{parent_id}-s{sigma:.2f}",
                            source_id=parent.source_id,
                            positions=positions,
                            meta={
                                "parent_stratum": stratum,
                                "parent_id": parent_id,
                                "sigma": sigma,
                                "rmsd": rmsd,
                            },
                        )
                    )
        self.funnel["written"] = len(out)
        return out[: self.n] if len(out) > self.n else out
