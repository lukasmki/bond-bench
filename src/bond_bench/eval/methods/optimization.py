"""Canonical O: coarse-grain the geometry to A(X), then select inside the fibre (§4.2).

Geometry enters only through the contact map. The prior is a lexicographic
list of linear objectives solved exactly by P4; all co-optimal structures are
returned and the point estimate is chosen equivariantly (P5).
"""

from __future__ import annotations

import numpy as np

from ..core.canon import pick, tiebreak_objectives, wl_colors
from ..core.geometry import CONTACT_SCALE, contact_map
from ..core.ilp import DecodeSpec, Objective, decode
from ..prediction import Prediction, Status
from ..query import Query
from ..tables import atom_penalty_fn
from .base import Capabilities, Method, register


def prior_objectives(preset: str, n_edges: int, z: np.ndarray, degree: np.ndarray) -> list[Objective]:
    if preset == "kim2015":
        return [
            Objective(edge=np.tile(np.arange(4.0), (n_edges, 1)), name="max ΣB"),
            Objective(atom=lambda i, st, eta: -abs(st.charge), name="min Σ|q|"),
        ]
    if preset == "atom_penalty":
        ap = atom_penalty_fn()
        return [
            Objective(atom=lambda i, st, eta: -ap(int(z[i]), int(degree[i] + eta), st), name="min ΣAP"),
            Objective(atom=lambda i, st, eta: -abs(st.charge), name="min Σ|q|"),
        ]
    raise ValueError(f"unknown preset {preset!r}")


@register
class CanonicalO(Method):
    name = "canonical-O"
    family = "optimization"
    capabilities = Capabilities(stripped_h=True, charged=True, radicals=True)
    defaults = {
        "preset": "kim2015",
        "table": "octet",
        "c_max": 1,
        "contact_scale": CONTACT_SCALE,
        "max_pool": 64,
        "time_limit": 30.0,
    }

    def predict(self, query: Query) -> Prediction:
        p = self.params
        z = query.atomic_numbers
        edges, margin = contact_map(z, query.positions, p["contact_scale"])
        allowed = np.ones((len(edges), 4), dtype=bool)
        allowed[:, 0] = False  # fibre constraint: supp B = A(X)
        spec = DecodeSpec(
            z,
            query.total_charge,
            edges,
            allowed=allowed,
            hydrogens=np.zeros(query.n_atoms, dtype=np.int64) if query.hydrogens_observed else None,
            radicals=query.unpaired_electrons,
            table=p["table"],
            c_max=p["c_max"],
            time_limit=p["time_limit"],
        )
        degree = np.bincount(edges.reshape(-1), minlength=query.n_atoms)
        prior = prior_objectives(p["preset"], len(edges), z, degree)
        res = decode(spec, prior, pool="all", max_pool=p["max_pool"])
        if res.status != "ok":
            return Prediction.fail(Status(res.status), res.reason)
        pool = [s.graph for s in res.solutions]
        colors = wl_colors(z, query.positions, edges)
        if res.truncated:  # pick inside the full co-optimal set, not the truncated pool
            tied = decode(spec, [*prior, *tiebreak_objectives(edges, query.positions, colors)], pool="all", max_pool=p["max_pool"])
            best = pick([s.graph for s in tied.solutions], colors, query.positions)
        else:
            best = pick(pool, colors, query.positions)
        flags = []
        if len(pool) > 1:
            flags.append("prior_tied")
        if res.truncated:
            flags.append("pool_truncated")
        bond_margin = dict(zip(map(tuple, edges.tolist()), margin))
        return Prediction.ok(
            best,
            alternatives=pool if len(pool) > 1 else [],
            margin=np.array([bond_margin[tuple(b)] for b in best.bonds.tolist()]),
            flags=flags,
            extra={"objective": list(res.solutions[0].values)},
        )
