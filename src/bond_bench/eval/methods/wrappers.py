"""Post-processing wrappers that turn any method into a sound one."""

from __future__ import annotations

import numpy as np

from ..core.canon import pick, tiebreak_objectives, wl_colors
from ..core.graph import Graph
from ..core.ilp import DecodeSpec, Objective, decode
from ..core.validity import is_valid
from ..prediction import Prediction, Status
from ..query import Query
from .base import Wrapper, register


def project_to_valid(
    query: Query,
    edges: np.ndarray,
    target: np.ndarray,
    target_h: np.ndarray | None = None,
    table: str = "octet",
    c_max: int = 1,
    max_pool: int = 64,
    time_limit: float = 30.0,
    keep_support: bool = True,
) -> tuple[Graph | None, list[Graph], str]:
    """Hamming-nearest point of 𝒢 to a (possibly fractional) bond-order guess.

    Minimizes Σ|k − B̂_ij| + Σ|q_i| (+ Σ|η_i − ĥ_i| over atoms with a finite
    ĥ_i when H are not observed)
    over 𝒢 ∩ {supp B = E} (``keep_support``) or 𝒢 ∩ {supp B ⊆ E}. Returns
    (point, co-optimal pool, failure reason).
    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float64)
    w = -np.abs(np.arange(4.0)[None, :] - target[:, None])
    hyd = np.zeros(query.n_atoms, dtype=np.int64) if query.hydrogens_observed else None
    if hyd is None and target_h is not None:
        th = np.nan_to_num(np.asarray(target_h, dtype=np.float64), nan=-1.0)
        atom = lambda i, st, eta: -abs(st.charge) - (abs(eta - th[i]) if th[i] >= 0 else 0.0)  # noqa: E731
    else:  # η is free: it fills whatever valence the bond orders leave
        atom = lambda i, st, eta: -abs(st.charge)  # noqa: E731
    allowed = np.ones((len(edges), 4), dtype=bool)
    allowed[:, 0] = not keep_support
    spec = DecodeSpec(
        query.atomic_numbers, query.total_charge, edges, allowed=allowed, hydrogens=hyd,
        radicals=query.unpaired_electrons, table=table, c_max=c_max, time_limit=time_limit,
    )
    objective = Objective(edge=w, atom=atom, name="projection")
    res = decode(spec, [objective], pool="all", max_pool=max_pool)
    if res.status != "ok":
        return None, [], f"{res.status}: {res.reason}"
    pool = [s.graph for s in res.solutions]
    colors = wl_colors(query.atomic_numbers, query.positions, edges)
    candidates = pool
    if res.truncated:
        tied = decode(spec, [objective, *tiebreak_objectives(edges, query.positions, colors)], pool="all", max_pool=max_pool)
        candidates = [s.graph for s in tied.solutions] or pool  # the tie-break stages can time out
    return pick(candidates, colors, query.positions), pool, ""


@register
class Projected(Wrapper):
    """R step 3 applied to any method: map invalid outputs to the nearest valid structure."""

    name = "project"
    family = "wrapper"
    defaults = {"table": "lewis", "c_max": 1, "keep_support": True}

    @property
    def capabilities(self):  # type: ignore[override]
        return self.inner.capabilities

    def finish(self, query: Query, pred: Prediction) -> Prediction:
        if pred.status is not Status.OK or pred.graph is None:
            return pred
        g = pred.graph
        if is_valid(query.atomic_numbers, query.total_charge, g):
            return pred
        best, pool, reason = project_to_valid(
            query, g.bonds, g.orders, None if query.hydrogens_observed else g.hydrogens,
            table=self.params["table"], c_max=self.params["c_max"], keep_support=self.params["keep_support"],
        )
        if best is None:
            return Prediction.fail(Status.INFEASIBLE, f"projection {reason}", runtime_s=pred.runtime_s)
        return Prediction.ok(best, alternatives=pool if len(pool) > 1 else [], flags=[*pred.flags, "projected"])
