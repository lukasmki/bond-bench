"""Canonical H: constrained MAP over 𝒢 under learned edge probabilities (§4.4).

``constrained_map`` decodes any factorized posterior q(B_ij = k | X) with P4.
:class:`Hybrid` wraps a method that reports ``edge_pairs``/``edge_probs`` (an
ML model's per-pair softmax) and replaces its entrywise argmax with the
constrained MAP.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..core.canon import pick, tiebreak_objectives, wl_colors
from ..core.elements import AtomState
from ..core.graph import Graph
from ..core.ilp import DecodeSpec, Objective, decode
from ..prediction import Prediction, Status
from ..query import Query
from .base import Wrapper, register

EPS = 1e-12


def constrained_map(
    query: Query,
    pairs: np.ndarray,
    edge_logp: np.ndarray,
    atom_logp: Callable[[int, AtomState, int], float] | None = None,
    allowed: np.ndarray | None = None,
    top_k: int = 16,
    alpha: float = 0.01,
    table: str = "octet",
    c_max: int = 1,
    time_limit: float = 30.0,
) -> Prediction:
    """argmax_{G ∈ 𝒢, supp B ⊆ E} Σ w_ij(B_ij) + Σ u_i(state_i)."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    spec = DecodeSpec(
        query.atomic_numbers, query.total_charge, pairs, allowed=allowed,
        hydrogens=np.zeros(query.n_atoms, dtype=np.int64) if query.hydrogens_observed else None,
        radicals=query.unpaired_electrons, table=table, c_max=c_max, time_limit=time_limit,
    )
    obj = Objective(edge=np.asarray(edge_logp, dtype=np.float64), atom=atom_logp, name="log q")
    ties = decode(spec, [obj], pool="all")
    if ties.status != "ok":
        return Prediction.fail(Status(ties.status), ties.reason)
    colors = wl_colors(query.atomic_numbers, query.positions, pairs)
    if ties.truncated:
        ties = decode(spec, [obj, *tiebreak_objectives(pairs, query.positions, colors)], pool="all")
    best = pick([s.graph for s in ties.solutions], colors, query.positions)
    ranked = decode(spec, [obj], pool=top_k)
    scores = np.array([s.values[-1] for s in ranked.solutions])
    top = scores[0]
    keep = [s for s, v in zip(ranked.solutions, scores) if top - v <= np.log(1 / alpha)]
    kernel = np.exp(scores - top)
    kernel /= kernel.sum()
    gap = float(top - scores[1]) if len(scores) > 1 else float("inf")
    flags = ["prior_tied"] if len(ties.solutions) > 1 else []
    return Prediction.ok(
        best,
        alternatives=[s.graph for s in keep] if len(keep) > 1 else [],
        flags=flags,
        extra={"score": float(top), "gap": gap, "kernel": kernel.round(6).tolist()},
    )


def graph_from_argmax(n: int, pairs: np.ndarray, probs: np.ndarray) -> Graph:
    """The M decoder: entrywise argmax (Hamming-loss Bayes estimator over 𝓜)."""
    return Graph(n, pairs, np.asarray(probs).argmax(axis=1), np.zeros(n), np.zeros(n), np.zeros(n))


@register
class Hybrid(Wrapper):
    """Constrained MAP decoding of an inner method's edge probabilities."""

    name = "hybrid"
    family = "hybrid"
    defaults = {"top_k": 16, "alpha": 0.01, "charge_penalty": 0.0, "table": "octet", "c_max": 1}

    def predict(self, query: Query) -> Prediction:
        inner = self.inner.predict(query)
        if inner.status is not Status.OK or inner.edge_probs is None:
            if inner.status is Status.OK:
                return Prediction.fail(Status.ERROR, f"{self.inner.name} reports no edge probabilities")
            return inner
        p = self.params
        lam = float(p["charge_penalty"])
        out = constrained_map(
            query, inner.edge_pairs, np.log(np.clip(inner.edge_probs, EPS, 1.0)),
            atom_logp=(lambda i, st, eta: -lam * abs(st.charge)) if lam else None,
            top_k=p["top_k"], alpha=p["alpha"], table=p["table"], c_max=p["c_max"],
        )
        if out.graph is not None and inner.graph is not None:
            a, b = out.graph.bond_dict(), inner.graph.bond_dict()
            out.extra["hamming_to_argmax"] = sum(a.get(k, 0) != b.get(k, 0) for k in set(a) | set(b))
        out.edge_pairs, out.edge_probs = inner.edge_pairs, inner.edge_probs
        return out
