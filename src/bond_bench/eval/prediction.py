"""Method outputs and their (de)serialization to flat rows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from .core.graph import Graph


class Status(StrEnum):
    OK = "ok"
    INFEASIBLE = "infeasible"  # ⊥: 𝒢(z, Q) ∩ {supp B ⊆ E} is empty
    UNSUPPORTED = "unsupported"  # the method lacks a needed capability
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class Prediction:
    status: Status
    graph: Graph | None = None
    reason: str = ""
    #: co-optimal or top-K structures (set-valued output); the point estimate is ``graph``.
    alternatives: list[Graph] = field(default_factory=list)
    #: probabilistic output: candidate pairs (m, 2) and P(B_ij = k), (m, 4).
    edge_pairs: np.ndarray | None = None
    edge_probs: np.ndarray | None = None
    #: per-bond confidence margin aligned with ``graph.bonds`` (method-specific units).
    margin: np.ndarray | None = None
    flags: list[str] = field(default_factory=list)
    runtime_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, graph: Graph, **kw) -> Prediction:
        return cls(Status.OK, graph, **kw)

    @classmethod
    def fail(cls, status: Status, reason: str, **kw) -> Prediction:
        return cls(Status(status), None, reason=reason, **kw)


def _graph_to_json(g: Graph) -> dict:
    return {
        "n": g.n,
        "bonds": g.bonds.tolist(),
        "orders": g.orders.tolist(),
        "charges": g.charges.tolist(),
        "hydrogens": g.hydrogens.tolist(),
        "radicals": g.radicals.tolist(),
    }


def _graph_from_json(d: dict) -> Graph:
    return Graph(d["n"], d["bonds"], d["orders"], d["charges"], d["hydrogens"], d["radicals"])


def to_row(structure_id: str, p: Prediction) -> dict[str, Any]:
    g = p.graph
    return {
        "structure_id": structure_id,
        "status": str(p.status),
        "reason": p.reason,
        "runtime_s": float(p.runtime_s),
        "bonds": g.bonds.reshape(-1).tolist() if g else None,
        "bond_orders": g.orders.tolist() if g else None,
        "formal_charges": g.charges.tolist() if g else None,
        "implicit_hydrogens": g.hydrogens.tolist() if g else None,
        "radical_electrons": g.radicals.tolist() if g else None,
        "margin": p.margin.tolist() if p.margin is not None else None,
        "n_alternatives": len(p.alternatives),
        "alternatives": json.dumps([_graph_to_json(a) for a in p.alternatives]) if p.alternatives else "",
        "edge_pairs": p.edge_pairs.reshape(-1).tolist() if p.edge_pairs is not None else None,
        "edge_probs": p.edge_probs.reshape(-1).tolist() if p.edge_probs is not None else None,
        "flags": list(p.flags),
        "extra": json.dumps(p.extra, default=str) if p.extra else "",
    }


def from_row(row: dict[str, Any], n_atoms: int) -> Prediction:
    graph = None
    if row.get("bond_orders") is not None:
        graph = Graph(
            n_atoms,
            np.asarray(row["bonds"], dtype=np.int64).reshape(-1, 2),
            row["bond_orders"],
            row["formal_charges"],
            row["implicit_hydrogens"],
            row["radical_electrons"],
        )
    alts = [_graph_from_json(d) for d in json.loads(row["alternatives"])] if row.get("alternatives") else []
    pairs = row.get("edge_pairs")
    probs = row.get("edge_probs")
    return Prediction(
        status=Status(row["status"]),
        graph=graph,
        reason=row.get("reason") or "",
        alternatives=alts,
        edge_pairs=np.asarray(pairs, dtype=np.int64).reshape(-1, 2) if pairs is not None else None,
        edge_probs=np.asarray(probs, dtype=np.float64).reshape(-1, 4) if probs is not None else None,
        margin=np.asarray(row["margin"]) if row.get("margin") is not None else None,
        flags=list(row.get("flags") or []),
        runtime_s=float(row.get("runtime_s") or 0.0),
        extra=json.loads(row["extra"]) if row.get("extra") else {},
    )
