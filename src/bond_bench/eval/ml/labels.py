"""Candidate pairs and training targets of canonical M (NumPy only).

A target holds the labels of one structure in one observation regime for
every member of its label class: the reference graph plus the structures that
differ from it only by labelling convention (same supp B and h, minimal Σ|q|;
Kekulé alternatives, the two nitro or carboxylate placements, ...). The
resonance-aware loss takes the minimum NLL over the members.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ...data.schema import MoleculeRecord
from ..core.elements import SUPPORTED_ELEMENTS
from ..core.geometry import contact_map
from ..core.graph import Graph
from ..core.ilp import DecodeSpec, Objective, decode
from ..prediction import ATOM_CHARGES, ATOM_HYDROGENS
from ..query import Observation

#: A6 per element pair: r_c(a, b) = 1.5 (r_a + r_b), above the contact tolerance τ_ab = 1.3 (r_a + r_b).
CUTOFF_SCALE = 1.5
#: Label of an entry the loss skips (ignored by cross-entropy).
IGNORE = -100
N_ATOM_CLASSES = len(ATOM_CHARGES) * len(ATOM_HYDROGENS)


def candidate_pairs(z: np.ndarray, positions: np.ndarray, scale: float = CUTOFF_SCALE) -> np.ndarray:
    """E: pairs i < j with d_ij ≤ scale · (r_i + r_j), sorted (a superset of the true bonds)."""
    return contact_map(z, positions, scale)[0]


def atom_class(charge: int, hydrogens: int) -> int:
    """Flat index into the (ATOM_CHARGES × ATOM_HYDROGENS) grid, IGNORE when outside it."""
    if charge not in ATOM_CHARGES or hydrogens not in ATOM_HYDROGENS:
        return IGNORE
    return ATOM_CHARGES.index(charge) * len(ATOM_HYDROGENS) + ATOM_HYDROGENS.index(hydrogens)


def pair_keys(pairs: np.ndarray, n: int) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return pairs[:, 0] * n + pairs[:, 1]


@dataclass
class Target:
    structure_id: str
    observation: str
    z: np.ndarray  # (n,)
    positions: np.ndarray  # (n, 3)
    total_charge: int
    bond_keys: np.ndarray  # (b,) sorted i·n + j of the true bonds (shared by all members)
    bond_orders: np.ndarray  # (M, b) order of each true bond, per member
    atom_labels: np.ndarray  # (M, n) atom_class(q_i, h_i), per member

    @property
    def n(self) -> int:
        return len(self.z)

    def edge_labels(self, pairs: np.ndarray) -> np.ndarray:
        """(M, m) order of each candidate pair per member (0 for non-bonds)."""
        keys = pair_keys(pairs, self.n)
        out = np.zeros((len(self.bond_orders), len(keys)), dtype=np.int64)
        if not len(self.bond_keys):
            return out
        pos = np.clip(np.searchsorted(self.bond_keys, keys), 0, len(self.bond_keys) - 1)
        hit = self.bond_keys[pos] == keys
        out[:, hit] = self.bond_orders[:, pos[hit]]
        return out

    def missed_bonds(self, pairs: np.ndarray) -> int:
        """True bonds outside the candidate set (unpredictable for this geometry)."""
        return int((~np.isin(self.bond_keys, pair_keys(pairs, self.n))).sum())


def trainable(record: MoleculeRecord) -> str | None:
    """Why ``record`` cannot be a training sample, or None."""
    if record.multiplicity > 1 or record.radical_electrons.any():
        return "open shell"
    if not set(np.unique(record.atomic_numbers).tolist()) <= SUPPORTED_ELEMENTS:
        return "unsupported elements"
    if not np.isin(record.bond_orders, (1, 2, 3)).all():
        return "bond orders outside {1,2,3}"
    return None


def label_class(record: MoleculeRecord, max_pool: int = 64, time_limit: float = 30.0) -> list[Graph]:
    """The reference graph, then the other minimal-Σ|q| octet structures of its resonance class.

    The ``octet`` state table keeps sextet forms (e.g. O⁺ with one bond) out of
    the class, and members more charge-separated than the reference are
    dropped, so a neutral S(IV)=O label is not paired with S⁺–O⁻.
    """
    g = Graph.from_record(record)
    allowed = np.ones((len(g.bonds), 4), dtype=bool)
    allowed[:, 0] = False
    spec = DecodeSpec(
        record.atomic_numbers, record.total_charge, g.bonds, allowed=allowed, hydrogens=g.hydrogens,
        table="octet", c_max=max(1, int(np.abs(g.charges).max(initial=0))), time_limit=time_limit,
    )
    res = decode(spec, [Objective(atom=lambda i, st, eta: -abs(st.charge), name="min Σ|q|")], pool="all", max_pool=max_pool)
    limit = int(np.abs(g.charges).sum())
    return [g, *(s.graph for s in res.solutions if s.graph != g and int(np.abs(s.graph.charges).sum()) <= limit)]


def _as_record(record: MoleculeRecord, g: Graph) -> MoleculeRecord:
    return replace(
        record, bonds=g.bonds, bond_orders=g.orders, aromatic=np.zeros(len(g.orders), dtype=bool),
        formal_charges=g.charges,
    )


def targets(
    record: MoleculeRecord,
    regimes: tuple[str, ...] = (Observation.EXPLICIT_H, Observation.STRIPPED),
    resonance: bool = True,
    max_pool: int = 64,
) -> list[Target]:
    """One target per observation regime the record supports."""
    members = label_class(record, max_pool) if resonance else [Graph.from_record(record)]
    out = []
    for regime in map(Observation, regimes):
        if regime is Observation.EXPLICIT_H and (record.implicit_hydrogens > 0).any():
            continue  # no H coordinates to observe
        graphs = []
        for g in members:
            rec = _as_record(record, g)
            graphs.append(Graph.from_record(rec.strip_hydrogens() if regime is Observation.STRIPPED else rec))
        ref = graphs[0]
        base = record.strip_hydrogens() if regime is Observation.STRIPPED else record
        out.append(Target(
            structure_id=record.structure_id,
            observation=str(regime),
            z=base.atomic_numbers.astype(np.int64),
            positions=base.positions.astype(np.float64),
            total_charge=int(record.total_charge),
            bond_keys=pair_keys(ref.bonds, ref.n),
            bond_orders=np.stack([g.orders for g in graphs]).astype(np.int64),
            atom_labels=np.array(
                [[atom_class(int(q), int(h)) for q, h in zip(g.charges, g.hydrogens)] for g in graphs], dtype=np.int64
            ).reshape(len(graphs), ref.n),
        ))
    return out
