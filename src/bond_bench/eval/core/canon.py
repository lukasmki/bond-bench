"""P5 equivariant tie-breaking and P6 canonicalization κ.

Atom colors come from Weisfeiler–Lehman refinement on (z, binned Δ), so
they depend only on the geometry orbit, never on atom indices. A structure's
canonical key is the sorted multiset of its bonds and atom labels written in
colors; picking the minimal key among co-optimal structures is equivariant.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np

from .graph import Graph

DISTANCE_BIN = 0.001  # Å; only exactly symmetric atoms share a color


def _h(*parts: object) -> int:
    digest = hashlib.blake2b(repr(parts).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def wl_colors(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    edges: np.ndarray | None = None,
    iterations: int | None = None,
    distance_bin: float = DISTANCE_BIN,
) -> np.ndarray:
    """Permutation-invariant atom colors: WL refinement on the weighted complete graph.

    Every pair contributes (color_j, binned d_ij), so atoms share a color only
    if their whole distance profile matches (exact symmetry, up to the bin).
    ``edges`` is accepted for call compatibility; refinement does not need it.
    """
    z = np.asarray(atomic_numbers)
    x = np.asarray(positions, dtype=np.float64)
    n = len(z)
    d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    # round first: rigid motions perturb d by ~1e-15, which must not move a bin edge
    bins = np.floor(np.round(d / distance_bin, 6)).astype(np.int64)
    colors = [_h(int(a)) for a in z]
    n_classes = len(set(colors))
    for _ in range(iterations or n):
        new = [
            _h(colors[i], tuple(sorted((colors[j], int(bins[i, j])) for j in range(n) if j != i)))
            for i in range(n)
        ]
        k = len(set(new))
        colors = new
        if k == n_classes:
            break
        n_classes = k
    return np.array(colors, dtype=np.uint64)


def ranks_from_colors(colors: np.ndarray) -> np.ndarray:
    """Dense rank of each atom's color (ties share a rank)."""
    _, inv = np.unique(colors, return_inverse=True)
    return inv.astype(np.int64)


def structure_key(g: Graph, colors: np.ndarray) -> tuple:
    """Index-free key of ``g``: bonds and atom labels written in atom colors."""
    c = [int(v) for v in colors]
    bonds = sorted((min(c[i], c[j]), max(c[i], c[j]), int(k)) for (i, j), k in zip(g.bonds.tolist(), g.orders))
    atoms = sorted((c[i], int(g.charges[i]), int(g.hydrogens[i]), int(g.radicals[i])) for i in range(g.n))
    return (tuple(bonds), tuple(atoms))


def bond_length_score(g: Graph, positions: np.ndarray) -> float:
    """Σ k_ij · d_ij: smaller when multiple bonds sit on shorter contacts (E(3)-invariant)."""
    if not len(g.bonds):
        return 0.0
    x = np.asarray(positions, dtype=np.float64)
    d = np.linalg.norm(x[g.bonds[:, 0]] - x[g.bonds[:, 1]], axis=1)
    return round(float((g.orders * d).sum()), 6)


def geometric_tiebreak(edges: np.ndarray, positions: np.ndarray):
    """P5 as a final ILP stage: maximize −Σ k_ij·d_ij (multiple bonds on shorter contacts).

    Linear in the one-hot order variables and E(3)-invariant, so it breaks
    ties equivariantly without enumerating the co-optimal set.
    """
    from .ilp import Objective

    x = np.asarray(positions, dtype=np.float64)
    edges = np.asarray(edges).reshape(-1, 2)
    d = np.linalg.norm(x[edges[:, 0]] - x[edges[:, 1]], axis=1) if len(edges) else np.zeros(0)
    return Objective(edge=-np.round(d, 4)[:, None] * np.arange(4.0)[None, :], name="P5 Σk·d")


def tiebreak_objectives(edges: np.ndarray, positions: np.ndarray, colors: np.ndarray) -> list:
    """Final lexicographic stages that make the optimum unique up to symmetry.

    Bonds: Σ k·d (geometric). Atom states: Σ ρ_i·q_i, then Σ ρ_i·η_i, with ρ
    the dense rank of the WL color, so charges and implicit H are placed by an
    index-free rule. Atoms with equal colors remain tied (genuine symmetry).
    """
    from .ilp import Objective

    rank = ranks_from_colors(colors).astype(float) + 1.0
    return [
        geometric_tiebreak(edges, positions),
        Objective(atom=lambda i, st, eta: rank[i] * st.charge, name="P5 Σρq"),
        Objective(atom=lambda i, st, eta: rank[i] * eta, name="P5 Σρη"),
    ]


def pick(graphs: Iterable[Graph], colors: np.ndarray, positions: np.ndarray | None = None) -> Graph:
    """P5: equivariant choice among co-optimal structures.

    Order: min Σ|q|, then the colored key, then (with ``positions``) the
    geometric score, which separates symmetry-equivalent keys unless the
    geometry itself is symmetric.
    """
    def key(g: Graph):
        geo = bond_length_score(g, positions) if positions is not None else 0.0
        return (int(np.abs(g.charges).sum()), structure_key(g, colors), geo)

    return min(graphs, key=key)


def same_resonance_class(g: Graph, ref: Graph) -> bool:
    """Same support of B and same h. Validity of ``g`` is checked separately."""
    return (
        g.n == ref.n
        and np.array_equal(g.bonds, ref.bonds)
        and np.array_equal(g.hydrogens, ref.hydrogens)
    )


def resonance_class(
    z: np.ndarray, total_charge: int, g: Graph, max_pool: int = 256, table: str = "lewis", c_max: int = 1
) -> list[Graph]:
    """Enumerate [g]∼: valid structures with supp B and h fixed (P6 step 1)."""
    from .ilp import DecodeSpec, Objective, decode

    allowed = np.ones((len(g.bonds), 4), dtype=bool)
    allowed[:, 0] = False
    spec = DecodeSpec(
        z, total_charge, g.bonds, allowed=allowed, hydrogens=g.hydrogens,
        radicals=int(g.radicals.sum()), table=table, c_max=c_max,
    )
    res = decode(spec, [Objective(name="feasibility")], pool="all", max_pool=max_pool)
    return [s.graph for s in res.solutions]


def canonicalize(z: np.ndarray, total_charge: int, g: Graph, colors: np.ndarray, **kw) -> Graph:
    """κ: the class member with minimal Σ|q|, then minimal colored key."""
    members = resonance_class(z, total_charge, g, **kw)
    return pick(members or [g], colors)
