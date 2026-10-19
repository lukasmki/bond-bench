"""P3 validity oracle: is G a valid Lewis structure for (z, Q)?"""

from __future__ import annotations

import numpy as np

from .elements import AtomState, states
from .graph import Graph

#: Λ used by the harness to judge predictions: the broad Lewis table.
ORACLE_TABLE = "lewis"
ORACLE_C_MAX = 3
ORACLE_R_MAX = 2


def check(
    z: np.ndarray,
    total_charge: int,
    g: Graph,
    table: str = ORACLE_TABLE,
    c_max: int = ORACLE_C_MAX,
    r_max: int = ORACLE_R_MAX,
) -> str | None:
    """Return ``None`` if ``g`` ∈ 𝒢(z, Q), else the first violated condition."""
    z = np.asarray(z)
    if g.n != len(z):
        return "atom count mismatch"
    if len(g.bonds):
        if (g.bonds[:, 0] == g.bonds[:, 1]).any():
            return "self bond"
        if g.bonds.min() < 0 or g.bonds.max() >= g.n:
            return "bond index out of range"
        if len({tuple(b) for b in g.bonds.tolist()}) != len(g.bonds):
            return "duplicate bond"
        if not np.isin(g.orders, (1, 2, 3)).all():
            return "bond order outside {1,2,3}"
    if (g.hydrogens < 0).any() or (g.radicals < 0).any():
        return "negative h or r"
    if int(g.charges.sum()) != int(total_charge):
        return "charge sum"
    d = g.valences()
    for i in range(g.n):
        allowed = states(int(z[i]), table, c_max, r_max)
        if not allowed:
            return f"unsupported element {int(z[i])}"
        if AtomState(int(g.charges[i]), int(d[i]), int(g.radicals[i])) not in allowed:
            return f"atom {i} state outside Λ"
    return None


def is_valid(z: np.ndarray, total_charge: int, g: Graph, **kw) -> bool:
    return check(z, total_charge, g, **kw) is None
