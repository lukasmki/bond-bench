"""Element state tables Λ(a) (assumption A0) and covalent radii (A1).

An atom state is ``(charge, valence, radicals)`` where valence is the bond
valence D_i = (B·1)_i + h_i. The Lewis conditions of the formalism (§2) give

    ν − c − d − r ≥ 0,   ν − c − d − r even,   ν − c + d ≤ 2·o,   |c| ≤ c_max

with ν the valence-electron count and o the shell capacity (in electron
pairs). Two tables are provided:

- ``lewis``: the conditions above with expanded shells (o = 6) for period ≥ 3.
  This is the broad definition of a valid Lewis structure and backs the
  harness validity oracle.
- ``octet``: exactly one valence per (charge, radicals), that of the
  isoelectronic octet (or duet) configuration, plus the hypervalent valences
  of the xyz2mol table (P(V), S(VI), …). It excludes carbenes, nitrenes and the
  spurious S(IV)/Cl(III) states that a bond-maximizing prior would otherwise
  pick (e.g. thiophene S=C).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from rdkit import Chem

_PT = Chem.GetPeriodicTable()

#: Main-group elements the canonical methods support (H, B–F, Al–Cl, Ga–Br, In–I).
SUPPORTED_ELEMENTS = frozenset({1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 31, 32, 33, 34, 35, 49, 50, 51, 52, 53})

# Neutral hypervalent valences added to the ``octet`` table (xyz2mol-style).
_HYPERVALENT = {
    15: (5,), 33: (5,), 51: (5,),
    16: (6,), 34: (6,), 52: (6,),
    17: (7,), 35: (3, 5, 7), 53: (3, 5, 7),
}


@dataclass(frozen=True, order=True)
class AtomState:
    charge: int
    valence: int
    radicals: int = 0


def period(z: int) -> int:
    for p, last in enumerate((2, 10, 18, 36, 54, 86, 118), start=1):
        if z <= last:
            return p
    raise ValueError(z)


def valence_electrons(z: int) -> int:
    return int(_PT.GetNOuterElecs(z))


def shell_capacity(z: int, expanded: bool) -> int:
    if z <= 2:
        return 1
    return 6 if expanded and period(z) >= 3 else 4


def octet_valence(z: int, charge: int, radicals: int = 0) -> int | None:
    """Valence of the closed-shell (octet/duet) configuration; None if impossible."""
    e = valence_electrons(z) - charge
    if z <= 2:
        d = 1 if e == 1 else 0 if e in (0, 2) else None
    elif 0 <= e <= 4:
        d = e
    elif 4 < e <= 8:
        d = 8 - e
    else:
        d = None
    if d is None or d - radicals < 0:
        return None
    return d - radicals


def _lewis_ok(nu: int, o: int, c: int, d: int, r: int) -> bool:
    nb = nu - c - d - r
    return nb >= 0 and nb % 2 == 0 and nu - c + d <= 2 * o


@lru_cache(maxsize=None)
def states(z: int, table: str = "lewis", c_max: int = 1, r_max: int = 0) -> tuple[AtomState, ...]:
    """Admissible states of element ``z``, sorted (charge, valence, radicals)."""
    if z not in SUPPORTED_ELEMENTS:
        return ()
    nu = valence_electrons(z)
    out = set()
    if table == "lewis":
        o = shell_capacity(z, expanded=True)
        for c in range(-c_max, c_max + 1):
            for r in range(r_max + 1):
                for d in range(0, 2 * o + 1):
                    if _lewis_ok(nu, o, c, d, r):
                        out.add(AtomState(c, d, r))
    elif table == "octet":
        for c in range(-c_max, c_max + 1):
            for r in range(r_max + 1):
                d = octet_valence(z, c, r)
                if d is not None and _lewis_ok(nu, shell_capacity(z, False), c, d, r):
                    out.add(AtomState(c, d, r))
        for d in _HYPERVALENT.get(z, ()):
            out.add(AtomState(0, d, 0))
    else:
        raise ValueError(f"unknown state table {table!r}")
    return tuple(sorted(out))


def max_hydrogens(z: int, table: str = "lewis", c_max: int = 1) -> int:
    """Upper bound on implicit H for the H-free regime."""
    if z == 1:
        return 0
    vals = [s.valence for s in states(z, table, c_max)]
    return min(4, max(vals)) if vals else 0


def covalent_radius(z: int) -> float:
    """Covalent radius in Å (RDKit table, as used by xyz2mol/DetermineBonds)."""
    return float(_PT.GetRcovalent(z))
