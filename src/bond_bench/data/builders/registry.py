"""Map stratum ids to builders."""

from __future__ import annotations

from .a1_pdb import A1Builder
from .a2_cod import A2Builder
from .b1_qmugs import B1Builder
from .b2_geom_drugs import B2Builder
from .b3_small import B3Builder
from .b5_tautomers import B5Builder
from .base import StratumBuilder
from .c1_noise import C1Builder
from .c5_offeq import C5Builder

BUILDERS: dict[str, type[StratumBuilder]] = {
    b.stratum_id: b for b in (A1Builder, A2Builder, B1Builder, B2Builder, B3Builder, B5Builder, C1Builder, C5Builder)
}


def get_builder(stratum_id: str) -> type[StratumBuilder]:
    try:
        return BUILDERS[stratum_id]
    except KeyError:
        raise NotImplementedError(f"no builder implemented for stratum {stratum_id}") from None


def is_implemented(stratum_id: str) -> bool:
    return stratum_id in BUILDERS
