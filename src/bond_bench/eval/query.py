"""What a method sees: a labelled point cloud, never the reference label."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from ..data.schema import MoleculeRecord


class Observation(StrEnum):
    EXPLICIT_H = "explicit_h"  # full 3D with H
    STRIPPED = "stripped"  # π_H(X): hydrogens deleted, counts become unknowns


@dataclass(frozen=True)
class Query:
    structure_id: str
    atomic_numbers: np.ndarray  # (n,)
    positions: np.ndarray  # (n, 3) Å
    total_charge: int
    multiplicity: int
    observation: Observation

    @classmethod
    def from_record(cls, record: MoleculeRecord, observation: Observation | str) -> Query:
        observation = Observation(observation)
        if observation is Observation.STRIPPED:
            record = record.strip_hydrogens()
        elif (record.implicit_hydrogens > 0).any():
            raise ValueError(f"{record.structure_id}: record has no explicit hydrogens")
        return cls(
            structure_id=record.structure_id,
            atomic_numbers=record.atomic_numbers.astype(np.int64).copy(),
            positions=record.positions.astype(np.float64).copy(),
            total_charge=int(record.total_charge),
            multiplicity=int(record.multiplicity),
            observation=observation,
        )

    @property
    def n_atoms(self) -> int:
        return len(self.atomic_numbers)

    @property
    def hydrogens_observed(self) -> bool:
        return self.observation is Observation.EXPLICIT_H

    @property
    def unpaired_electrons(self) -> int:
        return max(self.multiplicity - 1, 0)

    def transformed(
        self, perm: np.ndarray, rotation: np.ndarray, translation: np.ndarray
    ) -> Query:
        """(σ, g)·(z, X): new index of atom i is perm[i]; x ↦ R x + t (R may be improper)."""
        inv = np.argsort(perm)
        x = self.positions[inv] @ np.asarray(rotation).T + np.asarray(translation)
        return replace(self, atomic_numbers=self.atomic_numbers[inv], positions=x)

    def noisy(self, sigma: float, rng: np.random.Generator) -> Query:
        return replace(self, positions=self.positions + rng.normal(0.0, sigma, self.positions.shape))


def random_euclidean(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random element of E(3) (including reflections) as (R, t)."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if rng.random() < 0.5:
        q[:, 0] = -q[:, 0]
    return q, rng.normal(scale=5.0, size=3)
