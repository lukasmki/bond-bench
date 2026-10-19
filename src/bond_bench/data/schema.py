"""In-memory record format shared by all strata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

SCHEMA_VERSION = "0.1.0"

# Bond order codes. 1-3 are Kekulé orders; COORDINATION is reserved for
# metal-ligand bonds (B6), which are scored as a separate class.
SINGLE, DOUBLE, TRIPLE = 1, 2, 3
COORDINATION = 9

# Per-atom and per-bond array names, with on-disk dtypes.
ATOM_FIELDS = {
    "atomic_numbers": np.int8,
    "positions": np.float32,
    "formal_charges": np.int8,
    "implicit_hydrogens": np.int8,
    "radical_electrons": np.int8,
}
BOND_FIELDS = {
    "bonds": np.int32,
    "bond_orders": np.int8,
    "aromatic": np.bool_,
}


@dataclass
class MoleculeRecord:
    """One labeled structure.

    ``implicit_hydrogens`` is -1 for atoms whose hydrogens are explicit in the
    record; otherwise it is the number of H the reference label attaches to
    that atom (used for H-stripped strata).
    """

    structure_id: str
    source_id: str
    atomic_numbers: np.ndarray  # (N,) int8
    positions: np.ndarray  # (N, 3) float32, Angstrom
    formal_charges: np.ndarray  # (N,) int8
    bonds: np.ndarray  # (M, 2) int32, atom indices
    bond_orders: np.ndarray  # (M,) int8, Kekulé
    aromatic: np.ndarray  # (M,) bool
    implicit_hydrogens: np.ndarray | None = None  # (N,) int8
    radical_electrons: np.ndarray | None = None  # (N,) int8
    total_charge: int = 0
    multiplicity: int = 1
    smiles: str = ""
    inchikey: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.atomic_numbers)
        if self.implicit_hydrogens is None:
            self.implicit_hydrogens = np.full(n, -1, dtype=np.int8)
        if self.radical_electrons is None:
            self.radical_electrons = np.zeros(n, dtype=np.int8)
        for name, dtype in ATOM_FIELDS.items():
            setattr(self, name, np.asarray(getattr(self, name), dtype=dtype))
        for name, dtype in BOND_FIELDS.items():
            setattr(self, name, np.asarray(getattr(self, name), dtype=dtype))
        self.positions = self.positions.reshape(n, 3)
        self.bonds = self.bonds.reshape(-1, 2)

    @property
    def n_atoms(self) -> int:
        return len(self.atomic_numbers)

    @property
    def n_bonds(self) -> int:
        return len(self.bond_orders)

    @property
    def n_heavy(self) -> int:
        return int((self.atomic_numbers != 1).sum())

    def strip_hydrogens(self) -> MoleculeRecord:
        """Drop explicit H atoms, folding their count into ``implicit_hydrogens``."""
        is_h = self.atomic_numbers == 1
        if not is_h.any():
            return self
        keep = np.flatnonzero(~is_h)
        remap = np.full(self.n_atoms, -1, dtype=np.int64)
        remap[keep] = np.arange(len(keep))

        implicit = self.implicit_hydrogens.astype(np.int64).copy()
        implicit[implicit < 0] = 0
        a, b = self.bonds[:, 0], self.bonds[:, 1]
        np.add.at(implicit, a[is_h[b]], 1)
        np.add.at(implicit, b[is_h[a]], 1)

        heavy_bond = ~(is_h[a] | is_h[b])
        return MoleculeRecord(
            structure_id=self.structure_id,
            source_id=self.source_id,
            atomic_numbers=self.atomic_numbers[keep],
            positions=self.positions[keep],
            formal_charges=self.formal_charges[keep],
            bonds=remap[self.bonds[heavy_bond]],
            bond_orders=self.bond_orders[heavy_bond],
            aromatic=self.aromatic[heavy_bond],
            implicit_hydrogens=implicit[keep],
            radical_electrons=self.radical_electrons[keep],
            total_charge=self.total_charge,
            multiplicity=self.multiplicity,
            smiles=self.smiles,
            inchikey=self.inchikey,
            meta=dict(self.meta),
        )
