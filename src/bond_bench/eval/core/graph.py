"""Chemical graph G = (B, q, h, r) on the atoms of a query, in edge-list form."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...data.schema import MoleculeRecord


@dataclass
class Graph:
    """Bond-order assignment on ``n`` atoms.

    ``bonds`` is (m, 2) with i < j, sorted; only nonzero orders are stored.
    ``hydrogens`` holds implicit H counts (0 when H are explicit atoms).
    """

    n: int
    bonds: np.ndarray
    orders: np.ndarray
    charges: np.ndarray
    hydrogens: np.ndarray
    radicals: np.ndarray

    def __post_init__(self) -> None:
        bonds = np.asarray(self.bonds, dtype=np.int64).reshape(-1, 2)
        orders = np.asarray(self.orders, dtype=np.int64).reshape(-1)
        bonds = np.sort(bonds, axis=1)
        keep = orders != 0
        bonds, orders = bonds[keep], orders[keep]
        idx = np.lexsort((bonds[:, 1], bonds[:, 0])) if len(bonds) else np.zeros(0, dtype=np.int64)
        self.bonds, self.orders = bonds[idx], orders[idx]
        self.charges = np.asarray(self.charges, dtype=np.int64).reshape(self.n)
        self.hydrogens = np.asarray(self.hydrogens, dtype=np.int64).reshape(self.n)
        self.radicals = np.asarray(self.radicals, dtype=np.int64).reshape(self.n)

    @classmethod
    def from_record(cls, record: MoleculeRecord) -> Graph:
        h = record.implicit_hydrogens.astype(np.int64).copy()
        h[h < 0] = 0
        return cls(
            n=record.n_atoms,
            bonds=record.bonds,
            orders=record.bond_orders,
            charges=record.formal_charges,
            hydrogens=h,
            radicals=record.radical_electrons,
        )

    def bond_dict(self) -> dict[tuple[int, int], int]:
        return {(int(i), int(j)): int(k) for (i, j), k in zip(self.bonds, self.orders)}

    def valences(self) -> np.ndarray:
        """D = B·1 + h."""
        d = self.hydrogens.copy()
        np.add.at(d, self.bonds[:, 0], self.orders)
        np.add.at(d, self.bonds[:, 1], self.orders)
        return d

    def key(self) -> tuple:
        """Hashable identity (index-aligned)."""
        return (
            tuple(map(tuple, self.bonds.tolist())),
            tuple(self.orders.tolist()),
            tuple(self.charges.tolist()),
            tuple(self.hydrogens.tolist()),
            tuple(self.radicals.tolist()),
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Graph) and self.n == other.n and self.key() == other.key()

    def __hash__(self) -> int:
        return hash(self.key())

    def permuted(self, perm: np.ndarray) -> Graph:
        """Relabel atoms: new index of old atom ``i`` is ``perm[i]``."""
        perm = np.asarray(perm)
        inv = np.argsort(perm)
        return Graph(
            n=self.n,
            bonds=perm[self.bonds],
            orders=self.orders,
            charges=self.charges[inv],
            hydrogens=self.hydrogens[inv],
            radicals=self.radicals[inv],
        )
