"""Transition1x (doi:10.1038/s41597-022-01870-w) frames labeled with Grambow endpoint graphs.

Transition1x.h5 layout: ``/data/<formula>/rxnNNNN/{atomic_numbers, positions
(F,N,3) Å, wB97x_6-31G(d).energy (F,) eV}`` plus ``reactant``/``product``/
``transition_state`` subgroups. Frames 0-9 are the initial NEB band (0 =
reactant, 9 = product); later frames come in chunks of the 8 interior images,
so frame f >= 10 is image ``(f - 10) % 8 + 1`` (1 next to the reactant, 8
next to the product).

Reactions are renamed relative to Grambow et al. (doi:10.1038/s41597-020-0460-4)
and the index is unpublished, so each reaction is matched to Grambow's
atom-mapped ``rsmi``/``psmi`` by its element sequence and the geometric
connectivity of both endpoints. Atom i corresponds to map number i + 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

T1X_URL = "https://ndownloader.figshare.com/files/36035789"
T1X_MD5 = "faab0413cf6b042b6d50e6272dea9a8c"
GRAMBOW_URL = "https://zenodo.org/api/records/3715478/files/wb97xd3.csv/content"
ENERGY_KEY = "wB97x_6-31G(d).energy"

Edges = frozenset[tuple[int, int]]


def edges_of(mol: Chem.Mol) -> Edges:
    return frozenset(tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds())


def geometric_edges(numbers: np.ndarray, positions: np.ndarray) -> Edges | None:
    mol = Chem.RWMol()
    for z in numbers:
        mol.AddAtom(Chem.Atom(int(z)))
    conf = Chem.Conformer(len(numbers))
    for i, xyz in enumerate(np.asarray(positions, dtype=float)):
        conf.SetAtomPosition(i, xyz.tolist())
    mol.AddConformer(conf, assignId=True)
    try:
        rdDetermineBonds.DetermineConnectivity(mol)
    except Exception:  # noqa: BLE001
        return None
    return edges_of(mol)


def mapped_mol(smiles: str) -> Chem.Mol | None:
    """Atom-mapped SMILES -> mol with atoms ordered by map number (explicit H kept)."""
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        return None
    maps = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    if sorted(maps) != list(range(1, len(maps) + 1)):
        return None
    mol = Chem.RenumberAtoms(mol, [maps.index(k) for k in range(1, len(maps) + 1)])
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        atom.SetNoImplicit(True)
    return mol


@dataclass
class GrambowReaction:
    idx: int
    reactant: Chem.Mol
    product: Chem.Mol
    numbers: tuple[int, ...]


def load_grambow(path) -> dict[tuple, list[GrambowReaction]]:
    """Grambow reactions keyed by (element sequence, reactant edges, product edges)."""
    table: dict[tuple, list[GrambowReaction]] = {}
    for row in pl.read_csv(path, infer_schema=False).iter_rows(named=True):
        reactant, product = mapped_mol(row["rsmi"]), mapped_mol(row["psmi"])
        if reactant is None or product is None:
            continue
        numbers = tuple(a.GetAtomicNum() for a in reactant.GetAtoms())
        if numbers != tuple(a.GetAtomicNum() for a in product.GetAtoms()):
            continue
        key = (numbers, edges_of(reactant), edges_of(product))
        table.setdefault(key, []).append(GrambowReaction(int(row["idx"]), reactant, product, numbers))
    return table


def frame_image(frame: int) -> int:
    """NEB image index of a frame (0 = reactant, 9 = product)."""
    return frame if frame < 10 else (frame - 10) % 8 + 1
