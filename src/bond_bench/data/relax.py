"""3D embedding and GFN2-xTB relaxation (tblite + ASE)."""

from __future__ import annotations

import os

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

from .chem import LabelError


def embed(mol: Chem.Mol, seed: int) -> Chem.Mol:
    """ETKDG conformer + MMFF cleanup; returns a copy with explicit H."""
    mol = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed % (2**31)
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise LabelError("embedding failed")
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    return mol


def xtb_relax(mol: Chem.Mol, fmax: float = 0.05, steps: int = 500) -> tuple[np.ndarray, float, bool]:
    """GFN2-xTB geometry optimization. Returns (positions, energy eV, converged)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from ase import Atoms
    from ase.optimize import LBFGS
    from tblite.ase import TBLite

    n_unpaired = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    atoms = Atoms(
        numbers=[a.GetAtomicNum() for a in mol.GetAtoms()],
        positions=mol.GetConformer().GetPositions(),
    )
    atoms.calc = TBLite(method="GFN2-xTB", charge=Chem.GetFormalCharge(mol), multiplicity=n_unpaired + 1, verbosity=0)
    opt = LBFGS(atoms, logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=steps))
    return atoms.get_positions(), float(atoms.get_potential_energy()), converged


def same_connectivity(mol: Chem.Mol, positions: np.ndarray) -> bool:
    """Whether distance-based connectivity at ``positions`` matches the mol's graph."""
    probe = Chem.RWMol()
    for atom in mol.GetAtoms():
        probe.AddAtom(Chem.Atom(atom.GetAtomicNum()))
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, xyz in enumerate(np.asarray(positions, dtype=float)):
        conf.SetAtomPosition(i, xyz.tolist())
    probe.AddConformer(conf, assignId=True)
    rdDetermineBonds.DetermineConnectivity(probe)
    edges = lambda m: {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in m.GetBonds()}
    return edges(probe) == edges(mol)
