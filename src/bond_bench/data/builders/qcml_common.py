"""Converting QCML records (source SMILES graph as label)."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from ..chem import LabelError, mol_to_record
from ..relax import same_connectivity
from ..schema import MoleculeRecord
from .base import StratumBuilder

# Main-group elements; transition-metal graphs belong to B6.
MAIN_GROUP = {
    "H", "B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl", "Ga", "Ge", "As", "Se", "Br",
    "In", "Sn", "Sb", "Te", "I",
}


def hetero_signature(smiles: str) -> str:
    """Sorted element symbols other than C/H (sampling stratum for element coverage)."""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return "?"
    return "".join(sorted({a.GetSymbol() for a in mol.GetAtoms()} - {"C", "H"}))


def qcml_record(builder: StratumBuilder, rec: dict, structure_id: str, meta: dict) -> MoleculeRecord | None:
    """Label a QCML structure with its source graph; None (and a rejection) if inconsistent."""
    if rec["is_outlier"]:
        builder.reject("qcml outlier")
        return None
    if rec["multiplicity"] != 1:
        builder.reject("open shell")
        return None
    mol = Chem.MolFromSmiles(rec["smiles"])
    if mol is None:
        builder.reject("source smiles fails sanitization")
        return None
    if not {a.GetSymbol() for a in mol.GetAtoms()} <= MAIN_GROUP:
        builder.reject("element outside main group")
        return None
    mol = Chem.AddHs(mol)
    if [a.GetAtomicNum() for a in mol.GetAtoms()] != rec["atomic_numbers"].tolist():
        builder.reject("atom order differs from source smiles")
        return None
    if Chem.GetFormalCharge(mol) != rec["charge"]:
        builder.reject("formal charges disagree with total charge")
        return None
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        builder.reject("open shell")
        return None
    if not same_connectivity(mol, rec["positions"]):
        builder.reject("geometry connectivity differs from source graph")
        return None
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.Set3D(True)
    for i, (x, y, z) in enumerate(np.asarray(rec["positions"], dtype=float)):
        conf.SetAtomPosition(i, Point3D(x, y, z))
    mol.AddConformer(conf, assignId=True)
    try:
        record = mol_to_record(mol, structure_id=structure_id, source_id=f"qcml/{rec['key_hash']}", meta=meta)
    except LabelError as exc:
        builder.reject(f"label: {exc}")
        return None
    return record if builder.accept(record) else None
