"""RDKit conversions and chemistry checks for MoleculeRecord."""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

from .schema import COORDINATION, MoleculeRecord

RDLogger.DisableLog("rdApp.*")

_ORDER_TO_RDKIT = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    COORDINATION: Chem.BondType.DATIVE,
}
_RDKIT_TO_ORDER = {v: k for k, v in _ORDER_TO_RDKIT.items()}


class LabelError(ValueError):
    """The reference label of a record is chemically inconsistent."""


def record_to_mol(record: MoleculeRecord, sanitize: bool = True) -> Chem.Mol:
    """Build an RDKit molecule from the labeled graph and coordinates."""
    rw = Chem.RWMol()
    for z, q, nh, nrad in zip(
        record.atomic_numbers,
        record.formal_charges,
        record.implicit_hydrogens,
        record.radical_electrons,
    ):
        atom = Chem.Atom(int(z))
        atom.SetFormalCharge(int(q))
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(max(int(nh), 0))
        atom.SetNumRadicalElectrons(int(nrad))
        rw.AddAtom(atom)
    for (i, j), order in zip(record.bonds, record.bond_orders):
        try:
            bond_type = _ORDER_TO_RDKIT[int(order)]
        except KeyError:
            raise LabelError(f"unknown bond order code {order}") from None
        rw.AddBond(int(i), int(j), bond_type)

    conf = Chem.Conformer(record.n_atoms)
    conf.Set3D(True)
    for k, (x, y, z) in enumerate(record.positions.astype(float)):
        conf.SetAtomPosition(k, Point3D(x, y, z))
    rw.AddConformer(conf, assignId=True)

    mol = rw.GetMol()
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:  # RDKit raises several exception types
            raise LabelError(f"sanitization failed: {exc}") from None
    return mol


def mol_to_record(
    mol: Chem.Mol,
    structure_id: str,
    source_id: str,
    meta: dict | None = None,
    multiplicity: int = 1,
    explicit_h: bool = True,
) -> MoleculeRecord:
    """Convert a sanitized RDKit molecule with a conformer.

    With ``explicit_h`` every hydrogen must be a graph atom and
    ``implicit_hydrogens`` is -1 throughout; otherwise per-atom H counts are
    stored as the label for an H-stripped structure.
    """
    mol = Chem.Mol(mol)
    Chem.Kekulize(mol, clearAromaticFlags=False)
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    try:
        orders = [_RDKIT_TO_ORDER[b.GetBondType()] for b in mol.GetBonds()]
    except KeyError as exc:
        raise LabelError(f"unsupported bond type {exc}") from None
    atoms = list(mol.GetAtoms())
    if explicit_h:
        if any(a.GetTotalNumHs() for a in atoms):
            raise LabelError("implicit hydrogens in an explicit-H structure")
        implicit = [-1] * len(atoms)
    else:
        implicit = [a.GetTotalNumHs() for a in atoms]
    record = MoleculeRecord(
        structure_id=structure_id,
        source_id=source_id,
        atomic_numbers=[a.GetAtomicNum() for a in atoms],
        positions=mol.GetConformer().GetPositions(),
        formal_charges=[a.GetFormalCharge() for a in atoms],
        bonds=np.array(bonds, dtype=np.int32).reshape(-1, 2),
        bond_orders=orders,
        aromatic=[b.GetIsAromatic() for b in mol.GetBonds()],
        implicit_hydrogens=implicit,
        radical_electrons=[a.GetNumRadicalElectrons() for a in atoms],
        total_charge=Chem.GetFormalCharge(mol),
        multiplicity=multiplicity,
        meta=meta or {},
    )
    return set_identifiers(record)


def identifiers(mol: Chem.Mol) -> tuple[str, str]:
    """Canonical (heavy-atom) SMILES and standard InChIKey."""
    heavy = Chem.RemoveHs(mol, sanitize=False)
    return Chem.MolToSmiles(heavy), Chem.MolToInchiKey(mol)


def set_identifiers(record: MoleculeRecord) -> MoleculeRecord:
    record.smiles, record.inchikey = identifiers(record_to_mol(record))
    return record


def validate_record(record: MoleculeRecord) -> Chem.Mol:
    """Raise LabelError if the record's label is inconsistent; return the mol."""
    if record.n_atoms == 0:
        raise LabelError("empty record")
    if record.bonds.size and (record.bonds.min() < 0 or record.bonds.max() >= record.n_atoms):
        raise LabelError("bond index out of range")
    pairs = {tuple(sorted(p)) for p in record.bonds.tolist()}
    if len(pairs) != record.n_bonds:
        raise LabelError("duplicate bonds")
    if any(i == j for i, j in pairs):
        raise LabelError("self bond")
    if int(record.formal_charges.sum()) != record.total_charge:
        raise LabelError(
            f"formal charges sum to {int(record.formal_charges.sum())}, "
            f"total_charge is {record.total_charge}"
        )
    if not np.isfinite(record.positions).all():
        raise LabelError("non-finite coordinates")
    mol = record_to_mol(record)
    check_valence(mol, record.radical_electrons)
    return mol


# Lowest normal valence of elements that also have hypervalent states.
_BASE_VALENCE = {15: 3, 16: 2, 34: 2, 17: 1, 35: 1, 53: 1}


def check_valence(mol: Chem.Mol, radical_electrons: np.ndarray) -> None:
    """Catch implausible labels that still pass RDKit sanitization.

    - under-valent atoms must be declared as radicals in the record;
    - neutral hypervalent S/Se/P/halogens carry no H and at least one multiple
      bond, unless they are perfluorinated (SF5, PF6-like); ``S(N)(O)O`` for a
      sulfonamide or ``C=[SH]`` are labeling errors.
    """
    # AssignRadicals misreads aromatic S, so probe a Kekulé copy.
    probe = Chem.Mol(mol)
    Chem.Kekulize(probe, clearAromaticFlags=True)
    for atom in probe.GetAtoms():
        atom.SetNumRadicalElectrons(0)
    Chem.AssignRadicals(probe)
    implied = np.array([a.GetNumRadicalElectrons() for a in probe.GetAtoms()])
    if not np.array_equal(implied, radical_electrons):
        raise LabelError("under-valent atoms not declared as radicals")

    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z not in _BASE_VALENCE or atom.GetFormalCharge() != 0:
            continue
        if atom.GetTotalValence() <= _BASE_VALENCE[z]:
            continue
        multiple = any(b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE) for b in atom.GetBonds())
        fluorinated = sum(n.GetAtomicNum() == 9 for n in atom.GetNeighbors()) >= 4
        if atom.GetTotalNumHs(includeNeighbors=True) > 0 or not (multiple or fluorinated):
            raise LabelError("implausible hypervalent atom")


def inchikey_from_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol is not None else None
