"""Crystallography Open Database: metadata, CIF download, crystal -> molecules.

Reference labels come from COD's manually curated SMILES (``/cod/smi``),
which Merkys et al. (2023, doi:10.1186/s13321-023-00692-1) cross-checked by
graph isomorphism against the coordinate-derived structure and the deposited
chemical name. Molecules are extracted from the symmetry-expanded unit cell
(gemmi for symmetry/occupancy/disorder, ASE for periodic neighbour search)
and labeled by mapping the curated SMILES onto them.
"""

from __future__ import annotations

import csv
import io
import subprocess
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.neighborlist import neighbor_list
from rdkit import Chem

from ..chem import LabelError
from ..download import USER_AGENT

SMILES_URL = "https://www.crystallography.net/cod/smi/allcod.smi"
MERKYS_URL = (
    "https://static-content.springer.com/esm/art%3A10.1186%2Fs13321-023-00692-1/"
    "MediaObjects/13321_2023_692_MOESM2_ESM.gz"
)
REST_URL = "https://www.crystallography.net/cod/result?format=csv&id={ids}"
RSYNC_CIF = "rsync://www.crystallography.net/cif/"

BOND_SLACK = 0.45  # Angstrom added to the sum of covalent radii


def cif_relpath(cod_id: str) -> str:
    return f"{cod_id[0]}/{cod_id[1:3]}/{cod_id[3:5]}/{cod_id}.cif"


def load_smiles(path: Path) -> dict[str, str]:
    out = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or "\t" not in line:
                continue
            smiles, cod_id = line.rstrip("\n").split("\t")[:2]
            out[cod_id.strip()] = smiles.strip()
    return out


def query_metadata(ids: list[str], cache_dir: Path, chunk: int = 200, pause: float = 2.0) -> list[dict]:
    """COD REST metadata rows for ``ids`` (cached per chunk; polite pauses)."""
    rows = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(ids), chunk):
        part = ids[start : start + chunk]
        path = cache_dir / f"meta_{part[0]}_{part[-1]}_{len(part)}.csv"
        if not path.exists():
            request = urllib.request.Request(REST_URL.format(ids=",".join(part)), headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=300) as resp:
                path.write_bytes(resp.read())
            time.sleep(pause)
        text = "".join(line for line in io.StringIO(path.read_text()) if not line.startswith("#"))
        rows.extend(csv.DictReader(io.StringIO(text)))
    return rows


def rsync_cifs(ids: list[str], dest: Path) -> None:
    """Fetch CIFs via rsync (the access route COD recommends for bulk use)."""
    missing = [i for i in ids if not (dest / cif_relpath(i)).exists()]
    if not missing:
        return
    dest.mkdir(parents=True, exist_ok=True)
    listing = dest / "rsync_files.txt"
    listing.write_text("\n".join(cif_relpath(i) for i in missing) + "\n")
    subprocess.run(["rsync", "-a", f"--files-from={listing}", RSYNC_CIF, str(dest)], check=False)


@dataclass
class Component:
    mol: Chem.Mol  # heavy atoms + H, all single bonds, no charges
    positions: np.ndarray
    polymeric: bool


def unit_cell_atoms(path: Path) -> tuple[Atoms, list[str]]:
    """Symmetry-expanded unit cell; issues lists disqualifying features."""
    structure = gemmi.read_small_structure(str(path))
    sites = structure.get_all_unit_cell_sites()
    issues = []
    if not sites:
        issues.append("no sites")
    if any(s.occ < 0.99 for s in sites):
        issues.append("partial occupancy")
    if any(s.disorder_group != 0 for s in sites):
        issues.append("disorder")
    symbols = [s.element.name for s in sites]
    if any(s.element.atomic_number == 0 for s in sites):
        issues.append("unknown element")
    if "D" in symbols:
        issues.append("deuterium")
    if "H" not in symbols and "D" not in symbols:
        issues.append("no hydrogens")
    cell = structure.cell
    atoms = Atoms(
        symbols=["H" if s == "D" else s for s in symbols],
        scaled_positions=np.array([[s.fract.x, s.fract.y, s.fract.z] for s in sites]).reshape(-1, 3) % 1.0,
        cell=[cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma],
        pbc=True,
    )
    if len(atoms) > 1 and len(neighbor_list("i", atoms, 0.25)):  # sites closer than 0.5 A
        issues.append("overlapping sites")
    return atoms, issues


def components(atoms: Atoms) -> list[Component]:
    """Connected molecules with whole-molecule (unwrapped) coordinates."""
    cutoffs = [covalent_radii[atomic_numbers[s]] + BOND_SLACK / 2 for s in atoms.get_chemical_symbols()]
    i, j, shifts = neighbor_list("ijS", atoms, cutoffs)
    adjacency: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(len(atoms))]
    for a, b, s in zip(i, j, shifts):
        adjacency[a].append((b, s))

    image: list[np.ndarray | None] = [None] * len(atoms)
    positions, cell = atoms.get_positions(), atoms.cell.array
    out = []
    for start in range(len(atoms)):
        if image[start] is not None:
            continue
        image[start] = np.zeros(3, dtype=int)
        members, queue, polymeric = [start], deque([start]), False
        while queue:
            a = queue.popleft()
            for b, s in adjacency[a]:
                want = image[a] + s
                if image[b] is None:
                    image[b] = want
                    members.append(b)
                    queue.append(b)
                elif not np.array_equal(image[b], want):
                    polymeric = True
        index = {a: k for k, a in enumerate(members)}
        rw = Chem.RWMol()
        for a in members:
            atom = Chem.Atom(atoms[a].symbol)
            atom.SetNoImplicit(True)
            rw.AddAtom(atom)
        for a in members:
            for b, _ in adjacency[a]:
                p, q = index[a], index[b]
                if p < q and rw.GetBondBetweenAtoms(p, q) is None:
                    rw.AddBond(p, q, Chem.BondType.SINGLE)
        xyz = np.array([positions[a] + image[a] @ cell for a in members])
        out.append(Component(rw.GetMol(), xyz, polymeric))
    return out


def skeleton_key(mol: Chem.Mol) -> str:
    """Element + connectivity (explicit H atoms), ignoring orders/charges/stereo."""
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    return Chem.MolToSmiles(rw, canonical=True, allHsExplicit=True)


def reference_fragments(smiles: str) -> list[Chem.Mol] | None:
    """Kekulé fragments of the curated SMILES with explicit H atoms."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    return list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))


def transfer_label(component: Component, reference: Chem.Mol) -> Chem.Mol:
    """Map the reference's bond orders and charges onto the crystal component."""
    pattern = Chem.RWMol(reference)
    for bond in pattern.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
    for atom in pattern.GetAtoms():
        atom.SetFormalCharge(0)
    target = Chem.RWMol(component.mol)
    match = target.GetSubstructMatch(pattern.GetMol(), useChirality=False)
    if len(match) != target.GetNumAtoms():
        raise LabelError("reference does not map onto crystal component")
    for bond in reference.GetBonds():
        target.GetBondBetweenAtoms(match[bond.GetBeginAtomIdx()], match[bond.GetEndAtomIdx()]).SetBondType(bond.GetBondType())
    for atom in reference.GetAtoms():
        target.GetAtomWithIdx(match[atom.GetIdx()]).SetFormalCharge(atom.GetFormalCharge())
    mol = target.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.Set3D(True)
    for k, xyz in enumerate(component.positions):
        conf.SetAtomPosition(k, xyz.tolist())
    mol.AddConformer(conf, assignId=True)
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # noqa: BLE001
        raise LabelError(f"sanitization failed: {exc}") from None
    return mol
