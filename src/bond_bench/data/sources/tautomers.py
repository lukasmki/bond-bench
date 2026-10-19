"""Tautobase and the Dhaked-Nicklaus tautomer database, merged into groups.

Both sources are turned into *observations*: a tuple of tautomer structures
measured together under one condition, with per-tautomer populations when
known and the preferred tautomer(s). Tautomers are identified by their
FixedH InChI; groups are connected components of the "observed together"
graph, so overlapping pairs/tuples from both sources merge.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

TAUTOBASE_URL = "https://raw.githubusercontent.com/WahlOya/Tautobase/master/Tautobase_SMIRKS.txt"
DN_URL = "https://cactus.nci.nih.gov/download/tautomer/Tautomer_database_release_3a.xlsx"

_AQUEOUS = re.compile(r"\b(water|h2o|d2o|aqueous|buffer)\b", re.IGNORECASE)


@dataclass
class Observation:
    source: str
    ref: str
    solvent: str
    tautomers: list[str]  # FixedH InChI per tautomer
    populations: list[float | None]
    preferred: set[int] = field(default_factory=set)  # indices; empty = no clear preference

    @property
    def quantitative(self) -> bool:
        return all(p is not None for p in self.populations)

    @property
    def aqueous(self) -> bool:
        return bool(_AQUEOUS.search(self.solvent or ""))


def fixed_h_inchi(mol: Chem.Mol) -> str:
    return Chem.MolToInchi(mol, options="/FixedH")


def _parse(smiles: str) -> Chem.Mol | None:
    if not smiles or smiles == "nul":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolFromSmiles(Chem.MolToSmiles(mol))


def _float(text) -> float | None:
    if text is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(text))
    return float(match.group()) if match else None


def _argmax_set(values: list[float | None]) -> set[int]:
    known = [(v, i) for i, v in enumerate(values) if v is not None]
    if not known:
        return set()
    best = max(v for v, _ in known)
    winners = {i for v, i in known if math.isclose(v, best, abs_tol=1e-9)}
    return winners if len(winners) == 1 else set()


def read_tautobase(path: Path, structures: dict[str, Chem.Mol]) -> list[Observation]:
    """Pairs ``T1>>T2``; K = [T2]/[T1] (positive log K favours T2)."""
    df = pl.read_csv(path, separator="\t", infer_schema=False, quote_char=None)
    out = []
    for row in df.iter_rows(named=True):
        left, _, right = (row["Smirks"] or "").partition(">>")
        mols = [_parse(left), _parse(right)]
        if any(m is None for m in mols):
            continue
        keys = [fixed_h_inchi(m) for m in mols]
        if keys[0] == keys[1]:
            continue
        for key, mol in zip(keys, mols):
            structures.setdefault(key, mol)

        populations: list[float | None] = [None, None]
        log_k = _float(row["log K"])
        percent = _float(row["% (Tautomer1)"])
        if percent is not None:
            populations = [percent / 100, 1 - percent / 100]
        elif log_k is not None:
            p1 = 1 / (1 + 10**log_k)
            populations = [p1, 1 - p1]
        preferred = _argmax_set(populations)
        if not preferred and (row["Preferred"] or "").strip().rstrip("*") in ("1", "2"):
            preferred = {int(row["Preferred"].strip().rstrip("*")) - 1}
        out.append(Observation(
            source="Tautobase",
            ref=f"row{row['Row Nr']}",
            solvent=row["State/Solvent"] or "",
            tautomers=keys,
            populations=populations,
            preferred=preferred,
        ))
    return out


def read_dhaked_nicklaus(path: Path, structures: dict[str, Chem.Mol]) -> list[Observation]:
    """Tuples of 2-5 tautomers; ``Ref`` identifies the tuple, rows are conditions."""
    df = pl.read_excel(path, infer_schema_length=0)
    out = []
    for row in df.iter_rows(named=True):
        size = int(_float(row["Size"]) or 0)
        if size < 2:
            continue
        cells = [str(row.get(f"Quantitative_ratio_{k}")) for k in range(1, size + 1)]
        if any("Qualitative" in c for c in cells):
            continue  # rows with values shifted into the wrong columns
        mols = [_parse(row.get(f"SMILES_{k}")) for k in range(1, size + 1)]
        if any(m is None for m in mols):
            continue
        keys = [fixed_h_inchi(m) for m in mols]
        if len(set(keys)) != len(keys):
            continue
        for key, mol in zip(keys, mols):
            structures.setdefault(key, mol)
        ratios = [_float(row.get(f"Quantitative_ratio_{k}")) for k in range(1, size + 1)]
        categories = [_float(row.get(f"Prevalence_Category_{k}")) for k in range(1, size + 1)]
        populations: list[float | None] = ratios if all(r is not None for r in ratios) else [None] * size
        preferred = _argmax_set(populations) or _argmax_set(categories)
        out.append(Observation(
            source="Dhaked-Nicklaus",
            ref=f"Ref{row['Ref']}",
            solvent=row["Solvent"] if row["Solvent"] not in (None, "nul") else "",
            tautomers=keys,
            populations=populations,
            preferred=preferred,
        ))
    return out


def group_tautomers(observations: list[Observation]) -> dict[str, int]:
    """Connected components over 'observed together'; returns InChI -> group index."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for obs in observations:
        for key in obs.tautomers[1:]:
            a, b = find(obs.tautomers[0]), find(key)
            if a != b:
                parent[max(a, b)] = min(a, b)
    roots = sorted({find(k) for k in parent})
    root_index = {r: i for i, r in enumerate(roots)}
    return {k: root_index[find(k)] for k in parent}


def reference_observation(observations: list[Observation]) -> Observation | None:
    """The observation used for a group's preference label.

    Priority: has a clear preference, aqueous, quantitative, Tautobase, then
    lexicographic ref (deterministic).
    """
    usable = [o for o in observations if o.preferred]
    if not usable:
        return None
    return min(usable, key=lambda o: (not o.aqueous, not o.quantitative, o.source != "Tautobase", o.ref))
