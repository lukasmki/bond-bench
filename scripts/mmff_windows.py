"""Build the rule-based distance windows W_{ab k} from MMFF94 reference bond lengths.

Centers are MMFF94 bond-stretch r0 values (Halgren 1996, as shipped with
RDKit), read by MMFF-typing small prototype molecules. Each bond class k ∈
{1, 1.5 aromatic, 2, 3} of an element pair gets the mean r0 over its
prototypes. Window edges are the midpoints between neighbouring class centers,
so windows never overlap; the outermost edges extend by ``--delta`` Å. No
benchmark stratum is used.

    uv run python scripts/mmff_windows.py [--delta 0.1]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem

from bond_bench.eval.tables import TABLE_DIR

# (prototype SMILES, SMARTS for the bond to read, bond class). Several prototypes
# per class average over hybridization contexts (e.g. sp3–sp3 and sp2–sp3 singles).
PROTOTYPES: list[tuple[str, str, float]] = [
    # C–C
    ("CC", "[CX4]-[CX4]", 1), ("CC=C", "[CX4]-[CX3]", 1), ("C=CC=C", "[CX3]-[CX3]", 1), ("CC#C", "[CX4]-[CX2]", 1),
    ("c1ccccc1", "c:c", 1.5), ("c1ccncc1", "c:c", 1.5),
    ("C=C", "C=C", 2), ("CC=CC", "[CX3]=[CX3]", 2),
    ("C#C", "C#C", 3), ("CC#CC", "C#C", 3),
    # C–N
    ("CN", "[CX4]-[NX3]", 1), ("CC(=O)N", "[CX3]-[NX3]", 1), ("Nc1ccccc1", "[NX3]-c", 1), ("CN(C)C", "[CX4]-[NX3]", 1),
    ("c1ccncc1", "c:n", 1.5), ("c1cc[nH]c1", "c:[nH]", 1.5),
    ("C=N", "C=N", 2), ("CC(C)=NC", "[CX3]=[NX2]", 2),
    ("CC#N", "C#N", 3),
    # C–O
    ("CO", "[CX4]-[OX2]", 1), ("COC", "[CX4]-[OX2]", 1), ("CC(=O)O", "[CX3]-[OX2H]", 1), ("Oc1ccccc1", "[OX2]-c", 1),
    ("c1ccoc1", "c:o", 1.5),
    ("C=O", "C=O", 2), ("CC(C)=O", "[CX3]=O", 2), ("CC(=O)O", "[CX3]=O", 2),
    # C–S
    ("CS", "[CX4]-[SX2]", 1), ("CSC", "[CX4]-[SX2]", 1), ("Sc1ccccc1", "[SX2]-c", 1),
    ("c1ccsc1", "c:s", 1.5),
    ("C=S", "C=S", 2), ("CC(C)=S", "[CX3]=S", 2),
    # C–X single
    ("CF", "C-F", 1), ("Fc1ccccc1", "c-F", 1), ("CCl", "C-Cl", 1), ("Clc1ccccc1", "c-Cl", 1),
    ("CBr", "C-Br", 1), ("Brc1ccccc1", "c-Br", 1), ("CI", "C-I", 1), ("Ic1ccccc1", "c-I", 1),
    ("C[SiH3]", "C-[Si]", 1), ("CP", "C-P", 1), ("CP(C)(C)=O", "C-P", 1),
    # N–N, N–O
    ("NN", "N-N", 1), ("c1ccnnc1", "n:n", 1.5), ("N=N", "N=N", 2), ("CN=NC", "N=N", 2), ("CN=[N+]=[N-]", "[N+]=[N-]", 2),
    ("NO", "N-O", 1), ("CON(C)C", "N-O", 1), ("c1cnoc1", "n:o", 1.5), ("CN=O", "N=O", 2),
    ("C[N+](=O)[O-]", "[N+]=O", 2),
    # O–O, S–S, S–O, S–N, P–O
    ("OO", "O-O", 1), ("SS", "S-S", 1), ("CSSC", "S-S", 1),
    ("CS(=O)(=O)O", "[SX4]-[OX2H]", 1), ("CS(C)(=O)=O", "S=O", 2), ("CS(C)=O", "S=O", 2),
    ("CS(N)(=O)=O", "S-N", 1), ("c1cnsc1", "n:s", 1.5),
    ("COP(=O)(OC)OC", "[PX4]-[OX2]", 1), ("COP(=O)(OC)OC", "P=O", 2), ("CP(=S)(C)C", "P=S", 2),
    ("[SiH3]O", "[Si]-O", 1),
]


def r0_values() -> dict[tuple[int, int, float], list[float]]:
    out: dict[tuple[int, int, float], list[float]] = defaultdict(list)
    for smiles, smarts, k in PROTOTYPES:
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is None:
            raise SystemExit(f"MMFF cannot type prototype {smiles}")
        match = mol.GetSubstructMatch(Chem.MolFromSmarts(smarts))
        if not match:
            raise SystemExit(f"SMARTS {smarts} does not match prototype {smiles}")
        i, j = match[0], match[1]
        params = props.GetMMFFBondStretchParams(mol, i, j)
        if params is None:
            raise SystemExit(f"no MMFF bond-stretch parameters for {smarts} in {smiles}")
        a, b = sorted((mol.GetAtomWithIdx(i).GetAtomicNum(), mol.GetAtomWithIdx(j).GetAtomicNum()))
        out[(a, b, float(k))].append(float(params[2]))
    return out


def windows(r0: dict[tuple[int, int, float], list[float]], delta: float) -> list[dict]:
    by_pair: dict[tuple[int, int], list[tuple[float, float, int]]] = defaultdict(list)
    for (a, b, k), vals in r0.items():
        by_pair[(a, b)].append((k, float(np.mean(vals)), len(vals)))
    rows = []
    for (a, b), classes in sorted(by_pair.items()):
        classes.sort(key=lambda c: -c[1])  # longest (lowest order) first
        centers = [c[1] for c in classes]
        for idx, (k, center, count) in enumerate(classes):
            hi = center + delta if idx == 0 else (centers[idx - 1] + center) / 2
            lo = center - delta if idx == len(classes) - 1 else (center + centers[idx + 1]) / 2
            rows.append({"z_a": a, "z_b": b, "k": k, "lo": round(lo, 4), "hi": round(hi, 4),
                         "center": round(center, 4), "count": count, "source": "MMFF94 r0"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=0.1, help="outer-edge extension (Å)")
    ap.add_argument("--out", type=Path, default=TABLE_DIR / "bond_windows_mmff.csv")
    args = ap.parse_args()
    rows = windows(r0_values(), args.delta)
    pl.DataFrame(rows).sort(["z_a", "z_b", "k"]).write_csv(args.out)
    print(f"wrote {len(rows)} windows from {len(PROTOTYPES)} MMFF94 prototypes to {args.out}")


if __name__ == "__main__":
    main()
