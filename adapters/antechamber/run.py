"""bond-bench external adapter for AmberTools antechamber bond-type perception.

    python run.py QUERIES.jsonl PREDICTIONS.jsonl
    python run.py --version

Each query is written as a Gaussian Cartesian input (``-fi gcrt``: element
symbols, coordinates, charge and multiplicity, no bonds). antechamber finds the
connectivity from distances and assigns bond types with bondtype (``-j 4``,
the Wang et al. 2006 atomic-penalty search). Bond types are read from the
mol2 it writes; atom order is preserved. Aromatic (``ar``) bonds are
Kekulized with RDKit. antechamber assigns no formal charges, so none are
reported. Explicit hydrogens are required.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from bbadapter import AROMATIC, kekulize, result, serve  # noqa: E402
from rdkit import Chem  # noqa: E402

_PT = Chem.GetPeriodicTable()
_MOL2_ORDERS = {"1": 1, "2": 2, "3": 3, "am": 1, "ar": AROMATIC}


def write_gcrt(q, path):
    lines = ["#HF/6-31G* SP", "", "bond-bench query", "", f"{int(q['total_charge'])} {int(q['multiplicity'])}"]
    for z, (x, y, w) in zip(q["atomic_numbers"], q["positions"]):
        lines.append(f"{_PT.GetElementSymbol(int(z)):<2s} {x:14.6f} {y:14.6f} {w:14.6f}")
    path.write_text("\n".join(lines) + "\n\n")


def read_mol2_bonds(path):
    section, bonds, orders, flags = None, [], [], []
    for line in path.read_text().splitlines():
        if line.startswith("@<TRIPOS>"):
            section = line.strip()
            continue
        if section == "@<TRIPOS>BOND" and line.strip():
            _, a, b, kind = line.split()[:4]
            if kind not in _MOL2_ORDERS:
                flags.append(f"bond type {kind} read as single")
            bonds.append((int(a) - 1, int(b) - 1))
            orders.append(_MOL2_ORDERS.get(kind, 1))
    return bonds, orders, flags


def perceive(q):
    if q["observation"] != "explicit_h":
        raise ValueError("antechamber needs explicit hydrogens")
    with tempfile.TemporaryDirectory(prefix="bb-antechamber-") as tmp:
        tmp = Path(tmp)
        write_gcrt(q, tmp / "in.gcrt")
        cmd = [
            "antechamber", "-i", "in.gcrt", "-fi", "gcrt", "-o", "out.mol2", "-fo", "mol2",
            "-at", "sybyl", "-j", "4", "-nc", str(int(q["total_charge"])),
            "-m", str(int(q["multiplicity"])), "-dr", "no", "-pf", "y", "-s", "0",
        ]
        proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=600)
        out = tmp / "out.mol2"
        if proc.returncode != 0 or not out.exists():
            msg = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
            raise RuntimeError(f"antechamber exit {proc.returncode}: {msg[0]}")
        bonds, orders, flags = read_mol2_bonds(out)
    z = q["atomic_numbers"]
    orders, kflags = kekulize(z, bonds, orders, explicit_h=True)
    return result(z, bonds, orders, flags=flags + kflags)


def version():
    proc = subprocess.run(["antechamber", "-h"], capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if "antechamber" in line.lower() and any(c.isdigit() for c in line):
            return line.strip()
    return "antechamber (unknown version)"


if __name__ == "__main__":
    serve(perceive, version)
