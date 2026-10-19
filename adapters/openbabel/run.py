"""bond-bench external adapter for Open Babel bond perception.

    python run.py QUERIES.jsonl PREDICTIONS.jsonl
    python run.py --version
"""

import json
import sys
import time

from openbabel import openbabel as ob

ob.obErrorLog.SetOutputLevel(0)


def perceive(q):
    mol = ob.OBMol()
    mol.BeginModify()
    for z, (x, y, w) in zip(q["atomic_numbers"], q["positions"]):
        atom = mol.NewAtom()
        atom.SetAtomicNum(int(z))
        atom.SetVector(float(x), float(y), float(w))
    mol.EndModify()
    mol.ConnectTheDots()
    mol.PerceiveBondOrders()  # leaves Kekulé orders (OB 3.x)
    if int(q["total_charge"]):
        mol.AssignTotalChargeToAtoms(int(q["total_charge"]))
    bonds, orders = [], []
    for bond in ob.OBMolBondIter(mol):
        bonds.append([bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1])
        orders.append(int(bond.GetBondOrder()))
    atoms = [mol.GetAtom(i + 1) for i in range(mol.NumAtoms())]
    if q["observation"] == "stripped":
        for a in atoms:
            ob.OBAtomAssignTypicalImplicitHydrogens(a)
    return {
        "atomic_numbers": [a.GetAtomicNum() for a in atoms],
        "bonds": bonds,
        "bond_orders": orders,
        "formal_charges": [a.GetFormalCharge() for a in atoms],
        "implicit_hydrogens": [a.GetImplicitHCount() for a in atoms],
        "radical_electrons": [0] * len(atoms),
    }


def main():
    if sys.argv[1:] == ["--version"]:
        print(f"openbabel {ob.OBReleaseVersion()}")
        return
    src, dst = sys.argv[1:3]
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            q = json.loads(line)
            t0 = time.perf_counter()
            try:
                out = {"status": "ok", **perceive(q)}
            except Exception as exc:  # report per molecule, keep going
                out = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
            out["structure_id"] = q["structure_id"]
            out["runtime_s"] = time.perf_counter() - t0
            fout.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
