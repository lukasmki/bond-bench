"""bond-bench external adapter for indigo-bondorder (Welsh & Allison 2019).

    python run.py QUERIES.jsonl PREDICTIONS.jsonl [algorithm=fpt]
    python run.py --version

indigoX assigns electrons (bond orders and formal charges) to a given
molecular graph by minimizing its electron-position energy score. It does not
perceive connectivity, so the graph is the harness contact map (``contacts``,
the same one ``canonical-O`` uses). Hydrogens must be explicit and the graph
connected (the harness marks other queries unsupported). Settings are
upstream's defaults (def2-SVPD energy tables, electron pairs, max bond order 3)
except the algorithm, which can be ``fpt`` (default), ``astar`` or
``local_optimisation``. Of the degenerate minimum-score assignments, the first
(the one upstream applies) is reported.
"""

import glob
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"
sys.path.insert(0, str(HERE.parent / "_shared"))
for path in glob.glob(str(VENDOR / "build" / "src" / "python")):
    sys.path.insert(0, path)

import bbadapter  # noqa: E402
from bbadapter import result  # noqa: E402

_state = {}


def setup():
    if "pt" in _state:
        return _state["ix"], _state["pt"]
    import pyindigox as ix

    ix.Options.DATA_DIRECTORY = str(VENDOR / "indigo-bondorder" / "data")
    ae = ix.Options.AssignElectrons
    name = bbadapter.options.get("algorithm", "fpt").upper()
    ae.ALGORITHM = getattr(ae.Algorithm, name)
    ae.MAXIMUM_RESULT_COUNT = 0
    _state.update(ix=ix, pt=ix.PeriodicTable())  # keep the periodic table alive (weak singleton)
    return ix, _state["pt"]


def perceive(q):
    if q["observation"] != "explicit_h":
        raise ValueError("indigo-bondorder needs explicit hydrogens")
    ix, pt = setup()
    z = q["atomic_numbers"]
    mol = ix.Molecule()
    atoms = [mol.NewAtom(pt[int(a)]) for a in z]
    pairs = [tuple(p) for p in q["contacts"]]
    bonds = [mol.NewBond(atoms[i], atoms[j]) for i, j in pairs]
    mol.SetTotalCharge(int(q["total_charge"]))
    if mol.AssignElectrons() == 0:
        raise RuntimeError("no electron assignment (disconnected graph or element outside the energy tables)")
    if mol.GetMinimumElectronAssignmentScore() == ix.Options.AssignElectrons.INF:
        raise RuntimeError("no valid electron assignment")
    mol.ApplyElectronAssignment(0)
    orders = [int(b.GetOrder()) for b in bonds]
    if any(o not in (1, 2, 3) for o in orders):  # 6/7 = 1.5/2.5 (odd electron count), 8 undefined
        raise RuntimeError(f"non-integer bond orders {sorted(set(orders) - {1, 2, 3})}")
    return result(z, pairs, orders, charges=[a.GetFormalCharge() for a in atoms])


def version():
    import subprocess

    commit = subprocess.run(["git", "-C", str(VENDOR / "indigo-bondorder"), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    return f"indigo-bondorder {commit or 'unknown'} ({bbadapter.options.get('algorithm', 'fpt')})"


if __name__ == "__main__":
    bbadapter.serve(perceive, version)
