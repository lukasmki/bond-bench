"""bond-bench external adapter for mambax (Loschen 2018; github.com/CHLoschen/mamba).

    python run.py QUERIES.jsonl PREDICTIONS.jsonl
    python run.py --version

Writes each query as an xyz file and calls ``mambax.ml.predict_bonds`` with
the pretrained random forest ``clf.p``, as ``mambax --predict`` does. Every pair within
3.0 Å is classified as none / single / double / triple / aromatic; the bond is
the argmax (no valence repair). Aromatic bonds are Kekulized with RDKit.
``edge_probs`` are the forest's class probabilities, with p(aromatic) split
equally between orders 1 and 2. The total charge is the model's ``q``
feature. No formal charges or H counts are predicted. The shipped model was
trained with explicit H; stripped queries use the same model, as mambax's
own ``--noH`` option does.
"""

import os
import pickle
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor" / "mamba"
sys.path.insert(0, str(HERE.parent / "_shared"))
sys.path.insert(0, str(VENDOR / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")

from bbadapter import AROMATIC, kekulize, probs4, result, serve  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")
# clf.p is a scikit-learn 0.23.2 pickle: version and feature-name warnings are expected
warnings.filterwarnings("ignore", module="sklearn")
_PT = Chem.GetPeriodicTable()
_ORDERS = {0: 0, 1: 1, 2: 2, 3: 3, 4: AROMATIC}
_clf = None


def model():
    global _clf
    if _clf is None:
        with open(VENDOR / "clf.p", "rb") as f:
            _clf = pickle.load(f)
    return _clf


def write_xyz(q, path):
    """The xyz form mambax reads (``mambax.io.read_xyz``); the charge goes in the title line."""
    lines = [str(len(q["atomic_numbers"])), f"CHARGE={int(q['total_charge'])}"]
    for z, (x, y, w) in zip(q["atomic_numbers"], q["positions"]):
        lines.append(f"{_PT.GetElementSymbol(int(z))} {x:.6f} {y:.6f} {w:.6f}")
    path.write_text("\n".join(lines) + "\n")


def perceive(q):
    from mambax.ml import predict_bonds

    z = q["atomic_numbers"]
    df = None
    if len(z) >= 2:
        with tempfile.TemporaryDirectory(prefix="bb-mambax-") as tmp:
            path = Path(tmp) / "query.xyz"
            write_xyz(q, path)
            df = predict_bonds(str(path), clf=model())
    if df is None:  # fewer than two atoms or no pair within 3 Å
        return result(z, [], [], edge_pairs=[], edge_probs=[])
    pairs, p4, bonds, orders = [], [], [], []
    for a, b, c, *p in df[["id1", "id2", "bond", "p(X)", "p(-)", "p(=)", "p(#)", "p(a)"]].itertuples(index=False):
        i, j = sorted((int(a) - 1, int(b) - 1))
        pairs.append((i, j))
        p4.append(probs4(*p))
        if _ORDERS[int(c)]:
            bonds.append((i, j))
            orders.append(_ORDERS[int(c)])
    orders, flags = kekulize(z, bonds, orders, explicit_h=q["observation"] == "explicit_h")
    return result(z, bonds, orders, edge_pairs=pairs, edge_probs=p4, flags=flags)


def version():
    try:
        commit = subprocess.run(["git", "-C", str(VENDOR), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    return f"mambax {commit} (clf.p)"


if __name__ == "__main__":
    serve(perceive, version)
