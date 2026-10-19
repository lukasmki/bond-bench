"""bond-bench external adapter for BALL's bond order assignment (Dehof et al. 2011).

    python run.py QUERIES.jsonl PREDICTIONS.jsonl [strategy=fpt]
    python run.py --version

Runs BALL's ``assign_bond_orders`` utility (AssignBondOrderProcessor with the
BALL penalty table ``bond_lengths/BondOrder.xml``, all optimal solutions,
Kekulé rings) with ``--strategy`` ``fpt`` (the utility's default), ``a_star``
or ``ilp``. BALL assigns orders to a given connectivity: each query is written
as a mol2 file whose bonds are the harness contact map (``contacts``), all
typed single; atom types are element symbols, and one substructure keeps the
atom order. The first optimal solution is reported. BALL uses no total charge
and assigns no formal charges, so none are reported. Hydrogens must be explicit
and the graph connected (the harness marks other queries unsupported).
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"
sys.path.insert(0, str(HERE.parent / "_shared"))

import bbadapter  # noqa: E402
from bbadapter import result, symbol  # noqa: E402

STRATEGIES = ("fpt", "a_star", "ilp")


def executable():
    hits = sorted(VENDOR.glob("build/**/assign_bond_orders"))
    if not hits:
        raise FileNotFoundError("assign_bond_orders not built; run setup.sh")
    return hits[0]


def write_mol2(q, path):
    z, x, contacts = q["atomic_numbers"], q["positions"], q["contacts"]
    lines = ["@<TRIPOS>MOLECULE", q["structure_id"], f"{len(z)} {len(contacts)} 1", "SMALL", "NO_CHARGES", "",
             "@<TRIPOS>ATOM"]
    for k, (a, (px, py, pz)) in enumerate(zip(z, x), start=1):
        sym = symbol(a)
        lines.append(f"{k} {sym}{k} {px:.6f} {py:.6f} {pz:.6f} {sym} 1 MOL 0.0")
    lines.append("@<TRIPOS>BOND")
    for k, (i, j) in enumerate(contacts, start=1):
        lines.append(f"{k} {i + 1} {j + 1} 1")
    lines += ["@<TRIPOS>SUBSTRUCTURE", "1 MOL 1", ""]
    path.write_text("\n".join(lines))


def read_mol2_orders(path):
    orders, section = {}, None
    for line in path.read_text().splitlines():
        if line.startswith("@<TRIPOS>"):
            section = line.strip()
        elif section == "@<TRIPOS>BOND" and line.strip():
            _, a, b, kind = line.split()[:4]
            i, j = sorted((int(a) - 1, int(b) - 1))
            orders[(i, j)] = {"1": 1, "2": 2, "3": 3}.get(kind, kind)
    return orders


def perceive(q):
    if q["observation"] != "explicit_h":
        raise ValueError("BALL needs explicit hydrogens")
    strategy = bbadapter.options.get("strategy", "fpt")
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    env = {**os.environ, "BALL_DATA_PATH": str(VENDOR / "ball" / "data")}
    with tempfile.TemporaryDirectory(prefix="bb-ball-") as tmp:
        tmp = Path(tmp)
        write_mol2(q, tmp / "in.mol2")
        proc = subprocess.run([str(executable()), "in.mol2", "out", f"--strategy={strategy}"],
                              cwd=tmp, env=env, capture_output=True, text=True, timeout=600)
        out = tmp / "out_0.mol2"
        if not out.exists():
            msg = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no solution"]
            raise RuntimeError(f"assign_bond_orders exit {proc.returncode}: {msg[0]}")
        orders = read_mol2_orders(out)
    pairs = [tuple(sorted(p)) for p in q["contacts"]]
    bad = {orders.get(p) for p in pairs} - {1, 2, 3}
    if bad:
        raise RuntimeError(f"unexpected bond types in BALL output: {sorted(map(str, bad))}")
    return result(q["atomic_numbers"], pairs, [orders[p] for p in pairs])


def version():
    commit = subprocess.run(["git", "-C", str(VENDOR / "ball"), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    return f"BALL {commit or 'unknown'} assign_bond_orders ({bbadapter.options.get('strategy', 'fpt')})"


if __name__ == "__main__":
    bbadapter.serve(perceive, version)
