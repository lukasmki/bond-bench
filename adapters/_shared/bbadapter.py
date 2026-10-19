"""Helpers shared by the bond-bench adapters (each runs in its own environment).

Only the standard library is imported at module level; RDKit is imported
lazily by :func:`kekulize`, so an adapter environment needs it only if it
calls that. The protocol is described in
``src/bond_bench/eval/methods/external.py``.

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
    from bbadapter import serve
"""

import json
import math
import sys
import time

#: Placeholder order for aromatic bonds before Kekulization.
AROMATIC = 1.5

#: Element symbols by atomic number (index 0 is a placeholder).
SYMBOLS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
    "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf "
    "Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()


def symbol(z):
    return SYMBOLS[int(z)]


#: ``key=value`` arguments after the two paths (forwarded method params), filled by :func:`serve`.
options = {}


def serve(perceive, version, batch=None):
    """Run the protocol: ``run.py QUERIES.jsonl PREDICTIONS.jsonl [key=value ...]`` or ``run.py --version``.

    ``perceive(q) -> dict`` handles one query. ``batch(queries) -> list[dict | Exception]``,
    if given, replaces the per-query loop (for models that batch on a GPU); runtime is then
    split evenly over the batch. Errors are reported per molecule.
    """
    args = sys.argv[1:]
    flags = [a for a in args if "=" not in a]
    options.update(a.split("=", 1) for a in args if "=" in a)
    if flags == ["--version"]:
        print(version() if callable(version) else version)
        return
    src, dst = flags[:2]
    with open(src) as fin:
        queries = [json.loads(line) for line in fin if line.strip()]
    if batch is not None:
        t0 = time.perf_counter()
        try:
            outs = batch(queries)
        except Exception as exc:  # the whole batch failed
            outs = [exc] * len(queries)
        dt = (time.perf_counter() - t0) / max(len(queries), 1)
        timed = [(o, dt) for o in outs]
    else:
        timed = []
        for q in queries:
            t0 = time.perf_counter()
            try:
                out = perceive(q)
            except Exception as exc:
                out = exc
            timed.append((out, time.perf_counter() - t0))
    with open(dst, "w") as fout:
        for q, (out, dt) in zip(queries, timed):
            if isinstance(out, Exception):
                out = {"status": "error", "reason": f"{type(out).__name__}: {out}"}
            else:
                out = {"status": "ok", **out}
            out["structure_id"] = q["structure_id"]
            out["runtime_s"] = dt
            fout.write(json.dumps(out) + "\n")


def heavy_atoms(z):
    return [i for i, a in enumerate(z) if a != 1]


def attach_hydrogens(z, positions, exclude=()):
    """Bond every H to its nearest non-H atom (nearest atom if there is none).

    For adapters whose model only sees heavy atoms: the H of an explicit-H query
    keep their positions, and each gets one single bond. Returns ``[(i, j), ...]``.
    """
    heavy = heavy_atoms(z) or list(range(len(z)))
    out, seen = [], set(exclude)
    for h, a in enumerate(z):
        if a != 1:
            continue
        cands = [k for k in heavy if k != h]
        if not cands:
            continue
        k = min(cands, key=lambda k: math.dist(positions[h], positions[k]))
        pair = (min(h, k), max(h, k))
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def probs4(p_none, p1, p2, p3, p_aromatic=0.0):
    """P(B_ij = k) for k = 0..3; an aromatic class is split equally between 1 and 2."""
    return [p_none, p1 + p_aromatic / 2, p2 + p_aromatic / 2, p3]


def kekulize(z, bonds, orders, charges=None, explicit_h=True, hydrogens=None):
    """Kekulé orders for ``orders`` that may contain :data:`AROMATIC`.

    ``hydrogens`` (predicted implicit H counts) fixes each atom's H; otherwise
    H are explicit atoms (``explicit_h``) or left to RDKit's default valence.

    Returns ``(orders, flags)``. If RDKit cannot Kekulize, aromatic bonds become
    single and the flag ``kekulize_failed`` is set; ``+project`` can repair that.
    """
    if not any(o == AROMATIC for o in orders):
        return [int(o) for o in orders], []
    from rdkit import Chem

    types = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, AROMATIC: Chem.BondType.AROMATIC}
    rw = Chem.RWMol()
    for k, a in enumerate(z):
        atom = Chem.Atom(int(a))
        if charges is not None:
            atom.SetFormalCharge(int(charges[k]))
        if hydrogens is not None:
            atom.SetNumExplicitHs(int(hydrogens[k]))
        atom.SetNoImplicit(bool(explicit_h) or hydrogens is not None)
        rw.AddAtom(atom)
    for (i, j), o in zip(bonds, orders):
        rw.AddBond(int(i), int(j), types[o])
        if o == AROMATIC:
            rw.GetAtomWithIdx(int(i)).SetIsAromatic(True)
            rw.GetAtomWithIdx(int(j)).SetIsAromatic(True)
    mol = rw.GetMol()
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        return [1 if o == AROMATIC else int(o) for o in orders], ["kekulize_failed"]
    out = []
    for i, j in bonds:
        b = mol.GetBondBetweenAtoms(int(i), int(j))
        out.append(int(b.GetBondTypeAsDouble()))
    return out, []


def result(z, bonds, orders, charges=None, hydrogens=None, edge_pairs=None, edge_probs=None, flags=()):
    """A protocol output record (status and timing are added by :func:`serve`)."""
    out = {
        "atomic_numbers": [int(a) for a in z],
        "bonds": [[int(i), int(j)] for i, j in bonds],
        "bond_orders": [int(o) for o in orders],
        "formal_charges": None if charges is None else [int(c) for c in charges],
        "implicit_hydrogens": None if hydrogens is None else [int(h) for h in hydrogens],
        "radical_electrons": [0] * len(z),
        "flags": list(flags),
    }
    if edge_probs is not None:
        out["edge_pairs"] = [[int(i), int(j)] for i, j in edge_pairs]
        out["edge_probs"] = [[float(p) for p in row] for row in edge_probs]
    return out
