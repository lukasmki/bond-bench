"""bond-bench external adapter for YuelBond (Wang & Dokholyan 2026; github.com/hust220/yuel_bond).

    python run.py QUERIES.jsonl PREDICTIONS.jsonl [checkpoint=geom_kekulized_3d]
    python run.py --version [checkpoint=...]

Runs the model in-process as ``yuel_bond.py`` does in 3D mode: heavy atoms
only (the model's element vocabulary has no H), directed edges between all
heavy atoms within 3 Å, one molecule per forward pass (upstream batching mixes
indices across molecules of different size). The decision rule is upstream's
(``create_molecule_from_predictions``): a pair is bonded if either direction's
argmax is not "No Bond", and its type comes from the first such direction in
edge order. Dative/ionic/hydrogen/three-centre/unspecified predictions are
dropped (upstream writes them as bond code 0) and flagged. Aromatic bonds
(sanitized checkpoints) are Kekulized with RDKit. ``edge_probs`` average the
two directions' softmax: P(0) collects "No Bond" and the exotic classes, and
P(aromatic) is split equally between orders 1 and 2. For explicit-H queries,
each H is bonded to its nearest heavy atom, and that pair is reported with
P(single) = 1. No formal charges or H counts.

Checkpoints: ``geom_kekulized_3d`` (default, clean geometry), ``geom_kekulized_cdg``
(trained with σ = 0.2 Å coordinate noise), ``geom_3d``, ``geom_cdg`` (aromatic labels).
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor" / "yuel_bond"
sys.path.insert(0, str(HERE.parent / "_shared"))
sys.path.insert(0, str(VENDOR))  # upstream imports its modules as ``src.*``

import bbadapter  # noqa: E402
from bbadapter import AROMATIC, attach_hydrogens, heavy_atoms, kekulize, probs4, result  # noqa: E402

CHECKPOINTS = ("geom_kekulized_3d", "geom_kekulized_cdg", "geom_3d", "geom_cdg")
_EXOTIC = (4, 5, 6, 7, 8)  # dative, ionic, hydrogen, three-centre, unspecified
_ORDERS = {0: 1, 1: 2, 2: 3, 3: AROMATIC}
_NO_BOND = 9
_state = {}


def load():
    if "model" in _state:
        return _state["model"], _state["device"]
    import torch
    from src.lightning import YuelBond

    name = bbadapter.options.get("checkpoint", CHECKPOINTS[0])
    if name not in CHECKPOINTS:
        raise ValueError(f"unknown checkpoint {name!r}; choose from {CHECKPOINTS}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Lightning checkpoint of upstream's YuelBond (fetched with a pinned checksum by setup.sh).
    ckpt = torch.load(HERE / "vendor" / "models" / f"{name}.ckpt", map_location=device, weights_only=False)
    model = YuelBond(**ckpt["hyper_parameters"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    _state.update(model=model, device=device)
    return model, device


def perceive(q):
    import numpy as np
    import torch
    from rdkit import Chem
    from src.datasets import BondDataset, atom_one_hot, collate

    z, x = q["atomic_numbers"], q["positions"]
    heavy = heavy_atoms(z)
    h_bonds = attach_hydrogens(z, x) if q["observation"] == "explicit_h" else []
    h_probs = [[0.0, 1.0, 0.0, 0.0]] * len(h_bonds)  # the attached H, as certain single bonds
    if len(heavy) < 2:
        return result(z, h_bonds, [1] * len(h_bonds), edge_pairs=h_bonds, edge_probs=h_probs)
    model, device = load()
    pt = Chem.GetPeriodicTable()
    one_hot = np.array([atom_one_hot(pt.GetElementSymbol(int(z[i]))) for i in heavy])
    ds = BondDataset(raw_data=[{"name": q["structure_id"], "positions": np.asarray(x, float)[heavy],
                                "atoms": one_hot, "bonds": np.array([])}],
                     device=device, has_bonds=False, progress_bar=False)
    if len(ds) == 0:  # no heavy-atom pair within 3 Å
        return result(z, h_bonds, [1] * len(h_bonds), edge_pairs=h_bonds, edge_probs=h_probs)
    data = collate([ds[0]])
    with torch.no_grad():
        logits = model(data)[0]
    prob = torch.softmax(logits, dim=-1).cpu().numpy()
    cls = prob.argmax(axis=1)
    edges = data["edge_index"][0].long().cpu().numpy()

    first, by_pair = {}, {}
    for e, (a, b) in enumerate(edges.tolist()):
        pair = (min(a, b), max(a, b))
        by_pair.setdefault(pair, []).append(prob[e])
        if cls[e] != _NO_BOND and pair not in first:
            first[pair] = int(cls[e])
    pairs, p4 = [], []
    for (a, b), ps in by_pair.items():
        p = np.mean(ps, axis=0)
        pairs.append((heavy[a], heavy[b]))
        p4.append(probs4(p[_NO_BOND] + p[list(_EXOTIC)].sum(), p[0], p[1], p[2], p[3]))
    bonds, orders, flags = [], [], []
    for (a, b), c in first.items():
        if c in _EXOTIC:
            flags.append("exotic_bond_dropped")
            continue
        bonds.append((heavy[a], heavy[b]))
        orders.append(_ORDERS[c])
    orders, kflags = kekulize(z, bonds, orders, explicit_h=False)
    return result(z, bonds + h_bonds, orders + [1] * len(h_bonds), edge_pairs=pairs + h_bonds,
                  edge_probs=p4 + h_probs, flags=sorted(set(flags)) + kflags)


def version():
    commit = subprocess.run(["git", "-C", str(VENDOR), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    return f"yuel_bond {commit or 'unknown'} ({bbadapter.options.get('checkpoint', CHECKPOINTS[0])})"


if __name__ == "__main__":
    bbadapter.serve(perceive, version)
