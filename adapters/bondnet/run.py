"""bond-bench external adapter for BondNet (Zhang 2026; github.com/zxc8848/BondNet).

    python run.py QUERIES.jsonl PREDICTIONS.jsonl [seed=42]
    python run.py --version [seed=42]

Upstream ships evaluation scripts but no predictor, so this follows the
release's own inference (``evaluate.py`` / ``scripts/export_per_molecule_stats.py``):

- models: ``joint_explicit_h`` for explicit-H queries, ``joint_heavy_only`` for
  stripped ones, loaded with upstream's ``evaluate.load_model``;
- candidates: built with the featurizer's graph helpers at a 3.0 Å envelope and
  masked to d ≤ 2.5 Å (heavy–heavy and H–heavy; never H–H), as the release does;
- connectivity: sigmoid ≥ 0.5 on either direction (OR); H–heavy bonds are single;
- bond type: argmax of the direction-symmetrized stage-3 logits over the bonded
  heavy–heavy edges (single / double / triple / aromatic); if the two directions
  disagree the higher order is kept, as ``scripts/evaluate_chemical_validity.py``
  does. Aromatic bonds are Kekulized with RDKit.

``edge_probs`` cover every candidate pair: P(bond) = max of the two directions'
sigmoids (consistent with the OR rule), and for heavy–heavy pairs P(bond) is split
by the stage-3 softmax over all candidates, with P(aromatic) split equally between
orders 1 and 2. BondNet reads no charge and predicts no formal charges or H counts.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor" / "BondNet"
sys.path.insert(0, str(HERE.parent / "_shared"))
sys.path.insert(0, str(VENDOR))

import bbadapter  # noqa: E402
from bbadapter import AROMATIC, kekulize, probs4, result  # noqa: E402

ENVELOPE, CUTOFF, THRESHOLD = 3.0, 2.5, 0.5
_ORDERS = (1, 2, 3, AROMATIC)
_models = {}


def load(explicit_h):
    import torch

    from evaluate import load_model

    key = "joint_explicit_h" if explicit_h else "joint_heavy_only"
    if key not in _models:
        seed = bbadapter.options.get("seed", "42")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        path = HERE / "vendor" / "checkpoints" / key / f"seed{seed}" / "best_e2e.pt"
        _models[key] = (load_model(str(path), device), device)
    return _models[key]


def perceive(q):
    import torch

    from bondnet.data.featurizer import MoleculeFeaturizer as MF
    from bondnet.model.bondnet import _symmetrize_logits

    z_list = q["atomic_numbers"]
    explicit_h = q["observation"] == "explicit_h"
    model, device = load(explicit_h)
    z = torch.tensor(z_list, dtype=torch.long)
    pos = torch.tensor(q["positions"], dtype=torch.float32)
    if explicit_h:
        ei, ed, dd, _, _ = MF._candidate_graph(pos, z, ENVELOPE, ENVELOPE)
    else:
        ei, ed, dd = MF._radius_graph(pos, ENVELOPE)
    if ei.shape[0] == 0:
        return result(z_list, [], [], edge_pairs=[], edge_probs=[])
    z, pos, ei, ed, dd = (t.to(device) for t in (z, pos, ei, ed, dd))
    with torch.no_grad():
        _, _, feat = model.backbone(z, pos, ei, ed, dd)
        is_h = z == 1
        hh = ~is_h[ei[:, 0]] & ~is_h[ei[:, 1]]
        active = dd <= CUTOFF  # hh and H–X pairs alike; H–H pairs are never built
        p_dir = torch.sigmoid(model.stage1(feat))
        bonded = (p_dir >= THRESHOLD) & active
        readout = bonded & hh
        types = torch.zeros(0, dtype=torch.long, device=device)
        if readout.any():
            logits, _ = model.stage3(feat[readout])
            types = _symmetrize_logits(logits, ei[readout]).argmax(-1)
        cand_hh = active & hh
        soft = torch.zeros((0, 4), device=device)
        if cand_hh.any():
            logits, _ = model.stage3(feat[cand_hh])
            soft = torch.softmax(_symmetrize_logits(logits, ei[cand_hh]), -1)

    ei, p_dir, active, hh = ei.cpu().tolist(), p_dir.cpu().tolist(), active.cpu().tolist(), hh.cpu().tolist()
    bonded, readout, cand_hh = bonded.cpu().tolist(), readout.cpu().tolist(), cand_hh.cpu().tolist()
    types, soft = iter(types.cpu().tolist()), iter(soft.cpu().tolist())

    order, p_bond, p_type = {}, {}, {}
    for (a, b), p, act, is_hh, bnd, ro, ch in zip(ei, p_dir, active, hh, bonded, readout, cand_hh):
        if not act:
            continue
        pair = (min(a, b), max(a, b))
        p_bond[pair] = max(p_bond.get(pair, 0.0), p)
        if ch:
            p_type.setdefault(pair, next(soft))
        if bnd:
            k = _ORDERS[next(types)] if ro else 1
            order[pair] = max(order.get(pair, 0), k)  # as scripts/evaluate_chemical_validity.py
    pairs = sorted(p_bond)
    probs = []
    for pair in pairs:
        pb = p_bond[pair]
        s = p_type.get(pair, (1.0, 0.0, 0.0, 0.0))  # H–heavy pairs are single when bonded
        probs.append(probs4(1.0 - pb, pb * s[0], pb * s[1], pb * s[2], pb * s[3]))
    bonds = sorted(order)
    orders, flags = kekulize(z_list, bonds, [order[b] for b in bonds], explicit_h=explicit_h)
    return result(z_list, bonds, orders, edge_pairs=pairs, edge_probs=probs, flags=flags)


def version():
    commit = subprocess.run(["git", "-C", str(VENDOR), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    return f"BondNet {commit or 'unknown'} (v4 joint, seed {bbadapter.options.get('seed', '42')})"


if __name__ == "__main__":
    bbadapter.serve(perceive, version)
