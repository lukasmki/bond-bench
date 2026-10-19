"""bond-bench external adapter for MPerformer (Wang et al. 2023; github.com/FanmengWang/MPerformer).

    python run.py QUERIES.jsonl PREDICTIONS.jsonl
    python run.py --version

The model is built through upstream's Uni-Core task with the arguments
``predict.py`` passes to ``MPerformer/infer.py``, except the classification head
name: ``predict.py`` uses ``cache``, which the checkpoint does not contain, so
upstream silently runs a randomly initialized formal-charge head. Here the
checkpoint's trained head (``total_data``) is used and every weight must load.
Inputs follow upstream's dataset pipeline for
inference: hydrogens removed (MPerformer never sees H), coordinates centred,
[CLS]/[SEP] tokens at the centroid, Gaussian-basis pair features from the full
distance matrix. On CUDA the model runs in fp16 as upstream does; on CPU in fp32.

Decoding is upstream's (``predict.py``), from the continuous heads:

- bond: < 0.5 none, > 2.5 triple, < 1.25 single, > 1.75 double, else aromatic
  (Kekulized with RDKit using the predicted charges and H counts);
- formal charge: < -0.5 → -1, > 0.5 → +1, else 0;
- H count: < 0.5 → 0, < 1.5 → 1, > 2.5 → 3, else 2. Upstream computes these but
  does not write them; here they are the implicit H counts of stripped queries.

For explicit-H queries each H is bonded to its nearest heavy atom. Total charge
is not used (upstream parses it but never passes it to the model).
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor" / "MPerformer"
sys.path.insert(0, str(HERE.parent / "_shared"))

import bbadapter  # noqa: E402
from bbadapter import AROMATIC, attach_hydrogens, heavy_atoms, kekulize, result  # noqa: E402

MAX_HEAVY = 256  # upstream's CroppingDataset subsamples atoms above this
#: The checkpoint's charge head is ``total_data``. predict.py asks for ``cache``, and because
#: infer.py loads with strict=False, upstream's own pipeline runs that head untrained.
HEAD = "total_data"
_state = {}


def load():
    if "model" in _state:
        return _state["model"], _state["dictionary"], _state["device"], _state["half"]
    import torch
    from unicore import checkpoint_utils, options, tasks

    argv = [
        str(VENDOR), "--user-dir", str(VENDOR / "MPerformer"), "--task", "MPerformer",
        "--loss", "finetune_focal_loss_all_constraint_infer", "--arch", "MPerformer_base",
        "--classification-head-name", HEAD, "--num-classes", "3", "--dict-name", "dict.txt",
        "--conf-size", "1", "--only-polar", "0", "--noise-valid", "0", "--batch-size", "1",
        "--path", str(HERE / "vendor" / "weight" / "checkpoint.pt"),
    ]
    parser = options.get_validation_parser()
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser, argv)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing {missing[:3]}, unexpected {unexpected[:3]}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half = device.type == "cuda" and bbadapter.options.get("fp16", "auto") != "no"
    if half:
        model.half()
    model.eval().to(device)
    _state.update(model=model, dictionary=task.dictionary, device=device, half=half)
    return model, task.dictionary, device, half


def bond_order(v):
    if v < 0.5:
        return 0
    if v > 2.5:
        return 3
    if v < 1.25:
        return 1
    if v > 1.75:
        return 2
    return AROMATIC


def charge(v):
    return -1 if v < -0.5 else 1 if v > 0.5 else 0


def h_count(v):
    if v > 2.5:
        return 3
    if v < 0.5:
        return 0
    if v < 1.5:
        return 1
    return 2


def perceive(q):
    import numpy as np
    import torch
    from rdkit import Chem

    z, x = q["atomic_numbers"], q["positions"]
    explicit_h = q["observation"] == "explicit_h"
    heavy = heavy_atoms(z)
    if len(heavy) > MAX_HEAVY:
        raise ValueError(f"more than {MAX_HEAVY} heavy atoms")
    h_bonds = attach_hydrogens(z, x) if explicit_h else []
    if not heavy:
        return result(z, h_bonds, [1] * len(h_bonds))
    model, d, device, half = load()
    pt = Chem.GetPeriodicTable()
    tokens = torch.tensor([d.bos()] + [d.index(pt.GetElementSymbol(int(z[i]))) for i in heavy] + [d.eos()])
    xyz = np.asarray(x, dtype=np.float32)[heavy]
    xyz = xyz - xyz.mean(axis=0)
    xyz = torch.from_numpy(np.vstack([np.zeros((1, 3)), xyz, np.zeros((1, 3))]).astype(np.float32))
    n = len(tokens)
    length = -(-n // 8) * 8  # upstream pads to a multiple of 8
    tok = torch.full((1, length), d.pad(), dtype=torch.long)
    tok[0, :n] = tokens
    crd = torch.zeros((1, length, 3))
    crd[0, :n] = xyz
    dist = torch.zeros((1, length, length))
    dist[0, :n, :n] = torch.cdist(xyz, xyz)
    edge = torch.zeros((1, length, length), dtype=torch.long)
    edge[0, :n, :n] = tokens.view(-1, 1) * len(d) + tokens.view(1, -1)
    dtype = torch.half if half else torch.float32
    with torch.no_grad():
        out = model(src_tokens=tok.to(device), src_distance=dist.to(device, dtype),
                    src_coord=crd.to(device, dtype), src_edge_type=edge.to(device),
                    classification_head_name=HEAD)
    m = len(heavy)
    q_pred = out[4][0, 1:m + 1, 0].float().cpu().tolist()
    h_pred = out[5][0, 1:m + 1, 0].float().cpu().tolist()
    b_pred = out[6][0, 1:m + 1, 1:m + 1, 0].float().cpu().numpy()

    charges = [0] * len(z)
    hyd = [0] * len(z)
    for k, i in enumerate(heavy):
        charges[i] = charge(q_pred[k])
        hyd[i] = h_count(h_pred[k])
    bonds, orders = [], []
    for a in range(m):
        for b in range(a + 1, m):
            k = bond_order(float(b_pred[a, b]))
            if k:
                bonds.append((heavy[a], heavy[b]))
                orders.append(k)
    orders, flags = kekulize(z, bonds, orders, charges=charges, explicit_h=explicit_h,
                             hydrogens=None if explicit_h else hyd)
    return result(z, bonds + h_bonds, orders + [1] * len(h_bonds), charges=charges,
                  hydrogens=None if explicit_h else hyd, flags=flags)


def version():
    commit = subprocess.run(["git", "-C", str(VENDOR), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    return f"MPerformer {commit or 'unknown'}"


if __name__ == "__main__":
    bbadapter.serve(perceive, version)
