"""Train the network of canonical M (TRAIN M of assets/boa_canonical_algorithms.md).

The training measure P (A4) defaults to B2 (GEOM-DRUGS), the calibration
stratum that ``fit_windows.py`` also uses: it is labeled in-distribution and
is not used for headline ranking. A tenth of it (by structure-id hash) is held
out for model selection; the ids used for training are stored in the
checkpoint, and canonical-M marks them in its predictions (``in_training_set``).

    uv sync --extra ml
    uv run python scripts/train_canonical_m.py [--strata B2] [--epochs 150]
    uv run python scripts/train_canonical_m.py --sigma-aug 0.05 --out canonical-M-sigma0.05
    uv run bond-bench run canonical-M --param checkpoint=canonical-M-sigma0.05

Train on a GPU compute node, e.g. inside an allocation:
    srun -n 1 uv run python scripts/train_canonical_m.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from bond_bench.data.manifest import REPO_ROOT
from bond_bench.eval.methods.learned import DEFAULT_CHECKPOINT, checkpoint_path
from bond_bench.eval.ml.model import ModelConfig
from bond_bench.eval.ml.train import TrainConfig, save_checkpoint, train


def main() -> None:
    d, m = TrainConfig(), ModelConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--strata", nargs="+", default=["B2"], help="stratum ids (in --dataset) or .h5 paths")
    ap.add_argument("--dataset", type=Path, default=REPO_ROOT / "dataset")
    ap.add_argument("--out", default=DEFAULT_CHECKPOINT, help="checkpoint name (models/<name>.pt) or .pt path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--weight-decay", type=float, default=d.weight_decay)
    ap.add_argument("--sigma-aug", type=float, default=d.sigma_aug, help="Å; Gaussian coordinate noise")
    ap.add_argument("--regimes", nargs="+", default=list(d.regimes), choices=["explicit_h", "stripped"])
    ap.add_argument("--no-resonance", action="store_true", help="plain cross-entropy on the reference graph")
    ap.add_argument("--val-fraction", type=float, default=d.val_fraction)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--max-samples", type=int, default=None, help="cap on structures (smoke runs)")
    ap.add_argument("--loader-workers", type=int, default=d.loader_workers)
    ap.add_argument("--label-workers", type=int, default=d.label_workers)
    ap.add_argument("--hidden", type=int, default=m.hidden)
    ap.add_argument("--layers", type=int, default=m.layers)
    ap.add_argument("--r-msg", type=float, default=m.r_msg)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    strata = [Path(s) if s.endswith(".h5") else args.dataset / f"{s}.h5" for s in args.strata]
    cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        sigma_aug=args.sigma_aug, regimes=tuple(args.regimes), resonance=not args.no_resonance,
        val_fraction=args.val_fraction, seed=args.seed, max_samples=args.max_samples,
        loader_workers=args.loader_workers, label_workers=args.label_workers,
        model=ModelConfig(hidden=args.hidden, layers=args.layers, r_msg=args.r_msg),
    )
    ckpt = train(strata, cfg, device=args.device)
    out = save_checkpoint(ckpt, checkpoint_path(args.out))
    print(json.dumps({"checkpoint": str(out), "best": ckpt["train"]["best"]}, indent=2))


if __name__ == "__main__":
    main()
