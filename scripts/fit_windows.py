"""Fit the rule-based distance windows W_{ab k} from the B2 calibration stratum.

B2 (GEOM-DRUGS) is labeled in-distribution and is not used for headline
ranking, so fitting here does not leak into the headline strata. Aromatic
bonds form their own class k = 1.5 because Kekulé labels split them
arbitrarily between 1 and 2.

    uv run python scripts/fit_windows.py [--stratum dataset/B2.h5] [--q 0.005]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from bond_bench.data.store import StratumReader
from bond_bench.eval.tables import WINDOWS_CSV


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stratum", type=Path, default=Path("dataset/B2.h5"))
    ap.add_argument("--q", type=float, default=0.005, help="tail quantile trimmed on each side")
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--out", type=Path, default=WINDOWS_CSV)
    args = ap.parse_args()

    lengths: dict[tuple[int, int, float], list[float]] = defaultdict(list)
    for rec in StratumReader(args.stratum):
        x, z = rec.positions.astype(np.float64), rec.atomic_numbers
        for (i, j), k, arom in zip(rec.bonds, rec.bond_orders, rec.aromatic):
            a, b = sorted((int(z[i]), int(z[j])))
            cls = 1.5 if arom else float(k)
            lengths[(a, b, cls)].append(float(np.linalg.norm(x[i] - x[j])))

    rows = []
    for (a, b, k), vals in sorted(lengths.items()):
        if len(vals) < args.min_count:
            continue
        v = np.asarray(vals)
        lo, hi = np.quantile(v, [args.q, 1 - args.q])
        rows.append({"z_a": a, "z_b": b, "k": k, "lo": round(lo, 4), "hi": round(hi, 4),
                     "center": round(float(np.median(v)), 4), "count": len(v)})
    pl.DataFrame(rows).write_csv(args.out)
    print(f"wrote {len(rows)} windows from {args.stratum} to {args.out}")


if __name__ == "__main__":
    main()
