"""Stage the built dataset as GitHub release assets for ``bond-bench pull``.

Copies ``dataset/`` into a flat directory (strata, per-stratum reports,
``index.parquet``, ``LICENSE.md``) with a ``SHA256SUMS``, then prints the
``gh release create`` command that publishes it. Dev builds (``--n``) are
refused unless ``--allow-partial``.

    uv run python scripts/publish_dataset.py [--tag dataset-v0.1.0] [--staging build/release]
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from bond_bench.data.manifest import load_manifest
from bond_bench.data.release import RELEASE_REPO, RELEASE_TAG, pack


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("dataset"))
    ap.add_argument("--staging", type=Path, default=Path("build/release"))
    ap.add_argument("--repo", default=RELEASE_REPO)
    ap.add_argument("--tag", default=RELEASE_TAG)
    ap.add_argument("--allow-partial", action="store_true", help="also pack dev (--n) builds")
    args = ap.parse_args()

    assets = pack(args.dataset, args.staging, load_manifest(), allow_partial=args.allow_partial)
    for path in assets:
        print(f"{path.stat().st_size / 1e6:8.1f} MB  {path}")
    cmd = [
        "gh", "release", "create", args.tag, "--repo", args.repo,
        "--title", f"Dataset {args.tag.removeprefix('dataset-')}",
        "--notes", "Prebuilt bond-bench dataset. Fetch with `uv run bond-bench pull`. See LICENSE.md for data licenses.",
        *map(str, assets),
    ]
    print("\npublish with:\n  " + shlex.join(cmd))


if __name__ == "__main__":
    main()
