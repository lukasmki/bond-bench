"""Stage benchmark results as GitHub release assets for ``bond-bench pull --results``.

Regenerates ``results/summary.parquet``, archives ``results/`` into ``results.tar.gz`` (absolute
dataset paths in ``provenance.json`` are reduced to file names) and writes a ``SHA256SUMS``, then
prints the ``gh release create`` command that publishes it. Subsampled runs (``run --n``) are
refused unless ``--allow-partial``.

    uv run python scripts/publish_results.py [--tag results-v0.1.0] [--staging build/results-release]
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from bond_bench.data.release import RELEASE_REPO, RELEASE_TAG
from bond_bench.eval.release import PROVENANCE, RESULTS_TAG, pack_results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--staging", type=Path, default=Path("build/results-release"))
    ap.add_argument("--repo", default=RELEASE_REPO)
    ap.add_argument("--tag", default=RESULTS_TAG)
    ap.add_argument("--dataset-tag", default=RELEASE_TAG, help="dataset release the results were run on")
    ap.add_argument("--allow-partial", action="store_true", help="also pack subsampled (--n) runs")
    args = ap.parse_args()

    assets = pack_results(args.results, args.staging, allow_partial=args.allow_partial)
    for path in assets:
        print(f"{path.stat().st_size / 1e6:8.1f} MB  {path}")
    methods = sorted(p.parent.name for p in args.results.glob(f"*/{PROVENANCE}"))
    notes = (
        f"bond-bench results on dataset release `{args.dataset_tag}`. "
        "Fetch with `uv run bond-bench pull --results`.\n\n"
        "Methods: " + ", ".join(f"`{m}`" for m in methods)
    )
    create = [
        "gh", "release", "create", args.tag, "--repo", args.repo,
        "--title", f"Results {args.tag.removeprefix('results-')}",
        "--notes", notes,
        *map(str, assets),
    ]
    upload = ["gh", "release", "upload", args.tag, "--repo", args.repo, "--clobber", *map(str, assets)]
    print("\npublish with:\n  " + shlex.join(create))
    print("\nor replace the assets of an existing release:\n  " + shlex.join(upload))


if __name__ == "__main__":
    main()
