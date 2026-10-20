"""Benchmark results releases: pack ``results/`` into GitHub release assets, and pull them back.

A results release has three assets:

- ``results.tar.gz``   every file under ``results/`` except ``summary.parquet`` (predictions,
                       scores, provenance, plots), with the original ``<method_tag>/`` layout
- ``summary.parquet``  ``results/summary.parquet``, to read the headline numbers without unpacking
- ``SHA256SUMS``

Method folders such as ``canonical-R@windows=b2`` keep their names inside the archive; GitHub
would rename them as individual assets. ``summary.parquet`` is derived from the scores, so
``pull_results`` regenerates it locally instead of unpacking it.
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
import urllib.error
from pathlib import Path
from typing import Any

from ..data.download import sha256sum
from ..data.manifest import REPO_ROOT
from ..data.release import CHECKSUMS, RELEASE_REPO, ReleaseSession, github_token, release_error
from ..data.schema import SCHEMA_VERSION
from .report import report

log = logging.getLogger(__name__)

RESULTS_TAG = f"results-v{SCHEMA_VERSION}"
ARCHIVE = "results.tar.gz"
SUMMARY = "summary.parquet"
PROVENANCE = "provenance.json"


def result_files(results_dir: Path) -> list[Path]:
    """Files to archive: everything under ``results_dir`` but hidden files and ``summary.parquet``."""
    results_dir = Path(results_dir)
    return sorted(
        path for path in results_dir.rglob("*")
        if path.is_file()
        and path != results_dir / SUMMARY
        and not any(part.startswith(".") for part in path.relative_to(results_dir).parts)
    )


def scrub_provenance(prov: dict) -> dict:
    """Strip local paths: the repository root (this machine's, and any ``cwd`` recorded by adapter
    methods run elsewhere) becomes relative, and absolute dataset paths become the file name."""
    for run in prov.get("runs", {}).values():
        dataset = run.get("dataset")
        if dataset and Path(dataset).is_absolute():
            run["dataset"] = Path(dataset).name
    roots = {str(REPO_ROOT), *_find(prov.get("method", {}), "cwd")}

    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: strip(item) for key, item in value.items()}
        if isinstance(value, list):
            return [strip(item) for item in value]
        if isinstance(value, str):
            for root in sorted(roots, key=len, reverse=True):
                value = "." if value == root else value.replace(root + "/", "")
        return value

    return strip(prov)


def _find(value: Any, key: str) -> list[str]:
    """Every absolute-path string stored under ``key`` in nested dicts."""
    found = []
    if isinstance(value, dict):
        for k, item in value.items():
            if k == key and isinstance(item, str) and Path(item).is_absolute() and item != "/":
                found.append(item.rstrip("/"))
            found.extend(_find(item, key))
    return found


def partial_runs(results_dir: Path) -> list[str]:
    """``<method_tag>/<stratum> (n=...)`` for every run made on a subsample (``run --n``)."""
    partial = []
    for path in sorted(Path(results_dir).glob(f"*/{PROVENANCE}")):
        for stratum, run in json.loads(path.read_text()).get("runs", {}).items():
            if run.get("n") is not None:
                partial.append(f"{path.parent.name}/{stratum} (n={run['n']})")
    return partial


def pack_results(results_dir: Path, staging_dir: Path, allow_partial: bool = False) -> list[Path]:
    """Write ``results.tar.gz``, ``summary.parquet`` and ``SHA256SUMS`` into ``staging_dir``.

    ``summary.parquet`` is regenerated from the scores first, so it matches the archive.
    Refuses results from subsampled runs (``run --n``) unless ``allow_partial``.
    """
    results_dir, staging_dir = Path(results_dir), Path(staging_dir)
    if not list(results_dir.glob(f"*/{PROVENANCE}")):
        raise FileNotFoundError(f"no method results in {results_dir}")
    if not allow_partial:
        partial = partial_runs(results_dir)
        if partial:
            raise ValueError(f"subsampled runs, not packing: {', '.join(partial)}")
    report(results_dir)

    staging_dir.mkdir(parents=True, exist_ok=True)
    archive = staging_dir / ARCHIVE
    with tarfile.open(archive, "w:gz") as tar:
        for path in result_files(results_dir):
            info = tar.gettarinfo(path, path.relative_to(results_dir).as_posix())
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            if path.name == PROVENANCE:
                data = (json.dumps(scrub_provenance(json.loads(path.read_text())), indent=2) + "\n").encode()
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            else:
                with open(path, "rb") as f:
                    tar.addfile(info, f)
    summary = staging_dir / SUMMARY
    shutil.copy2(results_dir / SUMMARY, summary)
    assets = [archive, summary]
    checksums = staging_dir / CHECKSUMS
    checksums.write_text("".join(f"{sha256sum(path)}  {path.name}\n" for path in assets))
    return [*assets, checksums]


def pull_results(
    results_dir: Path,
    repo: str = RELEASE_REPO,
    tag: str = RESULTS_TAG,
    force: bool = False,
) -> dict[str, str]:
    """Download a results release into ``results_dir`` and regenerate ``summary.parquet``.

    The archive is checked against the release's ``SHA256SUMS`` and unpacked into a temporary
    directory first. Files already present with the same contents are skipped; local files that
    differ are only replaced with ``force``. Local results for other methods are kept, and the
    regenerated summary covers them too.

    Returns {path relative to ``results_dir``: "downloaded" | "up to date"}.
    """
    results_dir = Path(results_dir)
    token = github_token()
    try:
        session = ReleaseSession(repo, tag, token)
        results_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=results_dir, prefix=".pull-") as tmp:
            staging = Path(tmp)
            sums = session.checksums(staging)
            session.download(staging, {ARCHIVE: sums[ARCHIVE]})
            tree = staging / "tree"
            with tarfile.open(staging / ARCHIVE) as tar:
                tar.extractall(tree, filter="data")
            status, todo = _plan(results_dir, tree, force)
            for rel in todo:
                dest = results_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(tree / rel, dest)
                status[rel] = "downloaded"
                log.info("%s -> %s", rel, dest)
    except (urllib.error.URLError, KeyError) as exc:
        raise release_error("results", repo, tag, token, exc) from exc
    report(results_dir)
    return status


def _plan(results_dir: Path, tree: Path, force: bool) -> tuple[dict[str, str], list[str]]:
    """Split the unpacked files into up-to-date ones and the relative paths to move into place."""
    status, todo, conflicts = {}, [], []
    for path in sorted(p for p in tree.rglob("*") if p.is_file()):
        rel = path.relative_to(tree).as_posix()
        dest = results_dir / rel
        if dest.exists() and sha256sum(dest) == sha256sum(path):
            status[rel] = "up to date"
        elif dest.exists() and not force:
            conflicts.append(str(dest))
        else:
            todo.append(rel)
    if conflicts:
        more = f" and {len(conflicts) - 10} more" if len(conflicts) > 10 else ""
        raise FileExistsError(
            "local results differ from the release (pass --force to overwrite): "
            + ", ".join(conflicts[:10]) + more
        )
    return status, todo
