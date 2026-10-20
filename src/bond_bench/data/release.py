"""Prebuilt dataset releases: pack ``dataset/`` into GitHub release assets, and pull them back.

A release is a flat set of assets plus a ``SHA256SUMS`` file listing every one of them:

- ``<stratum_id>.h5``           -> ``dataset/<stratum_id>.h5``
- ``<stratum_id>.report.json``  -> ``dataset/reports/<stratum_id>.json``
- ``index.parquet``             -> ``dataset/index.parquet``
- ``LICENSE.md``                -> ``dataset/LICENSE.md``
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from .download import USER_AGENT, fetch, fetch_many, sha256sum
from .manifest import REPO_ROOT, StratumSpec
from .schema import SCHEMA_VERSION
from .store import StratumReader, write_index

log = logging.getLogger(__name__)

RELEASE_REPO = "lukasmki/bond-bench"
RELEASE_TAG = f"dataset-v{SCHEMA_VERSION}"
CHECKSUMS = "SHA256SUMS"
INDEX = "index.parquet"
LICENSE = "LICENSE.md"
REPORT_SUFFIX = ".report.json"


def asset_url(name: str, repo: str = RELEASE_REPO, tag: str = RELEASE_TAG) -> str:
    """Anonymous download URL (public repositories only)."""
    return f"https://github.com/{repo}/releases/download/{tag}/{name}"


def github_token() -> str | None:
    """``GITHUB_TOKEN``/``GH_TOKEN``, else the GitHub CLI's login, else None."""
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token
    if shutil.which("gh"):
        res = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=False)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    return None


def api_asset_urls(repo: str, tag: str, token: str) -> dict[str, str]:
    """{asset name: API download URL} for a release; works for private repositories."""
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/releases/tags/{tag}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=60) as resp:
        return {asset["name"]: asset["url"] for asset in json.load(resp)["assets"]}


class ReleaseSession:
    """Download verified assets of one release, via the API when a token is available."""

    def __init__(self, repo: str, tag: str, token: str | None):
        self.repo, self.tag = repo, tag
        if token:
            self._api_urls = api_asset_urls(repo, tag, token)
            self.headers = {"Authorization": f"Bearer {token}", "Accept": "application/octet-stream"}
        else:
            self._api_urls = None
            self.headers = None

    def url(self, name: str) -> str:
        return self._api_urls[name] if self._api_urls is not None else asset_url(name, self.repo, self.tag)

    def checksums(self, staging: Path) -> dict[str, str]:
        """Fetch the release's ``SHA256SUMS`` into ``staging`` and parse it."""
        fetch(self.url(CHECKSUMS), staging / CHECKSUMS, headers=self.headers)
        return parse_checksums((staging / CHECKSUMS).read_text())

    def download(self, staging: Path, wanted: dict[str, str]) -> None:
        """Fetch {asset: sha256} into ``staging`` and check every checksum."""
        errors = fetch_many(((self.url(name), staging / name) for name in wanted), headers=self.headers)
        failed = {path.name: exc for path, exc in errors.items() if exc is not None}
        if failed:
            raise RuntimeError(f"download failed: {failed}")
        for name, digest in wanted.items():
            got = sha256sum(staging / name)
            if got != digest:
                raise RuntimeError(f"checksum mismatch for {name}: expected {digest}, got {got}")


def release_error(what: str, repo: str, tag: str, token: str | None, exc: Exception) -> RuntimeError:
    hint = "" if token else " (private repositories need GITHUB_TOKEN or `gh auth login`)"
    return RuntimeError(f"cannot fetch {what} release {tag!r} from {repo}: {exc!r}{hint}")


def local_path(dataset_dir: Path, asset: str) -> Path:
    """Where a release asset lives inside the dataset directory."""
    if asset.endswith(REPORT_SUFFIX):
        return dataset_dir / "reports" / f"{asset.removesuffix(REPORT_SUFFIX)}.json"
    if asset.endswith(".h5") or asset in (INDEX, LICENSE):
        return dataset_dir / asset
    raise ValueError(f"unexpected release asset {asset!r}")


def parse_checksums(text: str) -> dict[str, str]:
    """Parse ``sha256sum``-style lines (``<hash>  <name>``) into {name: hash}."""
    sums = {}
    for line in text.splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.lstrip("*")] = digest
    return sums


def pack(
    dataset_dir: Path,
    staging_dir: Path,
    specs: dict[str, StratumSpec],
    license_path: Path = REPO_ROOT / LICENSE,
    allow_partial: bool = False,
) -> list[Path]:
    """Copy the built dataset into ``staging_dir`` as flat release assets with a ``SHA256SUMS``.

    Refuses dev builds (``n_requested`` below the manifest's ``n_structures``) unless
    ``allow_partial``, so a ``--n`` build is not published by accident.
    """
    dataset_dir, staging_dir = Path(dataset_dir), Path(staging_dir)
    strata = sorted(dataset_dir.glob("*.h5"))
    if not strata:
        raise FileNotFoundError(f"no stratum files in {dataset_dir}")
    if not allow_partial:
        partial = []
        for path in strata:
            spec = specs.get(path.stem)
            requested = int(StratumReader(path).attrs.get("n_requested", 0))
            if spec is not None and requested < spec.n_structures:
                partial.append(f"{path.stem} (n_requested={requested} < {spec.n_structures})")
        if partial:
            raise ValueError(f"dev builds, not packing: {', '.join(partial)}")

    sources = {path.name: path for path in strata}
    for path in strata:
        report = dataset_dir / "reports" / f"{path.stem}.json"
        if report.exists():
            sources[path.stem + REPORT_SUFFIX] = report
    for name, path in ((INDEX, dataset_dir / INDEX), (LICENSE, Path(license_path))):
        if not path.exists():
            raise FileNotFoundError(path)
        sources[name] = path

    staging_dir.mkdir(parents=True, exist_ok=True)
    assets = []
    for name, src in sorted(sources.items()):
        dest = staging_dir / name
        shutil.copy2(src, dest)
        assets.append(dest)
    lines = [f"{sha256sum(path)}  {path.name}\n" for path in assets]
    checksums = staging_dir / CHECKSUMS
    checksums.write_text("".join(lines))
    return [*assets, checksums]


def pull(
    dataset_dir: Path,
    strata: list[str] | None = None,
    repo: str = RELEASE_REPO,
    tag: str = RELEASE_TAG,
    force: bool = False,
) -> dict[str, str]:
    """Download a prebuilt dataset release into ``dataset_dir``.

    Every asset is checked against the release's ``SHA256SUMS``. Files already present with the
    right checksum are skipped. Local files that differ (e.g. a local rebuild) are only replaced
    with ``force``. Nothing is written into ``dataset_dir`` until every download has verified.
    With ``strata``, only those strata are pulled and ``index.parquet`` is regenerated from the
    stratum files present locally.

    Downloads go through the GitHub API when a token is available (see ``github_token``), which
    private repositories require; otherwise the anonymous release URLs are used.

    Returns {asset: "downloaded" | "up to date"}.
    """
    dataset_dir = Path(dataset_dir)
    token = github_token()
    try:
        session = ReleaseSession(repo, tag, token)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dataset_dir, prefix=".pull-") as tmp:
            staging = Path(tmp)
            wanted = _select(session.checksums(staging), strata)
            status, todo = _plan(dataset_dir, wanted, force)
            session.download(staging, {name: wanted[name] for name in todo})
            for name, dest in todo.items():
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging / name, dest)
                status[name] = "downloaded"
                log.info("%s -> %s", name, dest)
    except (urllib.error.URLError, KeyError) as exc:
        raise release_error("dataset", repo, tag, token, exc) from exc
    if strata is not None:
        write_index(dataset_dir)
    return status


def _select(sums: dict[str, str], strata: list[str] | None) -> dict[str, str]:
    """The released assets to pull: everything, or only the given strata (+ license)."""
    if strata is None:
        return sums
    released = {name.removesuffix(".h5") for name in sums if name.endswith(".h5")}
    missing = sorted(set(strata) - released)
    if missing:
        raise ValueError(f"strata not in release: {missing} (released: {sorted(released)})")
    names = {f"{s}.h5" for s in strata} | {f"{s}{REPORT_SUFFIX}" for s in strata} | {LICENSE}
    return {name: digest for name, digest in sums.items() if name in names}


def _plan(dataset_dir: Path, wanted: dict[str, str], force: bool) -> tuple[dict[str, str], dict[str, Path]]:
    """Split ``wanted`` into up-to-date assets and {asset: destination} to download."""
    status, todo, conflicts = {}, {}, []
    for name, digest in wanted.items():
        dest = local_path(dataset_dir, name)
        if dest.exists() and sha256sum(dest) == digest:
            status[name] = "up to date"
        elif dest.exists() and not force:
            conflicts.append(str(dest))
        else:
            todo[name] = dest
    if conflicts:
        raise FileExistsError(
            "local files differ from the release (pass --force to overwrite): " + ", ".join(conflicts)
        )
    return status, todo
