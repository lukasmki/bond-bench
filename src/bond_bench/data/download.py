"""Cached downloads into the raw-data cache directory."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .manifest import REPO_ROOT

log = logging.getLogger(__name__)

USER_AGENT = "bond-bench/0.1 (+https://github.com/lukasmki/bond-bench)"


class _NoCrossHostAuthRedirect(urllib.request.HTTPRedirectHandler):
    """Drop ``Authorization`` when a redirect leaves the original host (e.g. GitHub -> asset storage)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).hostname != urllib.parse.urlsplit(req.full_url).hostname:
            new.remove_header("Authorization")
        return new


_OPENER = urllib.request.build_opener(_NoCrossHostAuthRedirect)


def default_cache_dir() -> Path:
    return Path(os.environ.get("BOND_BENCH_CACHE", REPO_ROOT / "data" / "raw"))


def sha256sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def fetch(
    url: str, dest: Path, retries: int = 4, timeout: float = 120.0, headers: dict[str, str] | None = None
) -> Path:
    """Download ``url`` to ``dest`` unless it already exists.

    Writes a ``<dest>.sha256`` sidecar so builds can record source checksums.
    """
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    for attempt in range(retries):
        try:
            with _OPENER.open(request, timeout=timeout) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
            break
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 410) or attempt == retries - 1:
                tmp.unlink(missing_ok=True)
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                tmp.unlink(missing_ok=True)
                raise
        wait = 2**attempt
        log.warning("retrying %s in %ss", url, wait)
        time.sleep(wait)
    os.replace(tmp, dest)
    dest.with_name(dest.name + ".sha256").write_text(sha256sum(dest) + "\n")
    return dest


def fetch_many(
    items: Iterable[tuple[str, Path]], workers: int = 8, headers: dict[str, str] | None = None
) -> dict[Path, Exception | None]:
    """Fetch (url, dest) pairs concurrently. Returns the error (or None) per dest."""

    def one(item: tuple[str, Path]) -> tuple[Path, Exception | None]:
        url, dest = item
        try:
            fetch(url, dest, headers=headers)
            return dest, None
        except Exception as exc:  # noqa: BLE001 - report per-item failures
            return dest, exc

    with ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(one, list(items)))


def post_json(url: str, payload: dict, cache: Path | None = None, retries: int = 4) -> dict:
    """POST a JSON payload and return the decoded response, optionally cached."""
    if cache is not None and cache.exists():
        return json.loads(cache.read_text())
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=300) as resp:
                raw = resp.read()
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(raw)
    return json.loads(raw) if raw else {}


def source_checksums(paths: Iterable[Path]) -> dict[str, str]:
    out = {}
    for path in paths:
        sidecar = path.with_name(path.name + ".sha256")
        if sidecar.exists():
            out[path.name] = sidecar.read_text().strip()
    return out
