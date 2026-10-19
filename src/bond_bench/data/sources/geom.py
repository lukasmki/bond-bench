"""GEOM (doi:10.7910/DVN/JNGTDF) ``rdkit_folder`` access without a 50 GB download.

The archive is an uncompressed tar served from S3 (despite its ``.tar.gz``
name), with one pickle per molecule named by its SMILES (``/`` written as
``_``). A single streaming pass records every member's byte offset; individual
molecules are then fetched with HTTP range requests and cached locally.
"""

from __future__ import annotations

import hashlib
import io
import logging
import pickle
import tarfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import polars as pl
from rdkit import Chem

from ..download import USER_AGENT

log = logging.getLogger(__name__)

RDKIT_FOLDER_URL = "https://dataverse.harvard.edu/api/access/datafile/4327252"
_PREFIX = "rdkit_folder/"


def index_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "geom" / "rdkit_folder_index.parquet"


def build_index(out: Path, url: str = RDKIT_FOLDER_URL) -> Path:
    """Stream the archive once, recording (subset, smiles, offset, size) per pickle."""
    log.warning("streaming GEOM rdkit_folder (~50 GB) to index its members; this runs once")
    rows: dict[str, list] = {"subset": [], "smiles": [], "offset": [], "size": []}
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=600) as resp, tarfile.open(fileobj=resp, mode="r|*") as tar:
        for member in tar:
            if not (member.isfile() and member.name.startswith(_PREFIX) and member.name.endswith(".pickle")):
                continue
            subset, _, name = member.name[len(_PREFIX) :].partition("/")
            rows["subset"].append(subset)
            rows["smiles"].append(name[: -len(".pickle")].replace("_", "/"))
            rows["offset"].append(member.offset_data)
            rows["size"].append(member.size)
            if len(rows["smiles"]) % 50000 == 0:
                log.info("indexed %d GEOM members", len(rows["smiles"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    pl.DataFrame(rows).write_parquet(tmp)
    tmp.replace(out)
    return out


class _RangeReader:
    """Range reads from a (redirecting) URL, resolving the signed target once."""

    def __init__(self, url: str):
        self.url = url
        self.target = url
        self.size = 0
        self._resolve()

    def _resolve(self) -> None:
        request = urllib.request.Request(self.url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=120) as resp:
            self.target = resp.geturl()
            self.size = int(resp.headers["Content-Range"].rsplit("/", 1)[1])

    def read(self, offset: int, length: int) -> bytes:
        end = min(offset + length, self.size) - 1
        for attempt in range(5):
            try:
                request = urllib.request.Request(self.target, headers={"User-Agent": USER_AGENT, "Range": f"bytes={offset}-{end}"})
                with urllib.request.urlopen(request, timeout=120) as resp:
                    return resp.read()
            except (OSError, urllib.error.URLError):
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
                self._resolve()  # the signed URL may have expired
        raise AssertionError("unreachable")


def _header(block: bytes) -> tarfile.TarInfo | None:
    if len(block) < 512 or block[257:262] != b"ustar":
        return None
    try:
        return tarfile.TarInfo.frombuf(block[:512], "utf-8", "surrogateescape")
    except tarfile.HeaderError:
        return None


def _blocks(size: int) -> int:
    return -(-size // 512) * 512


def _walk_segment(reader: _RangeReader, start: int, end: int) -> list[tuple[str, int, int]]:
    """(name, data offset, size) of members whose header starts in [start, end)."""
    # Find the first header at or after ``start`` (512-byte aligned scan).
    pos, found = start, False
    while pos < end and not found:
        chunk = reader.read(pos, 1 << 21)
        for k in range(0, len(chunk) - 511, 512):
            if _header(chunk[k : k + 512]) is not None:
                pos, found = pos + k, True
                break
        else:
            pos += len(chunk) - len(chunk) % 512
            if not chunk:
                break
    out = []
    long_name = None
    while found and pos < end + 4096:  # finish a long-name pair straddling ``end``
        block = reader.read(pos, 4096)
        info = _header(block)
        if info is None:
            break
        data_start = pos + 512
        if info.type in (tarfile.GNUTYPE_LONGNAME, tarfile.XHDTYPE, tarfile.XGLTYPE):
            payload = block[512 : 512 + info.size] if 512 + info.size <= len(block) else reader.read(data_start, info.size)
            if info.type == tarfile.GNUTYPE_LONGNAME:
                long_name = payload.rstrip(b"\0").decode("utf-8", "surrogateescape")
            elif info.type == tarfile.XHDTYPE:
                for record in payload.decode("utf-8", "surrogateescape").split("\n"):
                    if " path=" in record:
                        long_name = record.split(" path=", 1)[1]
            pos = data_start + _blocks(info.size)
            continue
        if pos >= end and long_name is None:
            break
        if info.isfile():
            out.append((long_name or info.name, data_start, info.size))
        long_name = None
        pos = data_start + _blocks(info.size)
    return out


def build_index_parallel(out: Path, url: str = RDKIT_FOLDER_URL, workers: int = 32, segments: int = 512) -> Path:
    """Index the archive by hopping between tar headers with parallel range requests.

    Downloads ~1 KB per member instead of streaming the whole archive.
    """
    reader = _RangeReader(url)
    bounds = [(reader.size * i // segments) // 512 * 512 for i in range(segments + 1)]
    log.info("indexing GEOM rdkit_folder (%d bytes) in %d segments", reader.size, segments)

    def run(i: int) -> list[tuple[str, int, int]]:
        return _walk_segment(_RangeReader(url) if i % workers == 0 else reader, bounds[i], bounds[i + 1])

    by_offset: dict[int, tuple[str, int]] = {}
    with ThreadPoolExecutor(workers) as pool:
        for k, members in enumerate(pool.map(run, range(segments))):
            for name, offset, size in members:
                # A member whose long-name header lies in the previous segment may be
                # seen twice; keep the full name.
                if offset not in by_offset or len(name) > len(by_offset[offset][0]):
                    by_offset[offset] = (name, size)
            if k % 32 == 0:
                log.info("indexed %d/%d segments (%d members)", k + 1, segments, len(by_offset))

    rows: dict[str, list] = {"subset": [], "smiles": [], "offset": [], "size": []}
    for offset in sorted(by_offset):
        name, size = by_offset[offset]
        if not (name.startswith(_PREFIX) and name.endswith(".pickle")):
            continue
        subset, _, base = name[len(_PREFIX) :].partition("/")
        rows["subset"].append(subset)
        rows["smiles"].append(base[: -len(".pickle")].replace("_", "/"))
        rows["offset"].append(offset)
        rows["size"].append(size)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    pl.DataFrame(rows).write_parquet(tmp)
    tmp.replace(out)
    return out


def load_index(cache_dir: Path) -> pl.DataFrame:
    path = index_path(cache_dir)
    if not path.exists():
        build_index_parallel(path)
    return pl.read_parquet(path)


class _MolOnlyUnpickler(pickle.Unpickler):
    """GEOM pickles only reference RDKit Mol; refuse everything else."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("rdkit.Chem.rdchem", "Mol"):
            return Chem.Mol
        raise pickle.UnpicklingError(f"disallowed global {module}.{name}")


_readers: dict[str, _RangeReader] = {}
_readers_lock = threading.Lock()


def _shared_reader(url: str) -> _RangeReader:
    """One resolved reader per URL: avoids hitting the Dataverse API per member."""
    with _readers_lock:
        if url not in _readers:
            _readers[url] = _RangeReader(url)
        return _readers[url]


def load_member(cache_dir: Path, subset: str, smiles: str, offset: int, size: int, url: str = RDKIT_FOLDER_URL) -> dict:
    """Fetch (or read from cache) and safely unpickle one GEOM molecule record."""
    digest = hashlib.sha1(f"{subset}/{smiles}".encode()).hexdigest()
    path = Path(cache_dir) / "geom" / "members" / subset / f"{digest}.pickle"
    if path.exists():
        data = path.read_bytes()
    else:
        data = _shared_reader(url).read(offset, size)
        if len(data) != size:
            raise OSError(f"short read for GEOM member {smiles!r}: {len(data)} of {size} bytes")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return _MolOnlyUnpickler(io.BytesIO(data)).load()


def lowest_energy_conformer(data: dict) -> dict | None:
    """The conformer with the highest Boltzmann weight."""
    conformers = [c for c in data.get("conformers") or [] if c.get("rd_mol") is not None]
    if not conformers:
        return None
    return max(conformers, key=lambda c: (c.get("boltzmannweight") or 0.0, -(c.get("relativeenergy") or 0.0)))
