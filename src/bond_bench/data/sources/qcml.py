"""QCML (doi:10.1038/s41597-025-04720-7) DFT records without TensorFlow.

The public bucket ``gs://qcml-datasets/tfds/qcml`` stores one TFDS config per
feature group; record *i* of shard *k* is the same structure in every
``dft_*`` config. Records are shuffled, so a byte prefix of a shard is a
random sample. We range-download aligned prefixes of ``dft_metadata``
(SMILES, conformation index) and ``dft_force_field`` (coordinates, charge,
multiplicity, outlier flag) and join them by record index.

``conformation_seq == 0`` is the GFN0-xTB optimized conformer; larger values
are normal-mode samples (0-1000 K) of it. Positions are in Bohr. The SMILES
(Open Babel canonical) lists heavy atoms in the order of the records'
atoms, followed by hydrogens as RDKit ``AddHs`` adds them.
"""

from __future__ import annotations

import struct
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

from ..download import USER_AGENT

BASE_URL = "https://storage.googleapis.com/qcml-datasets/tfds/qcml/{config}/1.0.0/qcml-full.tfrecord-{shard:05d}-of-00011"
BOHR_TO_ANGSTROM = 0.529177210903

# Bytes per record are ~340 (metadata) and ~570 (force field); prefixes are
# sized in proportion so that both cover (almost) the same records.
BYTES_PER_RECORD = {"dft_metadata": 345, "dft_force_field": 575}


def fetch_prefix(config: str, shard: int, n_bytes: int, dest: Path) -> Path:
    """Download the first ``n_bytes`` of a shard (extends a shorter cached prefix)."""
    have = dest.stat().st_size if dest.exists() else 0
    if have >= n_bytes:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": USER_AGENT, "Range": f"bytes={have}-{n_bytes - 1}"}
    request = urllib.request.Request(BASE_URL.format(config=config, shard=shard), headers=headers)
    with urllib.request.urlopen(request, timeout=600) as resp, open(dest, "ab") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    return dest


# -- TFRecord / tf.train.Example decoding ------------------------------------------


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            return result, i


def _fields(buf: bytes) -> Iterator[tuple[int, bytes | int]]:
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 2:
            n, i = _varint(buf, i)
            yield field, buf[i : i + n]
            i += n
        elif wire == 0:
            value, i = _varint(buf, i)
            yield field, value
        elif wire == 5:
            yield field, buf[i : i + 4]
            i += 4
        elif wire == 1:
            yield field, buf[i : i + 8]
            i += 8
        else:
            raise ValueError(f"unsupported wire type {wire}")


def _feature(buf: bytes):
    for field, value in _fields(buf):
        if field == 1:  # BytesList
            return [v for _, v in _fields(value)]
        if field == 2:  # FloatList (packed)
            parts = [v for _, v in _fields(value)]
            return np.frombuffer(b"".join(parts), "<f4")
        if field == 3:  # Int64List (packed varints)
            out = []
            for _, v in _fields(value):
                if isinstance(v, int):
                    out.append(v)
                else:
                    j = 0
                    while j < len(v):
                        x, j = _varint(v, j)
                        out.append(x)
            return np.array([x - (1 << 64) if x >= 1 << 63 else x for x in out], dtype=np.int64)
    return None


def parse_example(buf: bytes) -> dict:
    out = {}
    for _, features in _fields(buf):
        for _, entry in _fields(features):
            key = value = None
            for field, v in _fields(entry):
                if field == 1:
                    key = v.decode()
                elif field == 2:
                    value = _feature(v)
            out[key] = value
    return out


def read_records(path: Path) -> Iterator[dict]:
    """Complete records of a (possibly truncated) TFRecord file."""
    data = path.read_bytes()
    i = 0
    while i + 12 <= len(data):
        (length,) = struct.unpack("<Q", data[i : i + 8])
        if i + 16 + length > len(data):
            break
        yield parse_example(data[i + 12 : i + 12 + length])
        i += 16 + length


def _scalar(value):
    if isinstance(value, list):
        return value[0].decode() if value else ""
    return value.item() if len(value) == 1 else value


def load_records(
    cache_dir: Path, n_records: int, shard: int = 0, keep: Callable[[dict], bool] | None = None
) -> list[dict]:
    """Joined records among the first ``n_records``: smiles, seq, Z, positions (Å), charge, ...

    ``keep`` filters on the metadata fields (smiles, seq, parent_seq, key_hash)
    before coordinates are decoded.
    """
    paths = {}
    for config, per_record in BYTES_PER_RECORD.items():
        dest = Path(cache_dir) / "qcml" / f"{config}-{shard:05d}.prefix"
        paths[config] = fetch_prefix(config, shard, int(n_records * per_record * 1.05), dest)
    out = []
    for i, (meta, ff) in enumerate(zip(read_records(paths["dft_metadata"]), read_records(paths["dft_force_field"]))):
        if i >= n_records:
            break
        if _scalar(meta["key_hash"]) != _scalar(ff["key_hash"]):
            raise ValueError("QCML configs out of alignment")
        head = {
            "key_hash": str(_scalar(meta["key_hash"])),
            "smiles": _scalar(meta["smiles"]),
            "seq": int(_scalar(meta["conformation_seq"])),
            "parent_seq": int(_scalar(meta["conformation_parent_seq"])),
        }
        if keep is not None and not keep(head):
            continue
        n = int(_scalar(meta["num_atoms"]))
        out.append({
            **head,
            "charge": int(_scalar(ff["charge"])),
            "multiplicity": int(_scalar(ff["multiplicity"])),
            "is_outlier": bool(_scalar(ff["is_outlier"])),
            "atomic_numbers": ff["atomic_numbers"].astype(np.int64),
            "positions": ff["positions"].reshape(n, 3).astype(np.float64) * BOHR_TO_ANGSTROM,
        })
    return out
