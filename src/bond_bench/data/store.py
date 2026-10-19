"""HDF5 storage: one ragged-array file per stratum, plus a global index.

Layout of ``<stratum_id>.h5``::

    /molecules/{structure_id, source_id, smiles, inchikey, total_charge,
                multiplicity, n_atoms, n_bonds, atom_offset, bond_offset,
                meta_<key>...}
    /atoms/{atomic_numbers, positions, formal_charges,
            implicit_hydrogens, radical_electrons}
    /bonds/{bonds, bond_orders, aromatic}       # bond indices are per-molecule
    attrs: manifest row, schema_version, build parameters, provenance
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import polars as pl

from .schema import ATOM_FIELDS, BOND_FIELDS, SCHEMA_VERSION, MoleculeRecord

_STR = h5py.string_dtype()
_MOL_STR_FIELDS = ("structure_id", "source_id", "smiles", "inchikey")
_MOL_INT_FIELDS = ("total_charge", "multiplicity")


def _meta_column(values: list[Any]) -> np.ndarray:
    present = [v for v in values if v is not None]
    if present and all(isinstance(v, (bool, np.bool_)) for v in present):
        return np.array([bool(v) if v is not None else False for v in values])
    if present and all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in present):
        if len(present) == len(values):
            return np.array(values, dtype=np.int64)
    if present and all(isinstance(v, (int, float, np.number)) for v in present):
        return np.array([np.nan if v is None else float(v) for v in values])
    return np.array(["" if v is None else str(v) for v in values], dtype=object)


def write_stratum(
    path: Path | str,
    records: Sequence[MoleculeRecord],
    attrs: dict[str, Any],
) -> Path:
    """Write records to ``path`` atomically (via a temporary file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    ids = [r.structure_id for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate structure_id in records")

    n_atoms = np.array([r.n_atoms for r in records], dtype=np.int64)
    n_bonds = np.array([r.n_bonds for r in records], dtype=np.int64)
    meta_keys = sorted({k for r in records for k in r.meta})

    with h5py.File(tmp, "w") as f:
        mols = f.create_group("molecules")
        for name in _MOL_STR_FIELDS:
            mols.create_dataset(name, data=[getattr(r, name) for r in records], dtype=_STR)
        for name in _MOL_INT_FIELDS:
            mols.create_dataset(name, data=np.array([getattr(r, name) for r in records], dtype=np.int32))
        mols.create_dataset("n_atoms", data=n_atoms)
        mols.create_dataset("n_bonds", data=n_bonds)
        mols.create_dataset("atom_offset", data=np.concatenate([[0], np.cumsum(n_atoms)[:-1]]).astype(np.int64))
        mols.create_dataset("bond_offset", data=np.concatenate([[0], np.cumsum(n_bonds)[:-1]]).astype(np.int64))
        for key in meta_keys:
            col = _meta_column([r.meta.get(key) for r in records])
            mols.create_dataset(f"meta_{key}", data=col, dtype=_STR if col.dtype == object else col.dtype)

        for group_name, fields in (("atoms", ATOM_FIELDS), ("bonds", BOND_FIELDS)):
            group = f.create_group(group_name)
            for name, dtype in fields.items():
                parts = [getattr(r, name) for r in records]
                width = parts[0].shape[1:] if parts else ()
                data = np.concatenate(parts) if parts else np.zeros((0, *width))
                group.create_dataset(name, data=data.astype(dtype), compression="gzip", compression_opts=4)

        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["n_structures"] = len(records)
        for key, value in attrs.items():
            f.attrs[key] = value if isinstance(value, (str, int, float, bool)) else json.dumps(value)

    os.replace(tmp, path)
    return path


class StratumReader:
    """Random access to the records of one stratum file (loaded into memory)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        with h5py.File(self.path, "r") as f:
            self.attrs = dict(f.attrs)
            mols = f["molecules"]
            self._mol = {}
            for name, ds in mols.items():
                data = ds[()]
                if ds.dtype.kind == "O":
                    data = np.array([x.decode() if isinstance(x, bytes) else x for x in data], dtype=object)
                self._mol[name] = data
            self._atoms = {name: f["atoms"][name][()] for name in ATOM_FIELDS}
            self._bonds = {name: f["bonds"][name][()] for name in BOND_FIELDS}
        self.meta_keys = sorted(k[5:] for k in self._mol if k.startswith("meta_"))
        self._index = {sid: i for i, sid in enumerate(self._mol["structure_id"])}

    @property
    def stratum_id(self) -> str:
        return str(self.attrs.get("stratum_id", self.path.stem))

    def __len__(self) -> int:
        return len(self._mol["structure_id"])

    def ids(self) -> list[str]:
        return list(self._mol["structure_id"])

    def molecule_table(self) -> pl.DataFrame:
        return pl.DataFrame({k: (v.tolist() if v.dtype == object else v) for k, v in self._mol.items()})

    def get(self, structure_id: str, strip_h: bool = False) -> MoleculeRecord:
        record = self[self._index[structure_id]]
        return record.strip_hydrogens() if strip_h else record

    def __getitem__(self, i: int) -> MoleculeRecord:
        if not -len(self) <= i < len(self):
            raise IndexError(i)
        i = i % len(self)
        a0, na = int(self._mol["atom_offset"][i]), int(self._mol["n_atoms"][i])
        b0, nb = int(self._mol["bond_offset"][i]), int(self._mol["n_bonds"][i])
        meta = {}
        for key in self.meta_keys:
            value = self._mol[f"meta_{key}"][i]
            meta[key] = value.item() if isinstance(value, np.generic) else value
        return MoleculeRecord(
            structure_id=str(self._mol["structure_id"][i]),
            source_id=str(self._mol["source_id"][i]),
            smiles=str(self._mol["smiles"][i]),
            inchikey=str(self._mol["inchikey"][i]),
            total_charge=int(self._mol["total_charge"][i]),
            multiplicity=int(self._mol["multiplicity"][i]),
            meta=meta,
            **{k: v[a0 : a0 + na] for k, v in self._atoms.items()},
            **{k: v[b0 : b0 + nb] for k, v in self._bonds.items()},
        )

    def iter(self, strip_h: bool = False) -> Iterator[MoleculeRecord]:
        for i in range(len(self)):
            record = self[i]
            yield record.strip_hydrogens() if strip_h else record

    def __iter__(self) -> Iterator[MoleculeRecord]:
        return self.iter()


INDEX_META_KEYS = ("parent_id", "parent_stratum", "sigma", "res_bin", "resolution", "ccd_id", "pdb_id", "chembl_id")


def write_index(dataset_dir: Path | str, readers: Iterable[StratumReader] | None = None) -> Path:
    """(Re)generate ``index.parquet`` across all stratum files in ``dataset_dir``."""
    dataset_dir = Path(dataset_dir)
    if readers is None:
        readers = [StratumReader(p) for p in sorted(dataset_dir.glob("*.h5"))]
    frames = []
    for reader in readers:
        heavy = (reader._atoms["atomic_numbers"] != 1).astype(np.int64)
        offsets = reader._mol["atom_offset"]
        n_heavy = np.add.reduceat(heavy, offsets) if len(offsets) else np.zeros(0, dtype=np.int64)
        cols: dict[str, Any] = {
            "stratum_id": [reader.stratum_id] * len(reader),
            "track": [str(reader.attrs.get("track", ""))] * len(reader),
            "structure_id": reader._mol["structure_id"].tolist(),
            "source_id": reader._mol["source_id"].tolist(),
            "smiles": reader._mol["smiles"].tolist(),
            "inchikey": reader._mol["inchikey"].tolist(),
            "total_charge": reader._mol["total_charge"],
            "multiplicity": reader._mol["multiplicity"],
            "n_atoms": reader._mol["n_atoms"],
            "n_heavy": n_heavy,
        }
        for key in INDEX_META_KEYS:
            if key in reader.meta_keys:
                col = reader._mol[f"meta_{key}"]
                cols[key] = [str(x) for x in col] if col.dtype == object else col.astype(str).tolist()
        frames.append(pl.DataFrame(cols))
    out = dataset_dir / "index.parquet"
    df = pl.concat(frames, how="diagonal") if frames else pl.DataFrame()
    df.write_parquet(out)
    return out
