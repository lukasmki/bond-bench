"""Dataset-level consistency checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import polars as pl

from .chem import LabelError, identifiers, validate_record
from .manifest import StratumSpec
from .sampling import derive_seed
from .schema import ATOM_FIELDS, BOND_FIELDS, SCHEMA_VERSION
from .store import StratumReader


@dataclass
class ValidationResult:
    path: Path
    n_structures: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_stratum(
    path: Path,
    spec: StratumSpec | None = None,
    sample: int | None = 200,
    seed: int = 0,
) -> ValidationResult:
    result = ValidationResult(path)
    with h5py.File(path, "r") as f:
        if f.attrs.get("schema_version") != SCHEMA_VERSION:
            result.errors.append(f"schema_version {f.attrs.get('schema_version')!r} != {SCHEMA_VERSION}")
        for group, fields in (("atoms", ATOM_FIELDS), ("bonds", BOND_FIELDS)):
            for name in fields:
                if f"{group}/{name}" not in f:
                    result.errors.append(f"missing {group}/{name}")
    if result.errors:
        return result

    reader = StratumReader(path)
    result.n_structures = n = len(reader)
    mol = reader._mol
    if mol["n_atoms"].sum() != len(reader._atoms["atomic_numbers"]):
        result.errors.append("atom offsets inconsistent with atom arrays")
    if mol["n_bonds"].sum() != len(reader._bonds["bond_orders"]):
        result.errors.append("bond offsets inconsistent with bond arrays")
    if len(set(reader.ids())) != n:
        result.errors.append("duplicate structure_id")

    expected = int(reader.attrs.get("n_requested", spec.n_structures if spec else n))
    if n != expected:
        result.warnings.append(f"{n} structures, expected {expected}")

    indices = np.arange(n)
    if sample is not None and sample < n:
        rng = np.random.default_rng(derive_seed(seed, "validate", path.name))
        indices = np.sort(rng.choice(n, size=sample, replace=False))
    for i in indices:
        record = reader[int(i)]
        try:
            rdmol = validate_record(record)
        except LabelError as exc:
            result.errors.append(f"{record.structure_id}: {exc}")
            continue
        smiles, inchikey = identifiers(rdmol)
        if inchikey != record.inchikey:
            result.errors.append(f"{record.structure_id}: stored InChIKey does not match label")
        stripped = record.strip_hydrogens()
        try:
            validate_record(stripped)
        except LabelError as exc:
            result.errors.append(f"{record.structure_id}: H-stripped label invalid: {exc}")
    return result


def validate_index(dataset_dir: Path, results: list[ValidationResult]) -> list[str]:
    index_path = dataset_dir / "index.parquet"
    if not index_path.exists():
        return ["index.parquet missing"]
    index = pl.read_parquet(index_path)
    errors = []
    for res in results:
        stratum = res.path.stem
        count = index.filter(pl.col("stratum_id") == stratum).height
        if count != res.n_structures:
            errors.append(f"index.parquet has {count} rows for {stratum}, file has {res.n_structures}")
    return errors
