"""Sampling GEOM molecules (lowest-energy conformer, GEOM graph as label)."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Hashable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from rdkit import Chem

from ..chem import LabelError, mol_to_record
from ..sampling import derive_seed, stratified_sample
from ..schema import MoleculeRecord
from ..sources import geom
from .b1_qmugs import inchikey_no_stereo
from .base import StratumBuilder


def geom_features(cache_dir: Path, subset: str) -> pl.DataFrame:
    """Index rows of one GEOM subset with cheap descriptors from the SMILES (cached)."""
    cache = Path(cache_dir) / "geom" / f"{subset}_features.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    index = geom.load_index(cache_dir).filter(pl.col("subset") == subset)
    rows = []
    for row in index.iter_rows(named=True):
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            continue
        heavy_elements = sorted({a.GetSymbol() for a in mol.GetAtoms()} - {"C", "H"})
        rows.append({
            **row,
            "n_heavy": mol.GetNumHeavyAtoms(),
            "charged": any(a.GetFormalCharge() for a in mol.GetAtoms()),
            "hetero": "".join(heavy_elements),
            "inchikey_nostereo": inchikey_no_stereo(mol),
        })
    df = pl.DataFrame(rows).sort("smiles")
    df.write_parquet(cache)
    return df


def geom_record(builder: StratumBuilder, row: dict, data: dict, structure_id: str) -> MoleculeRecord | None:
    """Convert a GEOM molecule; the conformer graph must match the GEOM SMILES."""
    conformer = geom.lowest_energy_conformer(data)
    if conformer is None:
        builder.reject("no conformer")
        return None
    mol = Chem.Mol(conformer["rd_mol"])
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001 - RDKit raises several types
        builder.reject("conformer graph fails sanitization")
        return None
    if inchikey_no_stereo(mol) != row["inchikey_nostereo"]:
        # CREST occasionally changes connectivity; the conformer no longer
        # matches the molecule it belongs to.
        builder.reject("conformer graph differs from GEOM SMILES")
        return None
    if data.get("charge") is not None and Chem.GetFormalCharge(mol) != data["charge"]:
        builder.reject("formal charges disagree with GEOM charge")
        return None
    try:
        record = mol_to_record(
            mol,
            structure_id=structure_id,
            source_id=f"{row['subset']}/{row['smiles']}",
            meta={
                "geom_smiles": row["smiles"],
                "geom_id": int(conformer.get("geom_id", -1)),
                "boltzmann_weight": float(conformer.get("boltzmannweight") or 0.0),
            },
        )
    except LabelError as exc:
        builder.reject(f"label: {exc}")
        return None
    return record if builder.accept(record) else None


def sample_geom(
    builder: StratumBuilder,
    subset: str,
    n: int,
    stratum: Callable[[dict], Hashable],
    id_prefix: str,
    rows_filter: pl.Expr | None = None,
    workers: int = 8,
) -> list[MoleculeRecord]:
    """Stratified sample of ``n`` validated GEOM molecules from ``subset``."""
    features = geom_features(builder.cache_dir, subset)
    if rows_filter is not None:
        features = features.filter(rows_filter)
    builder.funnel[f"geom_{subset}_eligible"] = features.height
    rows = features.to_dicts()
    n_pre = int(n * builder.params.get("oversample", 1.3)) + 10
    pre = stratified_sample(rows, n_pre, derive_seed(builder.seed, builder.stratum_id, subset, "pre"), stratum, key=lambda r: r["smiles"])
    builder.funnel[f"geom_{subset}_preselected"] = len(pre)

    def load(row: dict) -> tuple[dict, dict | None]:
        try:
            return row, geom.load_member(builder.cache_dir, subset, row["smiles"], row["offset"], row["size"])
        except Exception:  # noqa: BLE001 - counted as a rejection below
            return row, None

    candidates, failures, successes = [], 0, 0
    with ThreadPoolExecutor(workers) as pool:
        for row, data in pool.map(load, pre):
            if data is None:
                failures += 1
                builder.reject("GEOM member download failed")
                if failures >= 25 and successes == 0:
                    raise RuntimeError("GEOM downloads are failing; aborting instead of retrying every member")
                continue
            successes += 1
            uid = hashlib.sha1(row["smiles"].encode()).hexdigest()[:12]
            record = geom_record(builder, row, data, f"{id_prefix}-{uid}")
            if record is not None:
                record.meta["_stratum"] = stratum(row)
                candidates.append(record)
    builder.funnel[f"geom_{subset}_valid"] = len(candidates)
    selected = stratified_sample(
        candidates, n, derive_seed(builder.seed, builder.stratum_id, subset, "final"),
        stratum=lambda r: r.meta["_stratum"], key=lambda r: r.structure_id,
    )
    for record in selected:
        record.meta.pop("_stratum")
    return selected
