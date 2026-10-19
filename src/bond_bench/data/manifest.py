"""Load the strata manifest (assets/benchmark_strata_manifest.csv)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "assets" / "benchmark_strata_manifest.csv"


@dataclass(frozen=True)
class StratumSpec:
    stratum_id: str
    track: str
    stratum: str
    sources: str
    source_doi: str
    n_structures: int
    geometry: str
    hydrogens_default: str
    chemistry: str
    reference_label: str
    notes: str
    access_urls: str

    def as_attrs(self) -> dict[str, str | int]:
        return asdict(self)


def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> dict[str, StratumSpec]:
    df = pl.read_csv(path, infer_schema=False)
    fields = StratumSpec.__dataclass_fields__
    specs = {}
    for row in df.iter_rows(named=True):
        kwargs = {k: (row.get(k) or "") for k in fields}
        kwargs["n_structures"] = int(kwargs["n_structures"])
        specs[row["stratum_id"]] = StratumSpec(**kwargs)
    return specs
