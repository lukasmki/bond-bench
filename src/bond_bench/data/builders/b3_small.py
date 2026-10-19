"""B3: small, elementally diverse molecules from QCML (equilibrium) and GEOM-QM9.

Half of the stratum comes from QCML GFN0-optimized conformers (<= 8 heavy
atoms, main-group elements up to Te/I) stratified by hetero-element
signature to maximize element coverage; the other half are GEOM-QM9
lowest-energy conformers (<= 9 heavy atoms, CHONF).
"""

from __future__ import annotations

import hashlib

import polars as pl

from ..sampling import derive_seed, stratified_sample
from ..schema import MoleculeRecord
from ..sources import geom, qcml
from .base import StratumBuilder
from .geom_common import sample_geom
from .qcml_common import hetero_signature, qcml_record


class B3Builder(StratumBuilder):
    stratum_id = "B3"
    source_name = "qcml"
    defaults = {
        "qcml_fraction": 0.5,
        # Records read from the start of QCML shard 0 (~0.45% are equilibrium).
        "qcml_records": 900_000,
        "oversample": 1.3,
        "workers": 8,
    }

    def fetch(self) -> None:
        geom.load_index(self.cache_dir)

    def _qcml_part(self, n: int) -> list[MoleculeRecord]:
        equilibrium = qcml.load_records(self.cache_dir, self.params["qcml_records"], keep=lambda r: r["seq"] == 0)
        self.funnel["qcml_equilibrium"] = len(equilibrium)
        # One conformer per graph: the lowest parent_seq (first GFN0 conformer).
        by_graph: dict[str, dict] = {}
        for r in sorted(equilibrium, key=lambda r: (r["smiles"], r["parent_seq"], r["key_hash"])):
            by_graph.setdefault(r["smiles"], r)
        self.funnel["qcml_equilibrium_graphs"] = len(by_graph)

        candidates = []
        for rec in by_graph.values():
            uid = hashlib.sha1(rec["key_hash"].encode()).hexdigest()[:12]
            record = qcml_record(self, rec, f"B3-qcml-{uid}", {"source": "QCML", "qcml_key": rec["key_hash"], "hetero": hetero_signature(rec["smiles"])})
            if record is not None:
                candidates.append(record)
        self.funnel["qcml_valid"] = len(candidates)
        return stratified_sample(
            candidates, n, derive_seed(self.seed, "B3", "qcml"),
            stratum=lambda r: r.meta["hetero"], key=lambda r: r.structure_id,
        )

    def records(self) -> list[MoleculeRecord]:
        n_qcml = round(self.n * self.params["qcml_fraction"])
        part_qcml = self._qcml_part(n_qcml)
        part_qm9 = sample_geom(
            self, "qm9", self.n - len(part_qcml),
            stratum=lambda r: (r["hetero"], r["charged"]),
            id_prefix="B3-qm9",
            rows_filter=pl.col("n_heavy") <= 9,
            workers=self.params["workers"],
        )
        for record in part_qm9:
            record.meta.update(source="GEOM-QM9", hetero=hetero_signature(record.meta["geom_smiles"]))
        return sorted(part_qcml + part_qm9, key=lambda r: r.structure_id)
