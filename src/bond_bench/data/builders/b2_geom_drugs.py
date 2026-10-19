"""B2: GEOM-DRUGS lowest-energy conformers (in-distribution reference)."""

from __future__ import annotations

from ..sampling import tertile, tertile_bounds
from ..schema import MoleculeRecord
from ..sources import geom
from .base import StratumBuilder
from .geom_common import geom_features, sample_geom


class B2Builder(StratumBuilder):
    stratum_id = "B2"
    source_name = "geom"
    defaults = {"oversample": 1.3, "workers": 8}

    def fetch(self) -> None:
        geom.load_index(self.cache_dir)

    def records(self) -> list[MoleculeRecord]:
        features = geom_features(self.cache_dir, "drugs")
        bounds = tertile_bounds(features["n_heavy"].to_list())
        self.extra_attrs["size_tertile_bounds"] = list(bounds)
        return sample_geom(
            self,
            "drugs",
            self.n,
            stratum=lambda r: (tertile(r["n_heavy"], bounds), r["charged"]),
            id_prefix="B2",
            workers=self.params["workers"],
        )
