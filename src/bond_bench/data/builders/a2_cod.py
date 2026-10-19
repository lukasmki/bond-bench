"""A2: organic small-molecule crystals from the COD, labeled by curated SMILES.

Candidates are entries whose curated SMILES Merkys et al. (2023) found
isomorphic with the coordinate-derived structure; entries where the deposited
chemical name also agrees ("name+coords") are preferred over "coords" only.
Each record is one formula unit of the curated SMILES (all ions/solvates in
their stoichiometric counts), assembled as a contact cluster from the
symmetry-expanded crystal with experimental hydrogen positions.
"""

from __future__ import annotations

import gzip
import itertools
import logging
from collections import Counter, defaultdict

import numpy as np
import polars as pl
from rdkit import Chem

from ..chem import LabelError, mol_to_record
from ..download import fetch
from ..sampling import derive_seed, stratified_sample, tertile, tertile_bounds
from ..schema import MoleculeRecord
from ..sources import cod
from .base import StratumBuilder

log = logging.getLogger(__name__)

ORGANIC = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I"}
COL_COORDS = "Curated SMILES / Coordinate-derived annot."
COL_NAME = "Curated SMILES / Chem. name from CIF"


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "", "?") else None
    except ValueError:
        return None


class A2Builder(StratumBuilder):
    stratum_id = "A2"
    source_name = "cod"
    defaults = {
        "max_r": 0.05,
        "max_heavy_atoms": 120,
        "oversample": 1.4,
    }

    def fetch(self) -> None:
        fetch(cod.SMILES_URL, self.raw_dir / "allcod.smi")
        fetch(cod.MERKYS_URL, self.raw_dir / "merkys2023_af2.csv.gz")

    # -- candidate selection -------------------------------------------------

    def _candidates(self) -> list[dict]:
        smiles = cod.load_smiles(self.raw_dir / "allcod.smi")
        with gzip.open(self.raw_dir / "merkys2023_af2.csv.gz", "rb") as f:
            verdicts = pl.read_csv(f.read(), infer_schema=False)
        verdicts = verdicts.filter(pl.col(COL_COORDS) == "isomorphic")
        self.funnel["isomorphic_coords"] = verdicts.height
        rows = []
        for row in verdicts.iter_rows(named=True):
            cod_id = row["id"]
            smi = smiles.get(cod_id)
            if smi is None:
                self.reject("not in current curated SMILES")
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                self.reject("curated SMILES unparsable")
                continue
            if not {a.GetSymbol() for a in mol.GetAtoms()} | {"H"} <= ORGANIC:
                self.reject("element outside organic set")
                continue
            n_heavy = mol.GetNumHeavyAtoms()
            if n_heavy > self.params["max_heavy_atoms"]:
                self.reject("too large")
                continue
            rows.append({
                "cod_id": cod_id,
                "smiles": smi,
                "agreement": "name+coords" if row[COL_NAME] == "isomorphic" else "coords",
                "n_heavy": n_heavy,
                "charged": any(a.GetFormalCharge() for a in mol.GetAtoms()),
            })
        self.funnel["organic_with_smiles"] = len(rows)
        return rows

    def _metadata_filter(self, rows: list[dict]) -> list[dict]:
        meta = {m["file"]: m for m in cod.query_metadata(sorted(r["cod_id"] for r in rows), self.raw_dir / "metadata")}
        kept = []
        for row in rows:
            m = meta.get(row["cod_id"])
            if m is None:
                self.reject("no metadata (retracted/duplicate/error)")
                continue
            r_factor = _float(m.get("Robs")) or _float(m.get("Rall"))
            flags = m.get("flags") or ""
            if m.get("method"):
                self.reject("not single-crystal diffraction")
            elif (m.get("radType") or "").lower().startswith("neutron"):
                self.reject("neutron diffraction")
            elif r_factor is None or r_factor > self.params["max_r"]:
                self.reject("R factor missing or too high")
            elif "disorder" in flags:
                self.reject("disorder flagged")
            elif "coordinates" not in flags:
                self.reject("no coordinates")
            else:
                row.update(r_factor=r_factor, temperature=_float(m.get("diffrtemp")) or _float(m.get("celltemp")),
                           year=m.get("year") or "", doi=m.get("doi") or "")
                kept.append(row)
        self.funnel["metadata_ok"] = len(kept)
        return kept

    # -- crystal processing ----------------------------------------------------

    def _record(self, row: dict) -> MoleculeRecord | None:
        path = self.raw_dir / "cif" / cod.cif_relpath(row["cod_id"])
        if not path.exists():
            self.reject("CIF download failed")
            return None
        try:
            atoms, issues = cod.unit_cell_atoms(path)
        except Exception:  # noqa: BLE001 - malformed CIFs
            self.reject("CIF unreadable")
            return None
        if issues:
            self.reject(f"crystal: {issues[0]}")
            return None
        try:
            comps = cod.components(atoms)
        except Exception:  # noqa: BLE001 - unexpected species in the cell
            self.reject("crystal: molecule extraction failed")
            return None
        if any(c.polymeric for c in comps):
            self.reject("crystal: polymeric component")
            return None
        fragments = cod.reference_fragments(row["smiles"])
        if fragments is None:
            self.reject("curated SMILES unparsable")
            return None

        needed = Counter(cod.skeleton_key(f) for f in fragments)
        reference = {cod.skeleton_key(f): f for f in fragments}
        by_key: dict[str, list[cod.Component]] = defaultdict(list)
        for comp in comps:
            by_key[cod.skeleton_key(comp.mol)].append(comp)
        if set(by_key) != set(needed):
            self.reject("crystal species differ from curated SMILES (e.g. missing H)")
            return None

        chosen = self._formula_unit(needed, by_key, atoms.cell.array)
        if chosen is None:
            self.reject("not enough copies for formula unit")
            return None
        try:
            mol = None
            for key, comp in chosen:
                labeled = cod.transfer_label(comp, reference[key])
                mol = labeled if mol is None else Chem.CombineMols(mol, labeled)
            Chem.SanitizeMol(mol)
            record = mol_to_record(
                mol,
                structure_id=f"A2-{row['cod_id']}",
                source_id=row["cod_id"],
                meta={
                    "cod_id": row["cod_id"],
                    "agreement": row["agreement"],
                    "r_factor": row["r_factor"],
                    "temperature": row["temperature"],
                    "n_components": len(chosen),
                    "doi": row["doi"],
                },
            )
        except LabelError as exc:
            self.reject(f"label: {str(exc).split(':')[0]}")
            return None
        except Exception:  # noqa: BLE001 - RDKit failures on odd chemistry
            self.reject("label: sanitization failed")
            return None
        return record if self.accept(record) else None

    @staticmethod
    def _formula_unit(
        needed: Counter, by_key: dict[str, list[cod.Component]], cell: np.ndarray
    ) -> list[tuple[str, cod.Component]] | None:
        """Pick stoichiometric copies forming a compact cluster (lattice images allowed)."""
        shifts = [np.array(n) @ cell for n in itertools.product((-1, 0, 1), repeat=3)]
        order = sorted(needed, key=lambda k: (-by_key[k][0].mol.GetNumAtoms(), k))
        chosen = [(order[0], by_key[order[0]][0])]
        used = {(order[0], 0)}
        cluster = chosen[0][1].positions
        for key in order:
            for _ in range(needed[key] - (key == order[0])):
                best, best_d = None, np.inf
                for idx, comp in enumerate(by_key[key]):
                    if (key, idx) in used:
                        continue
                    for shift in shifts:
                        moved = comp.positions + shift
                        d = np.min(np.linalg.norm(moved[:, None, :] - cluster[None, :, :], axis=-1))
                        if d < best_d:
                            best, best_d = (idx, cod.Component(comp.mol, moved, False)), d
                if best is None:
                    return None
                used.add((key, best[0]))
                chosen.append((key, best[1]))
                cluster = np.vstack([cluster, best[1].positions])
        return chosen

    def records(self) -> list[MoleculeRecord]:
        rows = self._metadata_filter(self._candidates())
        bounds = tertile_bounds([r["n_heavy"] for r in rows])
        self.extra_attrs["size_tertile_bounds"] = list(bounds)

        def stratum(r: dict) -> tuple:
            return (r["agreement"], r["charged"], tertile(r["n_heavy"], bounds))

        # Prefer "name+coords" agreement: sample it first, then fill with "coords".
        n_pre = int(self.n * self.params["oversample"]) + 10
        strict = [r for r in rows if r["agreement"] == "name+coords"]
        loose = [r for r in rows if r["agreement"] == "coords"]
        pre = stratified_sample(strict, n_pre, derive_seed(self.seed, "A2", "strict"), stratum, key=lambda r: r["cod_id"])
        if len(pre) < n_pre:
            pre += stratified_sample(loose, n_pre - len(pre), derive_seed(self.seed, "A2", "loose"), stratum, key=lambda r: r["cod_id"])
        self.funnel["preselected"] = len(pre)

        cod.rsync_cifs([r["cod_id"] for r in pre], self.raw_dir / "cif")
        candidates = []
        for row in pre:
            record = self._record(row)
            if record is not None:
                record.meta["_stratum"] = stratum(row)
                candidates.append(record)
        self.funnel["valid"] = len(candidates)

        strict_records = [r for r in candidates if r.meta["agreement"] == "name+coords"]
        loose_records = [r for r in candidates if r.meta["agreement"] == "coords"]
        key = lambda r: r.structure_id  # noqa: E731
        out = stratified_sample(strict_records, self.n, derive_seed(self.seed, "A2", "final-strict"), lambda r: r.meta["_stratum"], key)
        if len(out) < self.n:
            out += stratified_sample(loose_records, self.n - len(out), derive_seed(self.seed, "A2", "final-loose"), lambda r: r.meta["_stratum"], key)
        for record in out:
            record.meta.pop("_stratum")
        return sorted(out, key=key)
