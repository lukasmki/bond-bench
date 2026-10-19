"""A1: PDB ligand instances in three resolution bins, labeled from the CCD.

Pipeline per resolution bin:

1. RCSB Search: X-ray entries in the bin with at least one ligand instance
   passing the density-fit thresholds.
2. Entries are visited in a seeded random order. RCSB GraphQL supplies
   per-instance validation scores, which are filtered Platinum-style; per-CCD
   and per-entry caps are applied greedily, so the walk stops once enough
   candidates are found.
3. Candidate mmCIF files and CCD definitions are downloaded; ligand heavy
   atoms are matched to the CCD by atom name, and covalently attached or
   incomplete/disordered instances are dropped.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import gemmi
import numpy as np

from ..chem import set_identifiers
from ..download import fetch, fetch_many, post_json
from ..sampling import derive_seed
from ..schema import MoleculeRecord
from .base import StratumBuilder

log = logging.getLogger(__name__)

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"
MMCIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"
CCD_URL = "https://files.rcsb.org/ligands/view/{ccd_id}.cif"

RESOLUTION_BINS = {"le1.5": (None, 1.5), "1.5-2.5": (1.5, 2.5), "gt2.5": (2.5, None)}

ALLOWED_ELEMENTS = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Se"}

# Crystallization additives, buffers, detergents and solvents with >= 6 heavy
# atoms (smaller ones are removed by the heavy-atom filter).
EXCLUDED_CCD = {
    "GOL", "PEG", "PGE", "PG4", "1PE", "P6G", "2PE", "PE4", "PE5", "7PE", "XPE",
    "12P", "15P", "P33", "PE8", "PG0", "MPD", "MRD", "MES", "EPE", "TRS", "CIT",
    "FLC", "TAR", "TLA", "MLI", "B3P", "BTB", "CXS", "CPS", "CHT", "HED", "DTT",
    "DTU", "BOG", "LDA", "C8E", "HTG", "LMT", "DMU", "SOG", "OLC", "OLA", "OLB",
    "MYR", "PLM", "STE", "LMR", "SRT", "NHE", "MOH", "BU3", "PGO", "PGR", "TBU",
    "IPH", "BEN", "AE3", "AE4", "CAC", "EDT", "EDO", "DMS", "ACT", "FMT", "IMD",
}

GRAPHQL_QUERY = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    rcsb_entry_info { resolution_combined }
    nonpolymer_entities {
      nonpolymer_comp {
        chem_comp { id type formula pdbx_release_status }
        rcsb_chem_comp_info { atom_count_heavy }
      }
      nonpolymer_entity_instances {
        rcsb_nonpolymer_entity_instance_container_identifiers {
          auth_asym_id auth_seq_id asym_id comp_id
        }
        rcsb_nonpolymer_instance_validation_score {
          RSCC RSR completeness alt_id average_occupancy
        }
      }
    }
  }
}
"""


@dataclass(frozen=True)
class LigandInstance:
    pdb_id: str
    ccd_id: str
    asym_id: str
    chain: str
    resnum: str
    resolution: float
    res_bin: str
    rscc: float
    rsr: float

    @property
    def key(self) -> str:
        return f"{self.pdb_id}_{self.asym_id}_{self.ccd_id}"


@dataclass
class CCDComponent:
    """Heavy-atom chemistry of one CCD component."""

    ccd_id: str
    status: str
    atom_names: list[str]
    elements: list[str]
    charges: list[int]
    n_hydrogens: list[int]
    bonds: list[tuple[int, int]]
    orders: list[int]
    aromatic: list[bool]


_ORDERS = {"SING": 1, "DOUB": 2, "TRIP": 3}


def _element(symbol: str) -> str:
    return symbol[0].upper() + symbol[1:].lower()


def parse_ccd(path: Path) -> CCDComponent:
    block = gemmi.cif.read(str(path)).sole_block()
    status = block.find_value("_chem_comp.pdbx_release_status") or "?"
    atoms = block.find("_chem_comp_atom.", ["atom_id", "type_symbol", "charge"])
    names_all = [gemmi.cif.as_string(row[0]) for row in atoms]
    elements_all = [_element(gemmi.cif.as_string(row[1])) for row in atoms]
    charges_all = [gemmi.cif.as_int(row[2], 0) for row in atoms]
    heavy = [i for i, e in enumerate(elements_all) if e not in ("H", "D")]
    local = {names_all[i]: k for k, i in enumerate(heavy)}
    hydrogens = {names_all[i] for i, e in enumerate(elements_all) if e in ("H", "D")}

    n_h = [0] * len(heavy)
    bonds, orders, aromatic = [], [], []
    table = block.find("_chem_comp_bond.", ["atom_id_1", "atom_id_2", "value_order", "?pdbx_aromatic_flag"])
    for row in table:
        a, b = gemmi.cif.as_string(row[0]), gemmi.cif.as_string(row[1])
        order = _ORDERS.get(gemmi.cif.as_string(row[2]).upper(), 0)
        if a in hydrogens or b in hydrogens:
            heavy_name = b if a in hydrogens else a
            if heavy_name in local:
                n_h[local[heavy_name]] += 1
            continue
        if a not in local or b not in local:
            continue
        bonds.append((local[a], local[b]))
        orders.append(order)
        aromatic.append(row.has(3) and gemmi.cif.as_string(row[3]).upper() == "Y")
    return CCDComponent(
        ccd_id=block.name,
        status=gemmi.cif.as_string(status),
        atom_names=[names_all[i] for i in heavy],
        elements=[elements_all[i] for i in heavy],
        charges=[charges_all[i] for i in heavy],
        n_hydrogens=n_h,
        bonds=bonds,
        orders=orders,
        aromatic=aromatic,
    )


def formula_elements(formula: str) -> set[str]:
    return set(re.findall(r"[A-Z][a-z]?", formula or ""))


class A1Builder(StratumBuilder):
    stratum_id = "A1"
    source_name = "pdb"
    defaults = {
        "min_rscc": 0.9,
        "max_rsr": 0.3,
        "min_heavy_atoms": 6,
        "max_heavy_atoms": 100,
        "max_per_ccd": 3,
        "oversample": 1.5,
        "graphql_batch": 100,
        "workers": 8,
    }

    def fetch(self) -> None:
        """Downloads happen incrementally in :meth:`records` (only what is needed)."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # -- discovery -----------------------------------------------------------

    def _search_entries(self, res_bin: str) -> list[str]:
        lo, hi = RESOLUTION_BINS[res_bin]
        nodes = [
            {"type": "terminal", "service": "text",
             "parameters": {"attribute": "exptl.method", "operator": "exact_match", "value": "X-RAY DIFFRACTION"}},
            {"type": "terminal", "service": "text",
             "parameters": {"attribute": "rcsb_nonpolymer_instance_validation_score.RSCC",
                            "operator": "greater_or_equal", "value": self.params["min_rscc"]}},
        ]
        if lo is not None:
            nodes.append({"type": "terminal", "service": "text",
                          "parameters": {"attribute": "rcsb_entry_info.resolution_combined", "operator": "greater", "value": lo}})
        if hi is not None:
            nodes.append({"type": "terminal", "service": "text",
                          "parameters": {"attribute": "rcsb_entry_info.resolution_combined", "operator": "less_or_equal", "value": hi}})
        query = {
            "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
            "return_type": "entry",
            "request_options": {"return_all_hits": True, "results_content_type": ["experimental"]},
        }
        cache = self.raw_dir / "search" / f"{res_bin}_rscc{self.params['min_rscc']}.json"
        result = post_json(SEARCH_URL, query, cache=cache)
        snapshot = datetime.fromtimestamp(cache.stat().st_mtime, UTC).date().isoformat()
        self.extra_attrs.setdefault("pdb_search_dates", {})[res_bin] = snapshot
        return sorted(hit["identifier"] for hit in result.get("result_set", []))

    def _entry_details(self, pdb_ids: list[str]) -> dict[str, dict]:
        """GraphQL entry details, cached per entry."""
        cache_dir = self.raw_dir / "graphql"
        cache_dir.mkdir(parents=True, exist_ok=True)
        out, missing = {}, []
        for pdb_id in pdb_ids:
            path = cache_dir / f"{pdb_id}.json"
            if path.exists():
                out[pdb_id] = json.loads(path.read_text())
            else:
                missing.append(pdb_id)
        if missing:
            response = post_json(GRAPHQL_URL, {"query": GRAPHQL_QUERY, "variables": {"ids": missing}})
            for entry in (response.get("data") or {}).get("entries") or []:
                (cache_dir / f"{entry['rcsb_id']}.json").write_text(json.dumps(entry))
                out[entry["rcsb_id"]] = entry
        return out

    def _instances(self, entry: dict, res_bin: str) -> Iterator[LigandInstance]:
        resolution = (entry.get("rcsb_entry_info") or {}).get("resolution_combined") or [None]
        p = self.params
        for entity in entry.get("nonpolymer_entities") or []:
            comp = entity["nonpolymer_comp"]
            cc = comp["chem_comp"]
            ccd_id = cc["id"]
            heavy = (comp.get("rcsb_chem_comp_info") or {}).get("atom_count_heavy") or 0
            if ccd_id in EXCLUDED_CCD:
                self.reject("excluded additive/solvent")
                continue
            if cc.get("type", "").lower() != "non-polymer":
                self.reject("chem_comp type not non-polymer")
                continue
            if not p["min_heavy_atoms"] <= heavy <= p["max_heavy_atoms"]:
                self.reject("heavy atom count out of range")
                continue
            if not formula_elements(cc.get("formula")) <= ALLOWED_ELEMENTS:
                self.reject("element outside organic set")
                continue
            for inst in entity.get("nonpolymer_entity_instances") or []:
                scores = (inst.get("rcsb_nonpolymer_instance_validation_score") or [None])[0]
                ident = inst["rcsb_nonpolymer_entity_instance_container_identifiers"]
                if not scores or scores.get("RSCC") is None or scores.get("RSR") is None:
                    self.reject("no density validation")
                    continue
                if scores["RSCC"] < p["min_rscc"] or scores["RSR"] > p["max_rsr"]:
                    self.reject("density fit (RSCC/RSR)")
                    continue
                if (scores.get("completeness") or 0) < 1.0:
                    self.reject("incomplete instance")
                    continue
                if scores.get("alt_id") or (scores.get("average_occupancy") or 0) < 1.0:
                    self.reject("partial occupancy or altloc")
                    continue
                yield LigandInstance(
                    pdb_id=entry["rcsb_id"],
                    ccd_id=ccd_id,
                    asym_id=ident["asym_id"],
                    chain=ident["auth_asym_id"],
                    resnum=str(ident["auth_seq_id"]),
                    resolution=float(resolution[0]),
                    res_bin=res_bin,
                    rscc=float(scores["RSCC"]),
                    rsr=float(scores["RSR"]),
                )

    def _candidates(self, res_bin: str, quota: int, per_ccd: Counter[str]) -> list[LigandInstance]:
        entries = self._search_entries(res_bin)
        self.funnel[f"{res_bin}_entries"] = len(entries)
        rng = np.random.default_rng(derive_seed(self.seed, "A1", res_bin))
        order = [entries[i] for i in rng.permutation(len(entries))]
        target = int(quota * self.params["oversample"]) + 5
        batch = self.params["graphql_batch"]

        picked: list[LigandInstance] = []
        for start in range(0, len(order), batch):
            ids = order[start : start + batch]
            details = self._entry_details(ids)
            for pdb_id in ids:
                if pdb_id not in details:
                    continue
                best: dict[str, LigandInstance] = {}
                for inst in self._instances(details[pdb_id], res_bin):
                    if inst.ccd_id not in best or inst.rscc > best[inst.ccd_id].rscc:
                        best[inst.ccd_id] = inst
                for ccd_id in sorted(best):
                    if per_ccd[ccd_id] >= self.params["max_per_ccd"]:
                        self.reject("per-CCD cap")
                        continue
                    per_ccd[ccd_id] += 1
                    picked.append(best[ccd_id])
            if len(picked) >= target:
                break
        self.funnel[f"{res_bin}_candidates"] = len(picked)
        return picked

    # -- structures ----------------------------------------------------------

    def _ccd(self, ccd_id: str) -> CCDComponent:
        return parse_ccd(fetch(CCD_URL.format(ccd_id=ccd_id), self.raw_dir / "ccd" / f"{ccd_id}.cif"))

    def _mmcif_path(self, pdb_id: str) -> Path:
        return self.raw_dir / "mmcif" / f"{pdb_id}.cif.gz"

    def _record(self, inst: LigandInstance) -> MoleculeRecord | None:
        try:
            comp = self._ccd(inst.ccd_id)
        except Exception:  # noqa: BLE001
            self.reject("ccd download/parse failed")
            return None
        if comp.status.upper() != "REL":
            self.reject("ccd not released")
            return None
        if not comp.orders or 0 in comp.orders:
            self.reject("ccd bond order missing/unknown")
            return None

        structure = gemmi.read_structure(str(self._mmcif_path(inst.pdb_id)))
        structure.setup_entities()
        residue = None
        for chain in structure[0]:
            if chain.name != inst.chain:
                continue
            for res in chain:
                if res.name == inst.ccd_id and str(res.seqid.num) + res.seqid.icode.strip() == inst.resnum:
                    residue = res
                    break
        if residue is None:
            self.reject("ligand residue not found in mmCIF")
            return None

        for conn in structure.connections:
            if conn.type != gemmi.ConnectionType.Covale:
                continue
            for here, there in ((conn.partner1, conn.partner2), (conn.partner2, conn.partner1)):
                if (here.chain_name == inst.chain and here.res_id.name == inst.ccd_id
                        and str(here.res_id.seqid.num) + here.res_id.seqid.icode.strip() == inst.resnum
                        and not (there.chain_name == inst.chain and there.res_id.seqid == here.res_id.seqid)):
                    self.reject("covalently linked")
                    return None

        coords: dict[str, np.ndarray] = {}
        for atom in residue:
            if atom.is_hydrogen():
                continue
            if atom.has_altloc() or atom.occ < 1.0:
                self.reject("partial occupancy or altloc")
                return None
            coords[atom.name] = np.array(atom.pos.tolist())
        if set(coords) != set(comp.atom_names):
            self.reject("atom names do not match CCD heavy atoms")
            return None

        record = MoleculeRecord(
            structure_id=f"A1-{inst.key}",
            source_id=inst.key,
            atomic_numbers=[gemmi.Element(e).atomic_number for e in comp.elements],
            positions=np.stack([coords[name] for name in comp.atom_names]),
            formal_charges=comp.charges,
            bonds=np.array(comp.bonds, dtype=np.int32).reshape(-1, 2),
            bond_orders=comp.orders,
            aromatic=comp.aromatic,
            implicit_hydrogens=comp.n_hydrogens,
            total_charge=int(sum(comp.charges)),
            meta={
                "pdb_id": inst.pdb_id,
                "ccd_id": inst.ccd_id,
                "chain": inst.chain,
                "resnum": inst.resnum,
                "resolution": inst.resolution,
                "res_bin": inst.res_bin,
                "rscc": inst.rscc,
                "rsr": inst.rsr,
            },
        )
        if not self.accept(record):
            return None
        return set_identifiers(record)

    def records(self) -> list[MoleculeRecord]:
        share, extra = divmod(self.n, len(RESOLUTION_BINS))
        quotas = {b: share + (i < extra) for i, b in enumerate(RESOLUTION_BINS)}
        per_ccd: Counter[str] = Counter()
        out: list[MoleculeRecord] = []
        for res_bin, quota in quotas.items():
            candidates = self._candidates(res_bin, quota, per_ccd)
            pdb_ids = sorted({c.pdb_id for c in candidates})
            errors = fetch_many(
                ((MMCIF_URL.format(pdb_id=p), self._mmcif_path(p)) for p in pdb_ids),
                workers=self.params["workers"],
            )
            failed = {p.name.split(".")[0] for p, err in errors.items() if err is not None}
            fetch_many(
                ((CCD_URL.format(ccd_id=c), self.raw_dir / "ccd" / f"{c}.cif") for c in sorted({c.ccd_id for c in candidates})),
                workers=self.params["workers"],
            )
            kept = 0
            for inst in candidates:
                if kept >= quota:
                    break
                if inst.pdb_id in failed:
                    self.reject("mmCIF download failed")
                    continue
                record = self._record(inst)
                if record is not None:
                    out.append(record)
                    kept += 1
            self.funnel[f"{res_bin}_written"] = kept
        return out
