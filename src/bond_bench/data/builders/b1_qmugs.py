"""B1: drug-like molecules from QMugs (GFN2-xTB geometries), outside GEOM.

The reference graph is the ChEMBL 27 structure of each QMugs molecule; the
QMugs SDF graph is kept only if its stereo-free InChIKey matches it. Molecules
whose InChIKey connectivity layer occurs in GEOM-DRUGS are excluded to avoid
leakage into models trained on GEOM.
"""

from __future__ import annotations

import gzip
import logging
import tarfile
from collections.abc import Iterator
from pathlib import Path

import polars as pl
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from ..chem import LabelError, mol_to_record
from ..download import fetch, source_checksums
from ..sampling import derive_seed, stratified_sample, tertile, tertile_bounds
from ..schema import MoleculeRecord
from ..sources import geom
from .base import StratumBuilder

log = logging.getLogger(__name__)

QMUGS_SHARE = "https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM/download?path=%2F&files={name}"
CHEMBL_CHEMREPS_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_27/chembl_27_chemreps.txt.gz"
)

INCHI_NO_STEREO = "-SNon"


def inchikey_no_stereo(mol: Chem.Mol) -> str:
    return Chem.MolToInchiKey(mol, options=INCHI_NO_STEREO)


def connectivity_key(mol: Chem.Mol) -> str:
    return Chem.MolToInchiKey(mol).split("-")[0]


def load_geom_connectivity_keys(smiles: list[str], cache: Path) -> set[str]:
    if cache.exists():
        return set(cache.read_text().split())
    keys = set()
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            keys.add(connectivity_key(mol))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(sorted(keys)) + "\n")
    return keys


def reference_mol(smiles: str) -> Chem.Mol | None:
    """ChEMBL structure reduced to its largest fragment (QMugs drops counterions)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) > 1:
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    return mol


class B1Builder(StratumBuilder):
    stratum_id = "B1"
    source_name = "qmugs"
    defaults = {
        "conformer": "conf_00",
        # Pre-selection size relative to n, leaving room for rejected structures.
        "oversample": 1.5,
        # Path to a GEOM-DRUGS SMILES list (one per line); None uses the GEOM index.
        "geom_smiles": None,
    }

    def _path(self, name: str) -> Path:
        return self.raw_dir / name

    def _geom_smiles(self) -> list[str]:
        if self.params["geom_smiles"]:
            return Path(self.params["geom_smiles"]).read_text().split()
        index = geom.load_index(self.cache_dir)
        return index.filter(pl.col("subset") == "drugs")["smiles"].to_list()

    def fetch(self) -> None:
        for name in ("tarball_assignment.csv", "structures.tar.gz"):
            fetch(QMUGS_SHARE.format(name=name), self._path(name))
        fetch(CHEMBL_CHEMREPS_URL, self.cache_dir / "chembl" / "chembl_27_chemreps.txt.gz")
        if not self.params["geom_smiles"]:
            geom.load_index(self.cache_dir)

    def _reference_table(self) -> pl.DataFrame:
        """Per QMugs molecule: ChEMBL reference SMILES and cheap descriptors (cached)."""
        cache = self._path("chembl_reference.parquet")
        if cache.exists():
            return pl.read_parquet(cache)
        qmugs_ids = set(pl.read_csv(self._path("tarball_assignment.csv"))["chembl_id"])
        rows = []
        with gzip.open(self.cache_dir / "chembl" / "chembl_27_chemreps.txt.gz", "rt") as f:
            next(f)
            for line in f:
                chembl_id, smiles, *_ = line.rstrip("\n").split("\t")
                if chembl_id not in qmugs_ids:
                    continue
                mol = reference_mol(smiles)
                if mol is None:
                    continue
                rows.append({
                    "chembl_id": chembl_id,
                    "smiles": Chem.MolToSmiles(mol),
                    "n_heavy": mol.GetNumHeavyAtoms(),
                    "charged": any(a.GetFormalCharge() for a in mol.GetAtoms()),
                    "connectivity": connectivity_key(mol),
                    "inchikey_nostereo": inchikey_no_stereo(mol),
                })
        df = pl.DataFrame(rows).sort("chembl_id")
        df.write_parquet(cache)
        return df

    def _iter_sdfs(self, wanted: set[str]) -> Iterator[tuple[str, bytes]]:
        """Stream ``<chembl_id>/<conformer>.sdf`` members out of the tarball."""
        target = f"{self.params['conformer']}.sdf"
        with tarfile.open(self._path("structures.tar.gz"), "r|gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(target):
                    continue
                chembl_id = Path(member.name).parent.name
                if chembl_id in wanted:
                    handle = tar.extractfile(member)
                    if handle is not None:
                        yield chembl_id, handle.read()

    def records(self) -> list[MoleculeRecord]:
        ref = self._reference_table()
        self.funnel["qmugs_with_chembl_reference"] = ref.height
        keys_cache = self.raw_dir / "geom_drugs_connectivity_keys.txt"
        if self.params["geom_smiles"]:
            keys_cache = keys_cache.with_name(f"{Path(self.params['geom_smiles']).stem}_connectivity_keys.txt")
        geom_keys = load_geom_connectivity_keys(self._geom_smiles(), keys_cache)
        self.extra_attrs["n_geom_connectivity_keys"] = len(geom_keys)
        ref = ref.filter(~pl.col("connectivity").is_in(list(geom_keys)))
        self.funnel["out_of_geom"] = ref.height

        bounds = tertile_bounds(ref["n_heavy"].to_list())
        self.extra_attrs["size_tertile_bounds"] = list(bounds)

        def stratum(row: dict) -> tuple[int, bool]:
            return (tertile(row["n_heavy"], bounds), row["charged"])

        rows = ref.to_dicts()
        n_pre = int(self.n * self.params["oversample"]) + 10
        pre = stratified_sample(rows, n_pre, derive_seed(self.seed, "B1", "pre"), stratum, key=lambda r: r["chembl_id"])
        by_id = {r["chembl_id"]: r for r in pre}
        self.funnel["preselected"] = len(pre)

        candidates: list[MoleculeRecord] = []
        for chembl_id, block in self._iter_sdfs(set(by_id)):
            mol = Chem.MolFromMolBlock(block.decode(), removeHs=False, sanitize=True)
            if mol is None:
                self.reject("sdf unparsable")
                continue
            if inchikey_no_stereo(mol) != by_id[chembl_id]["inchikey_nostereo"]:
                self.reject("graph differs from ChEMBL reference")
                continue
            try:
                record = mol_to_record(
                    mol,
                    structure_id=f"B1-{chembl_id}",
                    source_id=f"{chembl_id}/{self.params['conformer']}",
                    meta={"chembl_id": chembl_id, "conf_id": self.params["conformer"]},
                )
            except LabelError as exc:
                self.reject(f"label: {exc}")
                continue
            if self.accept(record):
                candidates.append(record)
        self.funnel["parsed_and_checked"] = len(candidates)

        selected = stratified_sample(
            candidates,
            self.n,
            derive_seed(self.seed, "B1", "final"),
            stratum=lambda r: stratum(by_id[r.meta["chembl_id"]]),
            key=lambda r: r.structure_id,
        )
        self.extra_attrs["source_checksums"] = source_checksums(
            [self._path("structures.tar.gz"), self.cache_dir / "chembl" / "chembl_27_chemreps.txt.gz"]
        )
        return selected
