"""C5: off-equilibrium (non-reactive) geometries.

- QCML normal-mode samples (``conformation_seq > 0``) labeled with their
  source graph, kept only if the distorted geometry still has that graph's
  connectivity.
- Transition1x NEB frames adjacent to the reactant or product (images 1-2 /
  7-8 of converging NEB iterations) labeled with that endpoint's Grambow
  graph. Frames are dropped if their connectivity differs from the endpoint
  graph, if they lie within ``min_ts_gap`` eV of the transition state, or
  more than ``max_rel_energy`` eV above the endpoint. One frame per reaction.
"""

from __future__ import annotations

import hashlib
import logging

import h5py
import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from ..chem import LabelError, mol_to_record
from ..download import fetch
from ..sampling import derive_seed, stratified_sample
from ..schema import MoleculeRecord
from ..sources import qcml
from ..sources import transition1x as t1x
from .base import StratumBuilder
from .qcml_common import hetero_signature, qcml_record

log = logging.getLogger(__name__)


class C5Builder(StratumBuilder):
    stratum_id = "C5"
    source_name = "transition1x"
    defaults = {
        "qcml_fraction": 0.5,
        "qcml_records": 900_000,
        # ~50% of QCML normal-mode samples pass validation.
        "qcml_presample": 2.5,
        "max_rel_energy": 1.5,  # eV above the endpoint
        "min_ts_gap": 0.5,  # eV below the transition state
        "near_images": {"reactant": [1, 2], "product": [7, 8]},
    }

    def fetch(self) -> None:
        fetch(t1x.GRAMBOW_URL, self.raw_dir / "grambow_wb97xd3.csv")
        fetch(t1x.T1X_URL, self.raw_dir / "Transition1x.h5", timeout=1800)

    # -- QCML ------------------------------------------------------------------

    def _qcml_part(self, n: int) -> list[MoleculeRecord]:
        # Records are shuffled, so a hash-based ~10% subsample is a random subsample.
        displaced = qcml.load_records(
            self.cache_dir, self.params["qcml_records"],
            keep=lambda r: r["seq"] > 0 and derive_seed(self.seed, "C5", r["key_hash"]) % 10 == 0,
        )
        self.funnel["qcml_normal_mode_samples"] = len(displaced)
        # Pre-sample by element signature, then validate (validation is the slow part).
        pre = stratified_sample(
            displaced, int(n * self.params["qcml_presample"]) + 10, derive_seed(self.seed, "C5", "qcml-pre"),
            stratum=lambda r: hetero_signature(r["smiles"]), key=lambda r: r["key_hash"],
        )
        candidates = []
        for rec in pre:
            uid = hashlib.sha1(rec["key_hash"].encode()).hexdigest()[:12]
            record = qcml_record(self, rec, f"C5-qcml-{uid}", {
                "source": "QCML", "qcml_key": rec["key_hash"], "qcml_seq": rec["seq"], "hetero": hetero_signature(rec["smiles"]),
            })
            if record is not None:
                candidates.append(record)
        self.funnel["qcml_valid"] = len(candidates)
        return stratified_sample(candidates, n, derive_seed(self.seed, "C5", "qcml"), lambda r: r.meta["hetero"], lambda r: r.structure_id)

    # -- Transition1x ------------------------------------------------------------

    def _t1x_candidates(self) -> list[tuple[str, int, str, Chem.Mol, np.ndarray, dict]]:
        grambow = t1x.load_grambow(self.raw_dir / "grambow_wb97xd3.csv")
        near = {img: side for side, imgs in self.params["near_images"].items() for img in imgs}
        out = []
        with h5py.File(self.raw_dir / "Transition1x.h5", "r") as f:
            for formula in sorted(f["data"]):
                for name in sorted(f["data"][formula]):
                    rxn = f["data"][formula][name]
                    self.funnel["t1x_reactions"] += 1
                    numbers = rxn["atomic_numbers"][()]
                    positions = rxn["positions"][()]
                    energies = rxn[t1x.ENERGY_KEY][()]
                    key = (
                        tuple(int(z) for z in numbers),
                        t1x.geometric_edges(numbers, rxn["reactant"]["positions"][0]),
                        t1x.geometric_edges(numbers, rxn["product"]["positions"][0]),
                    )
                    matches = grambow.get(key, [])
                    if len(matches) != 1:
                        self.reject("t1x: no unique Grambow match" if not matches else "t1x: ambiguous Grambow match")
                        continue
                    match = matches[0]
                    ends = {
                        "reactant": (match.reactant, float(rxn["reactant"][t1x.ENERGY_KEY][0])),
                        "product": (match.product, float(rxn["product"][t1x.ENERGY_KEY][0])),
                    }
                    e_ts = float(rxn["transition_state"][t1x.ENERGY_KEY][0])
                    frames = []
                    for frame in range(10, len(energies)):
                        side = near.get(t1x.frame_image(frame))
                        if side is None:
                            continue
                        label, e_end = ends[side]
                        e = float(energies[frame])
                        if e - e_end > self.params["max_rel_energy"] or e_ts - e < self.params["min_ts_gap"]:
                            continue
                        if t1x.geometric_edges(numbers, positions[frame]) != t1x.edges_of(label):
                            continue
                        frames.append((frame, side, e - e_end))
                    if not frames:
                        self.reject("t1x: no frame near an endpoint")
                        continue
                    rng = np.random.default_rng(derive_seed(self.seed, "C5", name))
                    frame, side, rel = frames[rng.integers(len(frames))]
                    meta = {"source": "Transition1x", "t1x_reaction": f"{formula}/{name}", "grambow_idx": match.idx,
                            "frame": frame, "endpoint": side, "rel_energy_ev": rel, "hetero": hetero_signature(Chem.MolToSmiles(ends[side][0]))}
                    out.append((f"C5-t1x-{name}", frame, side, ends[side][0], positions[frame], meta))
        self.funnel["t1x_candidates"] = len(out)
        return out

    def _t1x_record(self, structure_id: str, label: Chem.Mol, positions: np.ndarray, meta: dict) -> MoleculeRecord | None:
        mol = Chem.Mol(label)
        try:
            Chem.SanitizeMol(mol)
        except Exception:  # noqa: BLE001
            self.reject("t1x: Grambow graph fails sanitization")
            return None
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.Set3D(True)
        for i, (x, y, z) in enumerate(np.asarray(positions, dtype=float)):
            conf.SetAtomPosition(i, Point3D(x, y, z))
        mol.AddConformer(conf, assignId=True)
        try:
            record = mol_to_record(mol, structure_id=structure_id, source_id=meta["t1x_reaction"], meta=meta)
        except LabelError as exc:
            self.reject(f"label: {exc}")
            return None
        return record if self.accept(record) else None

    def _t1x_part(self, n: int) -> list[MoleculeRecord]:
        candidates = []
        for structure_id, _, side, label, positions, meta in self._t1x_candidates():
            record = self._t1x_record(structure_id, label, positions, meta)
            if record is not None:
                candidates.append(record)
        self.funnel["t1x_valid"] = len(candidates)
        return stratified_sample(
            candidates, n, derive_seed(self.seed, "C5", "t1x"),
            stratum=lambda r: (r.meta["endpoint"], r.meta["hetero"]), key=lambda r: r.structure_id,
        )

    def records(self) -> list[MoleculeRecord]:
        # Any QCML shortfall is filled from Transition1x, which has a large surplus.
        part_qcml = self._qcml_part(round(self.n * self.params["qcml_fraction"]))
        part_t1x = self._t1x_part(self.n - len(part_qcml))
        return sorted(part_qcml + part_t1x, key=lambda r: r.structure_id)
