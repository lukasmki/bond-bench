"""B5: tautomer groups from Tautobase + Dhaked-Nicklaus, 3D-embedded and GFN2-xTB relaxed.

Every member of a selected group is included (the accepted tautomer set). A
group is dropped entirely if any member fails embedding, relaxation, or keeps
a different connectivity after relaxation (e.g. a proton hops back). Per
structure meta carries the preference label from one reference observation:
``pref_label`` 1 = preferred, 0 = not preferred, -1 = not covered by it.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from rdkit import Chem

from ..chem import LabelError, mol_to_record
from ..download import fetch
from ..relax import embed, same_connectivity, xtb_relax
from ..sampling import derive_seed, tertile, tertile_bounds
from ..schema import MoleculeRecord
from ..sources.tautomers import (
    DN_URL,
    TAUTOBASE_URL,
    group_tautomers,
    read_dhaked_nicklaus,
    read_tautobase,
    reference_observation,
)
from .base import StratumBuilder

log = logging.getLogger(__name__)

ALLOWED_ELEMENTS = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I", "Se"}


def skeleton(mol: Chem.Mol) -> str:
    """Canonical SMILES of the heavy-atom graph with bond orders/charges/H erased."""
    rw = Chem.RWMol(Chem.RemoveHs(mol))
    for atom in rw.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    return Chem.MolToSmiles(rw, canonical=True)


def _relax_one(task: tuple[str, str, int]) -> tuple[str, str | None, np.ndarray | None, float, bool]:
    """Worker: (key, smiles, seed) -> (key, molblock or error, positions, energy, converged)."""
    key, smiles, seed = task
    try:
        mol = embed(Chem.MolFromSmiles(smiles), seed)
        positions, energy, converged = xtb_relax(mol)
        if not same_connectivity(mol, positions):
            return key, "error: connectivity changed during relaxation", None, 0.0, False
        return key, Chem.MolToMolBlock(mol), positions, energy, converged
    except LabelError as exc:
        return key, f"error: {exc}", None, 0.0, False
    except Exception as exc:  # noqa: BLE001 - xtb/embedding failures are rejections
        return key, f"error: {type(exc).__name__}", None, 0.0, False


class B5Builder(StratumBuilder):
    stratum_id = "B5"
    source_name = "tautomers"
    defaults = {
        "max_heavy_atoms": 50,
        # Extra structures attempted to absorb relaxation failures.
        "oversample": 1.15,
        "workers": max(1, (os.cpu_count() or 2) - 1),
    }

    def fetch(self) -> None:
        fetch(TAUTOBASE_URL, self.raw_dir / "Tautobase_SMIRKS.txt")
        fetch(DN_URL, self.raw_dir / "Tautomer_database_release_3a.xlsx")

    def _groups(self):
        structures: dict[str, Chem.Mol] = {}
        observations = read_tautobase(self.raw_dir / "Tautobase_SMIRKS.txt", structures)
        observations += read_dhaked_nicklaus(self.raw_dir / "Tautomer_database_release_3a.xlsx", structures)
        self.funnel["observations"] = len(observations)
        self.funnel["unique_tautomers"] = len(structures)
        group_of = group_tautomers(observations)
        by_group: dict[int, list] = defaultdict(list)
        for obs in observations:
            by_group[group_of[obs.tautomers[0]]].append(obs)
        members: dict[int, list[str]] = defaultdict(list)
        for key, g in group_of.items():
            members[g].append(key)
        self.funnel["groups"] = len(members)

        groups = []
        for g in sorted(members):
            keys = sorted(members[g])
            mols = [structures[k] for k in keys]
            ref = reference_observation(by_group[g])
            if ref is None:
                self.reject("group: no preference label")
                continue
            elements = {a.GetSymbol() for m in mols for a in m.GetAtoms()}
            if not elements <= ALLOWED_ELEMENTS:
                self.reject("group: element outside organic set")
                continue
            n_heavy = max(m.GetNumHeavyAtoms() for m in mols)
            if n_heavy > self.params["max_heavy_atoms"]:
                self.reject("group: too large")
                continue
            heavy_graphs = {skeleton(m) for m in mols}
            groups.append({
                "group": g,
                "keys": keys,
                "smiles": [Chem.MolToSmiles(m) for m in mols],
                "n_heavy": n_heavy,
                "kind": "proton_transfer" if len(heavy_graphs) == 1 else "ring_chain_or_valence",
                "charged": any(a.GetFormalCharge() for m in mols for a in m.GetAtoms()),
                "ref": ref,
            })
        self.funnel["groups_eligible"] = len(groups)
        return groups

    def _order_groups(self, groups: list[dict]) -> list[dict]:
        """Seeded round-robin over strata, so any prefix is balanced."""
        bounds = tertile_bounds([g["n_heavy"] for g in groups])
        self.extra_attrs["size_tertile_bounds"] = list(bounds)
        strata: dict[tuple, list[dict]] = defaultdict(list)
        for g in groups:
            strata[(tertile(g["n_heavy"], bounds), g["kind"], g["charged"])].append(g)
        queues = []
        for name in sorted(strata, key=str):
            rng = np.random.default_rng(derive_seed(self.seed, "B5", name))
            items = strata[name]
            queues.append([items[i] for i in rng.permutation(len(items))])
        ordered = []
        while any(queues):
            for q in queues:
                if q:
                    ordered.append(q.pop(0))
        return ordered

    def records(self) -> list[MoleculeRecord]:
        ordered = self._order_groups(self._groups())
        budget = int(self.n * self.params["oversample"])
        attempt, total = [], 0
        for g in ordered:
            if total >= budget:
                break
            attempt.append(g)
            total += len(g["keys"])
        tasks = [(k, smi, derive_seed(self.seed, "B5", k)) for g in attempt for k, smi in zip(g["keys"], g["smiles"])]
        log.info("[B5] relaxing %d tautomers from %d groups with %d workers", len(tasks), len(attempt), self.params["workers"])

        results = {}
        with ProcessPoolExecutor(self.params["workers"]) as pool:
            for key, block, positions, energy, converged in pool.map(_relax_one, tasks, chunksize=4):
                results[key] = (block, positions, energy, converged)

        out: list[MoleculeRecord] = []
        for g in attempt:
            group_records = self._group_records(g, results)
            if group_records is None:
                continue
            if len(out) + len(group_records) > self.n:
                continue
            out.extend(group_records)
            if len(out) == self.n:
                break
        self.funnel["groups_written"] = len({r.meta["group_id"] for r in out})
        return out

    def _group_records(self, g: dict, results: dict) -> list[MoleculeRecord] | None:
        ref = g["ref"]
        pref = {}
        for i, key in enumerate(ref.tautomers):
            pref[key] = (1 if i in ref.preferred else 0, ref.populations[i])
        records = []
        for k, key in enumerate(g["keys"]):
            block, positions, energy, converged = results[key]
            if positions is None:
                self.reject(f"tautomer {block.removeprefix('error: ')}")
                self.reject("group dropped: member failed")
                return None
            mol = Chem.MolFromMolBlock(block, removeHs=False)
            conf = mol.GetConformer()
            for i, xyz in enumerate(positions):
                conf.SetAtomPosition(i, xyz.tolist())
            label, population = pref.get(key, (-1, None))
            try:
                record = mol_to_record(
                    mol,
                    structure_id=f"B5-G{g['group']:04d}-{k}",
                    source_id=key,
                    meta={
                        "group_id": f"G{g['group']:04d}",
                        "group_size": len(g["keys"]),
                        "tautomerism": g["kind"],
                        "pref_label": label,
                        "population": population,
                        "pref_source": ref.source,
                        "pref_ref": ref.ref,
                        "pref_solvent": ref.solvent,
                        "xtb_energy_ev": energy,
                        "xtb_converged": converged,
                    },
                )
            except LabelError as exc:
                self.reject(f"label: {exc}")
                return None
            if not self.accept(record):
                return None
            records.append(record)
        return records
