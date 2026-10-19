"""RDKit rdDetermineBonds: the xyz2mol algorithm of Kim & Kim (2015)."""

from __future__ import annotations

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

from ..core.graph import Graph
from ..prediction import Prediction, Status
from ..query import Query
from .base import Capabilities, Method, register

_ORDERS = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}


def query_to_mol(query: Query) -> Chem.RWMol:
    """Atoms and conformer only (no bonds), in query order."""
    rw = Chem.RWMol()
    for z in query.atomic_numbers:
        rw.AddAtom(Chem.Atom(int(z)))
    conf = Chem.Conformer(query.n_atoms)
    for k, (x, y, z) in enumerate(query.positions.astype(float)):
        conf.SetAtomPosition(k, Point3D(x, y, z))
    rw.AddConformer(conf, assignId=True)
    return rw


def mol_to_graph(mol: Chem.Mol, implicit_h: bool = False) -> Graph:
    mol = Chem.Mol(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    bonds, orders = [], []
    for b in mol.GetBonds():
        try:
            orders.append(_ORDERS[b.GetBondType()])
        except KeyError:
            raise ValueError(f"unsupported bond type {b.GetBondType()}") from None
        bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
    atoms = list(mol.GetAtoms())
    return Graph(
        n=len(atoms),
        bonds=np.array(bonds, dtype=np.int64).reshape(-1, 2),
        orders=orders,
        charges=[a.GetFormalCharge() for a in atoms],
        hydrogens=[a.GetTotalNumHs() if implicit_h else 0 for a in atoms],
        radicals=[a.GetNumRadicalElectrons() for a in atoms],
    )


@register
class RDKitDetermineBonds(Method):
    name = "rdkit-determinebonds"
    family = "optimization"
    capabilities = Capabilities(stripped_h=False, charged=True, radicals=False)
    defaults = {"use_huckel": False, "allow_charged_fragments": True, "cov_factor": 1.3}

    def version(self) -> str:
        return f"rdkit {rdBase.rdkitVersion}"

    def predict(self, query: Query) -> Prediction:
        mol = query_to_mol(query)
        try:
            rdDetermineBonds.DetermineBonds(
                mol,
                charge=query.total_charge,
                useHueckel=self.params["use_huckel"],
                allowChargedFragments=self.params["allow_charged_fragments"],
                covFactor=self.params["cov_factor"],
                embedChiral=False,
                useAtomMap=False,
            )
        except ValueError as exc:
            msg = str(exc)
            if "charge" in msg.lower():
                return Prediction.fail(Status.INFEASIBLE, f"charge: {msg}")
            return Prediction.fail(Status.ERROR, msg)
        return Prediction.ok(mol_to_graph(mol))
