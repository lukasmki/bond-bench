from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from bond_bench.data.chem import mol_to_record
from bond_bench.data.manifest import load_manifest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def manifest():
    return load_manifest()


def embedded_record(smiles: str, structure_id: str, **meta):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    return mol_to_record(mol, structure_id=structure_id, source_id=structure_id, meta=meta)


@pytest.fixture
def records():
    smiles = ["CC(=O)Nc1ccc(O)cc1", "C[N+](C)(C)CC(=O)[O-]", "O=[N+]([O-])c1ccccc1", "c1ccncc1", "CS(N)(=O)=O"]
    return [embedded_record(s, f"T-{i}", tag=f"t{i}", size=i) for i, s in enumerate(smiles)]
