from rdkit import Chem

from bond_bench.data.builders.b5_tautomers import skeleton
from bond_bench.data.sources.tautomers import (
    Observation,
    fixed_h_inchi,
    group_tautomers,
    read_tautobase,
    reference_observation,
)

HEADER = "Smirks\tpassed stoichiometry check\tlog K\t% (Tautomer1)\tPreferred\tState/Solvent\tTemperature\tReference\tAdd. Info\tMeas/Calc/Est\tID in text\tDate added\tRow Nr\n"


def _key(smiles):
    return fixed_h_inchi(Chem.MolFromSmiles(smiles))


def test_tautobase_log_k_sign(tmp_path):
    path = tmp_path / "tb.txt"
    path.write_text(
        HEADER
        + "[OH:1][c:2]1[cH:3][cH:4][cH:5][cH:7][n:6]1>>[O:1]=[c:2]1[cH:3][cH:4][cH:5][cH:7][nH:6]1\tyes\t3.5\t\t\tWater\t\t1\t\tE\t\t\t1\n"
        + "[CH3:1][C:2](=[O:3])[CH3:4]>>[CH2:1]=[C:2]([OH:3])[CH3:4]\tyes\t\t\t1\tGas\t\t1\t\tE\t\t\t2\n"
    )
    structures = {}
    obs = read_tautobase(path, structures)
    assert len(structures) == 4
    pyridone = obs[0]
    assert pyridone.preferred == {1}  # log K > 0 favours T2
    assert pyridone.populations[1] > 0.99
    assert obs[1].preferred == {0} and obs[1].populations == [None, None]


def test_grouping_merges_overlapping_observations():
    a, b, c, d = _key("Oc1ccccn1"), _key("O=c1cccc[nH]1"), _key("O=C1C=CC=CN1"), _key("CC(C)=O")
    obs = [
        Observation("Tautobase", "r1", "Water", [a, b], [0.1, 0.9], {1}),
        Observation("Dhaked-Nicklaus", "Ref1", "CDCl3", [b, c], [None, None], {0}),
        Observation("Tautobase", "r2", "Gas", [d, _key("C=C(C)O")], [None, None], set()),
    ]
    groups = group_tautomers(obs)
    assert groups[a] == groups[b] == groups[c] != groups[d]
    assert reference_observation(obs[:2]).ref == "r1"  # aqueous + quantitative wins
    assert reference_observation(obs[2:]) is None


def test_skeleton_ignores_bond_orders_and_h():
    assert skeleton(Chem.MolFromSmiles("Oc1ccccn1")) == skeleton(Chem.MolFromSmiles("O=c1cccc[nH]1"))
    assert skeleton(Chem.MolFromSmiles("CCC(C)=NCC(C)O")) != skeleton(Chem.MolFromSmiles("CCC1(C)NCC(C)O1"))
