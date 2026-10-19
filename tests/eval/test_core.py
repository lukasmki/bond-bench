import numpy as np
import pytest
from rdkit import Chem

from bond_bench.eval.core.canon import pick, same_resonance_class, wl_colors
from bond_bench.eval.core.elements import AtomState, states
from bond_bench.eval.core.geometry import contact_map
from bond_bench.eval.core.graph import Graph
from bond_bench.eval.core.ilp import DecodeSpec, Objective, decode
from bond_bench.eval.core.validity import check, is_valid

from ..conftest import embedded_record


def _fixed_support_spec(record, **kw):
    edges, _ = contact_map(record.atomic_numbers, record.positions)
    allowed = np.ones((len(edges), 4), dtype=bool)
    allowed[:, 0] = False
    kw.setdefault("hydrogens", np.zeros(record.n_atoms, dtype=np.int64))
    return DecodeSpec(record.atomic_numbers, record.total_charge, edges, allowed=allowed, **kw)


KIM = [
    lambda m: Objective(edge=np.tile(np.arange(4.0), (m, 1)), name="max ΣB"),
    lambda m: Objective(atom=lambda i, st, eta: -abs(st.charge), name="min Σ|q|"),
]


def kim(spec):
    return [f(len(spec.edges)) for f in KIM]


def test_lewis_states():
    n = states(7, "lewis", c_max=1)
    for c, d in [(0, 3), (1, 4), (-1, 2)]:
        assert AtomState(c, d) in n
    assert AtomState(0, 4) not in n  # odd lone-pair count
    assert AtomState(0, 6) in states(16, "lewis")  # expanded shell
    assert {s.valence for s in states(6, "octet", 0)} == {4}  # no carbenes
    assert AtomState(0, 6) in states(16, "octet") and AtomState(0, 4) not in states(16, "octet")
    assert states(26) == ()  # metals unsupported


def test_validity_oracle(records):
    for record in records:
        g = Graph.from_record(record)
        assert is_valid(record.atomic_numbers, record.total_charge, g)
        assert check(record.atomic_numbers, record.total_charge + 1, g) == "charge sum"
        stripped = record.strip_hydrogens()
        assert is_valid(stripped.atomic_numbers, stripped.total_charge, Graph.from_record(stripped))


def test_benzene_pool_has_both_kekule_forms():
    rec = embedded_record("c1ccccc1", "benzene")
    res = decode(_fixed_support_spec(rec), kim(_fixed_support_spec(rec)), pool="all")
    assert res.status == "ok" and len(res.solutions) == 2
    a, b = (s.graph for s in res.solutions)
    assert a != b and same_resonance_class(a, b)


@pytest.mark.parametrize("smiles,n_forms", [("C[N+](=O)[O-]", 2), ("CC(=O)[O-]", 2), ("c1ccsc1", 1)])
def test_resonance_pools(smiles, n_forms):
    rec = embedded_record(smiles, smiles)
    spec = _fixed_support_spec(rec)
    res = decode(spec, kim(spec), pool="all")
    assert len(res.solutions) == n_forms
    for s in res.solutions:
        assert is_valid(rec.atomic_numbers, rec.total_charge, s.graph)


def test_infeasible_charge_reports_reason():
    rec = embedded_record("CCO", "ethanol")
    spec = _fixed_support_spec(rec)
    spec.total_charge = 3
    res = decode(spec, kim(spec))
    assert res.status == "infeasible" and res.reason == "charge"


def test_hydrogen_free_decoding_recovers_h():
    rec = embedded_record("CC(=O)Nc1ccc(O)cc1", "paracetamol").strip_hydrogens()
    edges, _ = contact_map(rec.atomic_numbers, rec.positions)
    ref = Graph.from_record(rec)
    # score the true orders highest: the decoder must then return the label exactly
    target = ref.bond_dict()
    w = np.full((len(edges), 4), -5.0)
    for e, (i, j) in enumerate(edges.tolist()):
        w[e, target.get((i, j), 0)] = 0.0
    spec = DecodeSpec(rec.atomic_numbers, 0, edges, hydrogens=None)
    res = decode(spec, [Objective(edge=w, atom=lambda i, st, eta: -abs(st.charge))])
    assert res.best.graph == ref


def test_top_k_pool_is_sorted():
    rec = embedded_record("C=CC=O", "acrolein")
    spec = _fixed_support_spec(rec)
    res = decode(spec, [Objective(edge=np.tile(np.arange(4.0), (len(spec.edges), 1)))], pool=3)
    values = [s.values[-1] for s in res.solutions]
    assert values == sorted(values, reverse=True)
    assert len({s.graph for s in res.solutions}) == len(res.solutions)


def test_wl_colors_and_pick_are_permutation_invariant():
    rec = embedded_record("Cc1ccc(O)cc1", "cresol")
    edges, _ = contact_map(rec.atomic_numbers, rec.positions)
    colors = wl_colors(rec.atomic_numbers, rec.positions, edges)
    rng = np.random.default_rng(0)
    perm = rng.permutation(rec.n_atoms)
    inv = np.argsort(perm)
    z2, x2 = rec.atomic_numbers[inv], rec.positions[inv]
    edges2, _ = contact_map(z2, x2)
    colors2 = wl_colors(z2, x2, edges2)
    np.testing.assert_array_equal(colors2[perm], colors)

    spec = _fixed_support_spec(rec)
    pool = [s.graph for s in decode(spec, kim(spec), pool="all").solutions]
    best = pick(pool, colors, rec.positions)
    assert pick([g.permuted(perm) for g in pool], colors2, x2) == best.permuted(perm)


def test_graph_permutation_roundtrip():
    rec = embedded_record("CC(=O)[O-]", "acetate")
    g = Graph.from_record(rec)
    perm = np.random.default_rng(1).permutation(rec.n_atoms)
    assert g.permuted(perm).permuted(np.argsort(perm)) == g
    assert Chem.MolFromSmiles(rec.smiles) is not None
