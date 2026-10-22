import sys
import textwrap

import numpy as np
import pytest

from bond_bench.eval.core.canon import same_resonance_class
from bond_bench.eval.core.geometry import contact_map
from bond_bench.eval.core.graph import Graph
from bond_bench.eval.core.validity import is_valid
from bond_bench.eval.methods.base import get_method
from bond_bench.eval.methods.hybrid import constrained_map, graph_from_argmax
from bond_bench.eval.methods.optimization import prior_objectives
from bond_bench.eval.metrics import graph_smiles
from bond_bench.eval.prediction import Status
from bond_bench.eval.query import Observation, Query, random_euclidean

from ..conftest import embedded_record

SMILES = ["CC(=O)Nc1ccc(O)cc1", "C[N+](C)(C)CC(=O)[O-]", "O=[N+]([O-])c1ccccc1", "c1ccncc1", "CS(N)(=O)=O", "C#CC=C"]


def _query(smiles, obs=Observation.EXPLICIT_H):
    rec = embedded_record(smiles, smiles)
    return rec, Query.from_record(rec, obs)


def _smiles_of(query, g):
    return graph_smiles(query.atomic_numbers, query.positions, g, query.total_charge)


@pytest.mark.parametrize("smiles", SMILES)
def test_canonical_o_recovers_label(smiles):
    rec, q = _query(smiles)
    pred = get_method("canonical-O").predict(q)
    assert pred.status is Status.OK
    assert is_valid(q.atomic_numbers, q.total_charge, pred.graph)
    assert _smiles_of(q, pred.graph) == rec.smiles


@pytest.mark.parametrize("smiles", SMILES)
def test_canonical_r_is_sound_and_correct_on_clean_geometry(smiles):
    rec, q = _query(smiles)
    pred = get_method("canonical-R").predict(q)
    assert pred.status is Status.OK
    assert is_valid(q.atomic_numbers, q.total_charge, pred.graph)
    assert same_resonance_class(pred.graph, Graph.from_record(rec))


def test_query_hides_label():
    rec, q = _query("CC(=O)[O-]")
    assert not hasattr(q, "bonds") and not hasattr(q, "formal_charges")
    s = Query.from_record(rec, Observation.STRIPPED)
    assert (s.atomic_numbers != 1).all()
    stripped = rec.strip_hydrogens()
    with pytest.raises(ValueError):
        Query.from_record(stripped, Observation.EXPLICIT_H)


def test_capabilities_gate_unsupported():
    _, q = _query("CCO", Observation.STRIPPED)
    assert get_method("rdkit-determinebonds").unsupported_reason(q) == "stripped hydrogens"
    assert get_method("canonical-O").unsupported_reason(q) is None


def test_rdkit_adapter():
    rec, q = _query("CC(=O)Nc1ccc(O)cc1")
    pred = get_method("rdkit-determinebonds").predict(q)
    assert pred.status is Status.OK and _smiles_of(q, pred.graph) == rec.smiles


@pytest.mark.parametrize("method", ["canonical-O", "canonical-R"])
def test_equivariance(method):
    m = get_method(method)
    rng = np.random.default_rng(3)
    for smiles in SMILES:
        _, q = _query(smiles)
        perm = rng.permutation(q.n_atoms)
        rot, t = random_euclidean(rng)
        base = m.predict(q).graph
        moved = m.predict(q.transformed(perm, rot, t)).graph
        assert moved == base.permuted(perm), smiles


def _one_hot_logp(edges, target: dict, sharp=5.0):
    w = np.full((len(edges), 4), -sharp)
    for e, (i, j) in enumerate(edges.tolist()):
        w[e, target.get((i, j), 0)] = 0.0
    return w


def test_hybrid_reduces_to_o():
    """H with w(0) = −∞ on A(X) and O's (single-stage) prior equals O."""
    for smiles in ["C=CC=O", "c1ccncc1", "CC#N"]:
        _, q = _query(smiles)
        edges, _ = contact_map(q.atomic_numbers, q.positions)
        allowed = np.ones((len(edges), 4), dtype=bool)
        allowed[:, 0] = False
        sum_b = prior_objectives("kim2015", edges, q.atomic_numbers)[0].edge
        h = constrained_map(q, edges, sum_b, allowed=allowed)
        o = get_method("canonical-O").predict(q)
        assert same_resonance_class(h.graph, o.graph) and (h.graph.orders.sum() == o.graph.orders.sum())


def test_hybrid_unconstrained_equals_argmax_when_valid():
    rec, q = _query("CC(=O)Nc1ccc(O)cc1")
    edges, _ = contact_map(q.atomic_numbers, q.positions)
    logp = _one_hot_logp(edges, Graph.from_record(rec).bond_dict())
    h = constrained_map(q, edges, logp)
    assert h.graph == graph_from_argmax(q.n_atoms, edges, np.exp(logp))
    assert h.graph == Graph.from_record(rec)
    assert h.extra["gap"] > 0


def test_hybrid_repairs_invalid_argmax():
    """Entrywise argmax that mixes resonance forms (both N–O double) is projected onto 𝒢."""
    rec, q = _query("C[N+](=O)[O-]")
    edges, _ = contact_map(q.atomic_numbers, q.positions)
    logp = _one_hot_logp(edges, Graph.from_record(rec).bond_dict(), sharp=10.0)
    for e, (i, j) in enumerate(edges.tolist()):
        if {int(q.atomic_numbers[i]), int(q.atomic_numbers[j])} == {7, 8}:
            logp[e, 1], logp[e, 2] = -1.0, 0.0  # the model prefers N=O on both oxygens
    argmax = graph_from_argmax(q.n_atoms, edges, np.exp(logp))
    assert not is_valid(q.atomic_numbers, q.total_charge, argmax)
    h = constrained_map(q, edges, logp)
    assert is_valid(q.atomic_numbers, q.total_charge, h.graph)
    assert same_resonance_class(h.graph, Graph.from_record(rec))
    assert len(h.alternatives) == 2  # the two nitro resonance forms


def test_projected_wrapper_and_tags():
    m = get_method("canonical-O+project")
    assert m.tag == "canonical-O+project"
    _, q = _query("CC(=O)[O-]")
    assert m.predict(q).status is Status.OK
    assert get_method("canonical-O", {"preset": "kim2015", "c_max": 2}).tag == "canonical-O@c_max=2"
    with pytest.raises(KeyError):
        get_method("nope")


ECHO_ADAPTER = textwrap.dedent("""
    import json, sys
    src, dst, mode = sys.argv[1:4]
    with open(src) as f, open(dst, "w") as out:
        for line in f:
            q = json.loads(line)
            z = q["atomic_numbers"][::-1] if mode == "reverse" else q["atomic_numbers"]
            out.write(json.dumps({"structure_id": q["structure_id"], "status": "ok", "atomic_numbers": z,
                                  "bonds": [], "bond_orders": []}) + "\\n")
""")


@pytest.mark.parametrize("mode,expected", [("echo", Status.OK), ("reverse", Status.ERROR)])
def test_external_protocol(tmp_path, mode, expected):
    script = tmp_path / "adapter.py"
    script.write_text(ECHO_ADAPTER)
    m = get_method("external", {"command": f"{sys.executable} {script} {{input}} {{output}} {mode}"})
    _, q = _query("CC(=O)O")
    preds = m.predict_batch([q, q])
    assert [p.status for p in preds] == [expected, expected]
    if expected is Status.OK:
        assert len(preds[0].graph.bonds) == 0


def test_external_failure_is_reported(tmp_path):
    m = get_method("external", {"command": f"{sys.executable} -c 'import sys; sys.exit(3)' {{input}} {{output}}"})
    _, q = _query("CCO")
    assert m.predict(q).status is Status.ERROR


@pytest.mark.parametrize("smiles,symbol,context", [
    ("O=[N+]([O-])c1ccccc1", "N", "in nitro"),
    ("CC(=O)[O-]", "C", "in COO−"),
    ("[O-][n+]1ccccc1", "N", "in pyridine-1-oxide, etc."),
    ("[O-][n+]1ccccc1", "O", "in pyridine-1-oxide, etc."),
    ("CN=[N+]=[N-]", "N", "in N=N=R"),
    ("[C-]#[N+]C", "C", "in C≡N–R"),
    ("CS(N)(=O)=O", "S", "bonded to two O(X1) or S(X1)"),
    ("OP(=O)(O)O", "P", None),  # three O but only one terminal (X1) with explicit H
    ("CP(=O)([O-])[O-]", "P", "bonded to three O(X1) or S(X1)"),
])
def test_aps_contexts(smiles, symbol, context):
    from rdkit import Chem

    from bond_bench.eval.methods.optimization import aps_contexts

    rec, q = _query(smiles)
    edges, _ = contact_map(q.atomic_numbers, q.positions)
    ctx = aps_contexts(q.atomic_numbers, edges)
    z = Chem.GetPeriodicTable().GetAtomicNumber(symbol)
    found = set().union(*(ctx[i] for i in range(q.n_atoms) if q.atomic_numbers[i] == z))
    assert (context in found) if context else not any("O(X1)" in c for c in found)


@pytest.mark.parametrize("smiles", SMILES)
def test_atom_penalty_preset(smiles):
    rec, q = _query(smiles)
    pred = get_method("canonical-O", {"preset": "atom_penalty"}).predict(q)
    assert pred.status is Status.OK
    assert _smiles_of(q, pred.graph) == rec.smiles


def test_atom_penalty_unsupported_elements():
    _, q = _query("CB(C)C")
    assert "penalty" in get_method("canonical-O", {"preset": "atom_penalty"}).unsupported_reason(q)
    assert get_method("canonical-O").unsupported_reason(q) is None


def test_mmff_windows_are_ordered_and_disjoint():
    from collections import defaultdict

    from bond_bench.eval.tables import bond_windows

    by_pair = defaultdict(list)
    for (a, b, k), w in bond_windows("mmff").items():
        assert w.lo < w.center < w.hi
        by_pair[(a, b)].append(w)
    for ws in by_pair.values():
        ws.sort(key=lambda w: w.lo)
        assert all(x.hi <= y.lo + 1e-9 for x, y in zip(ws, ws[1:]))  # midpoint edges never overlap


@pytest.mark.parametrize("windows", ["mmff", "b2"])
def test_canonical_r_window_tables(windows):
    rec, q = _query("CC(=O)Nc1ccc(O)cc1")
    pred = get_method("canonical-R", {"windows": windows}).predict(q)
    assert pred.status is Status.OK and _smiles_of(q, pred.graph) == rec.smiles


@pytest.fixture
def tiebreak_times_out(monkeypatch):
    """Truncated co-optimal pools whose tie-break re-solve then times out (the A1 crash)."""
    from bond_bench.eval.core import ilp
    from bond_bench.eval.core.ilp import DecodeResult
    from bond_bench.eval.methods import hybrid, optimization, wrappers

    def stop(self):
        self.truncated = True
        self.stop_search()

    monkeypatch.setattr(ilp._Collector, "on_solution_callback", stop)

    def decode(spec, objectives, pool="none", max_pool=64):
        if any(o.name.startswith("P5") for o in objectives):
            return DecodeResult("timeout", reason="stage 1 (P5 Σk·d) not proven optimal")
        return ilp.decode(spec, objectives, pool=pool, max_pool=max_pool)

    for module in (hybrid, optimization, wrappers):
        monkeypatch.setattr(module, "decode", decode)


def test_tiebreak_timeout_falls_back_to_the_pool(tiebreak_times_out):
    from bond_bench.eval.methods.wrappers import project_to_valid

    rec, q = _query("c1ccncc1")
    edges, _ = contact_map(q.atomic_numbers, q.positions)
    best, pool, reason = project_to_valid(q, edges, np.full(len(edges), 1.5))
    assert best is not None and is_valid(q.atomic_numbers, q.total_charge, best) and reason == ""

    o = get_method("canonical-O").predict(q)
    assert o.status is Status.OK and "tiebreak_timeout" in o.flags

    h = constrained_map(q, edges, _one_hot_logp(edges, Graph.from_record(rec).bond_dict()))
    assert h.status is Status.OK and "tiebreak_timeout" in h.flags
    assert is_valid(q.atomic_numbers, q.total_charge, h.graph)


def test_wrapper_failure_is_that_structures_error():
    from bond_bench.eval.methods.base import Wrapper

    class Fails(Wrapper):
        name = "fails"
        family = "wrapper"

        def finish(self, query, pred):
            if query.structure_id == "bad":
                raise ValueError("min() iterable argument is empty")
            return pred

    queries = [_query(s)[1] for s in ("CCO", "c1ccncc1")]
    queries[0] = Query(**{**queries[0].__dict__, "structure_id": "bad"})
    out = Fails(get_method("canonical-O")).predict_batch(queries)
    assert out[0].status is Status.ERROR and "fails: ValueError" in out[0].reason
    assert out[1].status is Status.OK
