import numpy as np
import pytest

from bond_bench.eval.core.graph import Graph
from bond_bench.eval.core.validity import is_valid
from bond_bench.eval.ml.labels import IGNORE, atom_class, candidate_pairs, label_class, targets
from bond_bench.eval.prediction import ATOM_CHARGES, ATOM_HYDROGENS, Prediction, from_row, to_row
from bond_bench.eval.query import Observation, Query, random_euclidean

from ..conftest import embedded_record

SMILES = ["CC(=O)Nc1ccc(O)cc1", "C[N+](C)(C)CC(=O)[O-]", "O=[N+]([O-])c1ccccc1", "c1ccncc1", "CS(N)(=O)=O", "C#CC=C"]


def test_candidate_pairs_cover_bonds():
    for smiles in SMILES:
        rec = embedded_record(smiles, smiles)
        keys = {tuple(p) for p in candidate_pairs(rec.atomic_numbers, rec.positions).tolist()}
        assert {tuple(sorted(b)) for b in rec.bonds.tolist()} <= keys


@pytest.mark.parametrize(("smiles", "size"), [("c1ccccc1", 2), ("O=[N+]([O-])C", 2), ("CC(=O)[O-]", 2), ("CCO", 1)])
def test_label_class(smiles, size):
    rec = embedded_record(smiles, smiles)
    members = label_class(rec)
    ref = Graph.from_record(rec)
    assert members[0] == ref and len(members) == size
    for g in members:
        assert np.array_equal(g.bonds, ref.bonds) and is_valid(rec.atomic_numbers, rec.total_charge, g)


def test_targets_per_regime():
    rec = embedded_record("CC(=O)[O-]", "acetate")
    explicit, stripped = targets(rec)
    assert explicit.observation == Observation.EXPLICIT_H and stripped.observation == Observation.STRIPPED
    assert explicit.n == rec.n_atoms and stripped.n == rec.n_heavy
    assert explicit.bond_orders.shape == (2, rec.n_bonds)
    # both members put −1 on one oxygen; H counts are labels only when H are stripped
    q = Query.from_record(rec, Observation.STRIPPED)
    assert (stripped.atom_labels[:, q.atomic_numbers == 6] != atom_class(0, 0)).any()
    assert set(np.unique(explicit.atom_labels)) <= {atom_class(c, 0) for c in ATOM_CHARGES}
    pairs = candidate_pairs(stripped.z, stripped.positions)
    labels = stripped.edge_labels(pairs)
    assert labels.shape == (2, len(pairs)) and (labels > 0).sum(axis=1).tolist() == [3, 3]
    assert stripped.missed_bonds(pairs) == 0
    assert atom_class(3, 0) == IGNORE


def test_atom_probs_roundtrip():
    g = Graph(2, [[0, 1]], [1], [0, 0], [0, 0], [0, 0])
    probs = np.random.default_rng(0).dirichlet(np.ones(25), size=2).reshape(2, len(ATOM_CHARGES), len(ATOM_HYDROGENS))
    back = from_row(to_row("x", Prediction.ok(g, atom_probs=probs)), 2)
    assert np.allclose(back.atom_probs, probs)
    assert from_row(to_row("x", Prediction.ok(g)), 2).atom_probs is None


# ---------------------------------------------------------------- torch


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    pytest.importorskip("torch")
    from bond_bench.data.store import write_stratum
    from bond_bench.eval.ml.model import ModelConfig
    from bond_bench.eval.ml.train import TrainConfig, save_checkpoint, train

    tmp = tmp_path_factory.mktemp("canonical_m")
    records = [embedded_record(s, f"T-{i}") for i, s in enumerate(SMILES)]
    path = write_stratum(tmp / "T1.h5", records, {"stratum_id": "T1"})
    cfg = TrainConfig(epochs=2, batch_size=2, val_fraction=0.3, loader_workers=0, label_workers=1,
                      model=ModelConfig(hidden=16, layers=2))
    ckpt = train([path], cfg, device="cpu")
    assert ckpt["train"]["n_train_targets"] + ckpt["train"]["n_val_targets"] == 2 * len(SMILES)
    assert {1, 6, 7, 8, 16} == set(ckpt["seen"]["elements"])
    return save_checkpoint(ckpt, tmp / "m.pt")


def test_network_is_invariant(checkpoint):
    import torch

    from bond_bench.eval.ml.model import collate, graph_inputs
    from bond_bench.eval.ml.train import load_checkpoint, load_model

    model = load_model(load_checkpoint(checkpoint))
    rec = embedded_record("CC(=O)Nc1ccc(O)cc1", "paracetamol")
    q = Query.from_record(rec, Observation.EXPLICIT_H)
    rng = np.random.default_rng(1)
    perm = rng.permutation(q.n_atoms)
    rot, t = random_euclidean(rng)
    moved = q.transformed(perm, rot, t)

    def run(query):
        pairs = candidate_pairs(query.atomic_numbers, query.positions)
        batch = collate([graph_inputs(query.atomic_numbers, query.positions, 0, pairs, model.config.r_msg)], torch.float64)
        with torch.no_grad():
            e, a = model(batch)
        return {tuple(p): row for p, row in zip(pairs.tolist(), e.numpy())}, a.numpy()

    e0, a0 = run(q)
    e1, a1 = run(moved)
    assert np.allclose(a1[perm], a0, atol=1e-9)
    for (i, j), row in e0.items():
        assert np.allclose(e1[tuple(sorted((perm[i], perm[j])))], row, atol=1e-9)


def test_canonical_m_predicts_and_is_equivariant(checkpoint):
    from bond_bench.eval.methods.base import get_method

    m = get_method("canonical-M", {"checkpoint": str(checkpoint)})
    assert m.check_available() is None and "+ckpt." in m.version()
    rng = np.random.default_rng(3)
    for smiles in SMILES[:3]:
        rec = embedded_record(smiles, "T-0" if smiles == SMILES[0] else smiles)
        for obs in Observation:
            q = Query.from_record(rec, obs)
            pred = m.predict(q)
            assert pred.graph is not None and pred.atom_probs.shape == (q.n_atoms, 5, 5)
            assert np.allclose(pred.edge_probs.sum(axis=1), 1.0)
            assert ("invalid" in pred.flags) != is_valid(q.atomic_numbers, q.total_charge, pred.graph)
            assert len(pred.margin) == len(pred.graph.bonds)
            if obs is Observation.EXPLICIT_H:
                assert not pred.graph.hydrogens.any()
            perm = rng.permutation(q.n_atoms)
            rot, t = random_euclidean(rng)
            assert m.predict(q.transformed(perm, rot, t)).graph == pred.graph.permuted(perm)
    first = m.predict(Query.from_record(embedded_record(SMILES[0], "T-0"), Observation.EXPLICIT_H))
    assert first.extra["in_training_set"] == ("T-0" in m._train_ids)


def test_canonical_m_hybrid_is_valid(checkpoint):
    from bond_bench.eval.methods.base import get_method

    m = get_method("canonical-M+hybrid", {"checkpoint": str(checkpoint)})
    for obs in Observation:
        q = Query.from_record(embedded_record("O=[N+]([O-])c1ccccc1", "nb"), obs)
        pred = m.predict(q)
        assert is_valid(q.atomic_numbers, q.total_charge, pred.graph)
        assert pred.atom_probs is not None and "in_training_set" in pred.extra


def test_checkpoint_names_resolve_under_models():
    from bond_bench.eval.methods.base import get_method
    from bond_bench.eval.methods.learned import CHECKPOINT_DIR, checkpoint_path

    assert checkpoint_path("canonical-M") == CHECKPOINT_DIR / "canonical-M.pt"
    assert checkpoint_path("/abs/m.pt").as_posix() == "/abs/m.pt"
    m = get_method("canonical-M", {"checkpoint": "canonical-M-sigma0.05"})
    assert m.tag == "canonical-M@checkpoint=canonical-M-sigma0.05" and "/" not in m.tag
