import numpy as np

from bond_bench.eval.core.graph import Graph
from bond_bench.eval.metrics import CLASSES, f1_from_confusion, grade
from bond_bench.eval.prediction import Prediction, Status, from_row, to_row

from ..conftest import embedded_record


def _swap_kekule(rec):
    orders = rec.bond_orders.astype(int).copy()
    orders[rec.aromatic] = 3 - orders[rec.aromatic]  # 1 ↔ 2 on the aromatic ring
    return Graph(rec.n_atoms, rec.bonds, orders, rec.formal_charges, [0] * rec.n_atoms, rec.radical_electrons)


def test_kekule_swap_is_resonance_match_not_exact():
    rec = embedded_record("c1ccccc1", "benzene")
    row = grade(rec, "explicit_h", Prediction.ok(_swap_kekule(rec)))
    assert row["valid"] and not row["exact"] and row["resonance_match"] and row["smiles_match"]


def test_exact_and_confusion_bookkeeping():
    rec = embedded_record("C=CC#N", "acrylonitrile")
    row = grade(rec, "explicit_h", Prediction.ok(Graph.from_record(rec)))
    assert row["exact"] and row["valid"]
    c = np.array(row["confusion"]).reshape(len(CLASSES), len(CLASSES))
    assert c.trace() == c.sum() == 3  # heavy-atom bonds only
    f1 = f1_from_confusion(c)
    assert f1["f1_macro"] == 1.0 and f1["conn_precision"] == 1.0


def test_failed_prediction_counts_missing_bonds():
    rec = embedded_record("CCO", "ethanol")
    row = grade(rec, "explicit_h", Prediction.fail(Status.INFEASIBLE, "charge"))
    assert not row["valid"] and not row["exact"]
    assert f1_from_confusion(row["confusion"])["conn_recall"] == 0.0


def test_stripped_grading_uses_h_counts():
    rec = embedded_record("CC(=O)O", "acetic acid")
    stripped = rec.strip_hydrogens()
    g = Graph.from_record(stripped)
    assert grade(rec, "stripped", Prediction.ok(g))["exact"]
    wrong = Graph(g.n, g.bonds, g.orders, g.charges, g.hydrogens + 1, g.radicals)
    row = grade(rec, "stripped", Prediction.ok(wrong))
    assert not row["valid"] and row["h_correct"] == 0


def test_tautomer_fields():
    rec = embedded_record("Oc1ccccn1", "hp")
    other = embedded_record("O=c1cccc[nH]1", "pyridone")
    row = grade(rec, "explicit_h", Prediction.ok(Graph.from_record(rec)), {rec.smiles, other.smiles}, other.smiles)
    assert row["taut_accept"] and not row["taut_pref"]


def test_prediction_row_roundtrip():
    rec = embedded_record("CC(=O)[O-]", "acetate")
    g = Graph.from_record(rec)
    p = Prediction.ok(g, alternatives=[g, g], margin=np.ones(len(g.bonds)), flags=["prior_tied"], extra={"a": 1})
    back = from_row(to_row(rec.structure_id, p), rec.n_atoms)
    assert back.graph == g and back.alternatives == [g, g] and back.flags == ["prior_tied"] and back.extra == {"a": 1}
