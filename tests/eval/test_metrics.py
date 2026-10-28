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


def test_equivalent_resonance_forms_score_full_f1():
    rec = embedded_record("c1ccccc1[N+](=O)[O-]", "nitrobenzene")
    g = _swap_kekule(rec)
    nitro = np.isin(rec.atomic_numbers[g.bonds], (7, 8)).all(axis=1)
    orders = g.orders.copy()
    orders[nitro] = 3 - orders[nitro]  # the other nitro placement
    o = np.flatnonzero(rec.atomic_numbers == 8)
    charges = g.charges.copy()
    charges[o] = charges[o[::-1]]
    swapped = Graph(g.n, g.bonds, orders, charges, g.hydrogens, g.radicals)
    row = grade(rec, "explicit_h", Prediction.ok(swapped))
    assert not row["exact"] and row["smiles_match"]
    assert f1_from_confusion(row["confusion"])["f1_macro"] == 1.0


def test_charge_separated_form_is_not_equivalent():
    rec = embedded_record("CC(C)=O", "acetone")
    g = Graph.from_record(rec)
    co = np.isin(rec.atomic_numbers[g.bonds], (6, 8)).all(axis=1) & (g.orders == 2)
    i, j = g.bonds[co][0]
    orders, charges = g.orders.copy(), g.charges.copy()
    orders[co] = 1
    charges[[i, j]] = (1, -1) if rec.atomic_numbers[i] == 6 else (-1, 1)
    row = grade(rec, "explicit_h", Prediction.ok(Graph(g.n, g.bonds, orders, charges, g.hydrogens, g.radicals)))
    assert f1_from_confusion(row["confusion"])["f1_macro"] < 1.0


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
