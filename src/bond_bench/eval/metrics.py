"""Per-structure grading of a prediction against the reference label (§5, protocol E)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data.chem import LabelError, identifiers, record_to_mol
from ..data.schema import COORDINATION, MoleculeRecord
from .core.canon import same_resonance_class
from .core.graph import Graph
from .core.validity import check
from .prediction import Prediction, Status
from .query import Observation

#: Bond classes of the per-pair confusion matrix (0 = no bond).
CLASSES = (0, 1, 2, 3, COORDINATION)
_CLASS_INDEX = {k: i for i, k in enumerate(CLASSES)}


def reference(record: MoleculeRecord, observation: Observation | str) -> MoleculeRecord:
    return record.strip_hydrogens() if Observation(observation) is Observation.STRIPPED else record


def graph_smiles(z: np.ndarray, positions: np.ndarray, g: Graph, total_charge: int) -> str | None:
    """Canonical heavy-atom SMILES of a predicted graph (same path as the labels)."""
    rec = MoleculeRecord(
        structure_id="", source_id="", atomic_numbers=z, positions=positions,
        formal_charges=g.charges, bonds=g.bonds, bond_orders=g.orders,
        aromatic=np.zeros(len(g.orders), dtype=bool), implicit_hydrogens=g.hydrogens,
        radical_electrons=g.radicals, total_charge=total_charge,
    )
    try:
        return identifiers(record_to_mol(rec))[0]
    except (LabelError, Exception):  # RDKit raises several types
        return None


def confusion(pred: Graph | None, ref: Graph, heavy: np.ndarray) -> np.ndarray:
    """Per-pair confusion (true class × predicted class) over heavy-atom pairs in supp B ∪ supp B*."""
    c = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    r = {k: v for k, v in ref.bond_dict().items() if heavy[k[0]] and heavy[k[1]]}
    p = {k: v for k, v in (pred.bond_dict() if pred else {}).items() if heavy[k[0]] and heavy[k[1]]}
    for pair in set(r) | set(p):
        c[_CLASS_INDEX.get(r.get(pair, 0), 0), _CLASS_INDEX.get(p.get(pair, 0), 0)] += 1
    return c


def grade(
    record: MoleculeRecord,
    observation: Observation | str,
    pred: Prediction,
    group_smiles: set[str] | None = None,
    pref_smiles: str | None = None,
) -> dict[str, Any]:
    """Metric row for one prediction. Atoms are index-aligned with the label."""
    ref_rec = reference(record, observation)
    ref = Graph.from_record(ref_rec)
    z = ref_rec.atomic_numbers
    heavy = z != 1
    g = pred.graph if pred.status is Status.OK else None
    row: dict[str, Any] = {
        "structure_id": record.structure_id,
        "observation": str(Observation(observation)),
        "status": str(pred.status),
        "reason": pred.reason,
        "runtime_s": pred.runtime_s,
        "n_atoms": int(len(z)),
        "n_heavy": int(heavy.sum()),
        "prior_tied": "prior_tied" in pred.flags,
        "n_alternatives": len(pred.alternatives),
    }
    invalid = check(z, ref_rec.total_charge, g) if g is not None else "no prediction"
    row["valid"] = invalid is None
    row["invalid_reason"] = invalid or ""
    row["exact"] = g is not None and g == ref
    row["resonance_match"] = bool(row["valid"] and same_resonance_class(g, ref))
    row["connectivity_match"] = g is not None and np.array_equal(g.bonds, ref.bonds)
    row["alternatives_hit"] = any(a == ref for a in pred.alternatives) or row["exact"]
    smiles = graph_smiles(z, ref_rec.positions, g, ref_rec.total_charge) if g is not None else None
    row["smiles"] = smiles or ""
    row["smiles_match"] = smiles is not None and smiles == record.smiles
    row["confusion"] = confusion(g, ref, heavy).reshape(-1).tolist()
    if g is not None:
        row["charge_correct"] = int((g.charges[heavy] == ref.charges[heavy]).sum())
        row["h_correct"] = int((g.hydrogens[heavy] == ref.hydrogens[heavy]).sum())
    else:
        row["charge_correct"] = row["h_correct"] = 0
    if group_smiles is not None:
        row["taut_accept"] = smiles is not None and smiles in group_smiles
        row["taut_pref"] = smiles is not None and smiles == pref_smiles
    return row


def f1_from_confusion(c: np.ndarray) -> dict[str, float]:
    """Per-class F1 for bond classes 1–3 (and 9), macro F1, and connectivity P/R."""
    c = np.asarray(c, dtype=np.float64).reshape(len(CLASSES), len(CLASSES))
    out: dict[str, float] = {}
    f1s = []
    for idx, k in enumerate(CLASSES):
        if k == 0:
            continue
        tp, fp, fn = c[idx, idx], c[:, idx].sum() - c[idx, idx], c[idx, :].sum() - c[idx, idx]
        if tp + fp + fn == 0:
            continue
        f1 = 2 * tp / (2 * tp + fp + fn)
        out[f"f1_{k}"] = f1
        f1s.append(f1)
    out["f1_macro"] = float(np.mean(f1s)) if f1s else float("nan")
    bonded_true, bonded_pred = c[1:, :].sum(), c[:, 1:].sum()
    both = c[1:, 1:].sum()
    out["conn_precision"] = both / bonded_pred if bonded_pred else float("nan")
    out["conn_recall"] = both / bonded_true if bonded_true else float("nan")
    return out
