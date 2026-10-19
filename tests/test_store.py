import numpy as np
import pytest

from bond_bench.data.chem import LabelError, identifiers, record_to_mol, validate_record
from bond_bench.data.store import StratumReader, write_index, write_stratum
from bond_bench.data.validate import validate_index, validate_stratum

from .conftest import embedded_record


def test_roundtrip(tmp_path, records):
    path = write_stratum(tmp_path / "T1.h5", records, {"stratum_id": "T1", "params": {"a": 1}})
    reader = StratumReader(path)
    assert len(reader) == len(records)
    for original, loaded in zip(records, reader):
        assert loaded.structure_id == original.structure_id
        assert loaded.inchikey == original.inchikey
        assert loaded.meta == original.meta
        np.testing.assert_array_equal(loaded.bonds, original.bonds)
        np.testing.assert_array_equal(loaded.bond_orders, original.bond_orders)
        np.testing.assert_allclose(loaded.positions, original.positions, atol=1e-5)
    assert reader.get("T-1").total_charge == 0


def test_strip_hydrogens_preserves_graph(records):
    for record in records:
        stripped = record.strip_hydrogens()
        assert (stripped.atomic_numbers != 1).all()
        assert stripped.n_atoms == record.n_heavy
        assert stripped.implicit_hydrogens.sum() == (record.atomic_numbers == 1).sum()
        validate_record(stripped)
        assert identifiers(record_to_mol(stripped)) == (record.smiles, record.inchikey)


def test_duplicate_ids_rejected(tmp_path, records):
    with pytest.raises(ValueError):
        write_stratum(tmp_path / "T1.h5", records + records[:1], {})


def test_validate_and_index(tmp_path, records):
    write_stratum(tmp_path / "T1.h5", records, {"stratum_id": "T1", "n_requested": len(records)})
    write_index(tmp_path)
    result = validate_stratum(tmp_path / "T1.h5", sample=None)
    assert result.ok, result.errors
    assert not result.warnings
    assert validate_index(tmp_path, [result]) == []


@pytest.mark.parametrize("smiles", ["CS(N)(O)O", "O=C1N=C(N)[SH]=C1"])
def test_implausible_hypervalence_rejected(smiles):
    with pytest.raises(LabelError, match="hypervalent"):
        validate_record(embedded_record(smiles, "x"))


def test_undeclared_radical_rejected(records):
    record = records[3].strip_hydrogens()
    record.implicit_hydrogens[0] -= 1
    with pytest.raises(LabelError, match="radical"):
        validate_record(record)


def test_charge_mismatch_rejected(records):
    record = records[0]
    record.total_charge = 1
    with pytest.raises(LabelError, match="total_charge"):
        validate_record(record)
