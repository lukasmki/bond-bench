import io
import pickle
import shutil
import tarfile

import numpy as np
import polars as pl
import pytest
from rdkit import Chem

from bond_bench.data.builders.a1_pdb import A1Builder, LigandInstance, parse_ccd
from bond_bench.data.builders.b1_qmugs import inchikey_no_stereo, reference_mol
from bond_bench.data.builders.c1_noise import C1Builder
from bond_bench.data.builders.registry import get_builder, is_implemented
from bond_bench.data.sampling import sample, stratified_sample
from bond_bench.data.store import StratumReader, write_stratum

from .conftest import FIXTURES


def test_manifest_and_registry(manifest):
    assert len(manifest) == 17
    assert manifest["A1"].n_structures == 6000
    assert {s for s in manifest if is_implemented(s)} == {"A1", "A2", "B1", "B2", "B3", "B5", "C1", "C5"}
    with pytest.raises(NotImplementedError):
        get_builder("B6")


# -- sampling ---------------------------------------------------------------


def test_sampling_is_deterministic_and_order_independent():
    items = [f"m{i}" for i in range(100)]
    a = sample(items, 10, seed=3, key=str)
    assert a == sample(list(reversed(items)), 10, seed=3, key=str)
    assert a != sample(items, 10, seed=4, key=str)


def test_stratified_sample_redistributes():
    items = [("small", i) for i in range(3)] + [("big", i) for i in range(50)]
    out = stratified_sample(items, 20, 0, stratum=lambda x: x[0], key=str)
    assert len(out) == 20
    assert sum(x[0] == "small" for x in out) == 3


# -- A1 -----------------------------------------------------------------------


def test_parse_ccd():
    comp = parse_ccd(FIXTURES / "ETF.cif")
    assert comp.status == "REL"
    assert sorted(comp.elements) == ["C", "C", "F", "F", "F", "O"]
    assert sum(comp.n_hydrogens) == 3
    assert comp.orders == [1] * 5


def _a1_builder(tmp_path, manifest):
    builder = A1Builder(manifest["A1"], cache_dir=tmp_path / "cache", out_dir=tmp_path / "out")
    (builder.raw_dir / "ccd").mkdir(parents=True)
    (builder.raw_dir / "mmcif").mkdir(parents=True)
    shutil.copy(FIXTURES / "ETF.cif", builder.raw_dir / "ccd" / "ETF.cif")
    shutil.copy(FIXTURES / "4NXY.cif.gz", builder.raw_dir / "mmcif" / "4NXY.cif.gz")
    return builder


def test_a1_record_from_ccd_and_mmcif(tmp_path, manifest):
    builder = _a1_builder(tmp_path, manifest)
    inst = LigandInstance("4NXY", "ETF", "B", "A", "201", 1.45, "le1.5", 0.95, 0.1)
    record = builder._record(inst)
    assert record is not None, builder.rejections
    assert record.smiles == "OCC(F)(F)F"
    assert record.n_atoms == 6 and (record.atomic_numbers != 1).all()
    assert record.implicit_hydrogens.sum() == 3
    assert record.meta["pdb_id"] == "4NXY"


def test_a1_missing_residue_rejected(tmp_path, manifest):
    builder = _a1_builder(tmp_path, manifest)
    inst = LigandInstance("4NXY", "ETF", "B", "A", "999", 1.45, "le1.5", 0.95, 0.1)
    assert builder._record(inst) is None
    assert builder.rejections["ligand residue not found in mmCIF"] == 1


def test_a1_ccd_error_rejected(manifest):
    from bond_bench.data.chem import LabelError, validate_record
    from bond_bench.data.schema import MoleculeRecord

    comp = parse_ccd(FIXTURES / "GWE.cif")
    record = MoleculeRecord(
        "x", "x",
        atomic_numbers=[Chem.GetPeriodicTable().GetAtomicNumber(e) for e in comp.elements],
        positions=np.zeros((len(comp.elements), 3)),
        formal_charges=comp.charges,
        bonds=np.array(comp.bonds),
        bond_orders=comp.orders,
        aromatic=comp.aromatic,
        implicit_hydrogens=comp.n_hydrogens,
    )
    with pytest.raises(LabelError):
        validate_record(record)


# -- B1 -----------------------------------------------------------------------


def test_reference_mol_strips_counterions():
    mol = reference_mol("CC(=O)[O-].[Na+]")
    assert Chem.MolToSmiles(mol) == "CC(=O)[O-]"


def test_inchikey_ignores_stereo_but_not_bond_orders():
    a = Chem.MolFromSmiles("C/C=C/C(=O)O")
    b = Chem.MolFromSmiles("CC=CC(=O)O")
    c = Chem.MolFromSmiles("CCCC(=O)O")
    assert inchikey_no_stereo(a) == inchikey_no_stereo(b)
    assert inchikey_no_stereo(a) != inchikey_no_stereo(c)


def test_geom_index_and_member(tmp_path):
    from bond_bench.data.sources import geom

    archive = tmp_path / "rdkit_folder.tar"
    payload = pickle.dumps({"smiles": "C/C=C/C", "conformers": []})
    with tarfile.open(archive, "w") as tar:
        for name, data in [
            ("rdkit_folder/summary_drugs.json", b"{}"),
            ("rdkit_folder/drugs/C_C=C_C.pickle", payload),
            ("rdkit_folder/qm9/C.pickle", b"x"),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    index = pl.read_parquet(geom.build_index(tmp_path / "index.parquet", url=archive.as_uri()))
    assert index.sort("subset")["smiles"].to_list() == ["C/C=C/C", "C"]
    assert index.sort("subset")["subset"].to_list() == ["drugs", "qm9"]
    row = index.filter(pl.col("subset") == "drugs").row(0, named=True)
    raw = archive.read_bytes()[row["offset"] : row["offset"] + row["size"]]
    assert raw == payload


def test_geom_unpickler_refuses_arbitrary_globals():
    from bond_bench.data.sources.geom import _MolOnlyUnpickler

    with pytest.raises(pickle.UnpicklingError):
        _MolOnlyUnpickler(io.BytesIO(pickle.dumps(OSError("x")))).load()


# -- C1 -----------------------------------------------------------------------


def test_c1_noise_ladder(tmp_path, manifest, records):
    out = tmp_path / "out"
    write_stratum(out / "B1.h5", records, {"stratum_id": "B1"})
    builder = C1Builder(manifest["C1"], cache_dir=tmp_path, out_dir=out, n=12)
    noisy = builder.records()
    assert len(noisy) == 12
    assert builder.extra_attrs["parent_strata"] == {"B1": 3}

    parents = StratumReader(out / "B1.h5")
    by_sigma = {}
    for record in noisy:
        parent = parents.get(record.meta["parent_id"])
        np.testing.assert_array_equal(record.bond_orders, parent.bond_orders)
        assert record.inchikey == parent.inchikey
        by_sigma.setdefault(record.meta["sigma"], []).append(record.meta["rmsd"])
        if record.meta["sigma"] == 0:
            np.testing.assert_array_equal(record.positions, parent.positions)
    assert sorted(by_sigma) == [0.0, 0.05, 0.1, 0.2]
    assert np.mean(by_sigma[0.2]) == pytest.approx(0.2 * np.sqrt(3), rel=0.3)

    again = C1Builder(manifest["C1"], cache_dir=tmp_path, out_dir=out, n=12).records()
    for a, b in zip(noisy, again):
        np.testing.assert_array_equal(a.positions, b.positions)


def test_geom_record_from_member(tmp_path, manifest):
    from bond_bench.data.builders.b2_geom_drugs import B2Builder
    from bond_bench.data.builders.geom_common import geom_record
    from bond_bench.data.sources.geom import _MolOnlyUnpickler

    smiles = "COc1cc2c(cc1OC)-c1c(C)c3ccc(OC)c(OC)c3c[n+]1CC2"
    data = _MolOnlyUnpickler(open(FIXTURES / "geom_drugs_member.pickle", "rb")).load()
    builder = B2Builder(manifest["B2"], cache_dir=tmp_path, out_dir=tmp_path)
    row = {"subset": "drugs", "smiles": smiles, "inchikey_nostereo": inchikey_no_stereo(Chem.MolFromSmiles(smiles))}
    record = geom_record(builder, row, data, "B2-x")
    assert record is not None, builder.rejections
    assert record.total_charge == 1
    assert (record.atomic_numbers == 1).any()

    row["inchikey_nostereo"] = inchikey_no_stereo(Chem.MolFromSmiles("c1ccccc1"))
    assert geom_record(builder, row, data, "B2-y") is None
    assert builder.rejections["conformer graph differs from GEOM SMILES"] == 1


# -- A2 -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cod_id", "smiles", "charge_pattern"),
    [
        ("2006389", "S=C1N(C)C=CN1C", []),
        ("7106503", "[NH2+]1[C@H](C(=O)O)C[C@@H](O)C[C@H]1C.[Cl-].O", [-1, 1]),
    ],
)
def test_a2_record_from_cif(tmp_path, manifest, cod_id, smiles, charge_pattern):
    from bond_bench.data.builders.a2_cod import A2Builder
    from bond_bench.data.sources.cod import cif_relpath

    builder = A2Builder(manifest["A2"], cache_dir=tmp_path, out_dir=tmp_path)
    dest = builder.raw_dir / "cif" / cif_relpath(cod_id)
    dest.parent.mkdir(parents=True)
    shutil.copy(FIXTURES / f"{cod_id}.cif", dest)
    row = {"cod_id": cod_id, "smiles": smiles, "agreement": "coords", "r_factor": 0.04, "temperature": 100.0, "doi": ""}
    record = builder._record(row)
    assert record is not None, builder.rejections
    expected = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert record.n_atoms == expected.GetNumAtoms()
    assert record.total_charge == 0
    assert sorted(q for q in record.formal_charges.tolist() if q) == charge_pattern
    assert record.inchikey == Chem.MolToInchiKey(expected)
    # one formula unit forms a compact cluster
    assert np.ptp(record.positions, axis=0).max() < 20


# -- C5 -----------------------------------------------------------------------


def test_t1x_frame_images_and_mapped_mols():
    from bond_bench.data.sources.transition1x import edges_of, frame_image, mapped_mol

    assert [frame_image(f) for f in (0, 9, 10, 17, 18, 25)] == [0, 9, 1, 8, 1, 8]
    mol = mapped_mol("[C:2]([H:3])([H:4])([H:5])[O:1][H:6]")
    assert [a.GetSymbol() for a in mol.GetAtoms()] == ["O", "C", "H", "H", "H", "H"]
    assert (0, 1) in edges_of(mol) and (0, 5) in edges_of(mol)


def test_qcml_example_parsing():
    from bond_bench.data.sources.qcml import parse_example

    def varint(n):
        out = bytearray()
        while True:
            b = n & 0x7F
            n >>= 7
            out.append(b | (0x80 if n else 0))
            if not n:
                return bytes(out)

    def ld(field, payload):
        return varint(field << 3 | 2) + varint(len(payload)) + payload

    smiles = ld(1, ld(1, b"CO"))  # Feature.bytes_list
    seq = ld(3, ld(1, varint(0)))  # Feature.int64_list (packed)
    pos = ld(2, ld(1, np.array([0.5, 1.5], "<f4").tobytes()))  # Feature.float_list (packed)
    entries = b"".join(ld(1, ld(1, k) + ld(2, v)) for k, v in [(b"smiles", smiles), (b"seq", seq), (b"pos", pos)])
    example = parse_example(ld(1, entries))
    assert example["smiles"] == [b"CO"]
    assert example["seq"].tolist() == [0]
    assert example["pos"].tolist() == [0.5, 1.5]


@pytest.mark.parametrize("fmt", [tarfile.GNU_FORMAT, tarfile.PAX_FORMAT])
def test_geom_header_walk_handles_long_names_and_segments(tmp_path, fmt):
    from bond_bench.data.sources.geom import _walk_segment

    names = [f"rdkit_folder/drugs/{'C' * n}.pickle" for n in (5, 150, 20, 300, 7)]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=fmt) as tar:
        for i, name in enumerate(names):
            data = bytes([i]) * (700 * (i + 1))
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    raw = buf.getvalue()

    class Reader:
        size = len(raw)

        def read(self, offset, length):
            return raw[offset : offset + length]

    expected = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        for m in tar:
            expected[m.offset_data] = (m.name, m.size)
    # Split at every block boundary pair to exercise headers straddling segments.
    for cut in range(512, len(raw), 512):
        found = {}
        for start, end in ((0, cut), (cut, len(raw))):
            for name, offset, size in _walk_segment(Reader(), start, end):
                if offset not in found or len(name) > len(found[offset][0]):
                    found[offset] = (name, size)
        assert found == expected, cut
