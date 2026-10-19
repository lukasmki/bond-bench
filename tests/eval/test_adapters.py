"""Adapter plumbing that needs no adapter environment: protocol, shared helpers, registry."""

import importlib.util
import sys
import textwrap

import pytest

from bond_bench.data.manifest import REPO_ROOT
from bond_bench.eval.core.geometry import contact_map
from bond_bench.eval.methods.adapters import AdapterMethod, adapter_command
from bond_bench.eval.methods.external import ExternalMethod, query_to_json
from bond_bench.eval.prediction import Status
from bond_bench.eval.query import Observation, Query

from ..conftest import embedded_record


def _query(smiles, obs=Observation.EXPLICIT_H):
    rec = embedded_record(smiles, smiles)
    return rec, Query.from_record(rec, obs)


@pytest.fixture(scope="module")
def bb():
    spec = importlib.util.spec_from_file_location("bbadapter", REPO_ROOT / "adapters" / "_shared" / "bbadapter.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_contacts_only_when_requested():
    _, q = _query("CCO")
    assert "contacts" not in query_to_json(q)
    d = query_to_json(q, 1.3)
    edges, _ = contact_map(q.atomic_numbers, q.positions, 1.3)
    assert d["contacts"] == edges.tolist()


CONTACT_ECHO = textwrap.dedent("""
    import json, sys
    src, dst = sys.argv[1:3]
    with open(src) as f, open(dst, "w") as out:
        for line in f:
            q = json.loads(line)
            c = q.get("contacts")
            out.write(json.dumps({"structure_id": q["structure_id"], "status": "ok",
                                  "atomic_numbers": q["atomic_numbers"], "bonds": c or [],
                                  "bond_orders": [1] * len(c or [])}) + "\\n")
""")


@pytest.mark.parametrize("send", [False, True])
def test_send_contacts_flag(tmp_path, send):
    script = tmp_path / "adapter.py"
    script.write_text(CONTACT_ECHO)

    class Echo(ExternalMethod):
        name = "echo-contacts"
        send_contacts = send
        defaults = {**ExternalMethod.defaults, "contact_scale": 1.3}

    _, q = _query("CC(=O)O")
    pred = Echo(command=f"{sys.executable} {script} {{input}} {{output}}").predict(q)
    assert pred.status is Status.OK
    assert len(pred.graph.bonds) == (len(contact_map(q.atomic_numbers, q.positions)[0]) if send else 0)


def test_attach_hydrogens(bb):
    _, q = _query("CC(=O)O")
    z, x = q.atomic_numbers.tolist(), q.positions.tolist()
    pairs = bb.attach_hydrogens(z, x)
    assert len(pairs) == sum(a == 1 for a in z)
    edges = set(map(tuple, contact_map(q.atomic_numbers, q.positions)[0].tolist()))
    assert set(pairs) <= edges  # every H goes to the heavy atom it is bonded to
    assert bb.attach_hydrogens([1, 1], [[0, 0, 0], [0.74, 0, 0]]) == [(0, 1)]  # H2


def test_probs4(bb):
    assert bb.probs4(0.1, 0.2, 0.1, 0.0, 0.6) == pytest.approx([0.1, 0.5, 0.4, 0.0])


def test_kekulize(bb):
    _, q = _query("c1ccccc1", Observation.STRIPPED)
    edges = contact_map(q.atomic_numbers, q.positions)[0].tolist()
    orders, flags = bb.kekulize(q.atomic_numbers.tolist(), edges, [bb.AROMATIC] * 6, explicit_h=False)
    assert not flags and sorted(orders) == [1, 1, 1, 2, 2, 2]
    assert bb.kekulize([6, 6], [(0, 1)], [2]) == ([2], [])
    orders, flags = bb.kekulize([6, 6, 6], [(0, 1), (1, 2), (0, 2)], [bb.AROMATIC] * 3)
    assert flags == ["kekulize_failed"] and orders == [1, 1, 1]


def test_adapter_commands():
    assert "uv run --quiet --project" in adapter_command("openbabel")
    assert "pixi run --quiet --manifest-path" in adapter_command("antechamber", "pixi")
    with pytest.raises(ValueError):
        adapter_command("x", "docker")


def test_adapter_registry():
    from bond_bench.eval.methods.base import available_methods

    adapters = {n: c for n, c in available_methods().items() if issubclass(c, AdapterMethod)}
    for name, cls in adapters.items():
        m = cls()
        assert m.params["command"].endswith("{input} {output}"), name
        assert (REPO_ROOT / "adapters" / cls.adapter / "run.py").exists(), name
        reason = m.check_available()
        assert reason is None or isinstance(reason, str)


def test_missing_setup_is_reported():
    class Fake(AdapterMethod):
        name = "fake"
        adapter = "openbabel"
        requires = ("vendor/does-not-exist",)

    assert "setup.sh" in Fake().check_available()


def _load_run(name):
    """An adapter's run.py as a module (only those whose top level needs no adapter env)."""
    spec = importlib.util.spec_from_file_location(f"run_{name}", REPO_ROOT / "adapters" / name / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mperformer_decoding_thresholds():
    run = _load_run("mperformer")
    assert [run.bond_order(v) for v in (0.2, 0.9, 1.5, 1.9, 2.7)] == [0, 1, run.AROMATIC, 2, 3]
    assert [run.charge(v) for v in (-0.8, 0.1, 0.7)] == [-1, 0, 1]
    assert [run.h_count(v) for v in (0.1, 1.2, 1.8, 2.9)] == [0, 1, 2, 3]


def test_ball_mol2_round_trip(tmp_path):
    run = _load_run("ball")
    _, q = _query("C=CCl")
    d = query_to_json(q, 1.3)
    run.write_mol2(d, tmp_path / "in.mol2")
    text = (tmp_path / "in.mol2").read_text()
    assert " Cl 1 MOL 0.0" in text and text.count("\n") > q.n_atoms
    assert run.read_mol2_orders(tmp_path / "in.mol2") == {tuple(p): 1 for p in d["contacts"]}


def test_antechamber_files(tmp_path):
    run = _load_run("antechamber")
    _, q = _query("CC(=O)[O-]")
    d = query_to_json(q)
    run.write_gcrt(d, tmp_path / "in.gcrt")
    lines = (tmp_path / "in.gcrt").read_text().splitlines()
    assert lines[4] == "-1 1" and len([x for x in lines[5:] if x.strip()]) == q.n_atoms
    (tmp_path / "out.mol2").write_text("@<TRIPOS>ATOM\n1 C1 0 0 0 C.ar\n@<TRIPOS>BOND\n1 1 2 ar\n2 2 3 2\n3 3 4 du\n")
    bonds, orders, flags = run.read_mol2_bonds(tmp_path / "out.mol2")
    assert bonds == [(0, 1), (1, 2), (2, 3)] and orders == [run.AROMATIC, 2, 1] and flags


def test_forwarded_options(tmp_path):
    script = tmp_path / "echo.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(REPO_ROOT / "adapters" / "_shared")!r})
        import bbadapter
        bbadapter.serve(lambda q: bbadapter.result(q["atomic_numbers"], [], [], flags=[bbadapter.options["mode"]]),
                        lambda: "echo " + bbadapter.options.get("mode", "?"))
    """))

    class Echo(AdapterMethod):
        name = "echo-options"
        forwarded = ("mode",)
        defaults = {**AdapterMethod.defaults, "command": f"{sys.executable} {script} {{input}} {{output}}", "mode": "x"}

    _, q = _query("CO")
    pred = Echo(mode="fast").predict(q)
    assert pred.status is Status.OK and pred.flags == ["fast"]
    assert Echo(mode="fast").version() == "echo fast"
