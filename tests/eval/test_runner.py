import polars as pl

from bond_bench.data.store import write_stratum
from bond_bench.eval.report import report, wilson
from bond_bench.eval.runner import observations_for, rescore, run


def test_observations_for(manifest):
    assert [str(o) for o in observations_for(manifest["A1"])] == ["stripped"]
    assert [str(o) for o in observations_for(manifest["B1"])] == ["explicit_h"]
    assert [str(o) for o in observations_for(manifest["C1"])] == ["explicit_h", "stripped"]


def test_wilson_interval():
    lo, hi = wilson(90, 100)
    assert lo < 0.9 < hi and 0.8 < lo and hi < 0.96


def test_run_score_report(tmp_path, records, manifest):
    data = tmp_path / "dataset"
    write_stratum(data / "C1.h5", records, {"stratum_id": "C1"})
    out = run(
        "canonical-O", ["C1"], data, manifest, results_dir=tmp_path / "results",
        n=3, noise=[0.05], reps=2, equivariance=2,
    )
    files = sorted(p.name for p in out.iterdir())
    assert "C1__explicit_h.parquet" in files and "C1__stripped.scores.parquet" in files
    assert "C1__explicit_h__noise.s0.050.parquet" in files and "C1__explicit_h__equivariance.parquet" in files
    assert "provenance.json" in files
    scores = pl.read_parquet(out / "C1__explicit_h.scores.parquet")
    assert scores.height == 3 and scores["valid"].all()
    noise = pl.read_parquet(out / "C1__explicit_h__noise.s0.050.parquet")
    assert noise.height == 6
    eq = pl.read_parquet(out / "C1__explicit_h__equivariance.parquet")
    assert eq["strict"].all()

    summary = report(tmp_path / "results")
    allrow = summary.filter((pl.col("subgroup") == "all") & (pl.col("observation") == "explicit_h"))
    assert allrow["n"].item() == 3 and allrow["smiles_match"].item() == 1.0
    assert summary.filter(pl.col("subgroup") == "noise_sigma=0.05").height == 2
    assert rescore(tmp_path / "results", data) == 2


def test_timeout_pool_kills_and_replaces_workers(records):
    from bond_bench.eval.prediction import Status
    from bond_bench.eval.query import Query
    from bond_bench.eval.runner import Runner

    queries = [Query.from_record(r, "explicit_h") for r in records]
    # a limit no real solve can meet: every structure must come back as timeout, none lost
    hung = Runner("canonical-O", {}, workers=2, timeout=1e-6).predict(queries)
    assert [p.status for p in hung] == [Status.TIMEOUT] * len(queries)
    ok = Runner("canonical-O", {}, workers=2, timeout=60).predict(queries)
    assert all(p.status is Status.OK for p in ok)
