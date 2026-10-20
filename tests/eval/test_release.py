import json
import tarfile

import polars as pl
import pytest

from bond_bench.data import release as data_release
from bond_bench.data.store import write_stratum
from bond_bench.eval import release
from bond_bench.eval.runner import run


@pytest.fixture
def results(tmp_path, records, manifest):
    """Results for one method on one stratum, with an absolute dataset path and a plot."""
    data = tmp_path / "dataset"
    write_stratum(data / "C1.h5", records, {"stratum_id": "C1"})
    out = run("canonical-O", ["C1"], data, manifest, results_dir=tmp_path / "results")
    prov = json.loads((out / "provenance.json").read_text())
    prov["runs"]["C1"]["dataset"] = "/home/someone/bond-bench/dataset/C1.h5"
    (out / "provenance.json").write_text(json.dumps(prov))
    (tmp_path / "results" / "plots").mkdir()
    (tmp_path / "results" / "plots" / "strata.png").write_bytes(b"png")
    (tmp_path / "results" / ".DS_Store").write_bytes(b"")
    return tmp_path / "results"


@pytest.fixture
def published(tmp_path, results, monkeypatch):
    """Pack ``results`` into a local release served over file:// URLs."""
    staging = tmp_path / "release"
    release.pack_results(results, staging)
    monkeypatch.setattr(data_release, "asset_url", lambda name, repo, tag: (staging / name).as_uri())
    monkeypatch.setattr(release, "github_token", lambda: None)
    return staging


def test_pack_layout(results, published):
    sums = data_release.parse_checksums((published / data_release.CHECKSUMS).read_text())
    assert sorted(sums) == ["results.tar.gz", "summary.parquet"]
    with tarfile.open(published / "results.tar.gz") as tar:
        names = set(tar.getnames())
        prov = json.load(tar.extractfile("canonical-O/provenance.json"))
    assert names == {p.relative_to(results).as_posix() for p in release.result_files(results)}
    assert "plots/strata.png" in names and "summary.parquet" not in names and ".DS_Store" not in names
    assert prov["runs"]["C1"]["dataset"] == "C1.h5"
    assert len(prov["runs"]["C1"]["dataset_sha256"]) == 64


def test_pack_refuses_subsampled_runs(tmp_path, results, records, manifest):
    run("openbabel", ["C1"], tmp_path / "dataset", manifest, results_dir=results, n=2)
    with pytest.raises(ValueError, match=r"openbabel/C1 \(n=2\)"):
        release.pack_results(results, tmp_path / "release")
    release.pack_results(results, tmp_path / "release", allow_partial=True)


def test_scrub_provenance():
    prov = {
        "runs": {"B2": {"dataset": "/data/bb/B2.h5", "seed": 0}},
        "method": {"inner": {"params": {
            "cwd": "/home/gpu/bond-bench",
            "command": "uv run --project /home/gpu/bond-bench/adapters/x python /home/gpu/bond-bench/adapters/x/run.py {input}",
        }}},
    }
    out = release.scrub_provenance(prov)
    assert out["runs"]["B2"] == {"dataset": "B2.h5", "seed": 0}
    assert out["method"]["inner"]["params"] == {
        "cwd": ".", "command": "uv run --project adapters/x python adapters/x/run.py {input}",
    }


def test_pull_results(tmp_path, results, published):
    out = tmp_path / "pulled"
    status = release.pull_results(out)
    assert set(status.values()) == {"downloaded"}
    assert (out / "plots" / "strata.png").read_bytes() == b"png"
    scores = "canonical-O/C1__explicit_h.scores.parquet"
    assert (out / scores).read_bytes() == (results / scores).read_bytes()
    assert pl.read_parquet(out / "summary.parquet").equals(pl.read_parquet(results / "summary.parquet"))
    assert not list(out.glob(".pull-*"))
    assert set(release.pull_results(out).values()) == {"up to date"}


def test_pull_results_keeps_local_changes_without_force(tmp_path, published):
    out = tmp_path / "pulled"
    release.pull_results(out)
    (out / "plots" / "strata.png").write_bytes(b"local")
    (out / "mine").mkdir()
    (out / "mine" / "provenance.json").write_text("{}")
    with pytest.raises(FileExistsError, match="--force"):
        release.pull_results(out)
    assert (out / "plots" / "strata.png").read_bytes() == b"local"
    assert release.pull_results(out, force=True)["plots/strata.png"] == "downloaded"
    assert (out / "mine" / "provenance.json").exists()


def test_pull_results_rejects_bad_checksum(tmp_path, published):
    (published / "results.tar.gz").write_bytes(b"tampered")
    out = tmp_path / "pulled"
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        release.pull_results(out)
    assert not list(out.rglob("*.parquet"))
