import urllib.request
from types import SimpleNamespace

import polars as pl
import pytest

from bond_bench.data import download, release
from bond_bench.data.store import write_index, write_stratum


@pytest.fixture
def built(tmp_path, records):
    """A two-stratum dataset directory with reports and an index, plus a license file."""
    dataset = tmp_path / "dataset"
    for sid, recs in (("T1", records[:3]), ("T2", records[3:])):
        write_stratum(dataset / f"{sid}.h5", recs, {"stratum_id": sid, "n_requested": len(recs)})
        (dataset / "reports").mkdir(exist_ok=True)
        (dataset / "reports" / f"{sid}.json").write_text(f'{{"stratum_id": "{sid}"}}\n')
    write_index(dataset)
    license_path = tmp_path / "LICENSE.md"
    license_path.write_text("# Licenses\n")
    return dataset, license_path


@pytest.fixture
def published(tmp_path, built, monkeypatch):
    """Pack ``built`` into a local release served over file:// URLs."""
    dataset, license_path = built
    staging = tmp_path / "release"
    release.pack(dataset, staging, {}, license_path=license_path)
    monkeypatch.setattr(release, "asset_url", lambda name, repo, tag: (staging / name).as_uri())
    monkeypatch.setattr(release, "github_token", lambda: None)
    return staging


def test_pack_layout(published):
    sums = release.parse_checksums((published / release.CHECKSUMS).read_text())
    assert sorted(sums) == [
        "LICENSE.md", "T1.h5", "T1.report.json", "T2.h5", "T2.report.json", "index.parquet",
    ]


def test_pack_refuses_dev_build(tmp_path, built):
    dataset, license_path = built
    specs = {"T1": SimpleNamespace(n_structures=100)}
    with pytest.raises(ValueError, match="dev builds"):
        release.pack(dataset, tmp_path / "release", specs, license_path=license_path)
    release.pack(dataset, tmp_path / "release", specs, license_path=license_path, allow_partial=True)


def test_pull_all(tmp_path, built, published):
    dataset, _ = built
    out = tmp_path / "pulled"
    status = release.pull(out)
    assert set(status.values()) == {"downloaded"}
    for name in ("T1.h5", "T2.h5", "index.parquet", "LICENSE.md", "reports/T1.json", "reports/T2.json"):
        assert (out / name).read_bytes() == ((dataset if name != "LICENSE.md" else tmp_path) / name).read_bytes()
    assert not list(out.glob(".pull-*"))
    assert set(release.pull(out).values()) == {"up to date"}


def test_pull_subset_rebuilds_index(tmp_path, published):
    out = tmp_path / "pulled"
    release.pull(out, strata=["T2"])
    assert sorted(p.name for p in out.glob("*.h5")) == ["T2.h5"]
    assert pl.read_parquet(out / "index.parquet")["stratum_id"].unique().to_list() == ["T2"]
    with pytest.raises(ValueError, match="not in release"):
        release.pull(out, strata=["T9"])


def test_pull_keeps_local_changes_without_force(tmp_path, published):
    out = tmp_path / "pulled"
    release.pull(out, strata=["T1"])
    (out / "T1.h5").write_bytes(b"locally rebuilt")
    with pytest.raises(FileExistsError, match="--force"):
        release.pull(out, strata=["T1"])
    assert (out / "T1.h5").read_bytes() == b"locally rebuilt"
    assert release.pull(out, strata=["T1"], force=True)["T1.h5"] == "downloaded"


def test_redirect_drops_auth_across_hosts():
    handler = download._NoCrossHostAuthRedirect()
    req = urllib.request.Request("https://api.github.com/a", headers={"Authorization": "Bearer t"})
    same = handler.redirect_request(req, None, 302, "Found", {}, "https://api.github.com/b")
    other = handler.redirect_request(req, None, 302, "Found", {}, "https://objects.githubusercontent.com/c")
    assert same.has_header("Authorization")
    assert not other.has_header("Authorization")


def test_pull_rejects_bad_checksum(tmp_path, published):
    (published / "T1.h5").write_bytes(b"tampered")
    out = tmp_path / "pulled"
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        release.pull(out)
    assert not list(out.glob("*.h5"))
