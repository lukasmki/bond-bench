"""Aggregate per-structure scores into a summary table with 95% intervals.

Rates use all *supported* structures as the denominator: ⊥, errors and
timeouts count as wrong; ``unsupported`` structures are excluded and reported
as coverage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from .metrics import CLASSES, f1_from_confusion

RATES = ("valid", "exact", "resonance_match", "smiles_match", "connectivity_match", "taut_accept", "taut_pref")
#: Subgroup columns reported in addition to the stratum totals.
SUBGROUPS = ("meta_res_bin", "meta_sigma", "noise_sigma")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (center - half, center + half)


def bootstrap_f1(conf: np.ndarray, n_boot: int = 200, seed: int = 0) -> tuple[float, float]:
    """Percentile CI of macro F1, resampling structures."""
    if len(conf) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(conf), len(conf))
        vals.append(f1_from_confusion(conf[idx].sum(0))["f1_macro"])
    return tuple(np.nanpercentile(vals, [2.5, 97.5]).tolist())  # type: ignore[return-value]


def summarize(df: pl.DataFrame, ci: bool = True) -> dict:
    total = df.height
    supported = df.filter(pl.col("status") != "unsupported")
    n = supported.height
    out: dict = {"n_total": total, "n": n, "coverage": n / total if total else float("nan")}
    for status in ("ok", "infeasible", "error", "timeout", "unsupported"):
        out[f"n_{status}"] = int((df["status"] == status).sum())
    for col in RATES:
        if col not in supported.columns or supported[col].null_count() == n:
            continue
        k = int(supported[col].fill_null(False).sum())
        out[col] = k / n if n else float("nan")
        if ci:
            out[f"{col}_lo"], out[f"{col}_hi"] = wilson(k, n)
    if n:
        conf = np.array(supported["confusion"].to_list(), dtype=np.int64)
        out.update(f1_from_confusion(conf.sum(0)))
        if ci:
            out["f1_macro_lo"], out["f1_macro_hi"] = bootstrap_f1(conf)
        heavy = supported["n_heavy"].sum()
        out["charge_acc"] = supported["charge_correct"].sum() / heavy if heavy else float("nan")
        out["h_acc"] = supported["h_correct"].sum() / heavy if heavy else float("nan")
        out["prior_tied"] = float(supported["prior_tied"].mean())
        ok = supported.filter(pl.col("status") == "ok")
        out["runtime_median_s"] = float(ok["runtime_s"].median()) if ok.height else float("nan")
    return out


def load_scores(results_dir: Path) -> pl.DataFrame:
    frames = []
    for path in sorted(Path(results_dir).glob("*/*.scores.parquet")) + sorted(Path(results_dir).glob("*/*__noise.*.parquet")):
        method = path.parent.name
        stratum = path.name.split("__")[0]
        df = pl.read_parquet(path).with_columns(
            pl.lit(method).alias("method"), pl.lit(stratum).alias("stratum"),
            pl.lit("__noise" in path.name).alias("is_noise"),
        )
        frames.append(df)
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def report(results_dir: Path, ci: bool = True) -> pl.DataFrame:
    scores = load_scores(results_dir)
    if scores.is_empty():
        return scores
    rows = []
    keys = ["method", "stratum", "observation"]
    for (method, stratum, obs), df in scores.group_by(keys, maintain_order=True):
        clean = df.filter(~pl.col("is_noise"))
        if clean.height:
            rows.append({"method": method, "stratum": stratum, "observation": obs, "subgroup": "all", **summarize(clean, ci)})
        for col in SUBGROUPS:
            if col not in df.columns:
                continue
            src = df if col == "noise_sigma" else clean
            for (value,), sub in src.filter(pl.col(col).is_not_null()).group_by([col], maintain_order=True):
                rows.append({"method": method, "stratum": stratum, "observation": obs,
                             "subgroup": f"{col.removeprefix('meta_')}={value}", **summarize(sub, ci)})
    out = pl.DataFrame(rows, infer_schema_length=None).sort(["stratum", "observation", "subgroup", "method"])
    out.write_parquet(Path(results_dir) / "summary.parquet")
    return out


def equivariance_report(results_dir: Path) -> pl.DataFrame:
    rows = []
    for path in sorted(Path(results_dir).glob("*/*__equivariance.parquet")):
        df = pl.read_parquet(path)
        stratum, obs = path.name.split("__")[:2]
        asym = df.filter(~pl.col("symmetric")) if "symmetric" in df.columns else df
        rows.append({
            "method": path.parent.name, "stratum": stratum, "observation": obs, "n": df.height,
            "n_symmetric": df.height - asym.height,
            "violation_strict_asymmetric": 1 - float(asym["strict"].mean()) if asym.height else float("nan"),
            "violation_strict": 1 - float(df["strict"].mean()),
            "violation_set": 1 - float(df["set"].mean()),
            "violation_resonance": 1 - float(df["resonance"].mean()),
        })
    return pl.DataFrame(rows)


def format_table(summary: pl.DataFrame) -> str:
    cols = ["stratum", "observation", "subgroup", "method", "n", "coverage", "valid", "exact",
            "resonance_match", "smiles_match", "f1_macro", "taut_accept", "taut_pref", "n_infeasible", "n_error"]
    cols = [c for c in cols if c in summary.columns]
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, float_precision=3, tbl_hide_dataframe_shape=True):
        return str(summary.select(cols))


__all__ = ["CLASSES", "report", "equivariance_report", "format_table", "summarize", "wilson"]
