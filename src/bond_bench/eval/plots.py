"""Matplotlib figures from evaluation results (``bond-bench plot``).

Needs the ``plot`` extra: ``uv sync --extra plot``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from .report import equivariance_report, report

METRIC_LABELS = {
    "smiles_match": "SMILES match",
    "exact": "Exact match",
    "resonance_match": "Resonance-aware match",
    "valid": "Validity rate",
    "f1_macro": "Bond-order macro F1",
    "taut_accept": "Tautomer acceptance",
    "taut_pref": "Tautomer preference",
}


OBS_LABELS = {"explicit_h": "explicit H", "stripped": "H stripped"}


def _colors(methods: list[str]) -> dict[str, str]:
    """One fixed color per method across every figure (matplotlib's default cycle)."""
    cycle = [f"C{i}" for i in range(10)]
    return {m: cycle[i % len(cycle)] for i, m in enumerate(sorted(methods))}


def _has(df: pl.DataFrame, metric: str) -> bool:
    return metric in df.columns and df[metric].drop_nulls().len() > 0


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("plotting needs matplotlib: uv sync --extra plot") from None
    return plt


def _error(sub: pl.DataFrame, metric: str) -> np.ndarray | None:
    lo, hi = f"{metric}_lo", f"{metric}_hi"
    if lo not in sub.columns or hi not in sub.columns:
        return None
    v = sub[metric].to_numpy().astype(float)
    err = np.vstack([v - sub[lo].to_numpy().astype(float), sub[hi].to_numpy().astype(float) - v])
    return np.nan_to_num(np.clip(err, 0, None))


def plot_strata(summary: pl.DataFrame, metric: str, out: Path, fmt: str = "png") -> list[Path]:
    """Grouped bars: one panel per observation, strata on x, one bar per method (95% CI)."""
    plt = _plt()
    df = summary.filter(pl.col("subgroup") == "all")
    if metric not in df.columns:
        raise SystemExit(f"metric {metric!r} not in results; available: {sorted(METRIC_LABELS)}")
    if not _has(df, metric):
        return []
    methods = sorted(df["method"].unique().to_list())
    colors = _colors(summary["method"].unique().to_list())
    observations = [
        o for o in ("explicit_h", "stripped")
        if _has(df.filter(pl.col("observation") == o), metric)
    ]
    fig, axes = plt.subplots(1, len(observations), figsize=(6 * len(observations), 4), squeeze=False)
    width = 0.8 / max(len(methods), 1)
    for ax, obs in zip(axes[0], observations):
        sub = df.filter(pl.col("observation") == obs)
        strata = sorted(sub.filter(pl.col(metric).is_not_null())["stratum"].unique().to_list())
        x = np.arange(len(strata))
        for k, method in enumerate(methods):
            rows = sub.filter(pl.col("method") == method)
            # align to strata; missing or unsupported (n = 0) cells stay empty
            vals, errs = np.full(len(strata), np.nan), np.zeros((2, len(strata)))
            for s_i, s in enumerate(strata):
                r = rows.filter((pl.col("stratum") == s) & (pl.col("n") > 0))
                if r.height and r[metric][0] is not None:
                    vals[s_i] = r[metric][0]
                    e = _error(r, metric)
                    if e is not None:
                        errs[:, s_i] = e[:, 0]
            ax.bar(x + (k - (len(methods) - 1) / 2) * width, vals, width, yerr=errs, capsize=2,
                   color=colors[method], label=method)
        ax.set_xticks(x, strata)
        ax.set_ylim(0, 1.05)
        ax.set_title(OBS_LABELS.get(obs, obs))
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.grid(axis="y", alpha=0.3)
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    fig.suptitle(f"{METRIC_LABELS.get(metric, metric)} by stratum")
    fig.tight_layout()
    path = out / f"strata_{metric}.{fmt}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


def plot_noise(summary: pl.DataFrame, metric: str, out: Path, fmt: str = "png") -> list[Path]:
    """Robustness curves: metric vs σ, from C1 (``sigma=``) and on-the-fly noise (``noise_sigma=``)."""
    plt = _plt()
    paths = []
    colors = _colors(summary["method"].unique().to_list())
    for prefix, name in (("sigma=", "C1 noise ladder"), ("noise_sigma=", "on-the-fly noise")):
        df = summary.filter(pl.col("subgroup").str.starts_with(prefix) & (pl.col("n") > 0))
        if df.is_empty() or not _has(df, metric):
            continue
        df = df.with_columns(pl.col("subgroup").str.strip_prefix(prefix).cast(pl.Float64).alias("sigma"))
        keys = df.select("stratum", "observation").unique().sort(["stratum", "observation"]).rows()
        fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), squeeze=False)
        for ax, (stratum, obs) in zip(axes[0], keys):
            sub = df.filter((pl.col("stratum") == stratum) & (pl.col("observation") == obs))
            for method in sorted(sub["method"].unique().to_list()):
                rows = sub.filter(pl.col("method") == method).sort("sigma")
                ax.errorbar(rows["sigma"], rows[metric], yerr=_error(rows, metric), marker="o", capsize=3,
                            color=colors[method], label=method)
            ax.set_xlabel("σ (Å)")
            ax.set_ylabel(METRIC_LABELS.get(metric, metric))
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{stratum} · {OBS_LABELS.get(obs, obs)}")
            ax.grid(alpha=0.3)
        # one legend for the figure: panels can hold different methods (unsupported ones are absent)
        handles = {}
        for ax in axes[0]:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(lab, h)
        labels = sorted(handles)
        fig.legend([handles[lab] for lab in labels], labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
        fig.suptitle(f"Robustness ({name})")
        fig.tight_layout()
        path = out / f"noise_{'c1' if prefix == 'sigma=' else 'onthefly'}_{metric}.{fmt}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_equivariance(eq: pl.DataFrame, out: Path, fmt: str = "png") -> list[Path]:
    """Mean violation rate per method, at three levels of strictness."""
    if eq.is_empty():
        return []
    plt = _plt()
    levels = [("violation_strict_asymmetric", "strict (asymmetric inputs)"),
              ("violation_set", "set-valued"), ("violation_resonance", "up to resonance")]
    levels = [(c, lab) for c, lab in levels if c in eq.columns]
    agg = eq.group_by("method").agg([pl.col(c).mean() for c, _ in levels]).sort("method")
    methods = agg["method"].to_list()
    x = np.arange(len(methods))
    width = 0.8 / len(levels)
    fig, ax = plt.subplots(figsize=(1.6 * len(methods) + 2, 4))
    for k, (col, label) in enumerate(levels):
        bars = ax.bar(x + (k - (len(levels) - 1) / 2) * width, agg[col], width, label=label)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)  # zero bars are otherwise invisible
    ax.set_xticks(x, methods)
    ax.set_ylabel("Violation rate")
    ax.set_ylim(0, 1)
    ax.set_title("Equivariance violations (mean over strata)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out / f"equivariance.{fmt}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


def plot_all(results_dir: Path, out: Path | None = None, metrics: list[str] | None = None, fmt: str = "png") -> list[Path]:
    results_dir = Path(results_dir)
    out = Path(out) if out else results_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)
    summary = report(results_dir)
    if summary.is_empty():
        raise SystemExit(f"no results in {results_dir}")
    paths: list[Path] = []
    for metric in metrics or ["smiles_match", "valid", "f1_macro"]:
        paths += plot_strata(summary, metric, out, fmt)
        paths += plot_noise(summary, metric, out, fmt)
    paths += plot_equivariance(equivariance_report(results_dir), out, fmt)
    return paths
