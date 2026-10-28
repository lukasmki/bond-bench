"""Matplotlib figures from evaluation results (``bond-bench plot``).

Every figure is the same dot grid: one row per method, one panel per condition
(stratum, σ, or strictness level), panels grouped under a spanning header. Rows
carry the method name, so color only has to separate wrapper variants
(raw, ``+project``, ``+hybrid``), which keeps it to a handful of hues however
many methods are run.

Needs the ``plot`` extra: ``uv sync --extra plot``.
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

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

# categorical slots in fixed order (validated light-mode palette; the first three pass all-pairs CVD checks)
SERIES = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
OTHER = "#898781"
# known wrapper chains keep their slot whatever subset of methods is plotted
VARIANTS = ["", "project", "hybrid"]
# reference methods always shown alongside the top-ranked ones
CANONICAL = ("canonical-O", "canonical-R", "canonical-M")

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BAND = "#e1e0d9", "#f3f2ee"

# layout, in inches
ROW_H = 0.2
PANEL_W = 0.95
PANEL_GAP = 0.2
GROUP_GAP = 0.4
TITLE_H = 0.5  # title and subtitle
LEGEND_H = 0.2
GROUP_H = 0.28  # group header and its rule
LINE_H = 0.14  # one line of a panel title
FOOT_H = 0.45  # tick labels, x label
NOTE_W = 0.25  # per-row notes to the right of a panel


@dataclass
class Panel:
    title: str
    # method -> (value, ci_lo, ci_hi); missing methods (unsupported, n = 0) leave the cell blank
    values: dict[str, tuple[float, float | None, float | None]] = field(
        default_factory=dict
    )
    # method -> short text printed right of the panel on that method's row
    notes: dict[str, str] = field(default_factory=dict)


def _split(method: str) -> tuple[str, str, str]:
    """``canonical-O@preset=x+project`` -> (family, configured base, wrapper chain)."""
    base, *wrappers = method.split("+")
    return base.split("@")[0], base, "+".join(wrappers)


def _order(methods) -> list[str]:
    """Group methods by family, base config before tuned configs, raw before wrapped."""

    def key(m):
        family, base, variant = _split(m)
        return (
            family,
            base != family,
            base,
            VARIANTS.index(variant) if variant in VARIANTS else len(VARIANTS),
            variant,
        )

    return sorted(set(methods), key=key)


def _variant_colors(methods) -> dict[str, str]:
    seen = {_split(m)[2] for m in methods}
    variants = VARIANTS + sorted(seen - set(VARIANTS))
    return {v: SERIES[i] if i < len(SERIES) else OTHER for i, v in enumerate(variants)}


def _variant_label(variant: str) -> str:
    return "+" + variant if variant else "raw"


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


def _num(x) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def _panel(title: str, rows: pl.DataFrame, metric: str) -> Panel:
    """One cell per method from rows already filtered to a single condition."""
    lo, hi = f"{metric}_lo", f"{metric}_hi"
    panel = Panel(title)
    for r in rows.filter(pl.col(metric).is_not_null()).iter_rows(named=True):
        panel.values[r["method"]] = (r[metric], _num(r.get(lo)), _num(r.get(hi)))
    return panel


def _dot_grid(
    groups: list[tuple[str, list[Panel]]],
    title: str,
    subtitle: str,
    xlabel: str,
    path: Path,
    panel_w: float = PANEL_W,
    rows: list[str] | None = None,
    labels: list[str] | None = None,
    shade: set[str] | None = None,
    xlim: tuple[float, float] = (0.0, 1.0),
) -> Path:
    """``rows`` fixes the row order (default: every method, grouped by family) and ``labels`` their tick text;
    ``shade`` bands those rows (default: alternate families); ``xlim`` is shared by every panel.
    """
    plt = _plt()
    groups = [(g, ps) for g, ps in groups if ps]
    methods = rows or _order(m for _, ps in groups for p in ps for m in p.values)
    labels = labels or methods
    colors = _variant_colors(methods)
    present = [v for v in colors if any(_split(m)[2] == v for m in methods)]
    n = len(methods)

    # header rows, top down: title + subtitle, variant legend, group header, panel titles
    legend_h = LEGEND_H if len(present) > 1 else 0
    group_h = GROUP_H if any(g for g, _ in groups) else 0
    ptitle_h = 0.1 + LINE_H * max(
        p.title.count("\n") + 1 for _, ps in groups for p in ps
    )
    title_h = TITLE_H + LINE_H * subtitle.count("\n")
    x_lo, x_hi = xlim
    span = x_hi - x_lo
    major = [x_lo, x_lo + span / 2, x_hi]
    head_h = title_h + legend_h + group_h + ptitle_h
    gap = PANEL_GAP + (NOTE_W if any(p.notes for _, ps in groups for p in ps) else 0)

    rc = {
        "font.size": 8,
        "text.color": INK,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "xtick.labelsize": 7,
        "ytick.labelsize": 8,
    }
    with plt.rc_context(rc):
        n_panels = sum(len(ps) for _, ps in groups)
        width = (
            n_panels * panel_w
            + sum(len(ps) - 1 for _, ps in groups) * gap
            + (len(groups) - 1) * GROUP_GAP
        )
        height = head_h + n * ROW_H + FOOT_H
        fig = plt.figure(figsize=(width, height))
        fx, fy = 1 / width, 1 / height  # inches -> figure fraction
        bottom, top = FOOT_H * fy, 1 - head_h * fy

        # by default alternate families get a light band so rows can be tracked across panels
        if shade is None:
            families = list(dict.fromkeys(_split(m)[0] for m in methods))
            banded = [
                y for y, m in enumerate(methods) if families.index(_split(m)[0]) % 2
            ]
        else:
            banded = [y for y, m in enumerate(methods) if m in shade]

        x0, first = 0.0, None
        for g_i, (group, panels) in enumerate(groups):
            g_start = x0
            for p_i, panel in enumerate(panels):
                ax = fig.add_axes((x0 * fx, bottom, panel_w * fx, top - bottom))
                first = first or ax
                for y in banded:
                    ax.axhspan(y - 0.5, y + 0.5, color=BAND, lw=0, zorder=0)
                for y, m in enumerate(methods):
                    if m not in panel.values:
                        continue
                    v, lo, hi = panel.values[m]
                    c = colors[_split(m)[2]]
                    if lo is not None and hi is not None:
                        ax.plot(
                            [max(lo, x_lo), min(hi, x_hi)],
                            [y, y],
                            color=c,
                            lw=1.5,
                            alpha=0.45,
                            solid_capstyle="round",
                            zorder=2,
                        )
                    ax.plot(v, y, "o", ms=4.5, mfc=c, mec="white", mew=0.7, zorder=3)
                for y, m in enumerate(methods):
                    if m in panel.notes:
                        ax.text(
                            1.04,
                            y,
                            panel.notes[m],
                            transform=ax.get_yaxis_transform(),
                            ha="left",
                            va="center",
                            fontsize=7,
                            color=INK_2,
                        )
                ax.set_xlim(x_lo - 0.04 * span, x_hi + 0.04 * span)
                ax.set_ylim(n - 0.5, -0.5)
                ax.set_xticks(major, [f"{t:.4g}" for t in major])
                ax.set_xticks([x_lo + span / 4, x_lo + 3 * span / 4], minor=True)
                ax.grid(axis="x", which="both", color=GRID, lw=0.6, zorder=1)
                ax.tick_params(length=0, pad=3)
                ax.tick_params(axis="x", which="minor", length=0)
                ax.set_yticks(range(n), labels if ax is first else [])
                for s in ax.spines.values():
                    s.set_visible(False)
                ax.set_title(panel.title, fontsize=8, color=INK_2, pad=4)
                x0 += panel_w + (gap if p_i < len(panels) - 1 else 0)
            if group:
                mid, y = (g_start + x0) / 2 * fx, top + (ptitle_h + 0.05) * fy
                fig.text(
                    mid,
                    y + 0.05 * fy,
                    group,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=INK,
                )
                fig.add_artist(
                    plt.Line2D([g_start * fx, x0 * fx], [y, y], color=MUTED, lw=0.6)
                )
            x0 += GROUP_GAP if g_i < len(groups) - 1 else 0

        # header and footer are anchored to the left edge of the method labels
        fig.canvas.draw()
        left = first.get_tightbbox().x0 / fig.dpi * fx
        fig.text(
            left,
            1 - 0.05 * fy,
            title,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
        )
        fig.text(
            left,
            1 - 0.3 * fy,
            subtitle,
            ha="left",
            va="top",
            fontsize=8,
            color=INK_2,
            linespacing=1.4,
        )
        if legend_h:
            handles = [
                plt.Line2D(
                    [],
                    [],
                    ls="",
                    marker="o",
                    ms=4.5,
                    mfc=colors[v],
                    mec="white",
                    mew=0.7,
                )
                for v in present
            ]
            fig.legend(
                handles,
                [_variant_label(v) for v in present],
                loc="upper left",
                bbox_to_anchor=(left - 0.02 * fx, 1 - (title_h - 0.03) * fy),
                ncol=len(present),
                frameon=False,
                handletextpad=0.1,
                columnspacing=1.2,
                borderaxespad=0,
                fontsize=8,
            )
        fig.text(
            (first.get_position().x0 + x0 * fx) / 2,
            0.08 * fy,
            xlabel,
            ha="center",
            va="bottom",
            color=INK_2,
        )
        fig.savefig(
            path, dpi=200, bbox_inches="tight", pad_inches=0.15, facecolor="white"
        )
        plt.close(fig)
    return path


def _ci_note(df: pl.DataFrame, metric: str) -> str:
    ci = f"{metric}_lo" in df.columns and df[f"{metric}_lo"].drop_nulls().len() > 0
    return (
        "Dot = estimate, line = 95% CI." if ci else "Dot = estimate."
    ) + " Blank = not supported."


def _strata_groups(df: pl.DataFrame, metric: str) -> list[tuple[str, list[Panel]]]:
    groups = []
    for obs in ("explicit_h", "stripped"):
        sub = df.filter((pl.col("observation") == obs) & pl.col(metric).is_not_null())
        strata = sorted(sub["stratum"].unique().to_list())
        groups.append(
            (
                OBS_LABELS[obs],
                [_panel(s, sub.filter(pl.col("stratum") == s), metric) for s in strata],
            )
        )
    return groups


def plot_strata(
    summary: pl.DataFrame, metric: str, out: Path, fmt: str = "png"
) -> list[Path]:
    """Dot grid: the overall score, then one panel per stratum grouped by observation; one row per
    method (95% CI), sorted by :func:`rank_methods` with methods not run on every stratum last.
    """
    df = summary.filter((pl.col("subgroup") == "all") & (pl.col("n") > 0))
    if metric not in summary.columns:
        raise SystemExit(
            f"metric {metric!r} not in results; available: {sorted(METRIC_LABELS)}"
        )
    if not _has(df, metric):
        return []
    ranked, n_cells = rank_methods(summary, metric)
    complete = ranked.filter(pl.col("cells") == n_cells)["method"].to_list()
    incomplete = [m for m in ranked["method"] if m not in complete]
    rows = complete + incomplete
    rank = {m: f"{i + 1}." for i, m in enumerate(complete)}
    labels = [f"{rank.get(m, '–')} {m}" for m in rows]
    canonical = {m for m in rows if _split(m)[0] in CANONICAL}
    overall = Panel(
        "mean score",
        {m: (s, None, None) for m, s in ranked.select("method", "score").iter_rows()},
    )
    label = METRIC_LABELS.get(metric, metric)
    subtitle = (
        f"Sorted by mean score over the {n_cells} strata, with unsupported molecules counted as failures; "
        f"canonical methods shaded. {_ci_note(df, metric)}"
    )
    if incomplete:
        subtitle += (
            f"\nNot ranked (not run on every stratum): {', '.join(sorted(incomplete))}."
        )
    path = out / f"strata_{metric}.{fmt}"
    return [
        _dot_grid(
            [("Overall", [overall]), *_strata_groups(df, metric)],
            f"{label} by stratum",
            subtitle,
            label,
            path,
            rows=rows,
            labels=labels,
            shade=canonical,
        )
    ]


def rank_methods(summary: pl.DataFrame, metric: str) -> tuple[pl.DataFrame, int]:
    """Rank by the mean over strata of ``metric × coverage``, so unsupported structures count as failures.

    Each method is averaged over the (stratum, observation) cells it was run on; returns the ranking
    (``method``, ``score``, ``cells``) and the number of cells any method has the metric for. Scores
    of methods with ``cells`` below that number are not comparable.
    """
    df = summary.filter(pl.col("subgroup") == "all")
    cells = (
        df.filter(pl.col(metric).is_not_null())
        .select("stratum", "observation")
        .unique()
    )
    ranked = (
        df.join(cells, on=["stratum", "observation"])
        .filter(pl.col(metric).is_not_null() | (pl.col("n") == 0))
        .with_columns(
            (
                pl.col(metric).fill_null(0)
                * pl.col("coverage").fill_nan(0).fill_null(0)
            ).alias("score")
        )
        .group_by("method")
        .agg(pl.col("score").mean(), pl.len().alias("cells"))
        .sort(["score", "method"], descending=[True, False])
    )
    return ranked, cells.height


def plot_top(
    summary: pl.DataFrame, metric: str, out: Path, fmt: str = "png", k: int = 5
) -> list[Path]:
    """The ``k`` best methods by :func:`rank_methods`, plus the canonical methods, in rank order.

    Only methods run on every stratum are ranked; the rest are named in the subtitle.
    """
    df = summary.filter((pl.col("subgroup") == "all") & (pl.col("n") > 0))
    if not _has(df, metric):
        return []
    ranked, n_cells = rank_methods(summary, metric)
    complete = ranked.filter(pl.col("cells") == n_cells)["method"].to_list()
    incomplete = sorted(set(ranked["method"]) - set(complete))
    canonical = [m for m in ranked["method"] if _split(m)[0] in CANONICAL]
    rows = complete[:k] + [m for m in canonical if m not in complete[:k]]
    rank = {m: f"{i + 1}." for i, m in enumerate(complete)}
    labels = [f"{rank.get(m, '–')} {m}" for m in rows]
    overall = Panel(
        "mean score",
        {m: (s, None, None) for m, s in ranked.select("method", "score").iter_rows()},
    )
    label = METRIC_LABELS.get(metric, metric)
    subtitle = (
        f"Top {k} by mean score over the {n_cells} strata, with unsupported molecules counted as failures; "
        f"canonical methods shaded. {_ci_note(df, metric)}"
    )
    if incomplete:
        subtitle += f"\nNot ranked (not run on every stratum): {', '.join(incomplete)}."
    groups = [("Overall", [overall]), *_strata_groups(df, metric)]
    # dots don't encode length, so when every shown value is high, zoom in (to 0.1, at most 0.8) to
    # separate close leaders
    shown = [
        x
        for _, ps in groups
        for p in ps
        for m in rows
        if m in p.values
        for x in p.values[m]
        if x is not None
    ]
    x_lo = min(math.floor(min(shown, default=0) * 10) / 10, 0.8)
    path = out / f"top_{metric}.{fmt}"
    return [
        _dot_grid(
            groups,
            f"{label}: top {k} methods",
            subtitle,
            label,
            path,
            rows=rows,
            labels=labels,
            shade=set(canonical),
            xlim=(x_lo if x_lo >= 0.5 else 0.0, 1.0),
        )
    ]


def plot_overall(
    summary: pl.DataFrame,
    metrics: list[str],
    out: Path,
    fmt: str = "png",
    raw: bool = False,
) -> list[Path]:
    """Overall score of every method on each metric side by side, one panel per metric, with each
    method's :func:`rank_methods` rank on that metric printed beside its dot.

    Rows are sorted by mean rank over the metrics; methods not ranked on every metric (not run on
    every stratum) come last and are named in the subtitle. ``raw`` drops wrapped methods
    (``+project``, ``+hybrid``) before ranking, so base methods are ranked among themselves.
    """
    if raw:
        summary = summary.filter(
            pl.col("method").map_elements(
                lambda m: not _split(m)[2], return_dtype=pl.Boolean
            )
        )
    df = summary.filter((pl.col("subgroup") == "all") & (pl.col("n") > 0))
    metrics = [m for m in metrics if _has(df, m)]
    if not metrics:
        return []
    panels, ranks = [], {}
    for metric in metrics:
        ranked, n_cells = rank_methods(summary, metric)
        complete = ranked.filter(pl.col("cells") == n_cells)["method"].to_list()
        ranks[metric] = {m: i + 1 for i, m in enumerate(complete)}
        panels.append(
            Panel(
                METRIC_LABELS.get(metric, metric),
                {
                    m: (s, None, None)
                    for m, s in ranked.select("method", "score").iter_rows()
                },
                {m: f"#{r}" for m, r in ranks[metric].items()},
            )
        )
    methods = {m for p in panels for m in p.values}
    complete = [m for m in methods if all(m in ranks[k] for k in metrics)]
    mean_rank = {m: sum(ranks[k][m] for k in metrics) / len(metrics) for m in complete}
    # ties on mean rank go to the better best rank
    complete.sort(key=lambda m: (mean_rank[m], min(ranks[k][m] for k in metrics), m))
    incomplete = sorted(methods - set(complete))
    rows = complete + incomplete
    labels = [f"{i + 1}. {m}" for i, m in enumerate(complete)] + [
        f"– {m}" for m in incomplete
    ]
    subtitle = textwrap.fill(
        "Score = mean over strata of metric × coverage, with unsupported molecules counted as failures; "
        "#k = rank on that metric. Sorted by mean rank over the "
        f"{len(metrics)} metrics; canonical methods shaded.",
        100,
    )
    if incomplete:
        subtitle += "\n" + textwrap.fill(
            f"Not ranked (not run on every stratum): {', '.join(incomplete)}.", 100
        )
    path = out / f"overall{'_raw' if raw else ''}.{fmt}"
    return [
        _dot_grid(
            [("", panels)],
            "Overall scores and rankings" + (" (raw methods)" if raw else ""),
            subtitle,
            "Mean score",
            path,
            panel_w=1.3,
            rows=rows,
            labels=labels,
            shade={m for m in rows if _split(m)[0] in CANONICAL},
        )
    ]


def plot_noise(
    summary: pl.DataFrame, metric: str, out: Path, fmt: str = "png"
) -> list[Path]:
    """Robustness: one panel per σ, grouped by stratum and observation, from C1 (``sigma=``) and on-the-fly noise (``noise_sigma=``)."""
    paths = []
    for prefix, name in (
        ("sigma=", "C1 noise ladder"),
        ("noise_sigma=", "on-the-fly noise"),
    ):
        df = summary.filter(
            pl.col("subgroup").str.starts_with(prefix) & (pl.col("n") > 0)
        )
        if df.is_empty() or not _has(df, metric):
            continue
        df = df.with_columns(
            pl.col("subgroup").str.strip_prefix(prefix).cast(pl.Float64).alias("sigma")
        )
        groups = []
        for stratum, obs in (
            df.select("stratum", "observation")
            .unique()
            .sort(["stratum", "observation"])
            .rows()
        ):
            sub = df.filter(
                (pl.col("stratum") == stratum)
                & (pl.col("observation") == obs)
                & pl.col(metric).is_not_null()
            )
            panels = [
                _panel(f"σ = {s:g} Å", sub.filter(pl.col("sigma") == s), metric)
                for s in sorted(sub["sigma"].unique().to_list())
            ]
            groups.append((f"{stratum} · {OBS_LABELS.get(obs, obs)}", panels))
        label = METRIC_LABELS.get(metric, metric)
        path = (
            out / f"noise_{'c1' if prefix == 'sigma=' else 'onthefly'}_{metric}.{fmt}"
        )
        paths.append(
            _dot_grid(
                groups,
                f"{label} under coordinate noise ({name})",
                _ci_note(df, metric),
                label,
                path,
            )
        )
    return paths


def plot_equivariance(eq: pl.DataFrame, out: Path, fmt: str = "png") -> list[Path]:
    """Mean violation rate per method, one panel per level of strictness."""
    if eq.is_empty():
        return []
    levels = [
        ("violation_strict_asymmetric", "strict"),
        ("violation_set", "set-valued"),
        ("violation_resonance", "up to resonance"),
    ]
    levels = [(c, lab) for c, lab in levels if c in eq.columns]
    agg = eq.group_by("method").agg([pl.col(c).mean() for c, _ in levels])
    panels = [
        Panel(
            textwrap.fill(lab, 18),
            {
                m: (v, None, None)
                for m, v in agg.select("method", c).iter_rows()
                if v is not None
            },
        )
        for c, lab in levels
    ]
    path = out / f"equivariance.{fmt}"
    return [
        _dot_grid(
            [("", panels)],
            "Equivariance violations",
            "Mean violation rate over strata; lower is better. Blank = not measured.",
            "Violation rate",
            path,
            panel_w=1.1,
        )
    ]


def plot_all(
    results_dir: Path,
    out: Path | None = None,
    metrics: list[str] | None = None,
    fmt: str = "png",
) -> list[Path]:
    results_dir = Path(results_dir)
    out = Path(out) if out else results_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)
    summary = report(results_dir)
    if summary.is_empty():
        raise SystemExit(f"no results in {results_dir}")
    metrics = metrics or ["smiles_match", "valid", "f1_macro"]
    paths: list[Path] = []
    for metric in metrics:
        paths += plot_strata(summary, metric, out, fmt)
        paths += plot_top(summary, metric, out, fmt)
        paths += plot_noise(summary, metric, out, fmt)
    paths += plot_overall(summary, metrics, out, fmt)
    paths += plot_overall(summary, metrics, out, fmt, raw=True)
    paths += plot_equivariance(equivariance_report(results_dir), out, fmt)
    return paths
