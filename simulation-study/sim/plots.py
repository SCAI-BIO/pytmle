"""Figures for the Study A report.

Design follows the data-viz procedure: form chosen from the data's job, colour
assigned last and validated (the four categorical hues pass the lightness-band,
chroma-floor, CVD-separation, normal-vision and contrast checks against a light
surface). Two slots carry a contrast WARN against white, which obliges visible
labelling rather than colour alone -- so every figure has both a legend and
direct labels, and the estimator is also encoded by marker shape.

These commit to a single light look: they are static figures for a written
report, not a themed web page.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ["plot_bias_by_cell", "plot_coverage", "plot_bias_vs_n",
           "plot_runtime", "make_all"]

# validated categorical slots 1-4 (light surface #fcfcfb)
# categorical slots 1-5, assigned in fixed order and validated together against
# the light surface (lightness band, chroma floor, CVD separation, normal-vision
# floor, contrast)
SERIES = {"gcomp": "#2a78d6", "ipw": "#eb6834", "aipw": "#1baf7a",
          "tmle (PyTMLE)": "#eda100", "tmle (concrete)": "#e87ba4"}
MARKER = {"gcomp": "o", "ipw": "s", "aipw": "^", "tmle (PyTMLE)": "D", "tmle (concrete)": "v"}
ORDER = ["gcomp", "ipw", "aipw", "tmle (PyTMLE)", "tmle (concrete)"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
SURFACE = "#fcfcfb"
GRID = "#e3e2de"

#: which cells theory says each estimator should be biased in -- drawn as a
#: background band so the reader can check the result against the prediction
#: without holding the 2^3 table in their head
EXPECTED_BIAS = {
    "gcomp": {"C5", "C6", "C7", "C8"},
    "ipw": {"C2", "C3", "C4", "C6", "C7", "C8"},
    "aipw": {"C6", "C7", "C8"},
    "tmle (PyTMLE)": {"C6", "C7", "C8"},
    # same estimator, second implementation: the same prediction applies
    "tmle (concrete)": {"C6", "C7", "C8"},
}
CELL_SPEC = {
    "C1": "✓✓✓", "C2": "✓✓✗", "C3": "✓✗✓", "C4": "✓✗✗",
    "C5": "✗✓✓", "C6": "✗✓✗", "C7": "✗✗✓", "C8": "✗✗✗",
}


def _titles(fig, title: str, subtitle: str) -> None:
    """Draw title and subtitle above a laid-out figure.

    Called *after* ``tight_layout``: tight_layout repositions a ``suptitle`` to sit
    just above the axes, which silently collapses any gap left for a subtitle and
    prints the two on top of each other.
    """
    top = max(ax.get_position().y1 for ax in fig.axes)
    h = fig.get_figheight()
    fig.text(0.02, top + 0.62 / h, title, color=INK, fontsize=13, ha="left",
             va="bottom")
    fig.text(0.02, top + 0.30 / h, subtitle, color=MUTED, fontsize=8.5, ha="left",
             va="bottom")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=1.0)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def _log_x_ticks(ax, values: Sequence[float]) -> None:
    """Label a log x-axis at the sample sizes actually run, and only there.

    ``set_xticks`` replaces the major ticks but leaves the log locator's *minor*
    ticks in place, and matplotlib labels those too -- so "3x10^2, 4x10^2, ..."
    prints between 250 and 1000 alongside the real ones.
    """
    vals = sorted(set(values))
    ax.set_xticks(vals)
    ax.set_xticklabels([f"{int(v)}" for v in vals])
    ax.set_xticks([], minor=True)


def _cells(df: pd.DataFrame) -> list[str]:
    return sorted({c.split("_")[0] for c in df["cell"]},
                  key=lambda c: int(c[1:]))


def _ns(df: pd.DataFrame) -> list[int]:
    return sorted({int(c.split("_n")[1]) for c in df["cell"]})


#: The study reports cause 1 as the focal event. Cause 2's biases are
#: opposite-signed, so mixing them into one panel cancels the signal; it is
#: available via ``event=2`` when wanted.
FOCAL_EVENT = 1


#: The runner writes PyTMLE's estimator as plain ``tmle``; the figures name it
#: ``tmle (PyTMLE)`` so it reads as one of two implementations rather than as
#: "the" TMLE. Remapped on the way in, so stored results stay untouched and an
#: unmapped label cannot silently drop the series from a plot.
ESTIMATOR_LABEL = {"tmle": "tmle (PyTMLE)"}


def _prep(summary: pd.DataFrame, estimand: str = "rd",
          event: Optional[int] = FOCAL_EVENT) -> pd.DataFrame:
    d = summary[summary["estimand"] == estimand].copy()
    d["estimator"] = d["estimator"].replace(ESTIMATOR_LABEL)
    if event is not None:
        d = d[d["event"] == event]
    d["cellid"] = d["cell"].str.split("_").str[0]
    d["n"] = d["cell"].str.split("_n").str[1].astype(int)
    return d


def plot_bias_by_cell(summary: pd.DataFrame, out: Path, estimand: str = "rd",
                      event: int = FOCAL_EVENT,
                      estimators: Optional[Sequence[str]] = None) -> Path:
    """The headline: does each estimator fail exactly where theory says it should?

    Bias with 95 % Monte Carlo intervals, faceted by cause and sample size.

    One cause per figure. The two causes carry opposite-signed biases here, so
    pooling them cancels almost exactly and makes a badly biased estimator look
    clean -- `gcomp` in C5-C8 reads as 0.000 when its mean |bias| is 0.041.

    A dot plot rather than bars: the quantity is signed, carries uncertainty, and
    zero is the reference.

    `estimators` restricts and orders the series. Dropping `tmle (concrete)`
    gives the figure that answers "does each estimator behave as theory
    predicts"; keeping it also answers "is the port faithful", which is a
    different question and costs a fifth mark in every row. The offsets are
    derived from whatever is left, so the remaining dots respace rather than
    leaving a gap where concrete used to be.
    """
    d = _prep(summary, estimand, event)
    order = [e for e in (estimators if estimators is not None else ORDER)
             if e in set(d["estimator"])]
    if not order:
        raise ValueError("no estimators left to plot")
    d = d[d["estimator"].isin(order)]
    ns, cells = _ns(d), _cells(d)
    events = [event]
    agg = (d.groupby(["cellid", "n", "event", "estimator"])
             .apply(lambda g: pd.Series({
                 "bias": g["bias"].mean(),
                 "mc_se": float(np.sqrt((g["bias_mc_se"] ** 2).sum()) / len(g)),
             }), include_groups=False).reset_index())

    fig, axes = plt.subplots(len(events), len(ns), squeeze=False, sharey=True,
                             sharex=True, figsize=(4.1 * len(ns), 5.4),
                             facecolor=SURFACE)
    off = np.linspace(-0.30, 0.30, len(order))

    for ri, ev in enumerate(events):
        for ci, n in enumerate(ns):
            ax = axes[ri][ci]
            _style(ax)
            sub = agg[(agg["n"] == n) & (agg["event"] == ev)]
            for yi, cell in enumerate(cells):
                for k, est in enumerate(order):
                    r = sub[(sub.cellid == cell) & (sub.estimator == est)]
                    if r.empty:
                        continue
                    b, se = float(r["bias"].iloc[0]), float(r["mc_se"].iloc[0])
                    y = yi + off[k]
                    expected = cell in EXPECTED_BIAS[est]
                    ax.plot([b - 1.96 * se, b + 1.96 * se], [y, y],
                            color=SERIES[est], lw=2, solid_capstyle="round", zorder=2)
                    # filled = series fill with a surface ring (separates overlaps);
                    # hollow = surface fill with a series outline. A surface edge on
                    # a hollow mark erases it.
                    ax.plot(b, y, MARKER[est], ms=6.5,
                            markerfacecolor=SERIES[est] if expected else SURFACE,
                            markeredgecolor=SURFACE if expected else SERIES[est],
                            markeredgewidth=1.4, zorder=3)
            ax.axvline(0, color=INK2, lw=1.2, zorder=1)
            ax.set_yticks(range(len(cells)))
            ax.set_yticklabels([f"{c}  {CELL_SPEC[c]}" for c in cells], fontsize=9)
            if ri == 0:
                ax.set_title(f"n = {n}", color=INK, fontsize=11, pad=8)
            if ri == len(events) - 1:
                ax.set_xlabel("bias in risk difference", color=INK2, fontsize=9)

    axes[0][0].invert_yaxis()   # shared axis: invert exactly once, C1 at the top
    present = [e for e in order if e in set(agg.estimator)]
    handles = [plt.Line2D([], [], color=SERIES[e], marker=MARKER[e], ms=6.5, lw=2,
                          markeredgecolor=SURFACE, label=e) for e in present]
    handles.append(plt.Line2D([], [], marker="o", ms=6.5, lw=0,
                              markerfacecolor=SURFACE, markeredgecolor=MUTED,
                              markeredgewidth=1.4, label="theory: unbiased here"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    # _titles(fig,
    #         f"Double robustness: bias by specification cell (cause {event})",
    #         "cell label shows Q / \u03c0 / G  (\u2713 correct, \u2717 wrong).  "
    #         "Filled marks = theory predicts bias; hollow = predicts none.  "
    #         "PyTMLE and concrete overlapping = the port is faithful.")
    fig.savefig(out, dpi=400, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_coverage(summary: pd.DataFrame, out: Path, estimand: str = "rd",
                  event: int = FOCAL_EVENT) -> Path:
    """Wald coverage against nominal, with the same cell layout."""
    d = _prep(summary, estimand, event)
    d = d[d["coverage"].notna()]
    ns, cells = _ns(d), _cells(d)
    agg = (d.groupby(["cellid", "n", "estimator"])
             .apply(lambda g: pd.Series({
                 "coverage": g["coverage"].mean(),
                 "mc_se": float(np.sqrt((g["coverage_mc_se"] ** 2).sum()) / len(g)),
             }), include_groups=False).reset_index())

    fig, axes = plt.subplots(1, len(ns), figsize=(4.1 * len(ns), 5.4), sharey=True,
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    off = np.linspace(-0.30, 0.30, len(ORDER))
    ests = [e for e in ORDER if e in set(agg.estimator)]

    for ax, n in zip(axes, ns):
        _style(ax)
        sub = agg[agg["n"] == n]
        for yi, cell in enumerate(cells):
            for k, est in enumerate(ests):
                r = sub[(sub.cellid == cell) & (sub.estimator == est)]
                if r.empty:
                    continue
                c, se = float(r["coverage"].iloc[0]), float(r["mc_se"].iloc[0])
                y = yi + off[ORDER.index(est)]
                ax.plot([c - 1.96 * se, c + 1.96 * se], [y, y],
                        color=SERIES[est], lw=2, solid_capstyle="round", zorder=2)
                ax.plot(c, y, MARKER[est], ms=6.5, color=SERIES[est],
                        markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        ax.axvline(0.95, color=INK2, lw=1.2, zorder=1)
        ax.set_yticks(range(len(cells)))
        ax.set_yticklabels([f"{c}  {CELL_SPEC[c]}" for c in cells], fontsize=9)
        ax.set_title(f"n = {n}", color=INK, fontsize=11, pad=8)
        ax.set_xlabel("95 % Wald coverage", color=INK2, fontsize=9)
        ax.set_xlim(min(0.5, agg["coverage"].min() - 0.05), 1.02)

    axes[0].invert_yaxis()   # shared axis: invert exactly once, C1 at the top
    handles = [plt.Line2D([], [], color=SERIES[e], marker=MARKER[e], ms=6.5, lw=2,
                          markeredgecolor=SURFACE, label=e) for e in ests]
    fig.legend(handles=handles, loc="lower center", ncol=len(ests), frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    # _titles(fig,
    #         f"Wald interval coverage by specification cell (cause {event})",
    #         "Vertical line is the nominal 95 %. Double robustness buys consistency, "
    #         "not inference: under-coverage where a nuisance is wrong is expected.")
    fig.savefig(out, dpi=400, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_bias_vs_n(summary: pd.DataFrame, out: Path, estimand: str = "rd",
                   event: int = FOCAL_EVENT) -> Path:
    """Does the bias shrink with n? Consistency, cell by cell.

    |bias| on a log-log scale against n, with a 1/sqrt(n) guide. A line that
    parallels the guide is a vanishing second-order remainder; a flat line is a
    fixed asymptotic bias.
    """
    d = _prep(summary, estimand, event)
    cells = _cells(d)
    agg = (d.groupby(["cellid", "n", "estimator"])["bias"]
             .apply(lambda s: float(np.abs(s).mean())).reset_index(name="abs_bias"))

    ncol = 4
    nrow = int(np.ceil(len(cells) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.9 * nrow),
                             sharex=True, sharey=True, facecolor=SURFACE)
    axes = np.atleast_2d(axes)
    ns = np.array(_ns(d), dtype=float)

    for i, cell in enumerate(cells):
        ax = axes[i // ncol][i % ncol]
        _style(ax)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.9)
        sub = agg[agg.cellid == cell]
        for est in ORDER:
            s = sub[sub.estimator == est].sort_values("n")
            if s.empty:
                continue
            ax.plot(s.n, s.abs_bias, MARKER[est] + "-", color=SERIES[est], lw=2,
                    ms=5.5, markeredgecolor=SURFACE, markeredgewidth=1.0,
                    label=est if i == 0 else None)
        ref = 0.05 * np.sqrt(ns[0] / ns)
        ax.plot(ns, ref, ls=(0, (4, 3)), color=MUTED, lw=1.3,
                label=r"$1/\sqrt{n}$" if i == 0 else None)
        ax.set_xscale("log"); ax.set_yscale("log")
        # a log locator inserts labelled minor ticks (3x10^2, ...) between the
        # only sample sizes actually run, which reads as data that is not there
        _log_x_ticks(ax, ns)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"{cell}   {CELL_SPEC[cell]}", color=INK, fontsize=10, pad=5)
        if i % ncol == 0:
            ax.set_ylabel("|bias|", color=INK2, fontsize=9)
        if i // ncol == nrow - 1:
            ax.set_xlabel("n", color=INK2, fontsize=9)
    for j in range(len(cells), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.legend(loc="lower center", ncol=5, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    # _titles(fig,
    #         f"Is the bias vanishing? |bias| against sample size (cause {event})",
    #         "Log-log. Parallel to the dashed guide = shrinking like "
    #         r"$1/\sqrt{n}$; flat = a fixed asymptotic bias. "
    #         "Only the guide's slope is meaningful, not its height.")
    fig.savefig(out, dpi=400, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_runtime(runtimes: pd.DataFrame, out: Path) -> Path:
    """Second-stage cost: PyTMLE against concrete, the same algorithm twice.

    Median seconds per targeted update against n, log-log, with the interquartile
    range as a band. Only the update is timed -- initial estimates are injected in
    both -- so this is implementation cost, not nuisance-fitting cost.

    The two are *not* run under matched conditions: PyTMLE's replicates run inside
    a worker pool while concrete runs one process at a time, and the second stage
    is memory-bandwidth bound. Read the shapes and the ratio's trend, not the
    absolute gap.
    """
    d = runtimes.copy()
    d["impl"] = np.where(d["implementation"].astype(str).str.contains("concrete"),
                         "tmle (concrete)", "tmle (PyTMLE)")
    agg = (d.groupby(["impl", "n"])
             .agg(median_s=("median_s", "median"), q05=("q05_s", "median"),
                  q95=("q95_s", "median"), steps=("median_steps", "median"))
             .reset_index().sort_values(["impl", "n"]))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), facecolor=SURFACE)
    ax = axes[0]; _style(ax); ax.grid(True, axis="y", color=GRID, lw=0.8, alpha=0.9)
    for impl in ("tmle (PyTMLE)", "tmle (concrete)"):
        g = agg[agg.impl == impl]
        if g.empty:
            continue
        ax.fill_between(g.n, g.q05, g.q95, color=SERIES[impl], alpha=0.13, lw=0)
        ax.plot(g.n, g.median_s, MARKER[impl] + "-", color=SERIES[impl], lw=2,
                ms=6, markeredgecolor=SURFACE, markeredgewidth=1.1, label=impl)
        last = g.iloc[-1]
        ax.annotate(impl, (last.n, last.median_s), textcoords="offset points",
                    xytext=(6, 3), color=SERIES[impl], fontsize=8.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    _log_x_ticks(ax, agg.n.unique())
    ax.set_xlabel("n", color=INK2, fontsize=9)
    ax.set_ylabel("seconds per targeted update", color=INK2, fontsize=9)
    ax.set_title("Second-stage runtime", color=INK, fontsize=11, pad=8)

    ax = axes[1]; _style(ax); ax.grid(True, axis="y", color=GRID, lw=0.8, alpha=0.9)
    w = agg.pivot_table(index="n", columns="impl", values="median_s")
    if {"tmle (PyTMLE)", "tmle (concrete)"}.issubset(w.columns):
        ratio = w["tmle (concrete)"] / w["tmle (PyTMLE)"]
        ax.plot(ratio.index, ratio.values, "o-", color=INK2, lw=2, ms=6,
                markeredgecolor=SURFACE, markeredgewidth=1.1)
        for x, y in zip(ratio.index, ratio.values):
            ax.annotate(f"{y:.2f}×", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", color=INK2, fontsize=8.5)
    ax.axhline(1.0, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.set_xscale("log")
    _log_x_ticks(ax, agg.n.unique())
    ax.set_xlabel("n", color=INK2, fontsize=9)
    ax.set_ylabel("concrete ÷ PyTMLE", color=INK2, fontsize=9)
    ax.set_title("Relative cost", color=INK, fontsize=11, pad=8)

    fig.tight_layout(rect=[0, 0, 1, 0.86])
    # _titles(fig, "Targeted update: PyTMLE vs concrete",
    #         "Median seconds per update, band = 5th\u201395th percentile. Initial "
    #         "estimates injected in both, so nuisance fitting is excluded. "
    #         "Not a matched-conditions benchmark \u2014 workers differ; read the shapes.")
    fig.savefig(out, dpi=400, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def make_all(summary: pd.DataFrame, out_dir: Path, estimand: str = "rd",
             event: int = FOCAL_EVENT,
             runtimes: Optional[pd.DataFrame] = None) -> list[Path]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # Two versions of the headline figure. With concrete it answers "is the port
    # faithful"; without it, "does each estimator fail where theory says" -- and
    # that reading is easier with four marks per row instead of five.
    own = [e for e in ORDER if "concrete" not in e]
    figs = [
        plot_bias_by_cell(summary, out_dir / f"study_a_bias_{estimand}.png", estimand, event),
        plot_bias_by_cell(summary, out_dir / f"study_a_bias_{estimand}_pytmle_only.png",
                          estimand, event, estimators=own),
        plot_coverage(summary, out_dir / f"study_a_coverage_{estimand}.png", estimand, event),
        plot_bias_vs_n(summary, out_dir / f"study_a_bias_vs_n_{estimand}.png", estimand, event),
    ]
    if runtimes is not None and len(runtimes):
        figs.append(plot_runtime(runtimes, out_dir / "study_a_runtime.png"))
    return figs
