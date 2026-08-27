"""Figures for Study C.

Three questions, three figures, in the order the study answers them.

Estimator identity goes on the y-axis, where many labels cost nothing, and the
figures carry no tier, tolerance or expectation annotation. Which comparisons are
entitled to which tolerance, and which one is expected to disagree, are arguments
rather than data: they belong in the write-up beside the reasoning, and the
numbers behind them are in `study_c_agreement.csv`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import GRID, INK, INK2, MUTED, SERIES, SURFACE, _style, _titles

__all__ = ["plot_agreement", "plot_performance", "plot_score",
           "plot_stage2_runtime", "make_all_c"]

#: What the x-axis means for each comparable quantity. `se` is compared as a
#: log ratio because a standard error scales as n^-1/2, so a difference would
#: not be one quantity across the sample sizes.
_AGREE_XLABEL = {
    "est": "|difference| in the risk difference",
    "se": "|log ratio| of the standard error",
    "pn_eic": "|difference| in Pn D*",
}
_AGREE_TITLE = {
    "est": "Point estimates: agreement with PyTMLE",
    "se": "Standard errors: agreement with PyTMLE",
    "pn_eic": "Scores: agreement with PyTMLE",
}
#: PyTMLE's own estimators, which are the baseline rather than comparators.
OWN = ("tmle", "gcomp", "aipw", "ipw")


def plot_agreement(agr: pd.DataFrame, out: Path,
                   quantity: str = "est") -> Path:
    """Mean and maximum paired difference from PyTMLE, per implementation.

    A dot plot on a log axis: the dot is the mean absolute paired difference, the
    bar runs out to the maximum. Log scale because the differences span three
    orders of magnitude and a linear axis would collapse everything but the
    largest onto the origin.

    Deliberately unannotated. Which tier an implementation belongs to, what
    tolerance it is entitled to, and which row is *expected* to disagree are all
    matters of interpretation that belong in the write-up next to the argument
    for them -- not encoded in a figure where they would have to be decoded
    again. The numbers, with their tiers and tolerances, are in
    `study_c_agreement.csv`.

    One quantity per figure. The agreement table now holds absolute differences
    (`est`), log ratios (`se`) and scores (`pn_eic`); plotting them on one shared
    log axis would put three different units side by side.
    """
    d = agr.copy()
    if "quantity" in d.columns:
        d = d[d["quantity"] == quantity]
    if "skipped" in d.columns:
        d = d[~d["skipped"].fillna(False)]
    d = d[d["mean_abs_diff"].notna()]
    if d.empty:
        raise ValueError(f"no agreement rows to plot for quantity={quantity!r}")

    ns = sorted(d["n"].unique())
    labels = list(dict.fromkeys(
        d.sort_values(["implementation"])["implementation"]))
    ypos = {lab: i for i, lab in enumerate(labels)}
    colour = SERIES["gcomp"]

    fig, axes = plt.subplots(1, len(ns), figsize=(max(3.9 * len(ns), 7.0),
                                                  0.46 * len(labels) + 2.4),
                             sharey=True, squeeze=False, facecolor=SURFACE)
    for ci, n in enumerate(ns):
        ax = axes[0][ci]; _style(ax)
        sub = d[d["n"] == n]
        for _, r in sub.iterrows():
            y = ypos[r["implementation"]]
            ax.plot([r["mean_abs_diff"], r["max_abs_diff"]], [y, y],
                    color=colour, lw=2, solid_capstyle="round", alpha=0.45,
                    zorder=2)
            ax.plot(r["mean_abs_diff"], y, "o", ms=7, color=colour,
                    markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=4)
        ax.set_xscale("log")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_ylim(-0.7, len(labels) - 0.3)
        ax.set_xlabel(_AGREE_XLABEL.get(quantity, "|difference| from PyTMLE"),
                      color=INK2, fontsize=9)
        ax.set_title(f"n = {int(n)}", color=INK, fontsize=11, pad=8)
    axes[0][0].invert_yaxis()

    fig.tight_layout(rect=[0, 0.02, 1, 0.86])
    _titles(fig, _AGREE_TITLE.get(quantity, "Agreement with PyTMLE"),
            "Paired per replicate: every implementation runs on the same data, "
            "so replicate-to-replicate variation cancels.\n"
            "Dot = mean, bar = maximum.")
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_performance(perf: pd.DataFrame, out: Path, event: int = 1) -> Path:
    """Bias and coverage on the risk-difference scale, by estimator and n.

    Only estimators that target the risk difference appear. The conventional Cox
    is on a different scale entirely and has its own figure; putting it here
    would invite exactly the comparison the study exists to warn against.
    """
    d = perf[(perf["estimand"] == "rd") & (perf["event"] == event)].copy()
    d = d[d["estimator"].notna()]
    agg = (d.groupby(["estimator", "n"])
             .agg(bias=("bias", "mean"),
                  mc_se=("bias_mc_se", lambda x: float(np.sqrt((x ** 2).sum()) / len(x))),
                  cov=("coverage", "mean"),
                  cov_se=("coverage_mc_se", lambda x: float(np.sqrt((x ** 2).sum()) / len(x))))
             .reset_index())
    ns = sorted(agg["n"].unique())
    labels = sorted(agg["estimator"].unique(),
                    key=lambda e: (e not in OWN, e))
    ypos = {lab: i for i, lab in enumerate(labels)}
    off = np.linspace(-0.26, 0.26, len(ns))
    shade = {n: str(0.15 + 0.3 * i / max(len(ns) - 1, 1)) for i, n in enumerate(ns)}

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 0.42 * len(labels) + 2.8),
                             sharey=True, facecolor=SURFACE)
    for k, (col, se, ref, xlab, title) in enumerate([
            ("bias", "mc_se", 0.0, "bias in risk difference", "Bias"),
            ("cov", "cov_se", 0.95, "95 % Wald coverage", "Coverage")]):
        ax = axes[k]; _style(ax)
        ax.axvline(ref, color=INK, lw=1.2, zorder=1)
        for i, n in enumerate(ns):
            sub = agg[agg["n"] == n]
            for _, r in sub.iterrows():
                y = ypos[r["estimator"]] + off[i]
                ax.errorbar(r[col], y, xerr=1.96 * r[se], fmt="o", ms=5.5,
                            color=shade[n], lw=0, elinewidth=1.4,
                            markeredgecolor=SURFACE, markeredgewidth=0.9, zorder=3)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_ylim(-0.7, len(labels) - 0.3)
        ax.set_xlabel(xlab, color=INK2, fontsize=9)
        ax.set_title(title, color=INK, fontsize=11, pad=8)
    axes[0].invert_yaxis()
    h = [plt.Line2D([], [], marker="o", ls="", ms=6, color=shade[n],
                    markeredgecolor=SURFACE, label=f"n = {int(n)}") for n in ns]
    axes[1].legend(handles=h, frameon=False, fontsize=8.5, loc="lower left")

    fig.tight_layout(rect=[0, 0, 1, 0.84])
    _titles(fig, f"Estimator performance under correct specification (cause {event})",
            "Every estimator sees a correctly specified model, so differences here are the "
            "estimator's own contribution. Bars are 95 % Monte Carlo intervals.")
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_score(score: pd.DataFrame, out: Path) -> Path:
    """Did each implementation solve its own score equation, and how far?

    `|Pn D*|` against the criterion each package stops on. Dimensionless, so the
    two are comparable at every n, and free of the FINDINGS 8 grid offset -- that
    shifts the CIF level in both estimators alike but is not a property of the
    score. A point below 1 has met its own stopping rule.

    Drawn on a log y-axis because the interesting range spans two orders of
    magnitude, with the threshold as a solid line rather than a dash: it is a
    hard boundary, not a target.
    """
    d = score.copy()
    srcs = [s for s in ("pytmle", "concrete") if s in set(d["source"])]
    ns = sorted(d["n"].unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(3.4 * len(ns), 3.9),
                             sharey=True, facecolor=SURFACE, squeeze=False)
    colour = {"pytmle": SERIES["tmle (PyTMLE)"], "concrete": SERIES["tmle (concrete)"]}
    mark = {"pytmle": "D", "concrete": "v"}
    for j, n in enumerate(ns):
        ax = axes[0][j]
        ax.axhline(1.0, color=INK2, linewidth=1.2, zorder=2)
        sub = d[d["n"] == n]
        for src in srcs:
            g = sub[sub["source"] == src].sort_values("time")
            for ev, gg in g.groupby("event"):
                ax.plot(gg["time"], gg["median_ratio"], color=colour[src],
                        marker=mark[src], markersize=5, linewidth=1.5,
                        linestyle="-" if ev == 1 else (0, (3, 2)),
                        label=f"{src}, cause {int(ev)}", zorder=3)
        _style(ax)
        ax.set_yscale("log")
        ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.9)
        ax.set_title(f"n = {int(n)}", fontsize=9, color=INK, pad=6)
        ax.set_xlabel("tau", fontsize=9, color=INK2)
        if j == 0:
            ax.set_ylabel("|Pn D*| / stopping criterion", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.09, 1, 0.88))
    h, la = axes[0][0].get_legend_handles_labels()
    if h:
        fig.legend(h, la, loc="lower center", ncol=min(len(la), 4), frameon=False,
                   fontsize=8, bbox_to_anchor=(0.5, 0.005))
    _titles(fig, "Did the targeted update solve its own score equation?",
            "Below the line is convergence by each package's own rule. The score "
            "is free of the grid offset that shifts the CIF level in both alike.")
    out = Path(out)
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_stage2_runtime(rt: pd.DataFrame, out: Path) -> Path:
    """Matched second-stage cost, on the three units that are comparable.

    Seconds per fit is not interpretable alone: cost is O(n * n_times) per update
    step, and the two take different numbers of steps on *different* grids --
    concrete builds its own, coarser one. So the per-step and per-cell panels sit
    beside it, each implementation normalised by its own grid, and a fourth panel
    shows the step and grid counts without which the first three cannot be read.

    Every point here was measured single-threaded and serialised; the
    `stage2_seconds` in the main study was not, and the two must not be mixed.
    """
    rt = rt.copy()
    impls = sorted(rt["implementation"].unique())
    colour = {"tmle": SERIES["tmle (PyTMLE)"], "pytmle": SERIES["tmle (PyTMLE)"],
              "concrete": SERIES["tmle (concrete)"],
              "tmle (concrete)": SERIES["tmle (concrete)"]}
    panels = [("median_s", "seconds per fit"),
              ("median_s_per_step", "seconds per accepted step"),
              ("median_ns_per_step_cell", "ns per step x subject x grid point")]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.7), facecolor=SURFACE)
    for j, (col, lab) in enumerate(panels):
        ax = axes[j]
        for im in impls:
            g = rt[rt["implementation"] == im].sort_values("n")
            ax.plot(g["n"], g[col], marker="o", markersize=5, linewidth=1.6,
                    color=colour.get(im, MUTED), label=im, zorder=3)
            if col == "median_s" and {"q05_s", "q95_s"} <= set(g.columns):
                ax.fill_between(g["n"], g["q05_s"], g["q95_s"],
                                color=colour.get(im, MUTED), alpha=0.15,
                                linewidth=0, zorder=2)
        _style(ax)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(sorted(rt["n"].unique()))
        ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.9)
        ax.set_xlabel("n", fontsize=9, color=INK2)
        ax.set_ylabel(lab, fontsize=9, color=INK2)
    # ax = axes[3]
    # for im in impls:
    #     g = rt[rt["implementation"] == im].sort_values("n")
    #     ax.plot(g["n"], g["median_steps"], marker="o", markersize=5,
    #             linewidth=1.6, color=colour.get(im, MUTED), zorder=3)
    #     ax.plot(g["n"], g["median_n_times"], marker="s", markersize=4,
    #             linewidth=1.2, linestyle=(0, (3, 2)),
    #             color=colour.get(im, MUTED), alpha=0.75, zorder=3)
    # _style(ax)
    # ax.set_xscale("log"); ax.set_yscale("log")
    # ax.set_xticks(sorted(rt["n"].unique()))
    # ax.get_xaxis().set_major_formatter(
    #     matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    # ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    # ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    # ax.set_xlabel("n", fontsize=9, color=INK2)
    # ax.set_ylabel("accepted steps (solid), grid size (dashed)", fontsize=9, color=INK2)

    fig.tight_layout(rect=(0, 0.10, 1, 0.87))
    h, la = axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, la, loc="lower center", ncol=len(la), frameon=False,
                   fontsize=8.5, bbox_to_anchor=(0.5, 0.005))
    # _titles(fig, "Second-stage cost, matched conditions",
    #         "Single-threaded and serialised, nuisances injected in both. Each "
    #         "implementation is normalised by its own grid; steps counted are "
    #         "accepted ones, so both understate rejected iterations.")
    out = Path(out)
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def make_all_c(tabs: Dict[str, pd.DataFrame], out_dir: Path,
               rd_truth: Optional[Dict[int, float]] = None,
               event: int = 1) -> list[Path]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    figs = []
    agr = tabs.get("agreement", pd.DataFrame())
    if len(agr):
        # one figure per quantity: absolute differences, log ratios and scores
        # are three different units and do not share an axis
        quants = (sorted(set(agr["quantity"])) if "quantity" in agr.columns
                  else ["est"])
        for q in quants:
            name = "study_c_agreement.png" if q == "est" else f"study_c_agreement_{q}.png"
            try:
                figs.append(plot_agreement(agr, out_dir / name, quantity=q))
            except ValueError:
                continue          # nothing comparable for that quantity
    if len(tabs.get("performance", [])):
        figs.append(plot_performance(tabs["performance"],
                                     out_dir / "study_c_performance.png", event=event))
    if len(tabs.get("score", [])):
        figs.append(plot_score(tabs["score"], out_dir / "study_c_score.png"))
    if len(tabs.get("runtime", [])):
        figs.append(plot_stage2_runtime(tabs["runtime"],
                                        out_dir / "study_c_runtime.png"))
    return figs
