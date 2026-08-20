"""Figures for Study C.

Three questions, three figures, in the order the study answers them.

**Colour encodes tier, not estimator.** There are eleven estimator labels here,
far past the number of categorical hues that stay distinguishable, and the
distinction that actually governs how a row should be read is what the row shares
with PyTMLE -- byte-identical nuisances, the same fitted objects, or only the
model class. That is three levels, it maps onto an ordered scale, and it puts the
interpretive rule into the encoding rather than the caption. Estimator identity
goes on the y-axis, where eleven labels cost nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import GRID, INK, INK2, MUTED, SURFACE, _style, _titles

__all__ = ["plot_agreement", "plot_performance", "plot_loghr", "make_all_c"]

#: Ordered, so darker = more shared. Slots taken from the validated categorical
#: set in `plots.py` and used here as an ordinal ramp.
TIER_COLOUR = {1: "#1b5e9c", 2: "#2a78d6", 3: "#8fb8e8"}
TIER_LABEL = {1: "tier 1 — identical nuisances",
              2: "tier 2 — identical fitted objects",
              3: "tier 3 — same model class only"}
#: PyTMLE's own estimators, which are the baseline rather than comparators.
OWN = ("tmle", "gcomp", "aipw", "ipw")


def _tier_legend(fig, tiers: Sequence[int]) -> None:
    """Figure-level, below the axes.

    Inside a panel it lands on the data whenever the y-axis is short, which it is
    here: the row count is small and the bottom rows are exactly where a
    lower-corner legend goes.
    """
    h = [plt.Line2D([], [], marker="o", ls="", ms=7, color=TIER_COLOUR[t],
                    markeredgecolor=SURFACE, markeredgewidth=1.0,
                    label=TIER_LABEL[t]) for t in sorted(set(tiers))]
    h.append(plt.Line2D([], [], marker="o", ls="", ms=7, markerfacecolor=SURFACE,
                        markeredgecolor=INK2, markeredgewidth=1.6,
                        label="divergence expected"))
    fig.legend(handles=h, frameon=False, fontsize=8.5, ncol=len(h),
               loc="lower center", bbox_to_anchor=(0.5, -0.03))


def plot_agreement(agr: pd.DataFrame, out: Path) -> Path:
    """Does every implementation agree with PyTMLE, at its own tolerance?

    A dot plot on a log axis: mean absolute paired difference per comparator,
    with the tier's tolerance drawn as a tick, so the verdict is read as a
    position rather than inferred from a colour. Log scale because the
    differences span three orders of magnitude and a linear axis would collapse
    tiers 1 and 2 onto the origin.

    One row is *expected* to sit right of its tick. concrete 1.0.8 still carries
    the targeted-update defect PyTMLE has fixed, so the two now differ on the
    targeting increment by roughly the size of that defect; agreement there would
    be the surprising result. Those rows are drawn hollow and labelled, so a
    reader does not have to know which way to read each line.
    """
    d = agr.copy()
    if d.empty:
        raise ValueError("no agreement rows to plot")
    ns = sorted(d["n"].unique())
    labels = list(dict.fromkeys(d.sort_values(["tier", "implementation"])["implementation"]))
    ypos = {lab: i for i, lab in enumerate(labels)}

    fig, axes = plt.subplots(1, len(ns), figsize=(max(4.6 * len(ns), 8.0),
                                                  0.46 * len(labels) + 2.9),
                             sharey=True, squeeze=False, facecolor=SURFACE)
    for ci, n in enumerate(ns):
        ax = axes[0][ci]; _style(ax)
        sub = d[d["n"] == n]
        for _, r in sub.iterrows():
            y = ypos[r["implementation"]]
            c = TIER_COLOUR[int(r["tier"])]
            ax.plot([r["mean_abs_diff"], r["max_abs_diff"]], [y, y], color=c,
                    lw=2, solid_capstyle="round", alpha=0.45, zorder=2)
            diverge = str(r.get("expect", "agree")) == "diverge"
            ax.plot(r["mean_abs_diff"], y, "o", ms=7,
                    markerfacecolor=SURFACE if diverge else c,
                    markeredgecolor=c if diverge else SURFACE,
                    markeredgewidth=1.6 if diverge else 1.1, zorder=4)
            if diverge and ci == 0:
                ax.annotate("divergence expected", (r["mean_abs_diff"], y),
                            textcoords="offset points", xytext=(10, 0),
                            va="center", color=c, fontsize=8.5)
            if np.isfinite(r["tolerance"]):
                ax.plot(r["tolerance"], y, "|", color=INK, ms=13, mew=1.8, zorder=5)
        ax.set_xscale("log")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_ylim(-0.7, len(labels) - 0.3)
        ax.set_xlabel("|difference| from PyTMLE  (dot = mean, bar = max)",
                      color=INK2, fontsize=9)
        ax.set_title(f"n = {int(n)}", color=INK, fontsize=11, pad=8)
    axes[0][0].invert_yaxis()
    _tier_legend(fig, d["tier"].astype(int).tolist())

    fig.tight_layout(rect=[0, 0.06, 1, 0.82])
    _titles(fig, "Do the implementations agree, each at its own tolerance?",
            "Paired per replicate. Tick = the tolerance for that tier; filled dots should sit "
            "left of it.\nThe hollow dot should not — concrete 1.0.8 still carries the "
            "targeted-update defect (FINDINGS 9), and the gap estimates its size.")
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


def plot_loghr(loghr: pd.DataFrame, rd_truth: Dict[int, float], out: Path) -> Path:
    """The conventional analysis, on its own scale, next to what it is mistaken for.

    A cause-specific hazard ratio is conditional and non-collapsible: it is not an
    estimate of the marginal risk difference, and scoring it against one would be
    a category error. So the left panel scores it against the parameter it really
    estimates -- the DGP's own ``theta_j``, known exactly -- and the right panel
    states the marginal risk difference the same data imply. The gap between the
    two panels is the point: a practitioner reading the hazard ratio as "the
    treatment effect on incidence" is answering a different question.
    """
    d = loghr.copy()
    if d.empty:
        raise ValueError("no log-HR rows to plot")
    ns = sorted(d["n"].unique())
    causes = sorted(d["event"].unique())
    off = np.linspace(-0.22, 0.22, len(ns))
    shade = {n: str(0.15 + 0.3 * i / max(len(ns) - 1, 1)) for i, n in enumerate(ns)}

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.4), facecolor=SURFACE)

    ax = axes[0]; _style(ax)
    for i, n in enumerate(ns):
        sub = d[d["n"] == n]
        for _, r in sub.iterrows():
            y = causes.index(int(r["event"])) + off[i]
            ax.errorbar(r["bias"], y, xerr=1.96 * r["mc_se"], fmt="o", ms=6,
                        color=shade[n], lw=0, elinewidth=1.5,
                        markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
    ax.axvline(0, color=INK, lw=1.2, zorder=1)
    ax.set_yticks(range(len(causes)))
    ax.set_yticklabels([f"cause {c}\ntrue log HR "
                        f"{d[d.event == c]['true_loghr'].iloc[0]:+.2f}" for c in causes],
                       fontsize=9)
    ax.set_ylim(-0.6, len(causes) - 0.4)
    ax.invert_yaxis()   # cause 1 at the top, as in every other figure here
    ax.set_xlabel("bias in the conditional log hazard ratio", color=INK2, fontsize=9)
    ax.set_title("Scored against what it estimates", color=INK, fontsize=11, pad=8)
    h = [plt.Line2D([], [], marker="o", ls="", ms=6, color=shade[n],
                    markeredgecolor=SURFACE, label=f"n = {int(n)}") for n in ns]
    ax.legend(handles=h, frameon=False, fontsize=8.5, loc="lower right")

    ax = axes[1]; _style(ax)
    ax.grid(False)
    ax.axis("off")
    lines = ["The same data, on the scale the question is usually asked in:", ""]
    for c in causes:
        hr = float(d[d.event == c]["true_loghr"].iloc[0])
        rd = rd_truth.get(int(c))
        lines.append(f"   cause {c}:   hazard ratio {np.exp(hr):.2f}"
                     + (f"      marginal risk difference {rd:+.3f}" if rd is not None else ""))
    lines += ["",
              "The hazard ratio is conditional on covariates and",
              "non-collapsible; the risk difference is marginal.",
              "They are different parameters, and a correctly",
              "estimated hazard ratio is still not an answer to",
              "the question the risk difference asks."]
    ax.text(0.0, 0.95, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=9.5, color=INK2, linespacing=1.6)

    fig.tight_layout(rect=[0, 0, 1, 0.82])
    _titles(fig, "The conventional cause-specific Cox analysis",
            "It recovers its own parameter well. That parameter is not the marginal risk "
            "difference, which is the motivation for a targeted estimator in the first place.")
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def make_all_c(tabs: Dict[str, pd.DataFrame], out_dir: Path,
               rd_truth: Optional[Dict[int, float]] = None,
               event: int = 1) -> list[Path]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    figs = []
    if len(tabs.get("agreement", [])):
        figs.append(plot_agreement(tabs["agreement"], out_dir / "study_c_agreement.png"))
    if len(tabs.get("performance", [])):
        figs.append(plot_performance(tabs["performance"],
                                     out_dir / "study_c_performance.png", event=event))
    if len(tabs.get("loghr", [])):
        figs.append(plot_loghr(tabs["loghr"], rd_truth or {},
                               out_dir / "study_c_loghr.png"))
    return figs
