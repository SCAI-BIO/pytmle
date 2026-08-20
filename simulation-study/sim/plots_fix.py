"""Figures for the targeted-update fix: what it repaired, and what it cost.

Deliberately three panels rather than one. The bias panel alone would be a
misleading advertisement: the fix removes the systematic bias it was meant to
remove, but at n = 250 it also produces rare replicates that diverge wildly, and
the efficient-influence-curve standard error stops covering the spread. A figure
that showed only the first of those would be selling the change rather than
reporting it.

    python -m sim.plots_fix --before results/study_a --after results/study_a_postfix
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dgp import get_config
from .plots import CELL_SPEC, GRID, INK, INK2, MUTED, SURFACE, _style, _titles
from .report import collect, summarise_dir
from .runner import target_times_for
from .truth import closed_form

__all__ = ["fix_frame", "plot_fix", "plot_fix_tails"]

#: before is the faded past, after is the estimator's own colour from `plots.py`
BEFORE, AFTER = "#6b6a66", "#eda100"
TRIM_Q = 0.5  # per side, in percent


def fix_frame(before: Path | str, after: Path | str,
              cells: Optional[Sequence[str]] = None, event: int = 1,
              config: str = "threshold", n_mc: int = 4_000_000) -> pd.DataFrame:
    """One row per (cell, run) with bias, coverage and both RMSEs."""
    before, after = Path(before), Path(after)
    if cells is None:
        cells = sorted(p.name for p in after.iterdir()
                       if p.is_dir() and not p.name.startswith("_"))

    p = get_config(config)
    taus = target_times_for(p)
    tr = (closed_form(p, taus, n_mc=n_mc).drop_duplicates(["event", "time"])
          .set_index(["event", "time"])["rd"])

    rows: List[Dict] = []
    for label, d in (("before", before), ("after", after)):
        s = summarise_dir(d, cells=cells, n_mc=n_mc)
        s = s[(s.estimand == "rd") & (s.event == event) & (s.estimator == "tmle")]
        raw = collect(d, cells=cells)
        raw = raw[(raw.estimator == "tmle") & (raw.estimand == "rd")
                  & (raw.event == event)].copy()
        raw["truth"] = [tr.loc[(event, float(t))] for t in raw.time]
        raw["err"] = raw.est - raw.truth

        for cell, g in s.groupby("cell"):
            # trim whole replicates, not individual targets: a replicate that
            # diverges does so at every target, and dropping targets piecemeal
            # would mix different replicate sets into one number
            per = raw[raw.cell == cell].dropna(subset=["err"]).groupby("rep")["err"].mean()
            lo, hi = np.percentile(per, [TRIM_Q, 100 - TRIM_Q])
            kept = per[(per >= lo) & (per <= hi)]
            rows.append({
                "cell": cell.split("_")[0], "run": label,
                "bias": g.bias.mean(),
                "mc_se": float(np.sqrt((g.bias_mc_se ** 2).sum()) / len(g)),
                "bias_trim": float(kept.mean()),
                "rmse": float(np.sqrt((per ** 2).mean())),
                "rmse_trim": float(np.sqrt((kept ** 2).mean())),
                "coverage": g.coverage.mean(),
                "cov_mc_se": float(np.sqrt((g.coverage_mc_se ** 2).sum()) / len(g)),
                "se_ratio": g.se_ratio.mean(),
                "n_wild": int((per.abs() > 0.5).sum()),
                "reps": int(len(per)),
            })
    return pd.DataFrame(rows)


def _cellorder(d: pd.DataFrame) -> List[str]:
    return sorted(d.cell.unique(), key=lambda c: int(c[1:]))


def _ylabels(ax, cells: Sequence[str]) -> None:
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([f"{c}  {CELL_SPEC[c]}" for c in cells], fontsize=9)
    ax.set_ylim(-0.6, len(cells) - 0.4)


def plot_fix(d: pd.DataFrame, out: Path, n_label: str = "n = 250") -> Path:
    """Bias, spread and coverage, before against after."""
    cells = _cellorder(d)
    yy = {c: i for i, c in enumerate(cells)}
    off = 0.16

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3), facecolor=SURFACE)

    # --- 1. bias -----------------------------------------------------------
    ax = axes[0]; _style(ax)
    ax.axvline(0, color=INK, lw=1.2, zorder=1)
    for run, colour, sgn in (("before", BEFORE, -1), ("after", AFTER, +1)):
        g = d[d.run == run]
        y = [yy[c] + sgn * off for c in g.cell]
        ax.errorbar(g.bias, y, xerr=1.96 * g.mc_se, fmt="o", color=colour,
                    ms=6.5, lw=0, elinewidth=1.6, capsize=0,
                    markeredgecolor=SURFACE, markeredgewidth=1.1, label=run, zorder=3)
        # trimmed mean, where it differs visibly, as a hollow marker
        for _, r in g.iterrows():
            if abs(r.bias_trim - r.bias) > 1.5 * r.mc_se:
                ax.plot(r.bias_trim, yy[r.cell] + sgn * off, "o", ms=6.5,
                        markerfacecolor="none", markeredgecolor=colour,
                        markeredgewidth=1.7, zorder=4)
    _ylabels(ax, cells)
    ax.set_xlabel("bias in risk difference", color=INK2, fontsize=9)
    ax.set_title("Bias", color=INK, fontsize=11, pad=8)

    # --- 2. rmse, trimmed; untrimmed called out only where it diverges ------
    # The untrimmed RMSE spans 0.04 to 0.28, so putting both on one linear axis
    # would compress every real difference into the left margin. The trimmed
    # value is plotted and the untrimmed one is named where it is materially
    # larger -- which is exactly the cell the reader needs warned about.
    ax = axes[1]; _style(ax)
    for run, colour, sgn in (("before", BEFORE, -1), ("after", AFTER, +1)):
        g = d[d.run == run]
        y = [yy[c] + sgn * off for c in g.cell]
        ax.plot(g.rmse_trim, y, "o", color=colour, ms=6.5,
                markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=3)
        for _, r in g.iterrows():
            if r.rmse > r.rmse_trim * 1.2:
                ax.annotate(f"{r.rmse:.2f} untrimmed\n({r.n_wild} of {r.reps} reps)",
                            (r.rmse_trim, yy[r.cell] + sgn * off),
                            textcoords="offset points", xytext=(-8, -4),
                            ha="right", va="center", color=colour, fontsize=8.5,
                            fontweight="bold")
    _ylabels(ax, cells)
    ax.set_xlabel("RMSE, 1 % trimmed", color=INK2, fontsize=9)
    ax.set_title("Spread", color=INK, fontsize=11, pad=8)

    # --- 3. coverage --------------------------------------------------------
    ax = axes[2]; _style(ax)
    ax.axvline(0.95, color=INK, lw=1.2, zorder=1)
    for run, colour, sgn in (("before", BEFORE, -1), ("after", AFTER, +1)):
        g = d[d.run == run]
        y = [yy[c] + sgn * off for c in g.cell]
        ax.errorbar(g.coverage, y, xerr=1.96 * g.cov_mc_se, fmt="o", color=colour,
                    ms=6.5, lw=0, elinewidth=1.6, capsize=0,
                    markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=3)
    _ylabels(ax, cells)
    ax.set_xlabel("95 % Wald coverage", color=INK2, fontsize=9)
    ax.set_title("Coverage", color=INK, fontsize=11, pad=8)

    for ax in axes:
        ax.invert_yaxis()

    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=7,
                          markeredgecolor=SURFACE, markeredgewidth=1.1, label=l)
               for l, c in (("before the fix", BEFORE), ("after the fix", AFTER))]
    fig.legend(handles=handles, frameon=False, fontsize=9, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.05, 1, 0.84])
    _titles(fig, f"Targeted-update fix: what it repaired and what it cost  ({n_label})",
            "cause 1, 500 replicates per cell, identical seeds. Bars are 95 % Monte Carlo "
            "intervals. Hollow marks = mean after trimming 1 % of replicates; where a hollow "
            "and filled mark separate, the mean is being driven by a few divergent fits.")
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_fix_tails(before: Path | str, after: Path | str, out: Path,
                   cells: Optional[Sequence[str]] = None, event: int = 1,
                   config: str = "threshold") -> Path:
    """The per-replicate estimates themselves -- where the cost actually lives.

    The summary statistics hide this: in C5 the bulk of the post-fix distribution
    is centred on the truth, and two replicates out of 500 sit far outside it.
    A mean and an RMSE cannot show both facts at once; the raw points can.
    """
    before, after = Path(before), Path(after)
    if cells is None:
        cells = sorted(p.name for p in after.iterdir()
                       if p.is_dir() and not p.name.startswith("_"))
    p = get_config(config)
    taus = target_times_for(p)
    tau = max(taus)
    tr = (closed_form(p, taus, n_mc=4_000_000).drop_duplicates(["event", "time"])
          .set_index(["event", "time"])["rd"])
    truth = float(tr.loc[(event, float(tau))])

    order = sorted({c.split("_")[0] for c in cells}, key=lambda c: int(c[1:]))
    fig, ax = plt.subplots(figsize=(9.6, 4.4), facecolor=SURFACE)
    _style(ax)
    ax.grid(True, axis="x", color=GRID, lw=0.8, alpha=0.9)
    ax.axvline(truth, color=INK, lw=1.2, zorder=1)

    # Two post-fix replicates reach -6.25. Letting them set the axis would
    # compress every other point into a few pixels and hide the thing the figure
    # exists to show, so the axis is scaled to the bulk and the off-scale points
    # are drawn on the margin with their count -- visible, not silently dropped.
    series = {}
    for label, d, colour, sgn in (("before", before, BEFORE, -1),
                                  ("after", after, AFTER, +1)):
        raw = collect(d, cells=cells)
        raw = raw[(raw.estimator == "tmle") & (raw.estimand == "rd")
                  & (raw.event == event) & np.isclose(raw.time, tau)]
        series[label] = (raw, colour, sgn)

    allv = np.concatenate([r[r.est.notna()]["est"].to_numpy()
                           for r, _, _ in series.values()])
    lo, hi = np.percentile(allv, [0.2, 99.8])
    pad = 0.12 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    rng = np.random.default_rng(7)
    for label, (raw, colour, sgn) in series.items():
        for cell in order:
            e = raw[raw.cell.str.startswith(cell + "_")]["est"].dropna().to_numpy()
            if not len(e):
                continue
            yc = order.index(cell) + sgn * 0.19
            inside = (e >= lo) & (e <= hi)
            y = yc + rng.normal(0, 0.045, int(inside.sum()))
            ax.plot(e[inside], y, "o", ms=2.4, color=colour, alpha=0.35, mew=0,
                    zorder=2, label=label if cell == order[0] else None)
            for side, sel in ((lo, e < lo), (hi, e > hi)):
                k = int(sel.sum())
                if not k:
                    continue
                ax.plot([side], [yc], "<" if side == lo else ">", ms=8,
                        color=colour, mew=0, zorder=5, clip_on=False)
                ax.annotate(f"{k} off scale (min {e[sel].min():.2f})"
                            if side == lo else f"{k} off scale (max {e[sel].max():.2f})",
                            (side, yc), textcoords="offset points",
                            xytext=(12 if side == lo else -12, 0),
                            ha="left" if side == lo else "right", va="center",
                            color=colour, fontsize=8.5, fontweight="bold")
    ax.set_xlim(lo, hi)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{c}  {CELL_SPEC[c]}" for c in order], fontsize=9)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel(f"estimated risk difference at tau = {tau:.2f} "
                  f"(vertical line = truth)", color=INK2, fontsize=9)
    lg = ax.legend(frameon=False, fontsize=9, loc="lower left", markerscale=3)
    for h in lg.legend_handles:
        h.set_alpha(0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.82])
    _titles(fig, "Every replicate, not just the summary",
            "One point per replicate, 500 per cell, jittered vertically. The post-fix bulk "
            "sits on the truth in C5 where the pre-fix bulk did not — and a handful of "
            "post-fix replicates diverge far enough to move the mean on their own.")
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="sim.plots_fix", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", default="results/study_a")
    ap.add_argument("--after", default="results/study_a_postfix")
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--out-dir", default="results/figures")
    ap.add_argument("--event", type=int, default=1)
    a = ap.parse_args(argv)

    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    d = fix_frame(a.before, a.after, cells=a.cells, event=a.event)
    d.to_csv(out_dir.parent / "tables" / "fix_summary.csv", index=False)
    f1 = plot_fix(d, out_dir / "fix_comparison.png")
    f2 = plot_fix_tails(a.before, a.after, out_dir / "fix_replicates.png",
                        cells=a.cells, event=a.event)
    print(d.round(4).to_string(index=False))
    print(f"\nwrote {f1}\n      {f2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
