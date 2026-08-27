"""Figures for Study B.

**Coverage is never drawn alone.** That is the discipline borrowed from Fan et
al. (2024), whose debiased-LASSO holds coverage near nominal under heavy-tailed
errors only by inflating mean interval length from 0.483 to 1.418 -- a fact
coverage alone would have hidden. So every dose-response figure here is three
stacked panels sharing an x-axis:

    coverage    with a Monte Carlo band and the nominal line
    SE / SD     with a line at 1, which separates the two ways of reaching 95 %
    width       so a procedure that "fixes" coverage by widening is visible

The SE/SD panel is the diagnostic one and is why the three-panel form is worth
the vertical space. Coverage at 0.93 says something is wrong; SE/SD at 0.90 with
a standardised bias near zero says *what* is wrong, and points at the variance
estimator rather than the estimator.

Stress level runs along x in every figure, because the question is never "is this
cell broken" but "how far can it be pushed before it breaks" -- and a breakpoint
is a feature of a curve, not of a point.
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

__all__ = ["plot_axis", "plot_min_nuisance", "plot_procedures", "make_all_b"]

#: Sample size is the series dimension in the dose-response figures, so the ramp
#: is ordinal: darker = more data. Slots from the validated categorical set.
N_COLOUR = {250: "#8fb8e8", 500: "#2a78d6", 1000: "#1b5e9c", 2000: "#0d3a63"}
N_MARKER = {250: "o", 500: "s", 1000: "^", 2000: "D"}

#: Procedure colours, for the bootstrap comparison figure. Every procedure that
#: can appear needs its own entry: the three filtered percentile variants used to
#: fall through to `MUTED` together, which drew them in one grey and made three
#: legend entries indistinguishable.
PROC_COLOUR = {
    "wald": "#eda100", "logwald": "#e0b955", "logitwald": "#c98a00",
    "atanhwald": "#a87400",
    # Hue carries the construction, lightness the filter: the comparison the
    # procedures figure exists to make is construction against construction, and
    # the filter is the nuisance dimension.
    "pct_all": "#eb6834", "basic_all": "#1baf7a", "bca_all": "#e87ba4",
    "pct_convfilter": "#8a8985", "basic_convfilter": "#7fd3b4",
    "bca_convfilter": "#f2b6cc",
    "pct_dropmode1": "#6f5bb0", "basic_dropmode1": "#0e7a55",
    "bca_dropmode1": "#b8557c",
    "pct_strict": "#3d3c3a", "basic_strict": "#0a5c40",
    "bca_strict": "#8c3f5e",
}

AXIS_TITLE = {
    "overlap": ("Treatment positivity", "gamma scaled; P(min(e, 1-e) < 0.05) rises 0.00 -> 0.33"),
    "rare": ("Rare events", "cause-1 incidence at the last tau falls 0.29 -> 0.02; cause 2 stays common"),
    "censoring": ("Censoring positivity", "dependence rises at a fixed 30 % censored fraction"),
    "min_nuisance": ("Truncation", "min_nuisance traded against bias at the stressed levels"),
}

#: Levels in the order they were calibrated, not alphabetically -- a
#: dose-response axis read out of order is not a dose-response.
LEVEL_ORDER = ["base", "OV1", "OV2", "OV3", "OV4", "RA1", "RA2", "RA3",
               "CN0", "CN1", "CN2", "CN3", "CN4", "NULL"]


def _order_levels(levels: Sequence[str]) -> list:
    known = [x for x in LEVEL_ORDER if x in set(levels)]
    rest = sorted(set(levels) - set(known))
    return known + rest


def _one_cell_per_point(d: pd.DataFrame, what: str) -> pd.DataFrame:
    """Keep exactly one cell per plotted x-position.

    A bootstrap cell emits its own `wald` rows -- on purpose, so that
    Wald-vs-bootstrap is a paired contrast within one cell -- and those rows
    carry the same axis, level, n, arm and procedure as the Wald-only cell of
    the same condition. Left alone, a dose-response panel then draws two points
    at that level from different replicate counts (150 against 1000) and joins
    the line through whichever pandas happened to order first.

    The dose-response figures are about the Wald-only ladder, so the bootstrap
    cells are dropped here and appear only in `plot_procedures`, where the cell
    is the facet and no collision is possible.
    """
    if "n_bootstrap" in d:
        d = d[d["n_bootstrap"] == 0]
    key = [c for c in ("level", "arm", "n", "time", "min_nuisance") if c in d]
    dup = d.duplicated(subset=key, keep=False)
    if dup.any():  # pragma: no cover - guard, not expected to fire
        import warnings
        warnings.warn(
            f"{what}: {sorted(d.loc[dup, 'cell'].unique())} occupy the same "
            f"plot position; keeping the cell with most replicates",
            RuntimeWarning)
        d = (d.sort_values("reps", ascending=False)
              .drop_duplicates(subset=key, keep="first"))
    return d


def _spanning_series(d: pd.DataFrame, x: str = "level") -> tuple:
    """Keep only sample sizes that actually trace a curve along the axis.

    n = 1000 and n = 2000 were run for the base condition only -- the stress
    ladders stop at n = 500, because second-stage cost scales as n^2.05 and
    stressed cells run 6-11x base. Plotted, they appear as a single marker
    hovering over the leftmost level, which reads as a point on a curve that is
    not there.

    They are not discarded: they carry the base condition's n-trend and remain in
    `study_b_performance.csv` and in the breakpoint table. They are simply not
    drawn on an axis they do not span. The rule is "two or more levels", not a
    hardcoded list, so it stays correct if the design changes.
    """
    spans = d.groupby("n")[x].nunique()
    keep = sorted(spans[spans >= 2].index)
    dropped = sorted(spans[spans < 2].index)
    return d[d["n"].isin(keep)], dropped


#: Which bootstrap variant the dose-response figures mark.
#:
#: `pct_all` is the percentile interval over *every* draw. That is what PyTMLE
#: now produces: the per-target `Converged` filter at `bootstrap.py:87` has been
#: removed, and the study measured that line as costing up to 0.188 coverage at
#: OV3 -- it kept only draws that had solved the score equation, which is
#: selection on the outcome, so what survived was both smaller and systematically
#: narrower.
#:
#: `pct_convfilter` (the filtered variant) is retained in the tables as the
#: *historical* behaviour, and the paired `pct_all - pct_convfilter` contrast in
#: `study_b_attribution.csv` is what quantifies the fix. Both come from the same
#: resamples, so nothing had to be re-run to change which one is drawn.
BOOT_PROC = "pct_all"
BOOT_COLOUR = PROC_COLOUR[BOOT_PROC]


def _boot_markers(ax, boot: pd.DataFrame, xs: Sequence[str], tt: float,
                  value: str) -> bool:
    """Mark the bootstrap result wherever one exists, beside its paired Wald.

    A bootstrap cell runs 150 replicates against the Wald ladder's 1000, so a
    star read against the *curve* would confound the procedure with Monte Carlo
    noise. Each bootstrap cell emits its own `wald` rows from the same replicates
    and the same fits, though, so the honest comparison is within the cell.

    Both are therefore drawn -- a hollow circle for that cell's Wald, a star for
    its bootstrap -- joined by a thin connector. The connector is the finding:
    its length and direction are the paired procedure difference, and the offset
    of the pair from the curve is the replicate-count noise, kept visually
    separate.
    """
    xi = {lv: i for i, lv in enumerate(xs)}
    drawn = False
    for lv, g in boot[boot["time"] == tt].groupby("level"):
        if lv not in xi:
            continue
        w = g[g["procedure"] == "wald"]
        b = g[g["procedure"] == BOOT_PROC]
        if b.empty or value not in b:
            continue
        x = xi[lv]
        yb = float(b.iloc[0][value])
        if not np.isfinite(yb):
            continue
        if not w.empty and np.isfinite(float(w.iloc[0][value])):
            yw = float(w.iloc[0][value])
            ax.plot([x, x], [yw, yb], color=INK2, linewidth=0.9, alpha=0.55,
                    zorder=4, solid_capstyle="butt")
            ax.plot([x], [yw], marker="o", markersize=5.5, markerfacecolor="none",
                    markeredgecolor=INK2, markeredgewidth=1.1, linestyle="none",
                    zorder=5, label="_paired Wald (same reps)")
        ax.plot([x], [yb], marker="*", markersize=11, color=BOOT_COLOUR,
                markeredgecolor=SURFACE, markeredgewidth=0.6, linestyle="none",
                zorder=6, label="_bootstrap")
        drawn = True
    return drawn


def _panel(ax, sub: pd.DataFrame, xs: Sequence[str], value: str,
           err: Optional[str] = None) -> None:
    """One metric against stress level, one line per sample size."""
    xi = {lv: i for i, lv in enumerate(xs)}
    for n, g in sorted(sub.groupby("n")):
        g = g.copy()
        g["_x"] = g["level"].map(xi)
        g = g.dropna(subset=["_x", value]).sort_values("_x")
        if g.empty:
            continue
        c = N_COLOUR.get(int(n), MUTED)
        ax.plot(g["_x"], g[value], color=c, marker=N_MARKER.get(int(n), "o"),
                markersize=4.5, linewidth=1.6, label=f"n = {int(n)}", zorder=3)
        if err and err in g:
            lo = g[value] - 1.96 * g[err]
            hi = g[value] + 1.96 * g[err]
            if value == "coverage":
                # a coverage band is a band on a proportion; letting it run past
                # [0, 1] floods the panel at small replicate counts and implies
                # values that cannot occur
                lo, hi = lo.clip(0.0, 1.0), hi.clip(0.0, 1.0)
            ax.fill_between(g["_x"], lo, hi, color=c, alpha=0.16,
                            linewidth=0, zorder=2)
    _style(ax)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, fontsize=8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.9)


def plot_axis(perf: pd.DataFrame, axis: str, out: Path | str,
              procedure: str = "wald", typ: str = "rd", event: int = 1,
              arm: str = "correct") -> Optional[Path]:
    """Three-panel dose-response for one stress axis, faceted by tau."""
    sel = (perf["axis"].isin([axis, "base"]) & (perf["type"] == typ)
           & (perf["event"] == event) & (perf["arm"] == arm))
    d = perf[sel & (perf["procedure"] == procedure)].copy()
    # Built from `perf` rather than from `d`: `d` is already restricted to one
    # procedure, so the bootstrap rows would never survive it. Only PyTMLE's
    # default B = 100 is marked, since that is what users actually get.
    boot = perf[sel & (perf["n_bootstrap"] == 100)
                & perf["procedure"].isin([BOOT_PROC, "wald"])].copy()
    d = _one_cell_per_point(d, f"plot_axis[{axis}, {arm}]")
    d, dropped = _spanning_series(d)
    if d.empty:
        return None
    taus = sorted(d["time"].unique())
    xs = _order_levels(d["level"].unique())
    fig, axes = plt.subplots(3, len(taus), figsize=(3.5 * len(taus), 7.4),
                             sharex=True, facecolor=SURFACE, squeeze=False)

    for j, tt in enumerate(taus):
        sub = d[d["time"] == tt]
        # coverage, with the nominal line and a shaded tolerance band
        ax = axes[0][j]
        ax.axhspan(0.93, 0.97, color=GRID, alpha=0.55, linewidth=0, zorder=1)
        ax.axhline(0.95, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
        _panel(ax, sub, xs, "coverage", "coverage_mc_se")
        has_boot = _boot_markers(ax, boot, xs, tt, "coverage")
        lo_y = min(0.5, float(sub["coverage"].min()) - 0.05)
        if has_boot and len(boot):
            bl = boot.loc[boot["time"] == tt, "coverage"].min()
            if np.isfinite(bl):
                lo_y = min(lo_y, float(bl) - 0.05)
        ax.set_ylim(lo_y, 1.005)
        ax.set_title(f"tau = {tt:.2f}", fontsize=9, color=INK, pad=6)
        if j == 0:
            ax.set_ylabel("coverage", fontsize=9, color=INK2)

        # SE / SD -- the panel that says *why*
        ax = axes[1][j]
        ax.axhline(1.0, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
        _panel(ax, sub, xs, "se_ratio")
        _boot_markers(ax, boot, xs, tt, "se_ratio")
        if j == 0:
            ax.set_ylabel("mean SE / empirical SD", fontsize=9, color=INK2)

        # width, so coverage bought by widening is visible
        ax = axes[2][j]
        _panel(ax, sub, xs, "mean_width")
        _boot_markers(ax, boot, xs, tt, "mean_width")
        if j == 0:
            ax.set_ylabel("mean interval width", fontsize=9, color=INK2)
        ax.set_xlabel("stress level", fontsize=9, color=INK2)

    fig.tight_layout(rect=(0, 0.05, 1, 0.90))
    handles, labels = axes[0][0].get_legend_handles_labels()
    if len(boot):
        from matplotlib.lines import Line2D
        handles = list(handles) + [
            Line2D([], [], marker="o", markersize=5.5, markerfacecolor="none",
                   markeredgecolor=INK2, markeredgewidth=1.1, linestyle="none"),
            Line2D([], [], marker="*", markersize=11, color=BOOT_COLOUR,
                   markeredgecolor=SURFACE, markeredgewidth=0.6, linestyle="none"),
        ]
        labels = list(labels) + ["Wald, same reps as \u2605",
                                 "bootstrap (percentile, all draws, B = 100)"]
    if handles:
        # figure-level and below the axes: the interesting curves fall towards
        # the bottom-right of the coverage panel, which is where a legend goes
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.005))
    title, subtitle = AXIS_TITLE.get(axis, (axis, ""))
    note = (f" n = {', '.join(str(x) for x in dropped)}: base condition only, "
            f"not drawn." if dropped else "")
    _titles(fig, title, "")
    # _titles(fig, f"{title} — {procedure}, cause {event} {typ.upper()}",
    #         f"{subtitle}. Shaded band: +-1.96 Monte Carlo SE. "
    #         f"Nuisances: {arm}.{note}")
    out = Path(out)
    fig.savefig(out, dpi=680, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_min_nuisance(perf: pd.DataFrame, out: Path | str, typ: str = "rd",
                      event: int = 1, tau_index: int = -1) -> Optional[Path]:
    """Coverage against truncation, per stressed condition.

    The bias/variance trade the `concrete` paper names: "enforcing a lower bound
    decreases estimator variance at the cost of introducing bias but improving
    stability". Drawn as coverage and standardised bias together, because the
    trade is only visible if both ends of it are on the page.
    """
    # the sweep only ran on the estimated-nuisance arm; without this the oracle
    # cells of the same conditions land on the same x-positions
    d = perf[(perf["type"] == typ) & (perf["event"] == event)
             & (perf["procedure"] == "wald")
             & (perf["arm"] == "correct")].copy()
    d = _one_cell_per_point(d, "plot_min_nuisance")
    # the sweep ran at n = 250 only, so n = 500 is a lone point at the default
    d, dropped_n = _spanning_series(d, x="min_nuisance")
    if d.empty or d["min_nuisance"].nunique() < 2:
        return None
    taus = sorted(d["time"].unique())
    tt = taus[tau_index]
    d = d[d["time"] == tt]
    # a swept cell shares its parent's dataset, so group by the parent condition
    d["cond"] = d["level"].str.split("@").str[0]
    conds = _order_levels(d["cond"].unique())
    conds = [c for c in conds if d[d["cond"] == c]["min_nuisance"].nunique() > 1]
    if not conds:
        return None

    fig, axes = plt.subplots(2, len(conds), figsize=(3.3 * len(conds), 5.6),
                             sharex=True, facecolor=SURFACE, squeeze=False)
    for j, cond in enumerate(conds):
        sub = d[d["cond"] == cond]
        for row, (val, lab) in enumerate((("coverage", "coverage"),
                                          ("std_bias", "bias / SD"))):
            ax = axes[row][j]
            if val == "coverage":
                ax.axhline(0.95, color=INK2, linewidth=1.0,
                           linestyle=(0, (4, 3)), zorder=2)
            else:
                ax.axhline(0.0, color=INK2, linewidth=1.0,
                           linestyle=(0, (4, 3)), zorder=2)
            for n, g in sorted(sub.groupby("n")):
                g = g.sort_values("min_nuisance")
                ax.plot(g["min_nuisance"], g[val], color=N_COLOUR.get(int(n), MUTED),
                        marker=N_MARKER.get(int(n), "o"), markersize=4.5,
                        linewidth=1.6, label=f"n = {int(n)}", zorder=3)
            _style(ax)
            ax.set_xscale("log")
            ax.set_xticks(sorted(sub["min_nuisance"].unique()))
            ax.get_xaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            # log scales otherwise print unlabelled minor ticks between the four
            # swept values, which read as data points that do not exist
            ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.9)
            if row == 0:
                ax.set_title(cond, fontsize=9, color=INK, pad=6)
            if j == 0:
                ax.set_ylabel(lab, fontsize=9, color=INK2)
            if row == 1:
                ax.set_xlabel("min_nuisance", fontsize=9, color=INK2)

    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.005))
    note = (f" Swept at n = {', '.join(str(x) for x in sorted(d['n'].unique()))} "
            f"only." if dropped_n else "")
    # _titles(fig, f"Truncation trade-off — cause {event} {typ.upper()}, tau = {tt:.2f}",
    #         "More truncation buys stability and costs bias. Both panels are "
    #         "needed: coverage alone cannot distinguish the two ends of the "
    #         "trade." + note)
    out = Path(out)
    fig.savefig(out, dpi=680, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_procedures(perf: pd.DataFrame, out: Path | str, typ: str = "rd",
                    event: int = 1) -> Optional[Path]:
    """Every interval construction at the bootstrap cells, coverage vs width.

    Coverage on x and width on y, so the reader can see directly that two
    procedures reaching the same coverage are not equivalent -- the one lower on
    the page reached it with a shorter interval. The nominal line is vertical.
    """
    d = perf[(perf["n_bootstrap"] > 0) & (perf["type"] == typ)
             & (perf["event"] == event) & perf["coverage"].notna()].copy()
    if d.empty:
        return None
    cells = sorted(d["cell"].unique())
    taus = sorted(d["time"].unique())
    fig, axes = plt.subplots(len(taus), len(cells),
                             figsize=(3.1 * len(cells), 2.9 * len(taus)),
                             facecolor=SURFACE, squeeze=False, sharex=True)
    for i, tt in enumerate(taus):
        for j, cell in enumerate(cells):
            ax = axes[i][j]
            sub = d[(d["cell"] == cell) & (d["time"] == tt)]
            ax.axvline(0.95, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)),
                       zorder=2)
            for _, r in sub.iterrows():
                proc = str(r["procedure"])
                ax.scatter(r["coverage"], r["mean_width"], s=42, zorder=3,
                           color=PROC_COLOUR.get(proc, MUTED),
                           edgecolor=SURFACE, linewidth=0.8, label=proc)
            _style(ax)
            ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.9)
            if i == 0:
                ax.set_title(cell, fontsize=8.5, color=INK, pad=6)
            if j == 0:
                ax.set_ylabel(f"width  (tau = {tt:.2f})", fontsize=8.5, color=INK2)
            if i == len(taus) - 1:
                ax.set_xlabel("coverage", fontsize=9, color=INK2)

    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    seen: Dict[str, object] = {}
    for ax in fig.axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    if seen:
        fig.legend(list(seen.values()), list(seen), loc="lower center",
                   ncol=min(len(seen), 5), frameon=False, fontsize=8.5,
                   bbox_to_anchor=(0.5, 0.005))
    # _titles(fig, f"Interval constructions — cause {event} {typ.upper()}",
    #         "Same replicates, same resamples: differences isolate the "
    #         "construction. Lower is a shorter interval at the same coverage.")
    out = Path(out)
    fig.savefig(out, dpi=680, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def make_all_b(perf: pd.DataFrame, out_dir: Path | str) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made: Dict[str, Path] = {}
    for axis in ("overlap", "rare", "censoring"):
        for arm in ("correct", "oracle"):
            p = plot_axis(perf, axis, out_dir / f"study_b_{axis}_{arm}.png", arm=arm)
            if p:
                made[f"{axis}_{arm}"] = p
    p = plot_min_nuisance(perf, out_dir / "study_b_min_nuisance.png")
    if p:
        made["min_nuisance"] = p
    p = plot_procedures(perf, out_dir / "study_b_procedures.png")
    if p:
        made["procedures"] = p
    return made


def main(argv=None) -> int:
    import argparse

    from .study_b_report import collect_b, performance_b

    ap = argparse.ArgumentParser(prog="sim.plots_study_b", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="results/study_b")
    ap.add_argument("--fig-dir", default=None)
    a = ap.parse_args(argv)
    perf_csv = Path(a.output_dir) / "study_b_performance.csv"
    perf = (pd.read_csv(perf_csv) if perf_csv.exists()
            else performance_b(collect_b(a.output_dir)))
    made = make_all_b(perf, a.fig_dir or Path(a.output_dir) / "figures")
    for k, v in made.items():
        print(f"  {k:22s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
