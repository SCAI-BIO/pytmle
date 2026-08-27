"""Study B reporting: coverage, width, and where the two disagree.

Coverage is never reported alone. Fan et al. (2024) make the point this study
borrows: their debiased-LASSO keeps coverage near nominal under heavy-tailed
errors *only* by inflating mean interval length from 0.483 to 1.418, and coverage
alone would have hidden that. So every coverage figure here sits beside the mean
interval width and, more diagnostically, beside the ratio of the interval's
implied standard error to the estimator's actual empirical SD.

That ratio is what separates the two ways a procedure can reach 95 %:

    ratio ~ 1   the interval is the right width and covers for the right reason
    ratio > 1   it covers by being too wide -- honest coverage, wasted precision
    ratio < 1   it under-covers because the SE understates the true spread

For `wald` the implied SE is the stored EIC standard error. For the bootstrap
procedures there is no SE, so the like-for-like quantity is `width / (2 * 1.96)`,
which puts every procedure on one scale.

Non-coverage is split into left- and right-tail misses, because the split
identifies the cause: a symmetric shortfall points to variance underestimation,
an asymmetric one to skewness -- which percentile and BCa intervals can correct
and a symmetric Wald interval cannot.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .dgp import get_config
from .truth import closed_form

__all__ = ["collect_b", "performance_b", "conditions_b", "breakpoints_b",
           "type_i_error_b", "attribution_b", "failure_panel", "build"]

Z = 1.959963984540054

#: Order the procedures so the analytic intervals read first, then the
#: convergence-filtered variants (PyTMLE's behaviour before the filter was
#: removed), then the unfiltered default and the remaining diagnostics.
PROC_ORDER = ["wald", "logwald", "logitwald", "atanhwald",
              "pct_all", "basic_all", "bca_all",
              "pct_convfilter", "basic_convfilter", "bca_convfilter",
              "pct_dropmode1", "basic_dropmode1", "bca_dropmode1",
              "pct_strict", "basic_strict", "bca_strict"]

#: Procedure labels renamed after the shards were written. See `collect_b`.
#:
#: `basic` and `bca` used to be emitted **only** under the convergence filter,
#: so a stored `basic` row is a `basic_convfilter` row. Mapping them under their
#: true names keeps the old shards readable and stops them being mistaken for
#: the unfiltered constructions, which did not exist until the filter was
#: removed and every construction was emitted under every filter.
_LEGACY_PROCEDURES = {"pct_shipped": "pct_convfilter",
                      "basic": "basic_convfilter",
                      "bca": "bca_convfilter"}

#: Procedures built from the stored EIC standard error rather than from draws.
#: For these the implied SE *is* the stored SE; for the bootstrap procedures the
#: like-for-like quantity is width / (2 * z).
_SE_PROCS = ("wald", "logwald", "logitwald", "atanhwald")

#: Per-replicate condition descriptors. Averaged per cell and reported beside
#: every coverage figure, because a coverage number is uninterpretable without
#: knowing what stress the replicates actually experienced.
_DESCRIPTORS = ["ps_min", "ps_p01", "frac_ps_below_05", "piG_p01",
                "frac_weights_truncated", "frac_subjects_truncated",
                "censored_frac", "n_events_1", "n_events_2", "n_times",
                "ps_mae", "cumhaz_mae", "n_failed_fits",
                "tmle_steps", "tmle_converged", "seconds"]


def _derive_basic(out: pd.DataFrame) -> pd.DataFrame:
    """Add `basic_*` rows wherever only `pct_*` was stored.

    The reverse-percentile interval is a deterministic reflection of the
    percentile interval about the point estimate:

        basic = (2 * est - pct_hi,  2 * est - pct_lo)

    so it needs no resamples -- only the stored percentile bounds and the point
    estimate. Verified against shards that carry both: exact to 0.0 over 3391
    rows.

    This matters because the two are **identical in width** and differ only in
    location, which makes the pair a clean read on whether the bootstrap
    distribution is offset from the point estimate. It also means the earlier
    limitation -- `basic` and `bca` having been emitted under the convergence
    filter alone, so their showing measured the filter -- is now lifted for
    `basic` at no cost. `bca` is not derivable this way: its bias-correction
    needs the draw distribution and its acceleration the influence curve, and
    neither was stored.

    Rows already present are never overwritten, so a run that emitted
    `basic_*` natively passes through unchanged.
    """
    need = {"procedure", "est", "ci_lo", "ci_hi"}
    if not need.issubset(out.columns):
        return out
    have = set(out["procedure"].dropna().unique())
    made = []
    for proc in sorted(p for p in have if isinstance(p, str) and p.startswith("pct_")):
        target = "basic_" + proc[len("pct_"):]
        if target in have:
            continue
        src = out[out["procedure"] == proc].copy()
        if src.empty:
            continue
        lo, hi = src["ci_lo"].to_numpy(), src["ci_hi"].to_numpy()
        est = src["est"].to_numpy()
        src["ci_lo"], src["ci_hi"] = 2 * est - hi, 2 * est - lo
        src["procedure"] = target
        made.append(src)
    return pd.concat([out, *made], ignore_index=True) if made else out


def collect_b(output_dir: Path | str, cells: Optional[Sequence[str]] = None) -> pd.DataFrame:
    output_dir = Path(output_dir)
    frames = []
    for meta_path in sorted(output_dir.glob("*/meta.json")):
        name = meta_path.parent.name
        if cells and name not in cells:
            continue
        shards = sorted(meta_path.parent.glob("shard_*.parquet"))
        if not shards:
            continue
        df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
        meta = json.loads(meta_path.read_text())
        df["n_bootstrap"] = meta.get("n_bootstrap", 0)
        df["target_times"] = [tuple(meta["target_times"])] * len(df)
        # Carried so the report can group by hypothesis and, crucially, so each
        # cell's truth is computed from *its own* DGP: the rare-event and null
        # axes move the estimand, and scoring them against the base truth would
        # report the design as bias.
        df["axis"] = meta.get("axis", "base")
        df["level"] = meta.get("level", "base")
        df["dgp_config"] = meta.get("config", "base")
        df["dgp_override"] = [_override_key(meta.get("params_override", {}))] * len(df)
        df["min_nuisance"] = meta.get("min_nuisance", np.nan)
        df["max_updates"] = meta.get("max_updates", -1)
        df["seed_key"] = meta.get("seed_key", name)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no completed cells under {output_dir}")
    out = pd.concat(frames, ignore_index=True)

    # Shards written before the rename carry the old label. It was called
    # `pct_shipped` because the per-target `Converged` filter was then what
    # PyTMLE shipped; that filter has since been removed from `bootstrap.py`, so
    # the name now points at the opposite of what it means. Mapped here rather
    # than re-run: the resamples are unchanged, only their name is.
    if "procedure" in out:
        out["procedure"] = out["procedure"].replace(_LEGACY_PROCEDURES)
    out = _derive_basic(out)

    # A replicate must appear in exactly one shard. Duplicates would inflate every
    # count silently, and the way they arise is not hypothetical: shard `i` holds
    # replicates [i*chunk, (i+1)*chunk), so re-chunking a cell that already has
    # shards re-indexes them and recomputes replicates it already holds. The
    # chunk is pinned in meta.json to prevent that; this catches anything that
    # slips past, including two runs writing to one directory.
    key = ["cell", "rep", "procedure", "type", "event", "time", "group"]
    if all(k in out for k in key):
        dup = out.duplicated(subset=key, keep=False)
        if dup.any():
            cells = sorted(out.loc[dup, "cell"].unique())
            warnings.warn(
                f"{int(dup.sum())} duplicated rows across cells {cells}; "
                f"keeping the first of each. Check that no cell was re-chunked "
                f"or written by two concurrent runs.", RuntimeWarning)
            out = out[~out.duplicated(subset=key, keep="first")]
    return out


def _override_key(ov: Dict) -> str:
    """A hashable, stable identity for a params_override, for grouping/caching."""
    return json.dumps(ov or {}, sort_keys=True)


def _truth_map(config: str, override_key: str, taus: Sequence[float],
               n_mc: int = 4_000_000) -> Dict:
    """Ground truth for one cell's DGP.

    Keyed on the override as well as the config: `gamma` and the censoring
    parameters leave the estimand alone, but `alpha` (rarity) and `theta` (the
    null condition) do not, so a single global truth would score the rare-event
    cells against a target ~8x too large.
    """
    p = get_config(config)
    ov = json.loads(override_key)
    if ov:
        p = p.with_(**{k: (np.asarray(v, dtype=float) if isinstance(v, list) else v)
                       for k, v in ov.items()})
    tr = closed_form(p, list(taus), n_mc=n_mc)
    out: Dict = {}
    for _, r in tr.iterrows():
        out[("risk", int(r["event"]), float(r["time"]), int(r["arm"]))] = float(r["risk"])
    for _, r in tr.drop_duplicates(["event", "time"]).iterrows():
        out[("rd", int(r["event"]), float(r["time"]), -1)] = float(r["rd"])
        out[("rr", int(r["event"]), float(r["time"]), -1)] = float(r["rr"])
    return out


def performance_b(d: pd.DataFrame, config: str = "base",
                  n_mc: int = 4_000_000) -> pd.DataFrame:
    """One row per (cell, procedure, estimand, event, tau, group)."""
    ok_rows = d[d["procedure"].notna()].copy()
    # Replicates that produced no estimate at all: counted per cell so a coverage
    # figure is never read without knowing how many replicates it excludes
    # (FINDINGS 4 -- a guard that silently drops failures is worse than none).
    failed = (d[d["procedure"].isna()].groupby("cell")["rep"].nunique()
              if (d["procedure"].isna()).any() else pd.Series(dtype=int))
    attempted = d.groupby("cell")["rep"].nunique()

    d = ok_rows
    rows: List[Dict] = []
    _cache: Dict = {}
    group_keys = ["cell", "axis", "level", "arm", "n", "n_bootstrap",
                  "min_nuisance", "max_updates", "dgp_config", "dgp_override",
                  "procedure", "type", "event", "time", "group"]
    for taus, block in d.groupby("target_times", sort=False):
        for keys, g in block.groupby(group_keys, dropna=False):
            (cell, axis, level, arm, n, nb, mn, mu, dcfg, dov,
             proc, typ, ev, tt, grp) = keys
            ck = (dcfg, dov, taus)
            if ck not in _cache:
                _cache[ck] = _truth_map(dcfg, dov, list(taus), n_mc=n_mc)
            truth = _cache[ck].get((typ, int(ev), float(tt), int(grp)))
            if truth is None:
                continue
            est = g["est"].to_numpy(dtype=float)
            lo = g["ci_lo"].to_numpy(dtype=float)
            hi = g["ci_hi"].to_numpy(dtype=float)
            ok = np.isfinite(est)
            has_ci = np.isfinite(lo) & np.isfinite(hi)
            m = int(ok.sum())
            if m == 0:
                continue
            err = est[ok] - truth
            emp_sd = float(np.std(est[ok], ddof=1)) if m > 1 else np.nan
            width = hi[has_ci] - lo[has_ci]
            # Wald stores its SE; the bootstrap procedures do not, so the
            # like-for-like quantity is the SE the interval implies.
            if proc in _SE_PROCS and "se" in g and g["se"].notna().any():
                implied = float(np.nanmean(g["se"].to_numpy(dtype=float)))
            else:
                implied = float(np.mean(width) / (2 * Z)) if len(width) else np.nan
            cov = float(np.mean((lo[has_ci] <= truth) & (hi[has_ci] >= truth))) \
                if has_ci.any() else np.nan
            n_ci = int(has_ci.sum())
            bias = float(err.mean())
            rows.append({
                "cell": cell, "axis": axis, "level": level,
                "arm": arm, "n": int(n), "n_bootstrap": int(nb),
                "min_nuisance": mn, "max_updates": mu,
                "procedure": proc, "type": typ, "event": int(ev),
                "time": float(tt), "group": int(grp),
                "reps": m, "reps_with_ci": n_ci,
                "missing_ci": int(ok.sum() - n_ci),
                "reps_attempted": int(attempted.get(cell, m)),
                "reps_failed": int(failed.get(cell, 0)),
                "completion": float(m / attempted.get(cell, m)) if attempted.get(cell, m) else np.nan,
                "truth": truth,
                "bias": bias,
                "bias_mc_se": float(np.std(err, ddof=1) / np.sqrt(m)) if m > 1 else np.nan,
                "emp_sd": emp_sd,
                # Bias on the scale of the estimator's own spread. This is the
                # only fair bias scale on the rarity axis, where the truth itself
                # shrinks eightfold, and it is what separates the two failure
                # mechanisms: heavy-tailed influence curves give |std_bias| ~ 0
                # with se_ratio < 1, whereas a non-vanishing second-order
                # remainder gives a large |std_bias| with se_ratio ~ 1.
                "std_bias": bias / emp_sd if emp_sd and emp_sd > 0 else np.nan,
                "mean_se": implied,
                "se_ratio": implied / emp_sd if emp_sd and emp_sd > 0 else np.nan,
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "coverage": cov,
                "coverage_mc_se": float(np.sqrt(cov * (1 - cov) / n_ci))
                if n_ci and np.isfinite(cov) else np.nan,
                "mean_width": float(np.mean(width)) if len(width) else np.nan,
                "median_width": float(np.median(width)) if len(width) else np.nan,
                # An interval bound outside the parameter space is not merely
                # wide, it is not a statement about a risk at all. This is the
                # concrete form of the rare-event hypothesis and the motivation
                # for the transformed scales.
                #
                # Only meaningful where the support is bounded on that side: a
                # cumulative incidence lies in [0, 1], whereas a risk difference
                # is legitimately negative, so counting negative RD bounds would
                # report 1.000 everywhere and mean nothing.
                "frac_outside_support": float(np.mean(
                    (lo[has_ci] < 0.0) | (hi[has_ci] > 1.0)))
                if has_ci.any() and typ == "risk" else (
                    float(np.mean((lo[has_ci] < -1.0) | (hi[has_ci] > 1.0)))
                    if has_ci.any() and typ == "rd" else np.nan),
                # Type-I error: only meaningful where the truth is exactly zero.
                "excludes_null": float(np.mean((lo[has_ci] > 0.0) | (hi[has_ci] < 0.0)))
                if has_ci.any() and typ == "rd" else np.nan,
                # which side the interval misses on: symmetric misses indicate
                # variance underestimation, asymmetric ones skewness
                "miss_left": float(np.mean(lo[has_ci] > truth)) if has_ci.any() else np.nan,
                "miss_right": float(np.mean(hi[has_ci] < truth)) if has_ci.any() else np.nan,
                "mean_eff_b": float(g["eff_b"].mean()) if "eff_b" in g else np.nan,
                "min_eff_b": float(g["eff_b"].min()) if "eff_b" in g else np.nan,
            })
    out = pd.DataFrame(rows)
    if len(out):
        out["procedure"] = pd.Categorical(out["procedure"], PROC_ORDER, ordered=True)
        out = out.sort_values(["cell", "type", "event", "time", "procedure"])
    return out.reset_index(drop=True)


def conditions_b(d: pd.DataFrame) -> pd.DataFrame:
    """What each cell's replicates actually experienced.

    A stress axis is only an axis if it moved the quantity it is named after, and
    the place to check that is the realised data, not the design table. This is
    also where replicate loss surfaces: `n_failed_fits` and the completion rate
    must be read *before* any coverage number from the same cell.
    """
    if d.empty:
        return pd.DataFrame()
    per_rep = d.drop_duplicates(["cell", "rep"])
    cols = [c for c in _DESCRIPTORS if c in per_rep.columns]
    agg = per_rep.groupby(["cell", "axis", "level", "arm", "n", "min_nuisance",
                           "max_updates", "n_bootstrap"], dropna=False)[cols].median()
    agg = agg.rename(columns={c: f"med_{c}" for c in cols}).reset_index()
    counts = per_rep.groupby("cell").agg(
        reps_attempted=("rep", "nunique"),
        reps_failed=("procedure", lambda s: int(s.isna().sum())),
    ).reset_index()
    out = agg.merge(counts, on="cell", how="left")
    out["completion"] = 1.0 - out["reps_failed"] / out["reps_attempted"]
    return out.sort_values(["axis", "level", "n", "arm"]).reset_index(drop=True)


def breakpoints_b(perf: pd.DataFrame, threshold: float = 0.90,
                  procedure: str = "wald") -> pd.DataFrame:
    """The operating envelope: where on each axis does coverage first fail?

    Study D's headline deliverable, inherited. For each (axis, n, estimand, tau)
    it reports the first level at which coverage drops below `threshold`, i.e.
    clearly outside Monte Carlo noise at these replicate counts, together with
    the SE/SD ratio and standardised bias there so the *reason* travels with the
    breakpoint. `NaN` in `breakpoint_level` means the axis never broke, which is
    itself the result -- and the result the previous Study B got everywhere.
    """
    if perf.empty:
        return pd.DataFrame()
    # Wald-only cells define the ladder. A bootstrap cell contributes `wald`
    # rows for the same level from far fewer replicates, and including them
    # would make the breakpoint depend on which sorted first.
    p = perf[(perf["procedure"] == procedure) & perf["coverage"].notna()
             & (perf["n_bootstrap"] == 0)].copy()
    if p.empty:
        return pd.DataFrame()
    rows: List[Dict] = []
    for keys, g in p.groupby(["axis", "arm", "n", "type", "event", "time"],
                             dropna=False):
        axis, arm, n, typ, ev, tt = keys
        g = g.sort_values("level")
        bad = g[g["coverage"] < threshold]
        first = bad.iloc[0] if len(bad) else None
        rows.append({
            "axis": axis, "arm": arm, "n": int(n), "type": typ,
            "event": int(ev), "time": float(tt),
            "levels_tested": int(g["level"].nunique()),
            "min_coverage": float(g["coverage"].min()),
            "breakpoint_level": None if first is None else first["level"],
            "coverage_at_break": None if first is None else float(first["coverage"]),
            "se_ratio_at_break": None if first is None else float(first["se_ratio"]),
            "std_bias_at_break": None if first is None else float(first["std_bias"]),
        })
    return pd.DataFrame(rows).sort_values(["axis", "type", "event", "time", "n"]) \
                             .reset_index(drop=True)


def type_i_error_b(perf: pd.DataFrame) -> pd.DataFrame:
    """Rejection rate where the truth is exactly zero.

    Fan et al. report coverage separately on the true-zero coefficients; this is
    the survival analogue and the most direct reading of "is seeing believing" --
    how often does an interval exclude no-effect when there is no effect. Only
    the null cells qualify, so the table is empty until they have run.
    """
    if perf.empty or "axis" not in perf:
        return pd.DataFrame()
    p = perf[(perf["axis"] == "nulleffect") & (perf["type"] == "rd")
             & perf["excludes_null"].notna()].copy()
    if p.empty:
        return pd.DataFrame()
    p = p[["cell", "arm", "n", "procedure", "event", "time", "reps_with_ci",
           "truth", "excludes_null", "coverage", "mean_width", "se_ratio"]]
    p["type_i_mc_se"] = np.sqrt(
        p["excludes_null"] * (1 - p["excludes_null"]) / p["reps_with_ci"])
    return p.sort_values(["event", "time", "n", "procedure"]).reset_index(drop=True)


def attribution_b(perf: pd.DataFrame) -> pd.DataFrame:
    """What each bootstrap failure mode costs in coverage.

    Paired at the (cell, estimand, event, tau, group) level: the four percentile
    variants come from the same replicates and the same resamples, differing only
    in which draws are filtered out, so the contrast is the filtering and nothing
    else.
    """
    key = ["cell", "arm", "n", "n_bootstrap", "type", "event", "time", "group"]
    # bootstrap cells only: a Wald-only cell has no percentile variants, so it
    # would contribute a row of NaN for every target and bury the real ones
    perf = perf[perf["n_bootstrap"] > 0]
    if perf.empty:
        return pd.DataFrame()
    w = perf.pivot_table(index=key, columns="procedure",
                         values=["coverage", "mean_width"], observed=True)
    rows = []
    for idx, r in w.iterrows():
        def cov(p):
            try:
                return float(r[("coverage", p)])
            except Exception:
                return np.nan

        def wid(p):
            try:
                return float(r[("mean_width", p)])
            except Exception:
                return np.nan

        base = cov("pct_all")
        rows.append(dict(zip(key, idx)) | {
            "cov_all": base,
            "cov_convfilter": cov("pct_convfilter"),
            "mode2_effect": cov("pct_convfilter") - base,
            "cov_dropmode1": cov("pct_dropmode1"),
            "mode1_effect": cov("pct_dropmode1") - base,
            "cov_strict": cov("pct_strict"),
            "both_effect": cov("pct_strict") - base,
            "width_all": wid("pct_all"),
            "width_convfilter": wid("pct_convfilter"),
        })
    return pd.DataFrame(rows)


def failure_panel(d: pd.DataFrame) -> pd.DataFrame:
    """Per-cell rates of the three failure modes, plus effective B."""
    d = d[d["n_bootstrap"] > 0]
    if d.empty:
        return pd.DataFrame()
    per_rep = d.drop_duplicates(["cell", "rep"])
    rows = []
    for cell, g in per_rep.groupby("cell"):
        B = int(g["n_bootstrap"].iloc[0])
        eff = d[d["cell"] == cell].groupby(["rep", "type", "event", "time", "group"])[
            "eff_b"].min()
        rows.append({
            "cell": cell, "arm": g["arm"].iloc[0], "n": int(g["n"].iloc[0]),
            "B": B, "reps": len(g),
            "mode1_rate": float(g["mode1"].mean() / B),
            "mode3_rate": float(g["mode3"].mean() / B),
            "reps_any_mode3": float((g["mode3"] > 0).mean()),
            "median_steps": float(g["median_steps"].median()),
            "mean_eff_b": float(eff.mean()) if len(eff) else np.nan,
            "min_eff_b": float(eff.min()) if len(eff) else np.nan,
            "frac_eff_b_below_40": float((eff < 40).mean()) if len(eff) else np.nan,
            "first_error": next((e for e in g["first_error"].dropna()), None),
        })
    return pd.DataFrame(rows)


def build(output_dir: Path | str, out_dir: Optional[Path | str] = None,
          config: str = "base") -> Dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    out_dir = Path(out_dir or output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = collect_b(output_dir)
    perf = performance_b(d, config=config)
    tabs = {"performance": perf,
            "conditions": conditions_b(d),
            "breakpoints": breakpoints_b(perf),
            "type_i_error": type_i_error_b(perf),
            "attribution": attribution_b(perf),
            "failures": failure_panel(d)}
    for name, t in tabs.items():
        if len(t):
            t.to_csv(out_dir / f"study_b_{name}.csv", index=False)
    return tabs


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="sim.study_b_report", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="results/study_b")
    ap.add_argument("--config", default="base")
    a = ap.parse_args(argv)
    tabs = build(a.output_dir, config=a.config)
    print({k: len(v) for k, v in tabs.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
