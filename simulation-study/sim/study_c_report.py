"""Study C reporting: agreement between implementations, then performance.

Two questions, deliberately answered in this order.

**Does the harness agree with itself?** Every implementation gets the same
nuisances, so a disagreement is an implementation difference and nothing else.
The tolerance depends on the tier (see `sim/study_c.py`), and for `concrete` the
gate is the *targeting increment* rather than the CIF level: the two build
different time grids, which shifts the plug-in by ~1e-3 in both `tmle` and
`gcomp` alike and has nothing to do with the update step (FINDINGS 8).

**Then, how do the estimators perform?** Bias, coverage and RMSE against the
closed-form truth, on the risk-difference scale, for every estimator that targets
it.

The conventional cause-specific Cox is scored **separately and on its own
scale**. A cause-specific log hazard ratio is a conditional, non-collapsible
parameter; it is not an estimate of the marginal risk difference and comparing it
to the RD truth would be a category error, so it never enters the RD tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .dgp import get_config
from .metrics import summarise, truth_long
from .study_c import CONFIG, collect_study_c
from .truth import cached_truth

__all__ = ["agreement", "score_panel", "performance", "build"]

#: Reference implementation every other row is compared against. PyTMLE is the
#: package under study, so it is the baseline by definition, not by merit.
REFERENCE = "tmle"

#: Which comparison each implementation is entitled to, at what tolerance, and
#: -- since one row is now expected to *disagree* -- what agreement would mean.
#: Stated as data rather than prose so the gate cannot drift from the write-up.
#:
#: `expect="diverge"` is not a way of excusing a failure. Before the
#: targeted-update fix (FINDINGS 9) PyTMLE and concrete agreed on the targeting
#: increment to 0.00018, which is what justified the 2e-3 tolerance. PyTMLE has
#: since been fixed and concrete 1.0.8 has not, so the two now differ by roughly
#: the size of the defect, and *agreement* would be the result worth
#: investigating. The check still runs; only its expected direction changed.
TIERS: Dict[str, Dict] = {
    # `tol_se` is pinned from the first fully instrumented run (500/500/150
    # replicates at n = 500/1000/2000). The SE agrees far better than the point
    # estimate does, which is the expected shape: the FINDINGS 9 defect moves
    # `Psi` through the targeted update, but both packages build the influence
    # curve with the same formula, so the SE is untouched by it. Observed mean
    # |log ratio|: 6.6e-4, 3.3e-4, 1.6e-4 -- shrinking as O(1/n) and three times
    # inside the tolerance at the worst size.
    "tmle (concrete)": dict(tier=1, tol=2e-3, tol_se=2e-3, expect_se="agree",
                            against="tmle", increment=True,
                            expect="diverge",
                            note="concrete 1.0.8 still carries the FINDINGS 9 "
                                 "defect; divergence here is the expected result "
                                 "and its size estimates the defect",
                            note_se="same IC formula on both sides, so the SE "
                                    "must agree even though the point estimate "
                                    "diverges",
                            # The real gate on the score is not a cross-package
                            # tolerance but each package's own stopping rule,
                            # `|PnEIC| <= seEIC/(sqrt(n) log n)` -- both satisfy
                            # it on 100 % of targets and no row's cross-package
                            # difference exceeds it (see `score_panel` and
                            # FINDINGS 13). This tolerance is the coarser
                            # companion check on the paired mean, which runs
                            # 2.2e-05 / 2.6e-06 / 2.4e-06.
                            tol_pn_eic=1e-3, expect_pn_eic="agree",
                            note_pn_eic="both solve their own stopping "
                                        "criterion; the residual gap is the "
                                        "width of that band, not a discrepancy"),
    # The plug-in involves no targeting, so it is unaffected by the defect and
    # must still agree. It is the control that shows the divergence above is the
    # update step rather than the bridge.
    "gcomp (concrete)": dict(tier=1, tol=5e-3, against="gcomp", increment=False,
                             expect="agree",
                             note="plug-in on concrete's coarser grid; unaffected "
                                  "by the targeted-update defect, so this must agree"),
    "ate:GFORMULA": dict(expect="agree", tier=2, tol=1e-2, against="gcomp", increment=False,
                         note="exact Aalen-Johansen vs discrete plug-in"),
    # The point estimates agree to 1.6e-5 and the standard errors do not: a
    # stable ~13 % gap (mean |log ratio| 0.134 / 0.130 / 0.128) that does **not**
    # shrink with n, so it is structural. The empirical SD settles which side is
    # right, and it is riskRegression's -- the smaller of the two (0.877x) and
    # the calibrated one (SE/SD 0.93-1.01 against PyTMLE's 1.02-1.21, which
    # over-covers up to 0.986). `run_ipw` treats the weights as **known**, and an
    # IPW estimator with estimated propensities has a *smaller* asymptotic
    # variance than one using the true ones, so treating them as known overstates
    # it. See FINDINGS 14.
    "ate:IPTW": dict(expect="agree", tier=2, tol=1e-2, against="ipw", increment=False,
                     note="",
                     tol_se=1e-2, expect_se="diverge",
                     note_se="PyTMLE's IPW SE is conservative -- it treats the "
                             "weights as known; riskRegression propagates "
                             "nuisance estimation and is the calibrated one"),
    "ate:AIPTW": dict(expect="agree", tier=2, tol=1e-2, against="aipw", increment=False,
                      note="",
                      # observed mean |log ratio| 6.0e-3 / 3.8e-3 / 2.4e-3
                      tol_se=1e-2, expect_se="agree"),
}

#: What each comparable quantity is, and how a difference in it should be read.
#:
#: `est`  absolute difference, and the only one entitled to the targeting-
#:        increment trick: differencing `tmle - gcomp` across packages cancels
#:        the grid offset of FINDINGS 8, which is a property of the CIF *level*.
#: `se`   the ratio, on the log scale. A standard error scales as n^-1/2, so an
#:        absolute tolerance would silently loosen at n = 500 and tighten at
#:        n = 2000 while appearing to be one gate. The increment is undefined:
#:        `gcomp` has no standard error in *either* package, so `tmle - gcomp`
#:        would be all-NaN and the existing `if diff.empty: continue` would skip
#:        the row silently -- a gate that reports nothing and looks like a pass.
#: `pn_eic` the score. Also no increment: it is not a level plus an offset but a
#:        residual each implementation drives toward zero, and the gcomp analogue
#:        is the *pre*-update score, which is about the starting point rather
#:        than the update. Compared as a level, in units of the stopping
#:        criterion, and read beside `score_panel` -- see there for why a paired
#:        difference alone would be uninformative.
QUANTITIES: Dict[str, Dict] = {
    "est": dict(statistic="difference", scale="absolute", allow_increment=True,
                frame="estimates"),
    "se": dict(statistic="log ratio", scale="relative", allow_increment=False,
               frame="estimates"),
    "pn_eic": dict(statistic="difference", scale="criterion", allow_increment=False,
                   frame="score"),
}


def _rd(d: pd.DataFrame, event: Optional[int] = None) -> pd.DataFrame:
    out = d[(d["estimand"] == "rd") & d["est"].notna()]
    return out if event is None else out[out["event"] == event]


def agreement(d: pd.DataFrame, event: Optional[int] = None,
              quantity: str = "est") -> pd.DataFrame:
    """Paired per-replicate agreement of every implementation with its counterpart.

    Paired, because all implementations run on the same replicate: the
    replicate-to-replicate variation cancels and what is left is the
    implementation difference. Comparing marginal means instead would hide a
    disagreement that happens to average out.

    `quantity` selects what is compared -- the point estimate, the standard
    error, or the score. See `QUANTITIES` for why each gets the statistic and
    tolerance scale it does, and why only `est` may use the targeting increment.

    A comparison that cannot be made is emitted as a row with `pairs = 0` and a
    `skip_reason`, never dropped. The previous code fell through `if diff.empty:
    continue`, so a quantity that was NaN on one side vanished from the table
    and read as though it had passed.
    """
    if quantity not in QUANTITIES:
        raise ValueError(f"unknown quantity {quantity!r}; "
                         f"expected one of {sorted(QUANTITIES)}")
    spec = QUANTITIES[quantity]
    rows: List[Dict] = []

    src = d[d["estimand"] == "rd"] if spec["frame"] == "estimates" else d
    src = src[src[quantity].notna()] if quantity in src else src.iloc[:0]
    if event is not None and "event" in src:
        src = src[src["event"] == event]

    # `group` is NaN on every contrast row, and pivot_table silently drops index
    # rows containing NaN -- which would empty the table rather than fail.
    index = ["n", "rep", "event", "time"]
    if "group" in src.columns:
        src = src.copy()
        src["group"] = src["group"].fillna(-1.0)
        index.append("group")
    w = (src.pivot_table(index=index, columns="estimator", values=quantity)
         if len(src) else pd.DataFrame())

    for name, cfg in TIERS.items():
        base = cfg["against"]
        tol = cfg.get(f"tol_{quantity}", cfg["tol"] if quantity == "est" else np.nan)
        # `expect` is per *quantity*, not per implementation. concrete is the
        # case that forces this: its point estimate is expected to diverge
        # (FINDINGS 9 moves Psi) while its standard error is expected to agree,
        # because the two packages compute the IC with the same formula and the
        # defect does not touch it. A single shared `expect` would score the
        # SE's 6.6e-4 agreement as a failure.
        exp = cfg.get(f"expect_{quantity}", cfg.get("expect", "agree"))
        common = dict(implementation=name, vs=base, tier=cfg["tier"],
                      quantity=quantity, statistic=spec["statistic"],
                      tolerance=tol, expect=exp,
                      note=cfg.get(f"note_{quantity}", cfg["note"]))

        def _skip(reason: str, n_val=np.nan):
            rows.append({**common, "n": n_val, "compared": spec["statistic"],
                         "pairs": 0, "mean_diff": np.nan, "mean_abs_diff": np.nan,
                         "max_abs_diff": np.nan, "as_expected": None,
                         "skipped": True, "skip_reason": reason})

        if name not in w.columns or base not in w.columns:
            _skip(f"{quantity} not available for {name} or {base}")
            continue

        # `increment` is a property of the *estimate* comparison -- it cancels
        # the FINDINGS 8 grid offset in the CIF level. For the standard error
        # and the score it is undefined, so those fall through to a level
        # comparison rather than being skipped: refusing here would drop the
        # single most valuable SE comparison in the study, tmle vs concrete.
        use_increment = spec["allow_increment"] and cfg["increment"]

        if use_increment:
            gc, gcc = "gcomp", "gcomp (concrete)"
            if gc not in w.columns or gcc not in w.columns:
                _skip("gcomp missing on one side, so the increment is undefined")
                continue
            diff = (w[base] - w[gc]) - (w[name] - w[gcc])
            what = "targeting increment"
        elif spec["statistic"] == "log ratio":
            with np.errstate(divide="ignore", invalid="ignore"):
                diff = np.log(w[base] / w[name])
            diff = diff.replace([np.inf, -np.inf], np.nan)
            what = "log ratio"
        else:
            diff = w[base] - w[name]
            what = "level"

        diff = diff.dropna()
        if diff.empty:
            _skip(f"no paired {quantity} values for {name} vs {base}")
            continue

        for n_val, g in diff.groupby(level="n"):
            a = g.to_numpy(dtype=float)
            mad = float(np.abs(a).mean())
            rows.append({
                **common, "n": int(n_val), "compared": what, "pairs": len(a),
                "mean_diff": float(a.mean()), "mean_abs_diff": mad,
                "max_abs_diff": float(np.abs(a).max()),
                "as_expected": (None if not np.isfinite(tol) else bool(
                    (mad <= tol) == (exp == "agree"))),
                "skipped": False, "skip_reason": None,
            })
    return pd.DataFrame(rows)


#: How each quantity's discrepancy is labelled in the summary table. The unit
#: differs by quantity and stating it is not decoration: 1e-3 is negligible for
#: an absolute risk difference and enormous for a score whose target is zero.
_QUANTITY_LABEL = {
    "est": ("point estimate", "abs. difference"),
    "se": ("standard error", "abs. log ratio"),
    "pn_eic": ("score (PnEIC)", "abs. difference"),
}


def agreement_summary(agr: pd.DataFrame) -> pd.DataFrame:
    """The agreement table as one row per comparison and quantity, `n` across.

    The long table has one row per (comparison, quantity, n), which is the right
    shape to store and the wrong shape to read: the question it has to answer is
    "does this discrepancy shrink with `n`", and that is a comparison *along* a
    row. Pivoting `n` into columns puts the trend on one line.

    `mean_abs_diff` is the value shown. It is the paired mean, so it does not
    cancel a two-sided disagreement the way `mean_diff` would, and it is what
    `as_expected` is gated on.
    """
    if agr.empty:
        return pd.DataFrame()
    d = agr[~agr["skipped"].fillna(False)].copy()
    if d.empty:
        return pd.DataFrame()

    d["comparison"] = d["implementation"] + " vs " + d["vs"]
    d["quantity_label"] = d["quantity"].map(lambda q: _QUANTITY_LABEL.get(q, (q, ""))[0])
    d["unit"] = d["quantity"].map(lambda q: _QUANTITY_LABEL.get(q, (q, ""))[1])

    keys = ["comparison", "tier", "quantity", "quantity_label", "unit",
            "tolerance", "expect"]
    # groupby/unstack, not `pivot_table(..., dropna=False)`: the latter builds
    # the full cartesian product of every index level, so seven keys turned nine
    # real rows into hundreds of empty ones.
    wide = (d.groupby(keys + ["n"], dropna=False)["mean_abs_diff"].mean()
            .unstack("n").reset_index())
    wide.columns = [c if isinstance(c, str) else f"n={int(c)}"
                    for c in wide.columns]

    # A verdict per row rather than per (row, n): `as_expected` is already the
    # per-n gate, so the row passes only if every size did.
    verdict = (d.groupby(keys)["as_expected"]
               .agg(lambda s: None if s.isna().all() else bool(s.fillna(True).all()))
               .rename("as_expected").reset_index())
    wide = wide.merge(verdict, on=keys, how="left")

    # Does the discrepancy shrink with n? The distinguishing question: a
    # numerical difference decays, a structural one does not. Reported as the
    # ratio of the largest sample size's discrepancy to the smallest's.
    ncols = [c for c in wide.columns if c.startswith("n=")]
    ncols.sort(key=lambda c: int(c[2:]))
    if len(ncols) >= 2:
        first, last = wide[ncols[0]], wide[ncols[-1]]
        wide["shrink_ratio"] = last / first.replace(0, np.nan)

    order = {"est": 0, "se": 1, "pn_eic": 2}
    wide = wide.sort_values(["quantity", "tier", "comparison"],
                            key=lambda s: s.map(order) if s.name == "quantity" else s)
    return wide.drop(columns=["quantity"]).reset_index(drop=True)


def score_panel(eic: pd.DataFrame) -> pd.DataFrame:
    """Did each implementation solve *its own* score equation?

    The paired cross-package difference in `Pn D*` is not enough on its own: if
    both packages drive their score to zero the difference goes to zero too, and
    any tolerance passes for the wrong reason. What distinguishes them is how far
    each one actually got, measured against the criterion it stops on --
    `|PnEIC| <= seEIC/(sqrt(n) log n)`, which both compute identically.

    That ratio is dimensionless and free of the grid offset (FINDINGS 8), which
    shifts the CIF level in both estimators alike but is not a property of the
    score. So it is a cleaner read on the targeting step than the estimate scale
    can give.
    """
    if eic.empty:
        return pd.DataFrame()
    d = eic[eic["event"] > 0].copy()          # skip the synthetic event-free row
    d["ratio"] = d["pn_eic"].abs() / d["eic_crit"]
    g = d.groupby(["n", "source", "event", "time"], dropna=False)
    out = g.agg(rows=("ratio", "size"),
                median_ratio=("ratio", "median"),
                q95_ratio=("ratio", lambda x: float(np.quantile(x, 0.95))),
                frac_solved=("ratio", lambda x: float((x <= 1).mean())),
                mean_abs_pn_eic=("pn_eic", lambda x: float(np.abs(x).mean())),
                mean_se_eic=("se_eic", "mean")).reset_index()
    red = (d.drop_duplicates(["n", "source", "rep"])
           .groupby(["n", "source"])
           .apply(lambda x: float(np.nanmedian(x["norm_pn_eic_first"]
                                               / x["norm_pn_eic_last"])),
                  include_groups=False)
           .rename("median_score_reduction").reset_index())
    return out.merge(red, on=["n", "source"], how="left")


def performance(d: pd.DataFrame, config: str = CONFIG,
                cache_dir: Path | str = "results/study_c/_truth",
                n_mc: int = 4_000_000) -> pd.DataFrame:
    """Bias / coverage / RMSE against the closed-form truth, per (n, estimator)."""
    p = get_config(config)
    out: List[pd.DataFrame] = []
    for n_val, g in _rd(d).groupby("n"):
        taus = sorted(g["time"].dropna().unique())
        tr = cached_truth(p, taus, Path(cache_dir), n_mc=n_mc)
        g = g.assign(cell=f"n{int(n_val)}", group=np.nan)
        if "converged" in g:
            g = g.assign(converged=g["converged"].astype("boolean"))
        s = summarise(g, truth_long(tr))
        s["n"] = int(n_val)
        out.append(s)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def build(dirs: Sequence[Path | str], out_dir: Path | str = "results/study_c",
          event: int = 1, config: str = CONFIG) -> Dict[str, pd.DataFrame]:
    """Collect every n-directory, then write the three tables."""
    from .study_c import collect_study_c_eic

    d = pd.concat([collect_study_c(x) for x in dirs], ignore_index=True)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out_dir / "study_c_estimates.parquet", index=False)

    eic_parts = [e for e in (collect_study_c_eic(x) for x in dirs) if len(e)]
    eic = pd.concat(eic_parts, ignore_index=True) if eic_parts else pd.DataFrame()
    if len(eic):
        eic.to_parquet(out_dir / "study_c_eic.parquet", index=False)

    # One agreement table covering all three quantities: point estimate,
    # standard error and score. Rows grow rather than the shape changing, so
    # existing readers of `study_c_agreement.csv` keep working.
    agr = [agreement(d, event=event, quantity="est"),
           agreement(d, event=event, quantity="se")]
    if len(eic):
        e = eic.rename(columns={"source": "_src"})
        e["estimator"] = np.where(e["_src"] == "concrete",
                                  "tmle (concrete)", "tmle")
        agr.append(agreement(e, event=event, quantity="pn_eic"))
    tabs = {
        "agreement": pd.concat([a for a in agr if len(a)], ignore_index=True),
        "performance": performance(d, config=config,
                                   cache_dir=out_dir / "_truth"),
    }
    # The same content with `n` across the columns. The long form is the one to
    # store and query; this is the one to read.
    tabs["agreement_summary"] = agreement_summary(tabs["agreement"])
    if len(eic):
        tabs["score"] = score_panel(eic)
    for name, t in tabs.items():
        if len(t):
            t.to_csv(out_dir / f"study_c_{name}.csv", index=False)
            (out_dir / f"study_c_{name}.md").write_text(t.to_markdown(index=False))

    bench_f = out_dir / "_bench" / "bench_stage2.parquet"
    if bench_f.exists():
        from .bench_stage2 import summarise_bench
        rt = summarise_bench(pd.read_parquet(bench_f))
        if len(rt):
            tabs["runtime"] = rt
            rt.to_csv(out_dir / "study_c_runtime.csv", index=False)
            (out_dir / "study_c_runtime.md").write_text(rt.to_markdown(index=False))

    # the marginal truth, for the panel that sets the hazard ratio beside it
    from .plots_study_c import make_all_c
    from .runner import target_times_for
    from .truth import closed_form

    p = get_config(config)
    taus = target_times_for(p)
    tr = closed_form(p, taus, n_mc=2_000_000).drop_duplicates(["event", "time"])
    tau = max(taus)
    rd_truth = {int(r.event): float(r.rd)
                for _, r in tr[np.isclose(tr["time"], tau)].iterrows()}
    make_all_c(tabs, out_dir / "figures", rd_truth=rd_truth, event=event)
    return tabs


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="sim.study_c_report", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="one directory per sample size")
    ap.add_argument("--out-dir", default="results/study_c")
    ap.add_argument("--event", type=int, default=1)
    a = ap.parse_args(argv)
    tabs = build(a.dirs, a.out_dir, event=a.event)
    for name, t in tabs.items():
        print(f"\n### {name}\n")
        print(t.to_string(index=False) if len(t) else "(empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
