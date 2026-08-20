"""Guard: does a misspecification actually bias the *estimand*?

This exists because a nuisance-level discrepancy is not evidence of an
estimand-level one, and conflating the two is how a double-robustness simulation
fails silently.

Concretely: giving the threshold term the same coefficient in both arms produces
a large `cumhaz_mae` (0.54 -> 6.34) and yet moves the risk *difference* by
~0.002, because omitting it shifts `F^1` and `F^0` together and the contrast
survives. A guard that checks only `cumhaz_mae` passes that design happily and
the whole factorial comes out inert.

So the quantity to calibrate on is the plug-in's bias at large `n`, where
estimation noise has died away and what remains is the asymptotic bias the
misspecification induces. `gcomp` is the probe: it is a pure function of `Q`, it
costs nothing (it rides along on the TMLE fit), and it is the estimator whose
bias the outcome misspecification is supposed to create.

    python -m sim.calibrate --config threshold --n 4000 --reps 40
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .dgp import DGPParams, get_config, sample
from .nuisance import Spec, build
from .runner import default_n_jobs, target_times_for
from .truth import closed_form

__all__ = ["asymptotic_bias", "crude_vs_truth"]


def _one(args):
    p, taus, spec, seed, n, which = args
    from .estimators import run_all

    try:
        sm = sample(n, p, np.random.default_rng(seed))
        ie = build(sm, spec)
        r = run_all(sm.df, ie, list(taus), target_events=list(range(1, p.n_causes + 1)),
                    which=which, min_nuisance=0.01)
    except Exception as exc:  # nuisance construction itself failed
        return pd.DataFrame([{"estimator": e, "estimand": "rd", "event": np.nan,
                              "time": np.nan, "est": np.nan,
                              "error": f"{type(exc).__name__}: {exc}"} for e in which])
    keep = r["estimand"].isin(["rd", "rr"]) | r["error"].notna()
    return r.loc[keep, ["estimator", "estimand", "event", "time", "est", "error"]]


def asymptotic_bias(
    p: DGPParams,
    specs: Dict[str, Spec],
    n: int = 4000,
    reps: int = 40,
    seed: int = 20250301,
    which: Sequence[str] = ("gcomp", "ipw", "tmle"),
    n_mc_truth: int = 4_000_000,
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """Bias of each estimator under each specification, at a large sample size.

    Averaging a moderate number of large-`n` replicates separates the asymptotic
    bias (which does not shrink) from estimation noise (which does), and the
    reported MC-SE says whether the difference between specifications is real.
    """
    taus = target_times_for(p)
    truth = closed_form(p, taus, n_mc=n_mc_truth)
    tmap = {}
    for _, r in truth.drop_duplicates(["event", "time"]).iterrows():
        tmap[("rd", int(r["event"]), float(r["time"]))] = float(r["rd"])
        tmap[("rr", int(r["event"]), float(r["time"]))] = float(r["rr"])

    n_jobs = n_jobs or default_n_jobs(n)
    rows: List[pd.DataFrame] = []
    for label, spec in specs.items():
        tasks = [(p, taus, spec, seed + 1000 * i, n, which) for i in range(reps)]
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            got = [r for r in ex.map(_one, tasks) if r is not None]
        if not got:
            continue
        df = pd.concat(got, ignore_index=True)
        df["spec"] = label
        df["attempted"] = reps
        ok = df["error"].isna() & df["est"].notna()
        df["truth"] = np.nan
        df.loc[ok, "truth"] = [
            tmap[(e, int(ev), float(t))]
            for e, ev, t in zip(df.loc[ok, "estimand"], df.loc[ok, "event"],
                                df.loc[ok, "time"])
        ]
        rows.append(df)

    allr = pd.concat(rows, ignore_index=True)
    allr["err"] = allr["est"] - allr["truth"]

    def _agg(g):
        e = g["err"].to_numpy(dtype=float)
        e = e[np.isfinite(e)]
        m = len(e)
        t = g["truth"].dropna()
        return pd.Series({
            "reps": m,
            "attempted": int(g["attempted"].iloc[0]),
            "truth": float(t.iloc[0]) if len(t) else np.nan,
            "bias": float(e.mean()) if m else np.nan,
            "mc_se": float(np.std(e, ddof=1) / np.sqrt(m)) if m > 1 else np.nan,
        })

    ok = allr[allr["estimand"].isin(["rd", "rr"]) & allr["est"].notna()]
    out = (
        ok.groupby(["spec", "estimator", "estimand", "event", "time"], dropna=False)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    out["z"] = out["bias"] / out["mc_se"]
    out["completion"] = out["reps"] / out["attempted"]

    # A guard that silently drops failed replicates is worse than no guard: the
    # survivors are not a random subset, so every number computed from them is
    # conditioned on success. Surface it rather than bury it.
    worst = out.groupby(["spec", "estimator"])["completion"].min()
    bad = worst[worst < 0.95]
    if len(bad):
        lines = "\n".join(
            f"    {s:>12s} / {e:<6s} {100 * c:5.1f}% of replicates usable"
            for (s, e), c in bad.items()
        )
        warnings.warn(
            "asymptotic_bias: replicates failed and were excluded. The remaining "
            "estimates are conditional on success and are NOT comparable across "
            f"cells:\n{lines}",
            RuntimeWarning,
            stacklevel=2,
        )
    return out


def crude_vs_truth(p: DGPParams, n: int = 200_000, seed: int = 7) -> pd.DataFrame:
    """Is there real confounding to remove?

    Compares the unadjusted arm difference in observed sub-distribution
    probabilities against the truth. If these agree, the design has no
    confounding and every adjusted estimator will look identical to a crude one.
    """
    taus = target_times_for(p)
    truth = closed_form(p, taus, n_mc=2_000_000).drop_duplicates(["event", "time"])
    sm = sample(n, p, np.random.default_rng(seed))
    t, d, a = sm.event_times, sm.event_indicator, sm.group

    rows = []
    for _, r in truth.iterrows():
        ev, tau = int(r["event"]), float(r["time"])
        # naive: ignores censoring entirely, which is what "crude" means here
        p1 = float(((t <= tau) & (d == ev) & (a == 1)).sum() / max((a == 1).sum(), 1))
        p0 = float(((t <= tau) & (d == ev) & (a == 0)).sum() / max((a == 0).sum(), 1))
        rows.append({"event": ev, "time": tau, "truth_rd": float(r["rd"]),
                     "crude_rd": p1 - p0, "confounding_gap": (p1 - p0) - float(r["rd"])})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="sim.calibrate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="threshold")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20250301)
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="workers. runner.default_n_jobs optimises throughput per "
                         "CPU, which caps at 4 for n > 1000 -- right when many "
                         "cells are queued, wrong for a single latency-bound "
                         "calibration run on an idle machine. Override it here.")
    ap.add_argument("--study-a", action="store_true",
                    help="guard all eight cells of Study A's 2^3 factorial")
    # A leading "-" makes argparse read the value as a flag, so a negative first
    # coefficient needs --eta=-1.5,1.0. Accepting a leading "m" for minus keeps
    # the space-separated form usable too.
    ap.add_argument("--threshold", type=float, default=None,
                    help="location of u(W) = 1{w_cont > threshold}")
    ap.add_argument("--delta", default=None,
                    help="cause-specific threshold coefficients, e.g. 1.5,1.5")
    ap.add_argument("--eta", default=None,
                    help="override the treatment x threshold interaction. Use "
                         "--eta=-1.5,1.0 (the = is required when the first value "
                         "is negative), or --eta m1.5,1.0")
    args = ap.parse_args(argv)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    p = get_config(args.config)
    def _vec(spec):
        return np.array([float(x.replace("m", "-", 1) if x.startswith("m") else x)
                         for x in spec.split(",")])

    over = {}
    if args.threshold is not None:
        over["threshold"] = args.threshold
    if args.delta:
        over["delta"] = _vec(args.delta)
    if args.eta:
        over["eta"] = _vec(args.eta)
    if over:
        p = p.with_(**over)
        print(f"[calibrate] overrides -> threshold={p.threshold} "
              f"delta={p.delta} eta={p.eta}")

    print(f"=== {args.config}: is there confounding to remove? ===")
    print(crude_vs_truth(p).round(4).to_string(index=False))

    print(f"\n=== asymptotic bias by specification (n = {args.n}, {args.reps} reps) ===")
    if args.study_a:
        from .nuisance import STUDY_A_CELLS

        specs = dict(STUDY_A_CELLS)
        which = ("gcomp", "ipw", "aipw", "tmle")
    else:
        specs = {
            "Q ok": Spec("correct", "correct", "correct"),
            "Q wrong": Spec("wrong", "correct", "correct"),
            "G wrong": Spec("correct", "correct", "wrong"),
            "Q+G wrong": Spec("wrong", "correct", "wrong"),
        }
        which = ("gcomp", "ipw", "tmle")

    out = asymptotic_bias(p, specs, n=args.n, reps=args.reps, seed=args.seed,
                          which=which, n_jobs=args.n_jobs)
    out.to_parquet(f"results/calibration_{args.config}.parquet", index=False)
    rd = out[out["estimand"] == "rd"]

    if args.study_a:
        # One row per cell: the largest |bias| over targets, with its z-score.
        # A cell whose theory says "biased" must show a large z; one whose theory
        # says "unbiased" must show a small one. Anything else means the design is
        # not testing what it claims.
        from .nuisance import STUDY_A_CELLS as CELLS

        rows = []
        for cell, spec in CELLS.items():
            g = rd[rd["spec"] == cell]
            for est in which:
                gg = g[g["estimator"] == est]
                if gg.empty:
                    continue
                i = gg["bias"].abs().idxmax()
                rows.append({
                    "cell": cell, "Q": spec.Q, "pi": spec.pi, "G": spec.G,
                    "estimator": est,
                    "max_abs_bias": float(gg.loc[i, "bias"]),
                    "mc_se": float(gg.loc[i, "mc_se"]),
                    "z": float(gg.loc[i, "z"]),
                })
        tab = pd.DataFrame(rows)
        print(tab.pivot_table(index=["cell", "Q", "pi", "G"], columns="estimator",
                              values="max_abs_bias").round(4).to_string())
        print("\n--- z-scores (|z| > 3 means the bias is real, < 2 means it is not) ---")
        print(tab.pivot_table(index=["cell", "Q", "pi", "G"], columns="estimator",
                              values="z").round(1).to_string())
    else:
        print(rd[["spec", "estimator", "event", "time", "truth", "bias", "mc_se", "z"]]
              .sort_values(["estimator", "event", "time", "spec"]).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
