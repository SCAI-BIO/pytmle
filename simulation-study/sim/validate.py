"""The AdjCuminc validation ladder -- the gate for the whole harness.

Four rungs, each isolating one change:

1. **Truth agreement** against Hage et al.'s own ``trueCSH()``.
2. **Their estimators on their DGP**, reproducing the published bias pattern.
3. **My estimators on their DGP**, checked at the tier appropriate to each
   comparator.
4. **Informative censoring switched on** -- the extension over their study.

Rung 1 also quantifies a target-definition question. The paper defines the
estimand as ``I_k^(z) = P(T^z <= t, Delta^z = k)`` over the realised sample,
while ``trueCSH()`` integrates against the *unconditional* covariate law. Those
coincide in Scenarios 1 and 3 but not in 2 and 4, where the control-resampling
device makes the realised law a mixture. Both targets are computed and the gap
reported; the signature of getting it wrong is every estimator biased by the
same constant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .dgp import get_config
from .truth import closed_form

__all__ = ["SCENARIOS", "rung1_truth_agreement", "rung2_reference_pattern", "which_target"]

#: Their four scenarios and what each misspecifies.
SCENARIOS: Dict[str, Dict] = {
    "s1": dict(config="adjcuminc_s1", misspecified="none"),
    "s2": dict(config="adjcuminc_s2", misspecified="treatment"),
    "s3": dict(config="adjcuminc_s3", misspecified="outcome"),
    "s4": dict(config="adjcuminc_s4", misspecified="both"),
}


def rung1_truth_agreement(
    r_truth_path: Path | str,
    times: Sequence[float],
    n_mc: int = 4_000_000,
    seed: int = 20250301,
    tol: float = 1e-3,
) -> pd.DataFrame:
    """Compare the Python closed form against ``trueCSH()``, scenario by scenario.

    Returns one row per (scenario, arm, event, time) with both candidate targets
    and the reference value, plus the gap between them.
    """
    ref = pd.read_parquet(r_truth_path).rename(columns={"risk": "risk_r"})

    rows: List[pd.DataFrame] = []
    for name, meta in SCENARIOS.items():
        p = get_config(meta["config"])
        realised = closed_form(p, times, n_mc=n_mc, seed=seed, target="realised")
        unweighted = closed_form(p, times, n_mc=n_mc, seed=seed, target="unweighted")
        merged = realised[["arm", "event", "time", "risk", "mc_se"]].rename(
            columns={"risk": "risk_realised"}
        )
        merged = merged.merge(
            unweighted[["arm", "event", "time", "risk"]].rename(
                columns={"risk": "risk_unweighted"}
            ),
            on=["arm", "event", "time"],
        )
        merged["scenario"] = name
        merged["misspecified"] = meta["misspecified"]
        rows.append(merged)

    out = pd.concat(rows, ignore_index=True).merge(
        ref, on=["scenario", "arm", "event", "time"], how="left", validate="one_to_one"
    )
    out["gap_realised_vs_r"] = out["risk_realised"] - out["risk_r"]
    out["gap_unweighted_vs_r"] = out["risk_unweighted"] - out["risk_r"]
    out["gap_target_definitions"] = out["risk_realised"] - out["risk_unweighted"]
    out["agrees"] = out["gap_unweighted_vs_r"].abs() < tol
    cols = [
        "scenario", "misspecified", "arm", "event", "time",
        "risk_r", "risk_unweighted", "risk_realised", "mc_se",
        "gap_unweighted_vs_r", "gap_realised_vs_r", "gap_target_definitions", "agrees",
    ]
    return out[cols].sort_values(["scenario", "event", "arm", "time"]).reset_index(drop=True)


def _truths_for_scenarios(
    times: Sequence[float], n_mc: int = 4_000_000, seed: int = 20250301
) -> pd.DataFrame:
    """Both candidate targets, per scenario, in long per-arm form."""
    rows = []
    for name, meta in SCENARIOS.items():
        p = get_config(meta["config"])
        for target, col in (("realised", "truth_realised"), ("unweighted", "truth_unweighted")):
            t = closed_form(p, times, n_mc=n_mc, seed=seed, target=target)
            t = t[["arm", "event", "time", "risk"]].rename(columns={"risk": col})
            t["scenario"] = name
            rows.append(t.set_index(["scenario", "arm", "event", "time"]))
    realised = pd.concat([r for r in rows if "truth_realised" in r.columns])
    unweighted = pd.concat([r for r in rows if "truth_unweighted" in r.columns])
    return realised.join(unweighted).reset_index()


def rung2_reference_pattern(
    est_path: Path | str,
    times: Sequence[float],
    n_mc: int = 4_000_000,
) -> pd.DataFrame:
    """Bias of Hage et al.'s own estimators against *both* candidate targets.

    Their published pattern is: Scenario 1 all adjusted methods unbiased;
    Scenario 2 (treatment model wrong) IPW biased, OR and DR unbiased; Scenario 3
    (outcome model wrong) OR biased, IPW and DR unbiased; Scenario 4 none
    reliable. Reproducing that here establishes the reference result on this
    machine before anything of ours is trusted.

    Reporting bias against both targets simultaneously is what identifies which
    population the estimators actually estimate -- a question rung 1 raises but
    cannot answer.
    """
    est = pd.read_parquet(est_path)
    truth = _truths_for_scenarios(times, n_mc=n_mc)
    merged = est.merge(truth, on=["scenario", "arm", "event", "time"],
                       how="left", validate="many_to_one")
    if merged["truth_realised"].isna().any():
        raise ValueError("estimates with no matching truth; check the time grid")

    def _agg(g: pd.DataFrame) -> pd.Series:
        m = len(g)
        br = (g["risk"] - g["truth_realised"]).to_numpy()
        bu = (g["risk"] - g["truth_unweighted"]).to_numpy()
        return pd.Series({
            "reps": m,
            "mean_est": float(g["risk"].mean()),
            "truth_realised": float(g["truth_realised"].iloc[0]),
            "truth_unweighted": float(g["truth_unweighted"].iloc[0]),
            "bias_vs_realised": float(br.mean()),
            "bias_vs_unweighted": float(bu.mean()),
            "mc_se": float(np.std(br, ddof=1) / np.sqrt(m)) if m > 1 else np.nan,
            "rmse_vs_realised": float(np.sqrt(np.mean(br**2))),
            "mean_seconds": float(g["seconds"].mean()) if "seconds" in g else np.nan,
        })

    return (
        merged.groupby(["scenario", "estimator", "arm", "event", "time"], dropna=False)
        .apply(_agg, include_groups=False)
        .reset_index()
    )


def which_target(rung2: pd.DataFrame) -> pd.DataFrame:
    """Which target definition do the estimators actually track?

    Only informative in Scenarios 2 and 4, where the two definitions differ.
    Uses the adjusted estimators that theory says are consistent in each
    scenario, so a verdict is not contaminated by an estimator that is biased for
    other reasons: OR and DR in Scenario 2 (treatment model wrong), and none in
    Scenario 4, where the comparison is descriptive only.
    """
    consistent = {"s1": ["adjOR", "adjDR", "adjIPW"], "s2": ["adjOR", "adjDR"],
                  "s3": ["adjIPW", "adjDR"], "s4": []}
    rows = []
    for sc, g in rung2.groupby("scenario"):
        keep = consistent.get(sc, [])
        sub = g[g["estimator"].isin(keep)] if keep else g[g["estimator"] != "crude"]
        sep = (sub["truth_realised"] - sub["truth_unweighted"]).abs().max()
        rows.append({
            "scenario": sc,
            "estimators_used": ",".join(sorted(sub["estimator"].unique())),
            "targets_differ_by": float(sep),
            "mean_abs_bias_vs_realised": float(sub["bias_vs_realised"].abs().mean()),
            "mean_abs_bias_vs_unweighted": float(sub["bias_vs_unweighted"].abs().mean()),
            "verdict": (
                "targets indistinguishable" if sep < 1e-3 else
                "tracks REALISED (mixture) population"
                if sub["bias_vs_realised"].abs().mean() < sub["bias_vs_unweighted"].abs().mean()
                else "tracks UNCONDITIONAL population"
            ),
            "conclusive": bool(len(keep) > 0),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rung 3: our estimators on their data, with their fitted nuisances
# ---------------------------------------------------------------------------


def load_r_replicate(stub: Path | str) -> Dict:
    """Read one replicate exported by ``R/rung3_export.R``.

    Returns the dataset in PyTMLE's column convention plus the counterfactual
    cumulative hazards and propensity scores that R fitted, so the Python
    estimators run on inputs identical to the R comparators'.
    """
    stub = Path(stub)
    df = pd.read_parquet(f"{stub}_data.parquet")
    grid = pd.read_parquet(f"{stub}_grid.parquet")["time"].to_numpy()
    ps1 = pd.read_parquet(f"{stub}_ps.parquet")["ps1"].to_numpy()
    mats = {
        nm: pd.read_parquet(f"{stub}_{nm}.parquet").to_numpy(dtype=float)
        for nm in ("H1_1", "H1_0", "H2_1", "H2_0", "HC_1", "HC_0")
    }

    status = df["status"].astype(str).to_numpy()
    tidy = pd.DataFrame(
        {
            "event_time": df["obs_time"].to_numpy(dtype=float),
            "event_indicator": np.where(status == "0", 0,
                                        np.where(status == "1", 1, 2)).astype(int),
            "group": (df["treatment"].astype(str).to_numpy() == "treat").astype(int),
        }
    )
    return {"df": tidy, "grid": grid, "ps1": ps1, **mats}


def initial_estimates_from_r(rep: Dict, key_1: int = 1, key_0: int = 0) -> Dict:
    """Wrap R-fitted nuisances as PyTMLE ``InitialEstimates``."""
    from pytmle import InitialEstimates

    grid = rep["grid"]
    A = rep["df"]["group"].to_numpy().astype(float)

    def _arm(h1, h2, hc, ps, g_star) -> InitialEstimates:
        haz = np.stack(
            [np.diff(h1, prepend=0.0, axis=1), np.diff(h2, prepend=0.0, axis=1)],
            axis=-1,
        )
        return InitialEstimates(
            times=grid,
            g_star_obs=g_star,
            propensity_scores=ps,
            hazards=np.maximum(haz, 0.0),
            event_free_survival_function=np.exp(-(h1 + h2)),
            censoring_survival_function=np.exp(-hc),
        )

    return {
        key_1: _arm(rep["H1_1"], rep["H2_1"], rep["HC_1"], rep["ps1"], A),
        key_0: _arm(rep["H1_0"], rep["H2_0"], rep["HC_0"], 1.0 - rep["ps1"], 1.0 - A),
    }


def rung3_our_estimators(
    out_dir: Path | str,
    times: Sequence[float],
    scenarios: Optional[Sequence[str]] = None,
    min_nuisance: float = 0.01,
) -> pd.DataFrame:
    """Run tmle / gcomp / aipw / ipw on the R-generated replicates."""
    from .estimators import run_all

    out_dir = Path(out_dir)
    scenarios = list(scenarios) if scenarios else list(SCENARIOS)
    rows: List[pd.DataFrame] = []
    for sc in scenarios:
        for data_path in sorted(out_dir.glob(f"{sc}_rep*_data.parquet")):
            stub = str(data_path)[: -len("_data.parquet")]
            rep_id = int(stub.split("_rep")[-1])
            try:
                rep = load_r_replicate(stub)
                ie = initial_estimates_from_r(rep)
                res = run_all(rep["df"], ie, list(times), target_events=[1, 2],
                              min_nuisance=min_nuisance)
            except Exception as exc:  # a bad replicate must not stop the rung
                res = pd.DataFrame([{ "estimator": None, "estimand": None,
                                      "event": np.nan, "time": np.nan, "group": np.nan,
                                      "est": np.nan, "error": f"{type(exc).__name__}: {exc}"}])
            res = res.copy()
            res["scenario"] = sc
            res["rep"] = rep_id
            rows.append(res)
    return pd.concat(rows, ignore_index=True)


def rung3_compare(
    ours: pd.DataFrame,
    theirs_path: Path | str,
    times: Sequence[float],
    n_mc: int = 4_000_000,
) -> pd.DataFrame:
    """Put both sets of estimators side by side against the realised-target truth.

    Pairs are chosen so that each row compares implementations of the *same*
    estimator on identical data and identical nuisances:
    ``gcomp``/``adjOR`` (plug-ins), ``ipw``/``adjIPW``, ``aipw``/``adjDR``.
    ``tmle`` has no counterpart here -- its comparator is ``concrete``, which is
    a tier-1 comparison handled in Study C.
    """
    theirs = pd.read_parquet(theirs_path)
    mine = ours[(ours["estimand"] == "risk") & ours["estimator"].notna()].copy()
    mine = mine.rename(columns={"group": "arm", "est": "risk"})
    mine["arm"] = mine["arm"].astype(int)
    mine["event"] = mine["event"].astype(int)
    both = pd.concat(
        [mine[["scenario", "rep", "estimator", "arm", "event", "time", "risk"]],
         theirs[["scenario", "rep", "estimator", "arm", "event", "time", "risk"]]],
        ignore_index=True,
    )
    truth = _truths_for_scenarios(times, n_mc=n_mc)
    merged = both.merge(truth, on=["scenario", "arm", "event", "time"], how="left")

    def _agg(g: pd.DataFrame) -> pd.Series:
        b = (g["risk"] - g["truth_realised"]).to_numpy()
        return pd.Series({
            "reps": len(g),
            "mean_est": float(g["risk"].mean()),
            "truth": float(g["truth_realised"].iloc[0]),
            "bias": float(b.mean()),
            "mc_se": float(np.std(b, ddof=1) / np.sqrt(len(b))) if len(b) > 1 else np.nan,
            "sd": float(np.std(g["risk"], ddof=1)) if len(g) > 1 else np.nan,
        })

    return (
        merged.groupby(["scenario", "estimator", "arm", "event", "time"], dropna=False)
        .apply(_agg, include_groups=False)
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Rung 4: what an informative censoring mechanism changes
# ---------------------------------------------------------------------------


#: Columns printed for rung 4, as a module constant so the contract is testable.
#:
#: **Every pivot key of `rung4_censoring_effect` must appear here.** `q_model` was
#: a key and was left out, so each (estimator, estimand, event, time, regime) row
#: printed twice -- once for Q correct, once for Q wrong -- with nothing to tell
#: them apart. It is also the column the rung turns on: `bias_change` is ~0 in
#: every `correct` row, which is double robustness holding against a misspecified
#: censoring model, and can only be non-zero where Q is wrong.
RUNG4_KEYS = ("estimator", "estimand", "event", "time", "regime", "q_model")

RUNG4_COLUMNS = list(RUNG4_KEYS) + [
    "bias_correct", "bias_wrong", "bias_change",
    "coverage_correct", "coverage_wrong",
]


def rung4_censoring_effect(output_dir: Path | str, n_mc: int = 4_000_000) -> pd.DataFrame:
    """Does misspecifying the censoring model matter, and when?

    A 2x2x2: censoring uninformative/informative, outcome model correct/wrong,
    censoring model correct/wrong.

    The outcome dimension is essential and was missing from the first version of
    this rung. TMLE and the one-step are doubly robust, so with ``Q`` correct they
    stay consistent *even when ``G`` is wrong* -- asking a misspecified censoring
    model to bias them under a correct outcome model is asking the theory to fail.
    The informative signal lives in the ``Q`` wrong row, where the double-robust
    protection is switched off and ``G`` has to carry the estimate on its own.

    Expected pattern in the ``Q`` wrong row: misspecifying ``G`` is harmless when
    censoring is uninformative (nothing to miss) and biasing when it is
    informative. In the ``Q`` correct row it should be harmless in both regimes --
    which is double robustness working, and worth reporting as such.
    """
    from .report import summarise_dir

    s = summarise_dir(output_dir, n_mc=n_mc)
    s = s[s["estimand"].isin(["rd", "rr"])].copy()
    s["regime"] = np.where(s["cell"].str.contains("info"), "informative", "uninformative")
    s["q_model"] = np.where(s["cell"].str.contains("Qbad"), "wrong", "correct")
    s["g_model"] = np.where(s["cell"].str.endswith("Gbad"), "wrong", "correct")

    keys = ["estimator", "estimand", "event", "time", "regime", "q_model"]
    wide = s.pivot_table(index=keys, columns="g_model",
                         values=["bias", "coverage", "bias_mc_se", "ci_width"]).reset_index()
    wide.columns = [
        c[0] if not c[1] else f"{c[0]}_{c[1]}" for c in wide.columns.to_flat_index()
    ]
    wide["bias_change"] = wide["bias_wrong"] - wide["bias_correct"]
    wide["coverage_change"] = wide["coverage_wrong"] - wide["coverage_correct"]
    # paired MC-SE for the change, so "is this real?" can be answered directly
    wide["bias_change_mc_se"] = np.hypot(wide["bias_mc_se_correct"], wide["bias_mc_se_wrong"])
    wide["z"] = wide["bias_change"] / wide["bias_change_mc_se"]
    return wide.sort_values(["estimand", "estimator", "q_model", "regime", "event", "time"])



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    """Report whichever rungs have completed.

        python -m sim.validate --dir results/validation --times 0.25,0.5,1.0

    Exits non-zero if a completed rung fails its gate, so this can be used as a
    precondition for the studies rather than only as a report.
    """
    import argparse

    ap = argparse.ArgumentParser(prog="sim.validate", description=main.__doc__)
    ap.add_argument("--dir", type=Path, default=Path("results/validation"))
    ap.add_argument("--times", default="0.25,0.5,1.0")
    ap.add_argument("--truth-mc", type=int, default=4_000_000)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args(argv)

    times = [float(x) for x in args.times.split(",")]
    pd.set_option("display.width", 230, "display.max_columns", 40)
    failures: List[str] = []

    r_truth = args.dir / "truth_r.parquet"
    if r_truth.exists():
        print("=== rung 1: truth agreement vs trueCSH() ===")
        r1 = rung1_truth_agreement(r_truth, times, n_mc=args.truth_mc, tol=args.tol)
        summary = (
            r1.groupby("scenario")
            .agg(max_abs_diff=("gap_unweighted_vs_r", lambda s: s.abs().max()),
                 targets_differ_by=("gap_target_definitions", lambda s: s.abs().max()),
                 all_agree=("agrees", "all"))
            .reset_index()
        )
        print(summary.to_string(index=False))
        if not summary["all_agree"].all():
            failures.append("rung 1: closed form disagrees with trueCSH()")
    else:
        print("rung 1: not run (missing truth_r.parquet)")

    est = args.dir / "adjcuminc_est.parquet"
    if est.exists():
        print("\n=== rung 2: their estimators, their DGP ===")
        r2 = rung2_reference_pattern(est, times, n_mc=args.truth_mc)
        print(which_target(r2).to_string(index=False))
        # gate: the doubly robust estimator must be the one that survives
        # misspecification of either model
        for sc, lim in (("s1", 0.01), ("s2", 0.01), ("s3", 0.01)):
            dr = r2[(r2.scenario == sc) & (r2.estimator == "adjDR")]
            worst = dr["bias_vs_realised"].abs().max()
            ok = worst < lim
            print(f"  {sc}: max |bias| of adjDR = {worst:.4f}  {'OK' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"rung 2: adjDR biased in {sc} ({worst:.4f})")
    else:
        print("\nrung 2: not run (missing adjcuminc_est.parquet)")

    ours = sorted(args.dir.glob("rung3_ours*.parquet"))
    theirs = args.dir / "rung3" / "their_estimates.parquet"
    if ours and theirs.exists():
        print("\n=== rung 3: our estimators vs theirs, identical data and nuisances ===")
        o = pd.concat([pd.read_parquet(p) for p in ours], ignore_index=True)
        cmp = rung3_compare(o, theirs, times, n_mc=args.truth_mc)
        cmp.to_parquet(args.dir / "rung3_comparison.parquet", index=False)
        pairs = [("gcomp", "adjOR"), ("ipw", "adjIPW"), ("aipw", "adjDR")]
        for mine, hers in pairs:
            sub = cmp[cmp.estimator.isin([mine, hers])]
            w = sub.pivot_table(index=["scenario", "arm", "event", "time"],
                                columns="estimator", values="mean_est")
            if mine in w and hers in w:
                d = (w[mine] - w[hers]).abs()
                print(f"  {mine:6s} vs {hers:7s}: max |difference in mean estimate| = {d.max():.4f}")
    else:
        print("\nrung 3: not run")

    rung4 = args.dir / "rung4"
    # Glob rather than name one cell. This checked for `R4_info_Gbad`, a name
    # that stopped existing when the rung became a 2x2x2 and the cells were
    # renamed `R4_{none,info}_Q{ok,bad}_G{ok,bad}` -- so a rung 4 that *had* run
    # was silently reported as "not run". `rung4_censoring_effect` already parses
    # the current names; only this gate was left behind.
    if any(rung4.glob("R4_*/meta.json")):
        print("\n=== rung 4: informative censoring ===")
        try:
            r4 = rung4_censoring_effect(rung4, n_mc=args.truth_mc)
            # `q_model` is a pivot key, so without it every row appears twice --
            # once for Q correct, once for Q wrong -- and they are
            # indistinguishable. It is also the column the rung turns on: the
            # informative signal lives in the Q-wrong row, where double
            # robustness is switched off and G has to carry the estimate alone.
            print(r4[[c for c in RUNG4_COLUMNS if c in r4.columns]]
                  .round(4).to_string(index=False))
        except FileNotFoundError:
            print("  (incomplete)")
    else:
        print("\nrung 4: not run")

    print()
    if failures:
        for f in failures:
            print(f"GATE FAILED -- {f}")
        return 1
    print("All completed rungs pass.")
    return 0


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(main())
