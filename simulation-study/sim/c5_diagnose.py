"""Where does C5's TMLE bias come from? A decomposition, not a guess.

Cell C5 is Q wrong, pi correct, G correct. Both papers say the substitution
estimator is consistent there, and the algebra says why: the second-order
remainder for this parameter is an *exact product*,

    R2 = int E_L[ S0(t-) c_{j,l,t}(Q) (lam_l - lam_0l) (1 - (pi0/pi)(S0^c/S^c)) ] dt

so it vanishes for any lambda, however wrong, once g is correct. Chen et al.
report exactly this in their Section 7.1. Study A measures a bias of ~0.028 that
does not shrink between n = 250 and n = 1000, in PyTMLE *and* in concrete.

So something in the implemented procedure does not match the theory, and there
are two families of explanation. This module separates them with one
measurement, evaluating the efficient influence curve **at the final targeted
estimates**:

    A. the score equation is not actually solved. Then
       psi_tmle + Pn D*(Q*) lands on the truth, and the leftover Pn D*(Q*) *is*
       the bias.
    B. the score equation is solved (Pn D*(Q*) ~ 0) and the bias is a genuine
       remainder -- meaning the plug-in that gets reported and the influence
       curve that gets solved do not correspond to the same functional.

These predict opposite things about one number, so the measurement decides.

    python -m sim.c5_diagnose --n 500 --reps 40
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .dgp import get_config, sample
from .nuisance import Spec, build
from .runner import target_times_for
from .truth import closed_form

__all__ = ["diagnose_replicate", "run"]

#: C5: outcome model wrong, propensity and censoring both correct.
SPEC_C5 = Spec(Q="wrong", pi="correct", G="correct")


def _pn_eic(updated: Dict, event_times, event_indicator, target_events,
            target_times) -> pd.DataFrame:
    """Pn D* per (arm, event, time) at whatever estimates are handed in."""
    from pytmle.get_influence_curve import get_eic

    ue = get_eic(estimates=updated, event_times=event_times,
                 event_indicator=event_indicator, g_comp=True)
    rows = []
    for arm, est in ue.items():
        ic = est.ic
        m = ic.groupby(["Event", "Time"])["IC"].agg(["mean", "std", "size"])
        g = est.g_comp_est.set_index(["Event", "Time"])["Risk"]
        for (ev, t), r in m.iterrows():
            if ev not in target_events or not np.any(np.isclose(t, target_times)):
                continue
            se = float(r["std"]) / np.sqrt(r["size"])
            rows.append({
                "arm": arm, "event": int(ev), "time": float(t),
                "plugin": float(g.loc[(ev, t)]),
                "pn_eic": float(r["mean"]),
                # the loop's own stopping threshold, so "solved" is judged by
                # the criterion the implementation actually uses
                "threshold": float(np.sqrt(np.mean(
                    ic.loc[(ic.Event == ev) & np.isclose(ic.Time, t), "IC"] ** 2))
                    / (np.sqrt(r["size"]) * np.log(r["size"]))),
            })
    return pd.DataFrame(rows)


def _consistency_defect(est, taus) -> List[Dict]:
    """Do the implemented S and F describe the same distribution?

    The exact second-order structure needs the plug-in functional and the clever
    covariate to be the *same* discretisation. Both implementations build the
    survival multiplicatively,

        S(t) = exp(-sum_s dLambda(s))                (continuous convention)

    but the cumulative incidence additively,

        F_j(t) = sum_{s<=t} S(s-) dLambda_j(s)        (discrete convention)

    Each convention is internally consistent -- `sum_j F_j + S = 1` exactly --
    but *mixing* them is not, and the defect is

        sum_s S(s-) [1 - exp(-dLambda(s)) - dLambda(s)]  ~  -(1/2) sum_s S(s-) dLambda(s)^2

    which is second order in the hazard increments. That is negligible while the
    increments are small, and the targeting step is exactly what makes them not
    small: fixing a badly wrong Q multiplies the hazard by exp(eps * h) with h
    carrying the 1/(pi * G) factor, over dozens of steps.

    So this reports `sum_j F_j(tau) + S(tau) - 1` and the size of the hazard
    increments, before and after targeting.
    """
    times = est.times
    haz = est.hazards                       # (n, K, J)
    surv = est.event_free_survival_function  # (n, K)
    lag = np.concatenate([np.ones((surv.shape[0], 1)), surv[:, :-1]], axis=1)
    cif = np.cumsum(lag[:, :, None] * haz, axis=1)   # (n, K, J) plug-in CIF
    rows = []
    for tau in taus:
        k = int(np.searchsorted(times, tau, side="right") - 1)
        defect = cif[:, k, :].sum(axis=1) + surv[:, k] - 1.0
        rows.append({
            "time": float(tau),
            "mean_defect": float(defect.mean()),
            "max_abs_defect": float(np.abs(defect).max()),
            "mean_dhaz": float(haz[:, :k + 1, :].sum(axis=-1).mean()),
            "max_dhaz": float(haz[:, :k + 1, :].sum(axis=-1).max()),
            "sum_dhaz_sq": float((haz[:, :k + 1, :].sum(axis=-1) ** 2).sum(axis=1).mean()),
        })
    return rows


def diagnose_replicate(args) -> Optional[pd.DataFrame]:
    """One replicate: plug-in and residual score, before and after targeting."""
    p, taus, seed, n, min_nuisance, spec, eps, max_upd = args
    from pytmle import PyTMLE
    from pytmle.estimates import UpdatedEstimates

    try:
        sm = sample(n, p, np.random.default_rng(seed))
        ie = build(sm, spec)
        events = list(range(1, p.n_causes + 1))

        model = PyTMLE(sm.df, target_times=list(taus), initial_estimates=ie,
                       g_comp=True, evalues_benchmark=False, verbose=0)
        model.fit(min_nuisance=min_nuisance, max_updates=max_upd,
                  one_step_eps=eps)
        upd = model._updated_estimates
        conv = bool(getattr(model, "_tmle_converged", True))

        # --- after targeting: is the score equation solved? -----------------
        post = _pn_eic({k: v for k, v in upd.items()}, sm.event_times,
                       sm.event_indicator, events, taus)
        post["stage"] = "targeted"

        # --- before targeting: the one-step's correction --------------------
        init = {k: UpdatedEstimates.from_initial_estimates(
                    ie[k], target_events=events, target_times=list(taus),
                    min_nuisance=min_nuisance)
                for k in ie}
        pre = _pn_eic(init, sm.event_times, sm.event_indicator, events, taus)
        pre["stage"] = "initial"

        # --- internal consistency of the implemented (S, F) pair -------------
        defects = []
        for stage, ests in (("initial", init), ("targeted", upd)):
            for arm, est in ests.items():
                for r in _consistency_defect(est, taus):
                    defects.append({**r, "stage": stage, "arm": arm,
                                    "kind": "defect"})
        dd = pd.DataFrame(defects)

        out = pd.concat([pre, post, dd], ignore_index=True)
        out["rep"] = seed
        out["converged"] = conv
        out["n"] = n
        return out
    except Exception as exc:
        return pd.DataFrame([{"rep": seed, "n": n, "stage": "error",
                              "error": f"{type(exc).__name__}: {exc}"}])


def run(n: int = 500, reps: int = 40, seed: int = 20250301,
        config: str = "threshold", min_nuisance: float = 0.01,
        n_jobs: int = 8, spec: Spec = SPEC_C5, eps: float = 0.1,
        max_upd: int = 200) -> pd.DataFrame:
    p = get_config(config)
    taus = target_times_for(p)
    tasks = [(p, taus, seed + 1000 * i, n, min_nuisance, spec, eps, max_upd)
             for i in range(reps)]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        got = [r for r in ex.map(diagnose_replicate, tasks) if r is not None]
    raw = pd.concat(got, ignore_index=True)

    tr = closed_form(p, taus, n_mc=4_000_000)
    return raw, tr, taus


def summarise(raw: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    """Contrast the plug-in, the re-corrected plug-in, and the truth, on the RD."""
    d = raw[raw["stage"].isin(["initial", "targeted"])].copy()
    # risk difference: arm 1 minus arm 0, per replicate
    w = d.pivot_table(index=["rep", "stage", "event", "time"], columns="arm",
                      values=["plugin", "pn_eic", "threshold"])
    rd = pd.DataFrame({
        "plugin": w[("plugin", 1)] - w[("plugin", 0)],
        "pn_eic": w[("pn_eic", 1)] - w[("pn_eic", 0)],
        "thr": np.sqrt(w[("threshold", 1)] ** 2 + w[("threshold", 0)] ** 2),
    }).reset_index()
    rd["corrected"] = rd["plugin"] + rd["pn_eic"]

    truth = tr.drop_duplicates(["event", "time"]).set_index(["event", "time"])["rd"]
    rd["truth"] = [truth.loc[(int(e), float(t))]
                   for e, t in zip(rd["event"], rd["time"])]

    def _agg(g):
        return pd.Series({
            "reps": len(g),
            "truth": float(g["truth"].iloc[0]),
            "bias_plugin": float((g["plugin"] - g["truth"]).mean()),
            "bias_corrected": float((g["corrected"] - g["truth"]).mean()),
            "mean_pn_eic": float(g["pn_eic"].mean()),
            "mean_abs_pn_eic": float(g["pn_eic"].abs().mean()),
            "mean_threshold": float(g["thr"].mean()),
            "frac_over_threshold": float((g["pn_eic"].abs() > g["thr"]).mean()),
            "mc_se": float((g["plugin"] - g["truth"]).std(ddof=1) / np.sqrt(len(g))),
        })

    return (rd.groupby(["stage", "event", "time"])
              .apply(_agg, include_groups=False).reset_index())


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="sim.c5_diagnose", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--config", default="threshold")
    ap.add_argument("--g", default="correct", choices=["correct", "oracle"],
                    help="how pi and G are supplied. `oracle` plugs in the true\n"
                         "propensity and censoring survival, which makes the\n"
                         "second-order remainder exactly zero in theory -- so a\n"
                         "bias that survives it is not g-estimation error.")
    ap.add_argument("--one-step-eps", type=float, default=0.1,
                    help="step size dx along the universal least favourable\n"
                         "submodel. The papers call it `a prespecified small\n"
                         "step size`; both packages default to 0.1.")
    ap.add_argument("--max-updates", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    spec = Spec(Q="wrong", pi=a.g, G=a.g)
    raw, tr, taus = run(n=a.n, reps=a.reps, config=a.config,
                        n_jobs=a.n_jobs, spec=spec,
                        eps=a.one_step_eps, max_upd=a.max_updates)
    err = raw[raw["stage"] == "error"]
    if len(err):
        print(f"[{len(err)} replicate failures] "
              f"{err['error'].iloc[0][:120]}")
    s = summarise(raw, tr)
    print(f"\nC5 decomposition, n = {a.n}, g = {a.g}, eps = {a.one_step_eps}, cause 1\n")
    print(s.to_string(index=False))

    dd = raw[raw.get("kind", pd.Series(dtype=object)).eq("defect")]
    if len(dd):
        print("\nInternal consistency of the implemented (S, F) pair: "
              "sum_j F_j(tau) + S(tau) - 1\n")
        agg = (dd.groupby(["stage", "time"])
                 .agg(mean_defect=("mean_defect", "mean"),
                      max_abs_defect=("max_abs_defect", "max"),
                      mean_cum_haz=("mean_dhaz", "mean"),
                      max_dhaz=("max_dhaz", "max"),
                      mean_sum_dhaz_sq=("sum_dhaz_sq", "mean"))
                 .reset_index())
        print(agg.to_string(index=False))
    if a.out:
        raw.to_parquet(a.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
