"""Estimators, all computed from identical injected nuisances.

Because every estimator here starts from the same ``InitialEstimates``, any
difference between them is attributable to the estimator alone.

``tmle``    PyTMLE's targeted update
``gcomp``   the untargeted plug-in, free off the same fitted object
``aipw``    the one-step estimator, built from PyTMLE's *own* efficient
            influence curve so that TMLE-vs-one-step is exactly like-for-like
``ipw``     IPTW+IPCW-weighted Aalen-Johansen from the same pi and G

Every call is wrapped so that PyTMLE's documented failure modes -- the IC
overflow ``RuntimeError`` and the "survival reached zero" ``ValueError`` -- are
recorded as rows carrying an ``error`` rather than killing the replicate loop.
"""

from __future__ import annotations

import time
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from pytmle import InitialEstimates, PyTMLE
from pytmle.estimates import UpdatedEstimates
from pytmle.get_influence_curve import get_eic

__all__ = ["run_all", "COLUMNS"]

COLUMNS = [
    "estimator",
    "estimand",
    "event",
    "time",
    "group",
    "est",
    "se",
    "ci_lo",
    "ci_hi",
    "converged",
    "error",
    "seconds",
    "stage2_seconds",
]

#: Implementations whose second-stage runtime is comparable across packages.
#: ``stage2_seconds`` is the targeted-update step *only* -- initial estimates are
#: injected, so it excludes all nuisance fitting and is the like-for-like number
#: to compare against ``concrete``. ``seconds`` is the total including object
#: construction and prediction.
TIMED_IMPLEMENTATIONS = ("tmle", "concrete")


def _empty(estimator: str, error: str, seconds: float = float("nan")) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "estimator": estimator,
                "estimand": None,
                "event": np.nan,
                "time": np.nan,
                "group": np.nan,
                "est": np.nan,
                "se": np.nan,
                "ci_lo": np.nan,
                "ci_hi": np.nan,
                "converged": False,
                "error": error,
                "seconds": seconds,
                "stage2_seconds": np.nan,
            }
        ]
    )


def _tidy_risks(df: pd.DataFrame, estimator: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "estimator": estimator,
            "estimand": "risk",
            "event": df["Event"].to_numpy(),
            "time": df["Time"].to_numpy(),
            "group": df["Group"].to_numpy(),
            "est": df["Pt Est"].to_numpy(),
            "se": df.get("SE", pd.Series(np.nan, index=df.index)).to_numpy(),
            "ci_lo": df.get("CI_lower", pd.Series(np.nan, index=df.index)).to_numpy(),
            "ci_hi": df.get("CI_upper", pd.Series(np.nan, index=df.index)).to_numpy(),
            "converged": df.get("Converged", pd.Series(True, index=df.index)).to_numpy(),
        }
    )
    return out


def _tidy_contrast(df: pd.DataFrame, estimator: str, estimand: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "estimator": estimator,
            "estimand": estimand,
            "event": df["Event"].to_numpy(),
            "time": df["Time"].to_numpy(),
            "group": np.nan,
            "est": df["Pt Est"].to_numpy(),
            "se": df.get("SE", pd.Series(np.nan, index=df.index)).to_numpy(),
            "ci_lo": df.get("CI_lower", pd.Series(np.nan, index=df.index)).to_numpy(),
            "ci_hi": df.get("CI_upper", pd.Series(np.nan, index=df.index)).to_numpy(),
            "converged": df.get("Converged", pd.Series(True, index=df.index)).to_numpy(),
        }
    )


# ---------------------------------------------------------------------------
# TMLE + g-computation (one fit serves both)
# ---------------------------------------------------------------------------


def run_tmle(
    df: pd.DataFrame,
    initial_estimates: Dict[int, InitialEstimates],
    target_times: Sequence[float],
    min_nuisance: Optional[float] = 0.01,
    max_updates: int = 200,
    alpha: float = 0.05,
    key_1: int = 1,
    key_0: int = 0,
) -> pd.DataFrame:
    t0 = time.time()
    t_fit = float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = PyTMLE(
                df,
                target_times=list(target_times),
                initial_estimates=initial_estimates,
                g_comp=True,
                evalues_benchmark=False,
                key_1=key_1,
                key_0=key_0,
                verbose=0,
            )
            # Timed separately from construction and prediction: with initial
            # estimates injected, fit() *is* the second stage, which is the
            # quantity comparable against concrete's targeted update.
            t_fit0 = time.time()
            model.fit(min_nuisance=min_nuisance, max_updates=max_updates)
            t_fit = time.time() - t_fit0
    except (RuntimeError, ValueError) as exc:
        el = time.time() - t0
        return pd.concat(
            [_empty("tmle", f"{type(exc).__name__}: {exc}", el),
             _empty("gcomp", f"{type(exc).__name__}: {exc}", el)],
            ignore_index=True,
        )

    el = time.time() - t0
    parts = []
    for est_name, g_comp in (("tmle", False), ("gcomp", True)):
        parts.append(_tidy_risks(model.predict("risks", alpha=alpha, g_comp=g_comp), est_name))
        parts.append(_tidy_contrast(model.predict("rd", alpha=alpha, g_comp=g_comp), est_name, "rd"))
        parts.append(_tidy_contrast(model.predict("rr", alpha=alpha, g_comp=g_comp), est_name, "rr"))
    out = pd.concat(parts, ignore_index=True)
    out["error"] = None
    out["seconds"] = el
    out["stage2_seconds"] = np.nan
    is_tmle = out["estimator"] == "tmle"
    out.loc[is_tmle, "stage2_seconds"] = t_fit
    out.loc[out["estimator"] == "gcomp", "seconds"] = 0.0
    out.attrs["tmle_converged"] = bool(model.has_converged)
    out.attrs["tmle_steps"] = int(model.step_num)
    return out


# ---------------------------------------------------------------------------
# One-step / AIPW, from PyTMLE's own EIF
# ---------------------------------------------------------------------------


def run_aipw(
    df: pd.DataFrame,
    initial_estimates: Dict[int, InitialEstimates],
    target_times: Sequence[float],
    target_events: Sequence[int],
    min_nuisance: Optional[float] = 0.01,
    alpha: float = 0.05,
    key_1: int = 1,
    key_0: int = 0,
) -> pd.DataFrame:
    """psi_gcomp + Pn[EIF at the *initial* estimates], SE from the same IC.

    Reuses ``pytmle.get_influence_curve.get_eic`` rather than reimplementing the
    influence function, so TMLE and one-step share an identical EIF and any
    difference between them is purely the targeting step.
    """
    from scipy.stats import norm

    t0 = time.time()
    z = norm.ppf(1 - alpha / 2)
    try:
        ue = {
            k: UpdatedEstimates.from_initial_estimates(
                initial_estimates[k],
                target_events=list(target_events),
                target_times=list(target_times),
                min_nuisance=min_nuisance,
            )
            for k in (key_1, key_0)
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ue = get_eic(
                ue,
                event_times=df["event_time"].to_numpy(),
                event_indicator=df["event_indicator"].to_numpy(),
                g_comp=True,
            )
    except (RuntimeError, ValueError) as exc:
        return _empty("aipw", f"{type(exc).__name__}: {exc}", time.time() - t0)

    per_arm, ic_by_arm = {}, {}
    for k in (key_1, key_0):
        risk = ue[k].g_comp_est.set_index(["Event", "Time"])["Risk"]
        ic = ue[k].ic
        mean_ic = ic.groupby(["Event", "Time"])["IC"].mean()
        per_arm[k] = (risk + mean_ic).rename("est")
        ic_by_arm[k] = ic

    n = len(df)
    rows = []
    for k, grp in ((key_1, 1), (key_0, 0)):
        se = ic_by_arm[k].groupby(["Event", "Time"])["IC"].apply(
            lambda x: np.sqrt(np.mean(x**2) / len(x))
        )
        for (ev, tt), val in per_arm[k].items():
            s = float(se.loc[(ev, tt)])
            rows.append(
                dict(estimator="aipw", estimand="risk", event=ev, time=tt, group=grp,
                     est=float(val), se=s, ci_lo=val - z * s, ci_hi=val + z * s,
                     converged=True)
            )

    # contrasts, using the per-subject IC difference (matching PyTMLE's own
    # convention in predict_ate.ate_diff / ate_ratio)
    ic1 = ic_by_arm[key_1].set_index(["ID", "Event", "Time"])["IC"]
    ic0 = ic_by_arm[key_0].set_index(["ID", "Event", "Time"])["IC"]
    d_ic = (ic1 - ic0).reset_index()
    se_rd = d_ic.groupby(["Event", "Time"])["IC"].apply(
        lambda x: np.sqrt(np.mean(x**2) / len(x))
    )
    for (ev, tt), r1 in per_arm[key_1].items():
        r0 = float(per_arm[key_0].loc[(ev, tt)])
        rd = float(r1) - r0
        s = float(se_rd.loc[(ev, tt)])
        rows.append(
            dict(estimator="aipw", estimand="rd", event=ev, time=tt, group=np.nan,
                 est=rd, se=s, ci_lo=rd - z * s, ci_hi=rd + z * s, converged=True)
        )
        rr = float(r1) / r0 if r0 > 0 else np.nan
        a = ic1.loc[:, ev, tt].to_numpy()
        b = ic0.loc[:, ev, tt].to_numpy()
        s_rr = float(np.sqrt(np.mean((a / r0 - b * float(r1) / r0**2) ** 2) / n)) if r0 > 0 else np.nan
        rows.append(
            dict(estimator="aipw", estimand="rr", event=ev, time=tt, group=np.nan,
                 est=rr, se=s_rr, ci_lo=rr - z * s_rr, ci_hi=rr + z * s_rr, converged=True)
        )

    out = pd.DataFrame(rows)
    out["error"] = None
    out["seconds"] = time.time() - t0
    out["stage2_seconds"] = np.nan
    return out


# ---------------------------------------------------------------------------
# IPTW + IPCW weighted Aalen-Johansen
# ---------------------------------------------------------------------------


def run_ipw(
    df: pd.DataFrame,
    initial_estimates: Dict[int, InitialEstimates],
    target_times: Sequence[float],
    target_events: Sequence[int],
    min_nuisance: Optional[float] = 0.01,
    alpha: float = 0.05,
    key_1: int = 1,
    key_0: int = 0,
) -> pd.DataFrame:
    """Weighted Aalen-Johansen using only ``pi`` and ``G``.

    Weight for subject i at time t is ``1{A = a} / (pi_a(W) * G(t- | a, W))``,
    truncated at ``min_nuisance`` exactly as PyTMLE truncates its clever
    covariate denominator. Depends on no part of ``Q``, which is what makes it
    the mirror image of ``gcomp`` in the double-robustness table.
    """
    from scipy.stats import norm

    t0 = time.time()
    z = norm.ppf(1 - alpha / 2)
    times = initial_estimates[key_1].times
    obs_t = df["event_time"].to_numpy()
    obs_d = df["event_indicator"].to_numpy()
    n = len(df)
    if min_nuisance is None:
        min_nuisance = 5 / np.sqrt(n) / np.log(n)

    # index of each subject's observed time on the shared grid
    idx = np.searchsorted(times, obs_t)
    idx = np.clip(idx, 0, len(times) - 1)

    rows = []
    per_arm_infl: Dict[int, Dict[tuple, np.ndarray]] = {key_1: {}, key_0: {}}
    per_arm_est: Dict[int, Dict[tuple, float]] = {key_1: {}, key_0: {}}

    for k, a in ((key_1, 1), (key_0, 0)):
        ie = initial_estimates[k]
        pi = ie.propensity_scores
        G = ie.censoring_survival_function
        lagG = np.column_stack([np.ones(n), G[:, :-1]])
        denom = np.maximum(pi[:, None] * lagG, min_nuisance)
        in_arm = (df["group"].to_numpy() == a).astype(float)

        for ev in target_events:
            for tau in target_times:
                # sum over subjects with an observed event of type ev at or before tau
                hit = (obs_d == ev) & (obs_t <= tau)
                w = np.zeros(n)
                w[hit] = in_arm[hit] / denom[hit, idx[hit]]
                est = float(w.mean())
                per_arm_est[k][(ev, tau)] = est
                per_arm_infl[k][(ev, tau)] = w - est

    for ev in target_events:
        for tau in target_times:
            for k, grp in ((key_1, 1), (key_0, 0)):
                e = per_arm_est[k][(ev, tau)]
                s = float(np.sqrt(np.mean(per_arm_infl[k][(ev, tau)] ** 2) / n))
                rows.append(
                    dict(estimator="ipw", estimand="risk", event=ev, time=tau, group=grp,
                         est=e, se=s, ci_lo=e - z * s, ci_hi=e + z * s, converged=True)
                )
            r1 = per_arm_est[key_1][(ev, tau)]
            r0 = per_arm_est[key_0][(ev, tau)]
            d = per_arm_infl[key_1][(ev, tau)] - per_arm_infl[key_0][(ev, tau)]
            rd = r1 - r0
            s = float(np.sqrt(np.mean(d**2) / n))
            rows.append(
                dict(estimator="ipw", estimand="rd", event=ev, time=tau, group=np.nan,
                     est=rd, se=s, ci_lo=rd - z * s, ci_hi=rd + z * s, converged=True)
            )
            rr = r1 / r0 if r0 > 0 else np.nan
            if r0 > 0:
                infl = per_arm_infl[key_1][(ev, tau)] / r0 - per_arm_infl[key_0][(ev, tau)] * r1 / r0**2
                s_rr = float(np.sqrt(np.mean(infl**2) / n))
            else:
                s_rr = np.nan
            rows.append(
                dict(estimator="ipw", estimand="rr", event=ev, time=tau, group=np.nan,
                     est=rr, se=s_rr, ci_lo=rr - z * s_rr, ci_hi=rr + z * s_rr, converged=True)
            )

    out = pd.DataFrame(rows)
    out["error"] = None
    out["seconds"] = time.time() - t0
    out["stage2_seconds"] = np.nan
    return out


# ---------------------------------------------------------------------------


def run_all(
    df: pd.DataFrame,
    initial_estimates: Dict[int, InitialEstimates],
    target_times: Sequence[float],
    target_events: Sequence[int],
    which: Sequence[str] = ("tmle", "gcomp", "aipw", "ipw"),
    min_nuisance: Optional[float] = 0.01,
    max_updates: int = 200,
    alpha: float = 0.05,
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    meta = {}
    if "tmle" in which or "gcomp" in which:
        res = run_tmle(df, initial_estimates, target_times,
                       min_nuisance=min_nuisance, max_updates=max_updates, alpha=alpha)
        meta = dict(res.attrs)
        res = res[res["estimator"].isin(which)]
        parts.append(res)
    if "aipw" in which:
        parts.append(run_aipw(df, initial_estimates, target_times, target_events,
                              min_nuisance=min_nuisance, alpha=alpha))
    if "ipw" in which:
        parts.append(run_ipw(df, initial_estimates, target_times, target_events,
                             min_nuisance=min_nuisance, alpha=alpha))
    out = pd.concat(parts, ignore_index=True)[COLUMNS]
    # PyTMLE's `Converged` flag arrives as a plain bool on targeted rows but can
    # come back float/NaN on g-computation rows, which leaves an object column
    # that pyarrow refuses to write ("Could not convert 1.0 with type float").
    # Normalising here, at the single exit point for estimator results, keeps
    # every shard writable and preserves "unknown" rather than coercing it away.
    out["converged"] = pd.array(
        [None if pd.isna(v) else bool(v) for v in out["converged"]], dtype="boolean"
    )
    for col in ("est", "se", "ci_lo", "ci_hi", "seconds", "stage2_seconds"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out.attrs.update(meta)
    return out
