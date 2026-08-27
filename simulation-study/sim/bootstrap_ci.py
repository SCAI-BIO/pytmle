"""Bootstrap resampling for Study B, keeping the diagnostics PyTMLE's loop discards.

PyTMLE's `bootstrap_tmle_loop` returns summarised percentile quantiles and throws
the draws away, so it cannot answer the questions this study asks. This module
runs the same resampling -- calling `pytmle.tmle_update.tmle_update` with the
arguments `single_boot` uses -- and keeps everything: the raw draws, so several
interval constructions can be compared on *identical* resamples, and the
convergence information, which PyTMLE's own path silently drops in three separate
places.

**The RNG is ours.** `bootstrap.standard_bootstrap` draws from the unseeded
legacy global (`np.random.choice`), so PyTMLE's bootstrap intervals are not
reproducible run to run. Here the resample indices are generated from a seeded
`Generator` and passed in explicitly, which makes the study reproducible and lets
every interval construction see the same draws. The difference is in *which*
resamples are drawn, never in what is done with them.

Three failure modes are counted separately, because they have different causes
and different remedies:

    mode 1  the update loop exhausts its budget and returns has_converged=False.
            `single_boot` unpacks it as `_` and passes verbose=0, which also
            suppresses the warning. Completely silent.
    mode 2  a per-(Event, Time) `Converged` flag is False. `single_boot` drops
            those rows, so the effective B differs per target without saying so.
    mode 3  the resample raises. `bootstrap_tmle_loop` calls `f.result()` with no
            guard, so one bad resample takes down the whole fit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = ["BootstrapDraws", "bootstrap_draws", "intervals_from_draws",
           "wald_interval", "log_wald_interval", "logit_wald_interval",
           "atanh_wald_interval"]


@dataclass
class BootstrapDraws:
    """Raw draws plus the diagnostics needed to interpret them."""

    #: long frame: one row per (resample, type, Event, Time, Group)
    draws: pd.DataFrame
    n_requested: int
    mode1_nonconverged: int = 0          # loop exhausted its budget
    mode3_errors: int = 0                # resample raised
    first_error: Optional[str] = None
    steps: List[int] = field(default_factory=list)

    @property
    def n_usable(self) -> int:
        return int(self.draws["boot"].nunique()) if len(self.draws) else 0


def _one_resample(args) -> Tuple[Optional[pd.DataFrame], bool, int, Optional[str]]:
    """One resample. Returns (draws, loop_converged, steps, error)."""
    (initial_estimates, event_times, event_indicator, target_times,
     target_events, key_1, key_0, idx, kwargs) = args
    from pytmle.predict_ate import ate_diff, ate_ratio, get_counterfactual_risks
    from pytmle.tmle_update import tmle_update

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boot_ie = {k: initial_estimates[k][idx] for k in initial_estimates}
            upd, _, converged, steps = tmle_update(
                initial_estimates=boot_ie,
                event_times=event_times[idx],
                event_indicator=event_indicator[idx],
                target_times=list(target_times),
                target_events=list(target_events),
                verbose=0,
                **kwargs,
            )
            risks = get_counterfactual_risks(upd, key_1=key_1, key_0=key_0)[
                ["Event", "Time", "Group", "Pt Est", "Converged"]].copy()
            risks["type"] = "risk"
            rr = ate_ratio(upd, key_1=key_1, key_0=key_0)[
                ["Event", "Time", "Pt Est", "Converged"]].copy()
            rr["type"] = "rr"; rr["Group"] = -1
            rd = ate_diff(upd, key_1=key_1, key_0=key_0)[
                ["Event", "Time", "Pt Est", "Converged"]].copy()
            rd["type"] = "rd"; rd["Group"] = -1
        out = pd.concat([risks, rr, rd], ignore_index=True)
        return out, bool(converged), int(steps), None
    except Exception as exc:
        return None, False, -1, f"{type(exc).__name__}: {exc}"


def bootstrap_draws(
    initial_estimates: Dict,
    event_times: np.ndarray,
    event_indicator: np.ndarray,
    target_times: Sequence[float],
    target_events: Sequence[int],
    n_bootstrap: int = 100,
    rng: Optional[np.random.Generator] = None,
    key_1: int = 1,
    key_0: int = 0,
    stratify_by_event: bool = False,
    n_jobs: int = 1,
    **kwargs,
) -> BootstrapDraws:
    """`n_bootstrap` resamples of the second stage, keeping draws and diagnostics.

    Only the second stage is resampled and the nuisances are never refit, exactly
    as PyTMLE's bootstrap does and for the reason it cites (Coyle & van der
    Laan 2018; Tran et al. 2023). Under an `oracle` nuisance arm that is the
    correct bootstrap; under an estimated arm it omits a genuine source of
    variability, and measuring the size of that omission is one of this study's
    questions rather than an oversight to work around.
    """
    rng = rng or np.random.default_rng(0)
    n = len(event_indicator)

    idxs = []
    for _ in range(n_bootstrap):
        if stratify_by_event:
            parts = [rng.choice(np.where(event_indicator == ev)[0],
                                size=int((event_indicator == ev).sum()), replace=True)
                     for ev in np.unique(event_indicator)]
            idxs.append(np.concatenate(parts))
        else:
            idxs.append(rng.choice(n, size=n, replace=True))

    tasks = [(initial_estimates, event_times, event_indicator, list(target_times),
              list(target_events), key_1, key_0, i, kwargs) for i in idxs]

    if n_jobs and n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(_one_resample, tasks))
    else:
        results = [_one_resample(t) for t in tasks]

    frames, out = [], BootstrapDraws(draws=pd.DataFrame(), n_requested=n_bootstrap)
    for b, (df, conv, steps, err) in enumerate(results):
        if err is not None:
            out.mode3_errors += 1
            if out.first_error is None:
                out.first_error = err
            continue
        out.steps.append(steps)
        if not conv:
            out.mode1_nonconverged += 1
        df = df.copy()
        df["boot"] = b
        # Tagged per draw, not merely counted: attributing a coverage shortfall
        # to a failure mode means recomputing the interval with and without the
        # affected draws, which needs the flag to survive to the draw.
        df["loop_converged"] = bool(conv)
        df["steps"] = int(steps)
        frames.append(df)
    out.draws = (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["Event", "Time", "Group", "Pt Est", "Converged", "type", "boot",
                 "loop_converged", "steps"]))
    return out


# ---------------------------------------------------------------------------
# interval constructions, all from the same draws
# ---------------------------------------------------------------------------


def _bca_acceleration(ic: Optional[np.ndarray]) -> float:
    """Acceleration from the empirical influence function.

    The textbook BCa acceleration is a jackknife over observations, which here
    would mean `n` extra second-stage fits per replicate -- more expensive than
    the bootstrap it is correcting. For a smooth functional the jackknife
    influence values are asymptotically the influence function, which PyTMLE has
    already computed, so

        a = (1/6) * sum(L^3) / (sum(L^2))^(3/2)

    gives the same quantity for free. This is the standard empirical-influence
    form of BCa, not an approximation invented here.
    """
    if ic is None or len(ic) < 3:
        return 0.0
    L = np.asarray(ic, dtype=float) - float(np.mean(ic))
    denom = float(np.sum(L ** 2)) ** 1.5
    return float(np.sum(L ** 3) / (6.0 * denom)) if denom > 0 else 0.0


def intervals_from_draws(
    draws: np.ndarray,
    point: float,
    alpha: float = 0.05,
    ic: Optional[np.ndarray] = None,
) -> Dict[str, Tuple[float, float]]:
    """Percentile, basic and BCa intervals from one set of draws.

    All three see identical resamples, so a difference between them isolates the
    *interval construction* rather than the resampling.
    """
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        nan = (np.nan, np.nan)
        return {"percentile": nan, "basic": nan, "bca": nan}

    lo_q, hi_q = alpha / 2, 1 - alpha / 2
    pct = (float(np.quantile(d, lo_q)), float(np.quantile(d, hi_q)))
    # basic (reverse percentile): reflects the draws about the point estimate,
    # which corrects bias in the opposite direction to the percentile interval
    basic = (float(2 * point - np.quantile(d, hi_q)),
             float(2 * point - np.quantile(d, lo_q)))

    frac = float(np.mean(d < point))
    if frac <= 0 or frac >= 1:
        bca = pct   # z0 undefined; fall back rather than emit an infinite bound
    else:
        z0 = float(norm.ppf(frac))
        a = _bca_acceleration(ic)
        zl, zh = norm.ppf(lo_q), norm.ppf(hi_q)
        a1 = norm.cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl)))
        a2 = norm.cdf(z0 + (z0 + zh) / (1 - a * (z0 + zh)))
        if not (np.isfinite(a1) and np.isfinite(a2)) or a1 >= a2:
            bca = pct
        else:
            bca = (float(np.quantile(d, a1)), float(np.quantile(d, a2)))
    return {"percentile": pct, "basic": basic, "bca": bca}


def wald_interval(point: float, se: float, alpha: float = 0.05) -> Tuple[float, float]:
    z = norm.ppf(1 - alpha / 2)
    return point - z * se, point + z * se


def log_wald_interval(point: float, se: float, alpha: float = 0.05) -> Tuple[float, float]:
    """`exp(log R +- z * SE/R)` -- the Wald interval built on the log scale.

    PyTMLE's risk-ratio SE is a delta-method approximation on the *untransformed*
    ratio, so its symmetric interval can cross zero for a small risk and is
    skewed on the wrong scale. If moving to the log scale restores coverage, the
    finding is not "the asymptotics fail" but "the interval is built on the wrong
    scale", which is directly actionable.
    """
    if not (point > 0 and np.isfinite(se) and se >= 0):
        return np.nan, np.nan
    z = norm.ppf(1 - alpha / 2)
    se_log = se / point
    return float(np.exp(np.log(point) - z * se_log)), float(np.exp(np.log(point) + z * se_log))


def logit_wald_interval(point: float, se: float, alpha: float = 0.05) -> Tuple[float, float]:
    """Wald on the logit scale, back-transformed -- for a risk in (0, 1).

    A cumulative incidence is bounded, and near the lower boundary its sampling
    distribution is skewed and cannot be symmetric. A symmetric interval on the
    natural scale responds by putting its lower bound below zero, which is not a
    possible value for a CIF; the logit interval cannot leave (0, 1) at all.

    Delta method: ``Var(logit F) = Var(F) / (F(1-F))^2``. If moving to this scale
    restores coverage at the rare-event levels, the finding is "the interval is
    built on the wrong scale" rather than "the asymptotics fail" -- and the remedy
    is a one-line change in `predict_ate`, not a bootstrap.
    """
    if not (np.isfinite(point) and np.isfinite(se) and se >= 0 and 0.0 < point < 1.0):
        return np.nan, np.nan
    z = norm.ppf(1 - alpha / 2)
    se_logit = se / (point * (1.0 - point))
    eta = np.log(point / (1.0 - point))
    lo, hi = eta - z * se_logit, eta + z * se_logit
    return float(_expit_scalar(lo)), float(_expit_scalar(hi))


def atanh_wald_interval(point: float, se: float, alpha: float = 0.05) -> Tuple[float, float]:
    """Wald on the Fisher-z scale, back-transformed -- for a risk difference.

    The RD lives in (-1, 1), and the same boundary argument applies at both ends:
    where one arm's risk is near zero the RD is squeezed against its own bound and
    the sampling distribution is skewed. ``atanh`` is the natural variance-
    stabilising map for that support, with ``Var(atanh D) = Var(D) / (1 - D^2)^2``,
    and the back-transform is confined to (-1, 1) by construction.
    """
    if not (np.isfinite(point) and np.isfinite(se) and se >= 0 and -1.0 < point < 1.0):
        return np.nan, np.nan
    z = norm.ppf(1 - alpha / 2)
    se_t = se / (1.0 - point**2)
    eta = np.arctanh(point)
    return float(np.tanh(eta - z * se_t)), float(np.tanh(eta + z * se_t))


def _expit_scalar(x: float) -> float:
    """Overflow-safe logistic, for bounds that can land far out on the logit scale."""
    if x >= 0:
        return float(1.0 / (1.0 + np.exp(-x)))
    e = np.exp(x)
    return float(e / (1.0 + e))
