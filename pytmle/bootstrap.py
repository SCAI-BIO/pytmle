from typing import Dict, List, Optional, Tuple
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from .tmle_update import tmle_update
from .predict_ate import get_counterfactual_risks, ate_ratio, ate_diff
from .estimates import InitialEstimates

#: Grouping that identifies one estimand: a per-arm risk, or a contrast where
#: ``Group`` is -1.
_TARGET_KEYS = ["type", "Event", "Time", "Group"]


def standard_bootstrap(event_indicator):
    return np.random.choice(
        len(event_indicator), size=len(event_indicator), replace=True
    )


def stratified_bootstrap(event_indicator):
    """
    Generate bootstrap samples stratified by event indicator.
    """
    sample_indices_all = []
    for ev in np.unique(event_indicator):
        indices = np.where(event_indicator == ev)[0]
        sample_indices = np.random.choice(indices, size=len(indices), replace=True)
        sample_indices_all.append(sample_indices)
    return np.concatenate(sample_indices_all)


def single_boot(
    initial_estimates,
    event_times,
    event_indicator,
    target_times,
    target_events,
    key_1,
    key_0,
    stratify_by_event,
    **kwargs,
):
    """
    Perform a single bootstrap sample and call tmle_update.

    As pointed out by Coyle & van der Laan (2018; https://link.springer.com/chapter/10.1007/978-3-319-65304-4_28)
    and Tran et al. (2023; https://www.degruyter.com/document/doi/10.1515/jci-2021-0067/html?srsltid=AfmBOopT0k3YNof6ON7IWkEv49nuaK_bqgd_bCL8GSyYvmUNBDoGavDG),
    only the second stage of TMLE should be bootstrapped, not the first stage
    """
    # Create a bootstrap sample of indices
    if stratify_by_event:
        sample_indices = stratified_bootstrap(event_indicator)
    else:
        sample_indices = standard_bootstrap(event_indicator)

    # Resample initial estimates, event times and event indicator;
    boot_initial_estimates = {}
    for k in initial_estimates.keys():
        boot_initial_estimates[k] = initial_estimates[k][sample_indices]
    boot_event_times = event_times[sample_indices]
    boot_event_indicator = event_indicator[sample_indices]
    # Call tmle_update
    updated_estimates, _, _, _ = tmle_update(
        initial_estimates=boot_initial_estimates,
        event_times=boot_event_times,
        event_indicator=boot_event_indicator,
        target_times=target_times,
        target_events=target_events,
        verbose=0,
        **kwargs,
    )
    cf_risks = get_counterfactual_risks(updated_estimates, key_1=key_1, key_0=key_0)[
        ["Event", "Time", "Group", "Pt Est", "Converged"]
    ]
    cf_risks["type"] = "risks"
    ate_ratios = ate_ratio(updated_estimates, key_1=key_1, key_0=key_0)[
        ["Event", "Time", "Pt Est", "Converged"]
    ]
    ate_ratios["type"] = "rr"
    ate_ratios["Group"] = -1
    ate_diffs = ate_diff(updated_estimates, key_1=key_1, key_0=key_0)[
        ["Event", "Time", "Pt Est", "Converged"]
    ]
    ate_diffs["type"] = "rd"
    ate_diffs["Group"] = -1
    result_df = pd.concat([cf_risks, ate_ratios, ate_diffs])
    # # keep only estimates that converged
    # result_df = result_df[result_df["Converged"]]
    return result_df


def _bca_acceleration(ic: Optional[np.ndarray]) -> float:
    """Acceleration constant from the empirical influence function.

    The textbook BCa acceleration is a jackknife over observations, which would
    mean ``n`` extra second-stage fits per bootstrap -- more expensive than the
    bootstrap it corrects. For a smooth functional the jackknife influence values
    are asymptotically the influence function, which the targeted fit has already
    produced, so

        a = (1/6) * sum(L^3) / (sum(L^2))^(3/2),   L = IC - mean(IC)

    gives the same quantity for free. This is the standard empirical-influence
    form of BCa, not an approximation invented here.

    Note that ``a`` is a *standardised third moment*: it is invariant to the
    scale of the influence curve, so a mis-calibrated IC variance does not bias
    it. It is, however, dominated by the tails, which is exactly where the IC is
    least stable under near-violation of positivity -- so ``a`` is high-variance
    in the regime BCa is most often reached for. The returned value is recorded
    per estimand so that this can be inspected rather than assumed.
    """
    if ic is None:
        return 0.0
    L = np.asarray(ic, dtype=float)
    L = L[np.isfinite(L)]
    if len(L) < 3:
        return 0.0
    L = L - L.mean()
    denom = float(np.sum(L**2)) ** 1.5
    if denom <= 0:
        return 0.0
    return float(np.sum(L**3) / (6.0 * denom))


def _target_influence(
    updated_estimates: Dict, key_1: int = 1, key_0: int = 0
) -> Dict[Tuple, np.ndarray]:
    """Per-subject influence values for every estimand, keyed like the draws.

    Keys match the ``(type, Event, Time, Group)`` grouping of the bootstrap
    draws, so the acceleration for each interval is computed from the influence
    function of *that* estimand:

    ``risks``  the arm's own influence curve
    ``rd``     ``IC_1 - IC_0``
    ``rr``     the delta-method influence function of ``r_1 / r_0``,
               ``IC_1 / r_0 - r_1 * IC_0 / r_0^2``

    Returns an empty mapping if the influence curves are unavailable, in which
    case every interval falls back to the percentile bootstrap.
    """
    out: Dict[Tuple, np.ndarray] = {}
    try:
        ic1 = updated_estimates[key_1].ic
        ic0 = updated_estimates[key_0].ic
        if ic1 is None or ic0 is None:
            return out
        idx = ["ID", "Event", "Time"]
        s1 = ic1.set_index(idx)["IC"].sort_index()
        s0 = ic0.set_index(idx)["IC"].sort_index()
        r1 = updated_estimates[key_1].predict_mean_risks().set_index(["Event", "Time"])
        r0 = updated_estimates[key_0].predict_mean_risks().set_index(["Event", "Time"])

        for grp, s in ((key_1, s1), (key_0, s0)):
            for (ev, t), g in s.groupby(level=["Event", "Time"]):
                out[("risks", int(ev), float(t), int(grp))] = g.to_numpy()

        diff = (s1 - s0).dropna()
        for (ev, t), g in diff.groupby(level=["Event", "Time"]):
            out[("rd", int(ev), float(t), -1)] = g.to_numpy()

        for (ev, t), g1 in s1.groupby(level=["Event", "Time"]):
            try:
                p1 = float(r1.loc[(ev, t), "Pt Est"])
                p0 = float(r0.loc[(ev, t), "Pt Est"])
            except Exception:
                continue
            if not np.isfinite(p0) or p0 == 0:
                continue
            g0 = s0.loc[(slice(None), ev, t)]
            out[("rr", int(ev), float(t), -1)] = (
                g1.to_numpy() / p0 - p1 * g0.to_numpy() / p0**2
            )
    except Exception as exc:  # diagnostics must never break the fit
        warnings.warn(
            f"Could not build influence values for the BCa acceleration ({exc}); "
            "falling back to percentile bootstrap intervals.",
            RuntimeWarning,
        )
        return {}
    return out


def _bca_interval(
    draws: np.ndarray,
    point: float,
    ic: Optional[np.ndarray],
    alpha: float = 0.05,
) -> Dict[str, float]:
    """One bias-corrected and accelerated interval, with its diagnostics.

    Falls back to the plain percentile interval, and says so, when the bias
    correction is undefined. That happens when *every* draw lies on one side of
    the point estimate, so the fraction below it is 0 or 1 and ``z0`` is
    infinite -- which is not a rare pathology: it is the normal state when the
    estimand sits near the boundary of its support, exactly where rare events
    push it. The interval is still valid (it is the percentile one), but it is
    **not** BCa, and ``ci_method`` records which was used.
    """
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    res = {
        "mean_bootstrap": float(np.mean(d)) if len(d) else np.nan,
        "n_draws": int(len(d)),
        "bca_z0": np.nan,
        "bca_a": np.nan,
        "ci_method": "bca",
    }
    if len(d) < 2:
        return {**res, "CI_lower": np.nan, "CI_upper": np.nan,
                "ci_method": "undefined"}

    pct = (float(np.quantile(d, lo_q)), float(np.quantile(d, hi_q)))

    frac = float(np.mean(d < point)) if np.isfinite(point) else np.nan
    if not np.isfinite(frac) or frac <= 0.0 or frac >= 1.0:
        return {**res, "CI_lower": pct[0], "CI_upper": pct[1],
                "ci_method": "percentile (z0 undefined)"}

    z0 = float(norm.ppf(frac))
    a = _bca_acceleration(ic)
    zl, zh = norm.ppf(lo_q), norm.ppf(hi_q)
    with np.errstate(divide="ignore", invalid="ignore"):
        a1 = norm.cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl)))
        a2 = norm.cdf(z0 + (z0 + zh) / (1 - a * (z0 + zh)))
    res["bca_z0"], res["bca_a"] = z0, a
    if not (np.isfinite(a1) and np.isfinite(a2)) or a1 >= a2:
        return {**res, "CI_lower": pct[0], "CI_upper": pct[1],
                "ci_method": "percentile (levels invalid)"}
    return {**res,
            "CI_lower": float(np.quantile(d, a1)),
            "CI_upper": float(np.quantile(d, a2))}


def bootstrap_intervals(
    draws: pd.DataFrame,
    updated_estimates: Optional[Dict] = None,
    alpha: float = 0.05,
    key_1: int = 1,
    key_0: int = 0,
    method: str = "bca",
) -> pd.DataFrame:
    """Confidence intervals from stored bootstrap draws.

    Separated from the resampling because BCa needs two things the resampling
    cannot see: the **point estimate** of the original fit, for the bias
    correction, and its **influence curve**, for the acceleration. Neither
    exists until the targeted update has run, and the update mutates the initial
    estimates in place -- so the draws have to be collected first, from the
    un-targeted estimates, and turned into intervals afterwards.

    Falls back to percentile intervals for every estimand when
    ``updated_estimates`` is not supplied, or per estimand when the BCa
    corrections are undefined. ``ci_method`` in the result says which was used
    for each row.
    """
    if method not in ("bca", "percentile"):
        raise ValueError(f"method must be 'bca' or 'percentile', got {method!r}")

    points: Dict[Tuple, float] = {}
    influence: Dict[Tuple, np.ndarray] = {}
    if method == "bca" and updated_estimates is not None:
        try:
            # `itertuples` renames "Pt Est" -- it is not a valid identifier --
            # so the columns are zipped explicitly instead.
            risks = get_counterfactual_risks(updated_estimates, key_1=key_1, key_0=key_0)
            for ev, t, grp, est in zip(
                risks["Event"], risks["Time"], risks["Group"], risks["Pt Est"]
            ):
                points[("risks", int(ev), float(t), int(grp))] = float(est)
            for typ, fn in (("rd", ate_diff), ("rr", ate_ratio)):
                tab = fn(updated_estimates, key_1=key_1, key_0=key_0)
                for ev, t, est in zip(tab["Event"], tab["Time"], tab["Pt Est"]):
                    points[(typ, int(ev), float(t), -1)] = float(est)
        except Exception as exc:
            warnings.warn(
                f"Could not read point estimates for the BCa bias correction "
                f"({exc}); falling back to percentile bootstrap intervals.",
                RuntimeWarning,
            )
            points = {}
        influence = _target_influence(updated_estimates, key_1=key_1, key_0=key_0)

    rows = []
    for keys, g in draws.groupby(_TARGET_KEYS, sort=True):
        key = (str(keys[0]), int(keys[1]), float(keys[2]), int(keys[3]))
        d = g["Pt Est"].to_numpy(dtype=float)
        if method == "percentile" or key not in points:
            dd = d[np.isfinite(d)]
            rows.append({
                **dict(zip(_TARGET_KEYS, keys)),
                "mean_bootstrap": float(np.mean(dd)) if len(dd) else np.nan,
                "CI_lower": float(np.quantile(dd, alpha / 2)) if len(dd) else np.nan,
                "CI_upper": float(np.quantile(dd, 1 - alpha / 2)) if len(dd) else np.nan,
                "n_draws": int(len(dd)),
                "bca_z0": np.nan, "bca_a": np.nan,
                "ci_method": "percentile" if method == "percentile"
                else "percentile (no point estimate)",
            })
            continue
        rows.append({**dict(zip(_TARGET_KEYS, keys)),
                     **_bca_interval(d, points[key], influence.get(key), alpha)})

    out = pd.DataFrame(rows)
    if len(out) and "ci_method" in out:
        fell = out["ci_method"].str.startswith("percentile").sum()
        if method == "bca" and fell:
            warnings.warn(
                f"{fell} of {len(out)} bootstrap intervals fell back to the "
                f"percentile construction because the BCa corrections were "
                f"undefined; see the 'ci_method' column. This is expected when "
                f"an estimand sits near the boundary of its support.",
                RuntimeWarning,
            )
    return out


def bootstrap_tmle_loop(
    initial_estimates: Dict[int, InitialEstimates],
    event_times: np.ndarray,
    event_indicator: np.ndarray,
    target_times: List[float],
    target_events: List[int],
    n_bootstrap: int = 100,
    n_jobs: int = -1,
    alpha: float = 0.05,
    key_1: int = 1,
    key_0: int = 0,
    stratify_by_event: bool = False,
    verbose: int = 2,
    **kwargs,
) -> Optional[pd.DataFrame]:
    """
    Perform parallel bootstrapping and call tmle_update on each sample.

    Parameters
    ----------
    initial_estimates: Dict[int, InitialEstimates]
        Initial estimates for each group.
    event_times: np.ndarray
        Array of event times.
    event_indicator: np.ndarray
        Array of event indicators.
    target_times: List[float]
        List of target times.
    target_events: List[int]
        List of target events.
    n_bootstrap: int
        Number of bootstrap samples.
    n_jobs: int
        Number of parallel jobs for bootstrapping.
    alpha: float
        Significance level for confidence intervals.
    key_1: int
        Key for group 1.
    key_0: int
        Key for group 0.
    stratify_by_event: bool
        Stratify bootstrapping by event indicator.
    verbose: int
        Verbosity level.
    kwargs
        Additional arguments to pass to tmle_update.

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame with bootstrapped confidence intervals.

    Notes
    -----
    Retained for backward compatibility. It collects the draws and immediately
    reduces them to **percentile** intervals, because at this point the targeted
    update has not run and the point estimate and influence curve that BCa needs
    do not exist. `PyTMLE` therefore calls `bootstrap_draws` and
    `bootstrap_intervals` separately instead.
    """
    draws = bootstrap_draws(
        initial_estimates,
        event_times=event_times,
        event_indicator=event_indicator,
        target_times=target_times,
        target_events=target_events,
        n_bootstrap=n_bootstrap,
        n_jobs=n_jobs,
        key_1=key_1,
        key_0=key_0,
        stratify_by_event=stratify_by_event,
        verbose=verbose,
        **kwargs,
    )
    return bootstrap_intervals(draws, None, alpha=alpha, key_1=key_1, key_0=key_0,
                               method="percentile")


def bootstrap_draws(
    initial_estimates: Dict[int, InitialEstimates],
    event_times: np.ndarray,
    event_indicator: np.ndarray,
    target_times: List[float],
    target_events: List[int],
    n_bootstrap: int = 100,
    n_jobs: int = -1,
    key_1: int = 1,
    key_0: int = 0,
    stratify_by_event: bool = False,
    verbose: int = 2,
    **kwargs,
) -> pd.DataFrame:
    """The raw second-stage bootstrap draws, one row per resample and estimand.

    Kept separate from interval construction so that the draws can be collected
    *before* the targeted update -- which mutates the initial estimates in place,
    so a bootstrap run afterwards would resample already-targeted values -- while
    the intervals are built *after* it, when the point estimate and influence
    curve BCa needs are available.
    """
    with ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
        futures = [
            executor.submit(
                single_boot,
                initial_estimates,
                event_times,
                event_indicator,
                target_times,
                target_events,
                key_1,
                key_0,
                stratify_by_event,
                **kwargs,
            )
            for _ in range(n_bootstrap)
        ]
        results = []
        if verbose >= 2:
            futures_iter = tqdm(
                as_completed(futures), total=n_bootstrap, desc="Bootstrapping"
            )
        else:
            futures_iter = as_completed(futures)
        for f in futures_iter:
            result = f.result()
            if result is not None:
                results.append(result)
    if not results:
        return pd.DataFrame(columns=_TARGET_KEYS + ["Pt Est", "Converged"])
    return pd.concat(results, ignore_index=True)
