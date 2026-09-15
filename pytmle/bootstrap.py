from typing import Dict, List, Optional, Tuple
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from .tmle_update import tmle_update
from .predict_ate import (
    BOOTSTRAP_SUFFIXES,
    get_counterfactual_risks,
    ate_ratio,
    ate_diff,
)
from .estimates import InitialEstimates

#: Grouping that identifies one estimand: a per-arm risk, or a contrast where
#: ``Group`` is -1.
_TARGET_KEYS = ["type", "Event", "Time", "Group"]

#: Columns of the long interval table, in order.
_INTERVAL_COLUMNS = _TARGET_KEYS + [
    "bootstrap_method",
    "mean_bootstrap",
    "n_draws",
    "CI_lower",
    "CI_upper",
    "bc_z0",
    "ci_method",
]


def _as_rng(rng=None) -> np.random.Generator:
    """A `Generator`, never the legacy global.

    Both resamplers used to call `np.random.choice`, i.e. the unseeded *global*
    RNG. `ProcessPoolExecutor` forks its workers, and a forked child inherits
    the parent's global state -- so every worker walked the identical stream and
    the `i`-th task of each worker drew the *same* resample. The number of
    distinct resamples was therefore `n_bootstrap / n_jobs`, not `n_bootstrap`:
    at the defaults, 25 rather than 100. Measured before the fix: 12 resamples
    across 6 workers produced 2 distinct draws, each repeated six times.

    Taking an explicit generator fixes that and makes the bootstrap reproducible
    when a seed is supplied.
    """
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def standard_bootstrap(event_indicator, rng=None):
    rng = _as_rng(rng)
    return rng.choice(
        len(event_indicator), size=len(event_indicator), replace=True
    )


def stratified_bootstrap(event_indicator, rng=None):
    """
    Generate bootstrap samples stratified by event indicator.
    """
    rng = _as_rng(rng)
    sample_indices_all = []
    for ev in np.unique(event_indicator):
        indices = np.where(event_indicator == ev)[0]
        sample_indices = rng.choice(indices, size=len(indices), replace=True)
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
    seed=None,
    **kwargs,
):
    """
    Perform a single bootstrap sample and call tmle_update.

    As pointed out by Coyle & van der Laan (2018; https://link.springer.com/chapter/10.1007/978-3-319-65304-4_28)
    and Tran et al. (2023; https://www.degruyter.com/document/doi/10.1515/jci-2021-0067/html?srsltid=AfmBOopT0k3YNof6ON7IWkEv49nuaK_bqgd_bCL8GSyYvmUNBDoGavDG),
    only the second stage of TMLE should be bootstrapped, not the first stage
    """
    # Create a bootstrap sample of indices. `seed` is per-resample and comes
    # from a SeedSequence spawned in the parent, so each task draws its own
    # independent resample rather than inheriting the parent's global RNG state
    # at fork -- see `_as_rng`.
    rng = _as_rng(seed)
    if stratify_by_event:
        sample_indices = stratified_bootstrap(event_indicator, rng)
    else:
        sample_indices = standard_bootstrap(event_indicator, rng)

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


def _percentile_interval(draws: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    """The plain percentile interval: fixed quantiles of the draws.

    The default, and the reason is measured rather than conventional. Because it
    reads *fixed* central quantiles it inherits none of the estimation error that
    a data-dependent level carries, and under near-violation of positivity -- the
    one regime where a bootstrap earns its cost here -- it covers at 0.98 where
    the bias-corrected interval covers at 0.88, at a width ratio of 1.05. See
    `simulation-study/STUDY_B.md`.
    """
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return {"mean_bootstrap": float(np.mean(d)) if len(d) else np.nan,
                "CI_lower": np.nan, "CI_upper": np.nan,
                "n_draws": int(len(d)), "bc_z0": np.nan,
                "ci_method": "undefined"}
    return {"mean_bootstrap": float(np.mean(d)),
            "CI_lower": float(np.quantile(d, alpha / 2.0)),
            "CI_upper": float(np.quantile(d, 1.0 - alpha / 2.0)),
            "n_draws": int(len(d)), "bc_z0": np.nan,
            "ci_method": "percentile"}


def _bc_interval(draws: np.ndarray, point: float,
                 alpha: float = 0.05) -> Dict[str, float]:
    """One bias-corrected interval, with its diagnostic.

    Shifts the quantile levels by twice the median-bias correction,

        z0 = Phi^-1( mean(draw < point) ),
        a1 = Phi(2 z0 + z_{alpha/2}),   a2 = Phi(2 z0 + z_{1-alpha/2}),

    and reads the draws there. There is deliberately no acceleration term: it was
    measured on this DGP at an order of magnitude below the sampling error in
    `z0`, so it moved the levels by under 1 % while making them depend on the
    influence curve's third moment.

    Falls back to the percentile interval, and says so, when the bias correction
    is undefined: every draw on one side of the point estimate makes the fraction
    0 or 1 and `z0` infinite. That is not a rare pathology -- it is the normal
    state when the estimand sits near the boundary of its support, and it fired
    on 56 % of replicates in the rare-event arm of the measurements behind this
    choice. The interval returned is still valid, but it is not BC, and
    `ci_method` records which was used.
    """
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    pct = _percentile_interval(d, alpha)
    if pct["ci_method"] == "undefined":
        return pct

    frac = float(np.mean(d < point)) if np.isfinite(point) else np.nan
    if not np.isfinite(frac) or frac <= 0.0 or frac >= 1.0:
        return {**pct, "ci_method": "percentile (z0 undefined)"}

    z0 = float(norm.ppf(frac))
    zl, zh = norm.ppf(alpha / 2.0), norm.ppf(1.0 - alpha / 2.0)
    a1, a2 = norm.cdf(2 * z0 + zl), norm.cdf(2 * z0 + zh)
    if not (np.isfinite(a1) and np.isfinite(a2)) or a1 >= a2:
        return {**pct, "bc_z0": z0, "ci_method": "percentile (levels invalid)"}
    return {**pct, "bc_z0": z0, "ci_method": "bc",
            "CI_lower": float(np.quantile(d, a1)),
            "CI_upper": float(np.quantile(d, a2))}


#: The interval constructions this package offers, in the order it prefers them.
#: Both are always computed; the name selects one only where a single pair of
#: bounds has to be drawn, i.e. in the plotting functions.
BOOTSTRAP_METHODS = tuple(BOOTSTRAP_SUFFIXES)


def _point_estimates(updated_estimates: Dict, key_1: int = 1,
                     key_0: int = 0) -> Dict[Tuple, float]:
    """The fit's own point estimate per estimand, keyed like the draws.

    Only `bc` needs these -- the percentile interval reads fixed quantiles and is
    a function of the draws alone.
    """
    points: Dict[Tuple, float] = {}
    # `itertuples` renames "Pt Est" -- it is not a valid identifier -- so the
    # columns are zipped explicitly instead.
    risks = get_counterfactual_risks(updated_estimates, key_1=key_1, key_0=key_0)
    for ev, t, grp, est in zip(risks["Event"], risks["Time"],
                               risks["Group"], risks["Pt Est"]):
        points[("risks", int(ev), float(t), int(grp))] = float(est)
    for typ, fn in (("rd", ate_diff), ("rr", ate_ratio)):
        tab = fn(updated_estimates, key_1=key_1, key_0=key_0)
        for ev, t, est in zip(tab["Event"], tab["Time"], tab["Pt Est"]):
            points[(typ, int(ev), float(t), -1)] = float(est)
    return points


def select_bootstrap_method(results: pd.DataFrame,
                            method: str = "percentile") -> pd.DataFrame:
    """Keep the rows of one construction from a long interval table.

    `bootstrap_intervals` returns **both** constructions -- one row per estimand
    and per method, tagged by `bootstrap_method` -- because they are quantiles of
    the same draws and the second one is free. This narrows that table to one
    method, for the places that can only show a single pair of bounds.

    Switching construction, or comparing the two, therefore never requires
    re-running the bootstrap. Returns a copy; the stored table is untouched.
    """
    if method not in BOOTSTRAP_METHODS:
        raise ValueError(
            f"method must be one of {BOOTSTRAP_METHODS}, got {method!r}")
    if "bootstrap_method" not in results.columns:
        raise KeyError(
            "These bootstrap results carry no 'bootstrap_method' column; they "
            "were built by an older version that stored a single construction. "
            "Re-run the bootstrap.")
    out = results[results["bootstrap_method"] == method].reset_index(drop=True)
    if len(out):
        warn_on_bc_fallback(out)
    return out


def warn_on_bc_fallback(results: pd.DataFrame) -> int:
    """Warn about bias-corrected intervals that are not, in fact, bias-corrected.

    Returns how many of the `bc` rows fell back. See `_bc_interval` for when and
    why that happens; it is common enough to be worth saying out loud, and only
    worth saying when someone actually asks for `bc`.
    """
    if "ci_method" not in results.columns:
        return 0
    bc = (results[results["bootstrap_method"] == "bc"]
          if "bootstrap_method" in results.columns else results)
    if not len(bc):
        return 0
    fell = int(bc["ci_method"].astype(str).str.startswith(
        ("percentile", "unavailable", "undefined")).sum())
    if fell:
        warnings.warn(
            f"{fell} of {len(bc)} bias-corrected bootstrap intervals fall back "
            f"to another construction because the bias correction is undefined; "
            f"see the 'ci_method' column. This is expected when an estimand sits "
            f"near the boundary of its support.",
            RuntimeWarning,
        )
    return fell


def bootstrap_intervals(
    draws: pd.DataFrame,
    updated_estimates: Optional[Dict] = None,
    alpha: float = 0.05,
    key_1: int = 1,
    key_0: int = 0,
) -> pd.DataFrame:
    """Confidence intervals from stored bootstrap draws -- **both** constructions.

    Two are computed, from identical draws:

    ``percentile``
        the ``alpha/2`` and ``1 - alpha/2`` quantiles of the draws. The default
        everywhere a choice is made. A function of the draws alone, so it needs
        nothing from the fit.
    ``bc``
        bias-corrected: the same quantiles, shifted by twice
        ``z0 = Phi^-1(mean(draw < point))``. Needs the fit's **point estimate**,
        so it is filled in only when `updated_estimates` is given.

    There is no accelerated option; see `_bc_interval` for why.

    The result is **long**: one row per estimand *and per construction*, told
    apart by the ``bootstrap_method`` column, with the bounds always in
    ``CI_lower``/``CI_upper``. Nothing here picks a construction. Quantiles of an
    array that already exists are negligible beside the resampling that produced
    it, so keeping both costs nothing, and it means a fit carries the material
    for either interval and comparing them never needs a second bootstrap. Use
    `select_bootstrap_method` to narrow the table where only one can be shown.

    Interval construction is separate from resampling because `bc` needs the
    point estimate, which does not exist while the bootstrap runs -- and the
    targeted update mutates the initial estimates in place, so the draws must be
    collected *before* it and turned into intervals *after*.
    """
    points: Dict[Tuple, float] = {}
    if updated_estimates is not None:
        try:
            points = _point_estimates(updated_estimates, key_1, key_0)
        except Exception as exc:
            warnings.warn(
                f"Could not read point estimates for the bias correction "
                f"({exc}); only percentile intervals will be available.",
                RuntimeWarning,
            )

    rows = []
    for keys, g in draws.groupby(_TARGET_KEYS, sort=True):
        key = (str(keys[0]), int(keys[1]), float(keys[2]), int(keys[3]))
        d = g["Pt Est"].to_numpy(dtype=float)
        ident = dict(zip(_TARGET_KEYS, keys))
        pct = _percentile_interval(d, alpha)
        rows.append({**ident, "bootstrap_method": "percentile", **pct})
        if key in points:
            bc = _bc_interval(d, points[key], alpha)
        else:
            # Without the fit's point estimate there is no bias correction to
            # make; leave the bounds missing rather than quietly serving the
            # percentile ones under the `bc` label.
            bc = {**pct, "CI_lower": np.nan, "CI_upper": np.nan,
                  "bc_z0": np.nan, "ci_method": "unavailable (no point estimate)"}
        rows.append({**ident, "bootstrap_method": "bc", **bc})

    return pd.DataFrame(rows, columns=_INTERVAL_COLUMNS)


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
    seed=None,
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
    turns them into intervals, so only the **percentile** rows are populated: at
    this point the targeted update has not run, and the point estimate that the
    bias correction needs does not exist. `PyTMLE` therefore calls
    `bootstrap_draws` and `bootstrap_intervals` separately instead, which is what
    lets it fill in both constructions.
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
        seed=seed,
        **kwargs,
    )
    return bootstrap_intervals(draws, None, alpha=alpha, key_1=key_1, key_0=key_0)


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
    seed=None,
    **kwargs,
) -> pd.DataFrame:
    """The raw second-stage bootstrap draws, one row per resample and estimand.

    Kept separate from interval construction so that the draws can be collected
    *before* the targeted update -- which mutates the initial estimates in place,
    so a bootstrap run afterwards would resample already-targeted values -- while
    the intervals are built *after* it, when `bc` can see the point estimate.

    `seed` makes the whole bootstrap reproducible. One child seed is spawned per
    resample and handed to the task, which is also what makes the resamples
    *independent*: without it every forked worker inherits one global RNG state
    and repeats the same draws (see `_as_rng`).
    """
    seeds = np.random.SeedSequence(seed).spawn(n_bootstrap)
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
                child,
                **kwargs,
            )
            for child in seeds
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
