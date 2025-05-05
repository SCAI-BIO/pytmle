from typing import Dict, List, Optional
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from .tmle_update import tmle_update
from .predict_ate import get_counterfactual_risks, ate_ratio, ate_diff
from .estimates import InitialEstimates


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
    updated_estimates, _, converged, _ = tmle_update(
        initial_estimates=boot_initial_estimates,
        event_times=boot_event_times,
        event_indicator=boot_event_indicator,
        target_times=target_times,
        target_events=target_events,
        verbose=0,
        **kwargs,
    )
    if not converged:
        # if tmle_update did not converge, return None
        return
    cf_risks = get_counterfactual_risks(updated_estimates, key_1=key_1, key_0=key_0)[
        ["Event", "Time", "Group", "Pt Est"]
    ]
    cf_risks["type"] = "risks"
    ate_ratios = ate_ratio(updated_estimates, key_1=key_1, key_0=key_0)[
        ["Event", "Time", "Pt Est"]
    ]
    ate_ratios["type"] = "rr"
    ate_ratios["Group"] = -1
    ate_diffs = ate_diff(updated_estimates, key_1=key_1, key_0=key_0)[
        ["Event", "Time", "Pt Est"]
    ]
    ate_diffs["type"] = "rd"
    ate_diffs["Group"] = -1
    result_df = pd.concat([cf_risks, ate_ratios, ate_diffs])
    return result_df


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
    if len(results) == 0:
        if verbose >= 1:
            warnings.warn(
                "Not a single bootstrap samples converged. Bootstrapped CIs will not be available.",
                RuntimeWarning,
            )
        return None
    if verbose >= 2:
        print(
            f"TMLE converged for {len(results)} out of {n_bootstrap} bootstrap samples."
        )
    results_df = pd.concat(results)
    summary_df = (
        results_df.groupby(["type", "Event", "Time", "Group"])["Pt Est"].agg(
            mean_bootstrap="mean",
            CI_lower=lambda x: x.quantile(alpha / 2),
            CI_upper=lambda x: x.quantile(1 - alpha / 2),
        )
    ).reset_index()
    return summary_df
