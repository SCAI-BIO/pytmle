from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import warnings

from .estimates import InitialEstimates, UpdatedEstimates
from .get_influence_curve import get_eic, get_clever_covariate


def combine_summarized_eic(estimates):
    """
    Combines summarized Efficient Influence Curve (EIC) estimates from a dictionary of estimates.

    Args:
        estimates (dict): A dictionary where keys are treatment or exposure regime names and
                          values contain a key "SummEIC" with the summarized EIC data.

    Returns:
        pd.DataFrame: A DataFrame combining the summarized EIC estimates with an additional "Trt" column.
    """
    summ_eic_per_trt = {}

    for trt, est in estimates.items():
        summ_eic_per_trt[trt] = est.summ_eic

    combined_summ_eic = pd.concat(
        [df.assign(trt=key) for key, df in summ_eic_per_trt.items()], ignore_index=True
    )

    return pd.DataFrame(combined_summ_eic)


def calculate_norm_pn_eic(pn_eic: pd.Series) -> float:
    """
    Calculate the normalized PnEIC.

    Args:
        pn_eic (pd.Series): Series of PnEIC values.

    Returns:
        float: Normalized PnEIC value.
    """
    norm_pn_eic = np.sqrt(np.sum(pn_eic * pn_eic))

    return norm_pn_eic


def update_hazards(
    hazards: np.ndarray,
    total_surv,
    g_star,
    nuisance_weight,
    eval_times,
    pn_eic,
    norm_pn_eic,
    one_step_eps,
    target_event,
    target_time,
) -> np.ndarray:
    """
    Update hazards using the clever covariate and one-step TMLE update.

    Args:
        hazards (dict): Dictionary of hazard matrices for each event type.
        total_surv (np.ndarray): Total survival probability matrix.
        g_star (np.ndarray): Intervention vector.
        nuisance_weight (np.ndarray): Nuisance weights matrix.
        eval_times (np.ndarray): Evaluation time points.
        pn_eic (pd.DataFrame): DataFrame containing PnEIC values.
        norm_pn_eic (float): Normalized PnEIC value.
        one_step_eps (float): Step size for the TMLE update.
        target_event (list): List of target event types.
        target_time (list): List of target times.

    Returns:
        np.ndarray: Updated hazards.
    """
    updated_hazards = np.zeros_like(hazards)

    # survival function needs to lagged for CIF calculation
    lagged_total_surv = np.column_stack(
        [
            np.ones((total_surv.shape[0], 1)),
            total_surv[:, :-1],
        ],
    )

    for l in range(1, hazards.shape[-1] + 1):
        if l not in target_event:
            continue
        # Initialize the update term
        update_term = np.zeros_like(hazards[..., l - 1])
        hazard = hazards[..., l - 1]

        for j in target_event:
            # Compute F.j.t for the current event type
            f_j_t = np.cumsum(hazards[..., j - 1] * lagged_total_surv, axis=1)
            for tau in target_time:
                # Initialize matrices for clever covariate computation
                h_fs = np.zeros_like(f_j_t)
                clev_cov = np.zeros_like(f_j_t)

                # Compute h.FS for times <= tau
                mask = eval_times <= tau
                h_fs[:, mask] = (
                    f_j_t[:, eval_times == tau].repeat(np.sum(mask), axis=1)
                    - f_j_t[:, mask]
                ) / total_surv[:, mask]

                # Compute clever covariate using the helper function
                clev_cov[:, mask] = get_clever_covariate(
                    g_star=g_star,
                    nuisance_weight=nuisance_weight[:, mask],
                    h_fs=h_fs[:, mask],
                    leq_j=int(l == j),
                )

                # Weight the clever covariate by PnEIC
                pn_eic_weights = pn_eic[
                    (pn_eic["Time"] == tau) & (pn_eic["Event"] == j)
                ]["PnEIC"].values

                if pn_eic_weights.size > 0:
                    pn_eic_weights = pn_eic_weights[
                        :, np.newaxis
                    ]  # Add axis for broadcasting
                clev_cov *= pn_eic_weights
                update_term += clev_cov

        # Apply exponential update to the hazard function
        updated_hazard = hazard * np.exp(update_term * one_step_eps / norm_pn_eic)
        updated_hazards[..., l - 1] = updated_hazard

    return updated_hazards


def tmle_loop(
    estimates,
    t_tilde,
    delta,
    target_events,
    target_times,
    max_updates,
    one_step_eps,
    norm_pn_eic,
    verbose,
) -> Tuple[Dict[int, UpdatedEstimates], List[float], bool, int]:
    """
    Perform the TMLE update procedure for estimates.

    Args:
        estimates (dict): Dictionary of initial estimates for each treatment.
        t_tilde (np.ndarray): Time points.
        delta (np.ndarray): Event indicator.
        target_events (list): List of target events for evaluation.
        target_times (list): List of target times for evaluation.
        max_updates (int): Maximum number of TMLE update iterations.
        one_step_eps (float): Initial epsilon for one-step update.
        norm_pn_eic (float): Norm of the efficient influence curve.
        verbose (int): Sets logging level.

    Returns:
        dict: Updated estimates after TMLE procedure.
        list: List of normalized PnEIC values.
        bool: Flag indicating convergence.
        int: Number of TMLE update steps.
    """

    working_eps = one_step_eps
    norm_pn_eics = [norm_pn_eic]

    step_num = 0
    iter_num = 0

    while step_num < max_updates and iter_num < max_updates * 2:
        iter_num += 1
        if verbose >= 4:
            print(f"Iteration {iter_num}: Starting update step {step_num + 1}.")

        # Get updated hazards and EICs
        new_ests = {}
        for trt, est_a in estimates.items():
            eval_times = est_a.times
            if target_times is None:
                target_times = est_a.target_times
            new_hazards = update_hazards(
                g_star=est_a.g_star_obs,
                hazards=est_a.hazards,
                total_surv=est_a.event_free_survival_function,
                nuisance_weight=est_a.nuisance_weight,
                eval_times=eval_times,
                pn_eic=est_a.summ_eic,
                norm_pn_eic=norm_pn_eic,
                one_step_eps=working_eps,
                target_event=target_events,
                target_time=target_times,
            )
            # Replace NaN/NA values in hazards with zeros
            new_hazards = np.nan_to_num(new_hazards, nan=0.0)

            new_surv = np.exp(-np.cumsum(np.sum(new_hazards, axis=-1), axis=1))
            new_surv[new_surv < 1e-12] = 1e-12

            new_ests[trt] = UpdatedEstimates(
                times=eval_times,
                hazards=new_hazards,
                event_free_survival_function=new_surv,
                propensity_scores=est_a.propensity_scores,
                censoring_survival_function=est_a.censoring_survival_function,
                target_times=target_times,
                target_events=target_events,
                g_star_obs=est_a.g_star_obs,
                g_comp_est=est_a.g_comp_est,
            )

        if verbose >= 4:
            print("Updated hazards and survival functions computed.")

        # get EIC for updated estimates
        new_ests = get_eic(
            estimates=new_ests,
            event_times=t_tilde,
            event_indicator=delta,
        )

        if verbose >= 4:
            print("Efficient influence curves (EIC) computed for updated estimates.")

        # Check for improvement
        new_summ_eic = combine_summarized_eic(new_ests)
        new_norm_pn_eic = calculate_norm_pn_eic(
            new_summ_eic.loc[
                np.isin(new_summ_eic["Time"], target_times)
                & np.isin(new_summ_eic["Event"], target_events),
                "PnEIC",
            ]
        )

        if np.any(np.isnan(new_norm_pn_eic)):
            raise ValueError("Update failed: Survival reached zero.")

        if norm_pn_eic < new_norm_pn_eic:
            if verbose >= 4:
                print("No improvement in norm PnEIC, reducing epsilon.")
            working_eps /= 2
            continue

        step_num += 1
        if verbose >= 3:
            print(f"Step {step_num}: Norm PnEIC improved to {new_norm_pn_eic}.")

        # Update estimates
        estimates.update(new_ests)

        norm_pn_eic = new_norm_pn_eic
        norm_pn_eics.append(new_norm_pn_eic)

        # Check convergence
        new_summ_eic["check"] = new_summ_eic.apply(
            lambda x: abs(x["PnEIC"]) <= x["seEIC/(sqrt(n)log(n))"],
            axis=1,
        )

        if all(new_summ_eic["check"]):
            if verbose >= 2:
                print(f"TMLE converged at step {step_num}.")
            return new_ests, norm_pn_eics, True, step_num

    # Warning for non-convergence
    if verbose >= 1:
        warnings.warn(
            f"Warning: TMLE has not converged by step {max_updates}. Estimates may not have the desired asymptotic properties.",
            RuntimeWarning,
        )
    return estimates, norm_pn_eics, False, step_num


def tmle_update(
    initial_estimates: Dict[int, InitialEstimates],
    event_times: np.ndarray,
    event_indicator: np.ndarray,
    target_times: List[float],
    target_events: List[int] = [1],
    max_updates: int = 500,
    min_nuisance: Optional[float] = None,
    g_comp: bool = False,
    one_step_eps: float = 0.1,
    verbose: int = 2,
) -> Tuple[Dict[int, UpdatedEstimates], List[float], bool, int]:
    """
    Function to update the initial estimates using the TMLE algorithm.

    Parameters
    ----------
    initial_estimates : Dict[int, InitialEstimates]
        Dictionary of initial estimates.
    target_times : List[float]
        List of target times for which effects are estimated.
    target_events : List[int]
        List of target events for which effects are estimated. Default is [1].
    event_times : np.ndarray
        Array of event times.
    event_indicator : np.ndarray
        Array of event indicators (censoring is 0).
    max_updates : int
        Maximum number of updates to the estimates in the TMLE loop.
    min_nuisance : Optional[float]
        Value between 0 and 1 for truncating the g-related denomiator of the clever covariate.
    g_comp : bool
        Whether to return the g-computation estimates. Default is False.
    one_step_eps : float
        Initial epsilon for the one-step update. Default is 0.1.
    verbose : int
        Verbosity level. 0: Absolutely so logging at all, 1: only warnings, 2: major execution steps, 3: execution steps, 4: everything for debugging. Default is 2.


    Returns
    -------
    updated_estimates : Dict[int, UpdatedEstimates]
        Dictionary of updated estimates.
    norm_pn_eics : List[float]
        List of normalized PnEIC values.
    has_converged : bool
        Flag indicating convergence.
    step_num : int
        Number of TMLE update steps.
    """

    updated_estimates = {
        i: UpdatedEstimates.from_initial_estimates(
            initial_estimates[i], target_events, target_times, min_nuisance
        )
        for i in initial_estimates.keys()
    }
    updated_estimates = get_eic(
        estimates=updated_estimates,
        event_times=event_times,
        event_indicator=event_indicator,
        g_comp=g_comp,
    )

    summ_eic = combine_summarized_eic(updated_estimates)

    norm_pn_eic = calculate_norm_pn_eic(
        summ_eic.loc[
            np.isin(summ_eic["Time"], target_times)
            & np.isin(summ_eic["Event"], target_events),
            "PnEIC",
        ]
    )
    # Calculate the check columns
    summ_eic["check"] = summ_eic.apply(
        lambda x: abs(x["PnEIC"]) <= x["seEIC/(sqrt(n)log(n))"],
        axis=1,
    )

    # check if initial estimates already fulfill convergence criterion
    if all(summ_eic["check"]):
        if verbose >= 2:
            print("Initial estimates already fulfill convergence criterion.")
        return updated_estimates, [norm_pn_eic], True, 0

    return tmle_loop(
        estimates=updated_estimates,
        t_tilde=event_times,
        delta=event_indicator,
        target_events=target_events,
        target_times=target_times,
        max_updates=max_updates,
        one_step_eps=one_step_eps,
        norm_pn_eic=norm_pn_eic,
        verbose=verbose,
    )
