from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from pytmle.estimates import InitialEstimates, UpdatedEstimates
from pytmle.get_influence_curve import get_eic, get_ic, get_clever_covariate, summarize_ic

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

    combined_summ_eic = pd.concat([df.assign(trt=key) for key, df in summ_eic_per_trt.items()], ignore_index=True)

    return pd.DataFrame(combined_summ_eic)

def calculate_norm_pn_eic(summ_eic, sigma=None, target_time=None, target_event=None):
    """
    Calculate the normalized PnEIC from a summEIC DataFrame.

    Args:
        summ_eic (pd.DataFrame): DataFrame containing columns ['Trt', 'Time', 'Event', 'PnEIC'].
        sigma (np.ndarray, optional): Covariance matrix for weighting. Defaults to None.
        target_time (list or int, optional): Specific time(s) to filter. Defaults to None.
        target_event (list or int, optional): Specific event(s) to filter. Defaults to None.

    Returns:
        float: Normalized PnEIC value.
    """
    # Filter based on TargetTime and TargetEvent if provided
    if target_time is not None:
        if not isinstance(target_time, list):
            target_time = [target_time]
        summ_eic = summ_eic[summ_eic['Time'].isin(target_time)]

    if target_event is not None:
        if not isinstance(target_event, list):
            target_event = [target_event]
        summ_eic = summ_eic[summ_eic['Event'].isin(target_event)]

    # Extract the PnEIC column
    pn_eic = summ_eic['PnEIC'].values

    # Apply weighting with Sigma if provided
    if sigma is not None:
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            # Regularize Sigma if it's not invertible
            sigma_inv = np.linalg.inv(sigma + np.eye(sigma.shape[0]) * 1e-6)
            print("Warning: Regularization of Sigma needed for inversion.")

        weighted_pn_eic = np.dot(pn_eic, sigma_inv)
    else:
        weighted_pn_eic = pn_eic

    # Calculate the normalized PnEIC
    norm_pn_eic = np.sqrt(np.sum(pn_eic * weighted_pn_eic))

    return norm_pn_eic


def update_hazards(
    hazards, total_surv, g_star, nuisance_weight, eval_times, pn_eic, norm_pn_eic, one_step_eps, target_event, target_time
):
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
        dict: Updated hazards dictionary.
    """
    updated_hazards = {}

    for event_type, hazard in hazards.items(): # *** The current hazards object does not include an event type indicator ***
        # Initialize the update term
        update_term = np.zeros_like(hazard)

        for tau in target_time:
            # Compute F.j.t for the current event type
            f_j_t = np.cumsum(hazard * total_surv, axis=0)

            # Initialize matrices for clever covariate computation
            h_fs = np.zeros_like(f_j_t)
            clev_cov = np.zeros_like(f_j_t)

            # Compute h.FS for times <= tau
            mask = eval_times <= tau
            h_fs[mask] = (
                f_j_t[eval_times == tau] - f_j_t[mask]
            ) / total_surv[mask]

            # Compute clever covariate using the helper function
            clev_cov[mask] = get_clever_covariate(
                g_star=g_star,
                nuisance_weight=nuisance_weight[mask],
                h_fs=h_fs[mask],
                leq_j=int(event_type in target_event),
            )

            # Weight the clever covariate by PnEIC
            pn_eic_weights = pn_eic[(pn_eic['Time'] == tau) & (pn_eic['Event'] == event_type)]["PnEIC"].values

            if pn_eic_weights.size > 0:
                pn_eic_weights = pn_eic_weights[:, np.newaxis]  # Add axis for broadcasting
                clev_cov *= pn_eic_weights
                update_term += clev_cov

        # Apply exponential update to the hazard function
        updated_hazard = hazard * np.exp(update_term * one_step_eps / norm_pn_eic)
        updated_hazards[event_type] = updated_hazard

    return updated_hazards

def do_tmle_update(estimates, summ_eic, data, target_event, target_time,
                   max_update_iter, one_step_eps, norm_pn_eic, verbose):
    """
    Perform the TMLE update procedure for estimates.

    Args:
        estimates (dict): Dictionary of initial estimates for each treatment.
        summ_eic (pd.DataFrame): Summary of Efficient Influence Curve (EIC).
        data (dict): Dictionary with data attributes like event time and type.
        target_event (list): List of target events for evaluation.
        target_time (list): List of target times for evaluation.
        max_update_iter (int): Maximum number of TMLE update iterations.
        one_step_eps (float): Initial epsilon for one-step update.
        norm_pn_eic (float): Norm of the efficient influence curve.
        verbose (bool): If True, print detailed diagnostics during updates.

    Returns:
        dict: Updated estimates after TMLE procedure.
    """
    eval_times = estimates.get("times")
    t_tilde = data.get("event_time")
    delta = data.get("event_indicator")
    working_eps = one_step_eps
    norm_pn_eics = [norm_pn_eic]

    step_num = 1
    iter_num = 1

    while step_num <= max_update_iter and iter_num <= max_update_iter * 2:
        iter_num += 1
        if verbose:
            print(f"Starting step {step_num} with update epsilon = {working_eps}")

        # Get updated hazards and EICs
        new_ests = {}
        for trt, est_a in estimates.items():
            new_hazards = update_hazard(
                g_star=est_a.propensity_scores,
                hazards=est_a.hazards,
                total_surv=est_a.event_free_survival_function,
                nuisance_weight=est_a.nuisance_weight,
                eval_times=eval_times,
                t_tilde=t_tilde,
                delta=delta,
                pn_eic=est_a.summ_eic,
                norm_pn_eic=norm_pn_eic,
                one_step_eps=working_eps,
                target_event=target_event,
                target_time=target_time
            )
            # Replace NaN/NA values in hazards with zeros
            new_hazards = {k: np.nan_to_num(v, nan=0.0) for k, v in new_hazards.items()}

            new_surv = np.exp(-np.cumsum(np.sum(list(new_hazards.values()), axis=0), axis=0))
            new_surv[new_surv < 1e-12] = 1e-12

            new_ic = get_ic(
                g_star=est_a.propensity_scores,
                hazards=new_hazards,
                total_surv=new_surv,
                nuisance_weight=est_a.nuisance_weight,
                target_event=target_event,
                target_time=target_time,
                t_tilde=t_tilde,
                delta=delta,
                eval_times=eval_times,
                g_comp=False
            )

            new_ests[trt] = {
                "hazards": new_hazards,
                "event_free_survival_function": new_surv,
                "summ_eic": summarize_ic(new_ic),
                "ic": new_ic
            }

        # Check for improvement
        new_summ_eic = pd.concat([
            pd.DataFrame({"trt": trt, **ests["summ_eic"]}) for trt, ests in new_ests.items()
        ])
        new_norm_pn_eic = get_norm_pn_eic(
            new_summ_eic[(new_summ_eic["time"].isin(target_time)) & 
                         (new_summ_eic["event"].isin(target_event))]["pn_eic"]
        )

        if np.any(np.isnan(new_norm_pn_eic)):
            raise ValueError("Update failed: Survival reached zero.")

        if norm_pn_eic < new_norm_pn_eic:
            if verbose:
                print("Update increased ||PnEIC||, halving OneStepEps")
            working_eps /= 2
            continue

        step_num += 1

        # Update estimates
        for trt in estimates.keys():
            estimates[trt].update(new_ests[trt])

        summ_eic = new_summ_eic
        norm_pn_eic = new_norm_pn_eic
        norm_pn_eics.append(new_norm_pn_eic)

        # Check convergence
        one_step_stop = new_summ_eic.groupby(["trt", "time", "event"]).apply(
            lambda x: {
                "check": abs(x["pn_eic"]) <= x["se_eic"] / (np.sqrt(len(x)) * np.log(len(x))),
                "ratio": abs(x["pn_eic"]) / (x["se_eic"] / (np.sqrt(len(x)) * np.log(len(x))))
            }
        )

        if verbose:
            print("Diagnostics:", one_step_stop)

        if all(one_step_stop.apply(lambda x: x["check"])):
            estimates["tmle_converged"] = {"converged": True, "step": step_num}
            estimates["norm_pn_eics"] = norm_pn_eics
            return estimates

    # Warning for non-convergence
    if verbose:
        print(f"Warning: TMLE has not converged by step {max_update_iter}")
    estimates["tmle_converged"] = {"converged": False, "step": step_num}
    estimates["norm_pn_eics"] = norm_pn_eics
    return estimates


#def tmle_update(
#    initial_estimates: Dict[int, InitialEstimates],
#    event_times: np.ndarray,
#    event_indicator: np.ndarray,
#    target_times: List[float],
#    target_events: List[int] = [1],
#    max_updates: int = 500,
#    min_nuisance: Optional[float] = None,
#    g_comp: bool = False,
#) -> Dict[int, UpdatedEstimates]:
#    """
#    Function to update the initial estimates using the TMLE algorithm.
#
#    Parameters
#    ----------
#    initial_estimates : Dict[int, InitialEstimates]
#        Dictionary of initial estimates.
#    target_times : List[float]
#        List of target times for which effects are estimated.
#    target_events : List[int]
#        List of target events for which effects are estimated. Default is [1].
#    event_times : np.ndarray
#        Array of event times.
#    event_indicator : np.ndarray
#        Array of event indicators (censoring is 0).
#    max_updates : int
#        Maximum number of updates to the estimates in the TMLE loop.
#    min_nuisance : Optional[float]
#        Value between 0 and 1 for truncating the g-related denomiator of the clever covariate.
#    g_comp : bool
#        Whether to return the g-computation estimates. Default is False.
#
#    Returns
#    -------
#    updated_estimates : Dict[int, UpdatedEstimates]
#        Dictionary of updated estimates.
#    """
#
#   updated_estimates = {
#        i: UpdatedEstimates.from_initial_estimates(
#            initial_estimates[i], target_events, target_times, min_nuisance
#        )
#        for i in initial_estimates.keys()
#    }
#    updated_estimates = get_eic(
#        estimates=updated_estimates,
#        event_times=event_times,
#        event_indicator=event_indicator,
#        g_comp=g_comp,
#    )
#
#    norm_pn_eic = get_norm_pn_eic(
#        summ_eic[
#            np.isin(summ_eic[:, 1], target_times) & np.isin(summ_eic[:, 2], target_events), 
#            3
#        ]
#    )
#
#
#    
#
#    # TODO: Implement TMLE update loop
#    return updated_estimates
