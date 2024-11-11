from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from pytmle.get_influence_curve import get_ic, summarize_ic, get_norm_pn_eic, get_clever_covariate

from pytmle.estimates import InitialEstimates, UpdatedEstimates


def tmle_update(
    initial_estimates: Dict[int, InitialEstimates],
    target_times: List[float],
    event_times: np.ndarray,
    event_indicator: np.ndarray,
    max_updates: int = 500,
    min_nuisance: Optional[float] = None,
) -> Dict[int, UpdatedEstimates]:
    """
    Function to update the initial estimates using the TMLE algorithm.

    Parameters
    ----------
    initial_estimates : Dict[int, InitialEstimates]
        Dictionary of initial estimates.
    target_times : List[float]
        List of target times for which effects are estimated.
    event_times : np.ndarray
        Array of event times.
    event_indicator : np.ndarray
        Array of event indicators (censoring is 0).
    max_updates : int
        Maximum number of updates to the estimates in the TMLE loop.
    min_nuisance : Optional[float]
        Value between 0 and 1 for truncating the g-related denomiator of the clever covariate.

    Returns
    -------
    updated_estimates : Dict[int, UpdatedEstimates]
        Dictionary of updated estimates.
    """
    raise NotImplementedError("TMLE update algorithm not yet implemented.")

    updated_estimates = {
        i: UpdatedEstimates.from_initial_estimates(initial_estimates[i], min_nuisance)
        for i in initial_estimates.keys()
    }
    # TODO: Implement TMLE algorithm
    return updated_estimates


def do_tmle_update(estimates, summ_eic, data, target_event, target_time,
                   max_update_iter, one_step_eps, norm_pn_eic, verbose=False):
    """
    Performs TMLE updates on hazard estimates for each target event and time.

    Args:
        estimates (list): Initial estimates, containing hazard, survival, and influence curve summaries.
        summ_eic (pd.DataFrame): Summary of the efficient influence curve.
        data (pd.DataFrame): DataFrame containing event time and event type columns.
        target_event (list): List of target event indices.
        target_time (list): List of target time points.
        max_update_iter (int): Maximum number of TMLE update iterations.
        one_step_eps (float): Step size for the one-step TMLE update.
        norm_pn_eic (float): Initial norm of the efficient influence curve.
        verbose (bool): If True, prints diagnostic information at each iteration.

    Returns:
        dict: Updated estimates with hazards, survival, and convergence information.
    """
    eval_times = estimates["Times"]
    t_tilde = data[data.attrs.get("EventTime")]
    delta = data[data.attrs.get("EventType")]
    working_eps = one_step_eps
    norm_pn_eic_values = [norm_pn_eic]
    
    # Initialize step counters
    step_num = 1
    iter_num = 1

    while step_num <= max_update_iter and iter_num <= max_update_iter * 2:
        iter_num += 1
        if verbose:
            print(f"Starting step {step_num} with update epsilon = {working_eps}")

        # Update hazards and efficient influence curves
        new_estimates = []
        for est in estimates:
            new_hazards = update_hazard(
                g_star=est["PropScore"].attrs.get("g_star_obs"),
                hazards=est["Hazards"],
                total_survival=est["EvntFreeSurv"],
                nuisance_weight=est["NuisanceWeight"],
                eval_times=eval_times,
                t_tilde=t_tilde,
                delta=delta,
                pn_eic=est["SummEIC"],
                norm_pn_eic=norm_pn_eic,
                one_step_eps=working_eps,
                target_event=target_event,
                target_time=target_time
            )
            
            # Remove NaNs from hazard updates
            new_hazards = [np.nan_to_num(hazard) for hazard in new_hazards]
            
            # Calculate updated survival probabilities
            new_survival = np.exp(-np.cumsum(sum(new_hazards), axis=1))
            new_survival = np.maximum(new_survival, 1e-12)
            
            # Update influence curve and summary
            new_ic = get_ic(
                g_star=est["PropScore"].attrs.get("g_star_obs"),
                hazards=new_hazards,
                total_survival=new_survival,
                nuisance_weight=est["NuisanceWeight"],
                target_event=target_event,
                target_time=target_time,
                t_tilde=t_tilde,
                delta=delta,
                eval_times=eval_times,
                gcomp=False
            )
            new_estimates.append({
                "Hazards": new_hazards,
                "EvntFreeSurv": new_survival,
                "SummEIC": summarize_ic(new_ic),
                "IC": new_ic
            })

        # Summarize efficient influence curve (EIC)
        new_summ_eic = pd.concat([
            pd.DataFrame({"Trt": [name], **new_est["SummEIC"]}) 
            for name, new_est in zip(estimates.keys(), new_estimates)
        ])
        new_norm_pn_eic = get_norm_pn_eic(
            new_summ_eic[(new_summ_eic["Time"].isin(target_time)) & 
                         (new_summ_eic["Event"].isin(target_event))]["PnEIC"]
        )
        
        # Check for improvement
        if np.isnan(new_norm_pn_eic).any():
            raise RuntimeError("TMLE update breaking because survival probability -> 0")

        if norm_pn_eic < new_norm_pn_eic:
            if verbose:
                print("Update increased ||PnEIC||, halving OneStepEps")
            working_eps /= 2
            continue
        
        # Apply updates to estimates
        for i, est in enumerate(estimates):
            est.update(new_estimates[i])

        summ_eic = new_summ_eic
        norm_pn_eic = new_norm_pn_eic
        norm_pn_eic_values.append(new_norm_pn_eic)
        
        # Check convergence
        one_step_stop = new_summ_eic.groupby(["Trt", "Time", "Event"]).apply(
            lambda x: pd.Series({
                "check": np.abs(x["PnEIC"]) <= x["seEIC/(sqrt(n)log(n))"],
                "ratio": np.abs(x["PnEIC"]) / x["seEIC/(sqrt(n)log(n))"]
            })
        )
        
        if verbose:
            print_one_step_diagnostics(one_step_stop, norm_pn_eic)
        
        if one_step_stop["check"].all():
            estimates["TmleConverged"] = {"converged": True, "step": step_num}
            estimates["NormPnEICs"] = norm_pn_eic_values
            return estimates
        
        step_num += 1

    # Convergence warning if maximum iterations are reached
    print("Warning: TMLE has not converged by step", max_update_iter)
    estimates["TmleConverged"] = {"converged": False, "step": step_num}
    estimates["NormPnEICs"] = norm_pn_eic_values
    return estimates


def update_hazard(g_star, hazards, total_survival, nuisance_weight, eval_times, t_tilde,
                  delta, pn_eic, norm_pn_eic, one_step_eps, target_event, target_time):
    """
    Updates hazard estimates using one-step TMLE for each target event and time.

    Args:
        g_star (np.ndarray): Array representing the treatment regime.
        hazards (list): List of hazard matrices for each event type.
        total_survival (np.ndarray): Array of survival probabilities at each time point.
        nuisance_weight (np.ndarray): Matrix of nuisance weights for stability.
        eval_times (np.ndarray): Array of evaluation times for the hazard update.
        t_tilde (np.ndarray): Observed times for censoring or event.
        delta (np.ndarray): Indicator array for observed events (1 if observed, 0 if censored).
        pn_eic (np.ndarray): Efficient influence curve.
        norm_pn_eic (float): Normalizing constant for the influence curve.
        one_step_eps (float): Step size for the TMLE update.
        target_event (list): List of target event indices.
        target_time (list): List of target time points.

    Returns:
        list: Updated list of hazard matrices, one for each event type.
    """
    iterative = False  # Set to False for one-step TMLE
    
    g_star = np.asarray(g_star).flatten()
    if np.min(total_survival) == 0:
        raise ValueError("Survival probability reaching zero makes the clever covariate explode.")
    if iterative:
        print("Warning: Iterative TMLE not implemented. Performing one-step TMLE instead.")

    # Initialize updated hazard list
    updated_hazards = []
    
    for hazard_matrix in hazards:
        l = hazard_matrix["event_type"]  # assuming each hazard matrix has an event type attribute
        update_matrix = np.zeros_like(hazard_matrix)
        
        for event in target_event:
            f_j_t = np.cumsum(hazards[event] * total_survival, axis=1)
            
            for tau in target_time:
                # Create and populate clever covariate and h.FS matrix
                h_fs = np.zeros_like(f_j_t)
                h_fs[eval_times <= tau, :] = (
                    np.tile(f_j_t[eval_times == tau, :], (f_j_t[eval_times <= tau, :].shape[0], 1)) - 
                    f_j_t[eval_times <= tau, :]
                ) / total_survival[eval_times <= tau, :]
                
                clever_covariate = np.zeros_like(f_j_t)
                clever_covariate[eval_times <= tau, :] = get_clever_covariate(
                    g_star=g_star,
                    nuisance_weight=nuisance_weight[eval_times <= tau, :],
                    h_fs=h_fs[eval_times <= tau, :],
                    leq_j=int(l == event)
                )
                
                # Update calculation
                update_matrix += clever_covariate * pn_eic[(eval_times == tau) & (delta == event)]

        # Apply the TMLE update
        new_hazard_matrix = hazard_matrix * np.exp(update_matrix * one_step_eps / norm_pn_eic)
        new_hazard_matrix["event_type"] = l
        updated_hazards.append(new_hazard_matrix)

    return updated_hazards

def print_one_step_diagnostics(one_step_stop, norm_pn_eic):
    """
    Prints diagnostics for the TMLE one-step update, including the highest ratios and the norm of PnEIC.

    Args:
        one_step_stop (pd.DataFrame): DataFrame containing diagnostic checks and ratios for each step.
        norm_pn_eic (float): Norm of the efficient influence curve (PnEIC).
    """
    # Filter out the "check" column and sort by "ratio" in descending order
    worst = one_step_stop.drop(columns=["check"]).copy()
    worst["ratio"] = worst["ratio"].round(2)
    worst = worst.sort_values(by="ratio", ascending=False)

    # Print the top three rows with the highest ratios
    print(worst.head(3))
    print(f"Norm PnEIC = {norm_pn_eic}")