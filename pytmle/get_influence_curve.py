import pandas as pd
import numpy as np
from pytmle.clever_covariate import get_clever_covariate, get_haz_ls



def get_eic(estimates, data, regime, target_event, target_time, min_nuisance, g_comp=False):
    """
    Calculates the efficient influence curve (EIC) and optionally performs G-computation for each estimate.
    
    Args:
        estimates (list): List of initial estimates for each treatment regime.
        data (pd.DataFrame): DataFrame containing event time and event type columns.
        regime (list): List of regimes for each treatment.
        target_event (list): List of target event indices.
        target_time (list): List of target time points.
        min_nuisance (float): Minimum threshold for nuisance weights.
        g_comp (bool): If True, performs G-computation.
    
    Returns:
        list: Updated list of estimates with EIC and G-computation information.
    """
    eval_times = estimates["Times"]
    t_tilde = data[data.attrs.get("EventTime")]
    delta = data[data.attrs.get("EventType")]
    
    for i, estimate in enumerate(estimates):
        nuisance_weight = estimate["NuisanceWeight"]
        g_star = estimate["PropScore"].attrs.get("g_star_obs")
        hazards = estimate["Hazards"]
        total_survival = estimate["EvntFreeSurv"]
        
        # Calculate the influence curve (IC) for the estimate
        ic_data = get_ic(
            g_star=g_star,
            hazards=hazards,
            total_survival=total_survival,
            nuisance_weight=nuisance_weight,
            target_event=target_event,
            target_time=target_time,
            t_tilde=t_tilde,
            delta=delta,
            eval_times=eval_times,
            g_comp=g_comp
        )
        
        # If G-computation is requested, calculate G-computed estimates
        if g_comp:
            estimate["GCompEst"] = get_g_comp(eval_times, hazards, total_survival, target_time)
        
        # Summarize the influence curve (IC) and store results
        estimate["SummEIC"] = summarize_ic(ic_data)
        estimate["IC"] = ic_data

    return estimates


    

def get_ic(g_star, hazards, total_survival, nuisance_weight, target_event, target_time, 
           t_tilde, delta, eval_times, g_comp):
    """
    Calculates the influence curve (IC) for each target event and time using hazard and survival data.
    
    Args:
        g_star (np.ndarray): Array representing the intervention regime.
        hazards (dict): Dictionary of hazard matrices for each event type.
        total_survival (np.ndarray): Array of survival probabilities at each time point.
        nuisance_weight (np.ndarray): Matrix of nuisance weights.
        target_event (list): List of target events.
        target_time (list): List of target times.
        t_tilde (np.ndarray): Observed times for censoring or event.
        delta (np.ndarray): Array indicating event type (1 if observed, 0 if censored).
        eval_times (np.ndarray): Array of evaluation times.
        g_comp (bool): Indicator for G-computation.

    Returns:
        pd.DataFrame: DataFrame containing influence curve values for each target event and time.
    """
    # Define the grid of target events and times
    target_grid = pd.DataFrame({"Time": target_time, "Event": target_event})
    unique_events = sorted(set(delta) - {0})
    g_star = np.asarray(g_star).flatten()

    ic_data = []

    for event in target_event:
        # Calculate cumulative incidence function for the event type
        f_j_t = np.cumsum(hazards[str(event)] * total_survival, axis=0)

        for tau in target_time:
            # Calculate the clever covariate adjustment
            h_fs = (np.tile(f_j_t[eval_times == tau, :], (np.sum(eval_times <= tau), 1)) - 
                    f_j_t[eval_times <= tau, :]) / total_survival[eval_times <= tau, :]

            # Calculate IC for the event and time
            ic_j_tau = sum(
                np.sum(get_clever_covariate(
                    g_star=g_star, 
                    nuisance_weight=nuisance_weight[eval_times <= tau, :],
                    h_fs=h_fs,
                    leq_j=int(hazard_name == event)
                ) * (np.array([
                    (eval_times == t_tilde[i]) & (i in np.where(delta == hazard_name)[0])
                    for i in range(len(t_tilde))
                ]) - get_haz_ls(
                    t_tilde=t_tilde, 
                    eval_times=eval_times[eval_times <= tau], 
                    hazard_l=hazards[hazard_name][eval_times <= tau, :]
                )), axis=1)
                for hazard_name in hazards.keys()
            ) + f_j_t[eval_times == tau, :] - np.mean(f_j_t[eval_times == tau, :])

            # Handle potential NaN values in IC
            if np.isnan(ic_j_tau).any():
                raise ValueError("IC overflow: increase MinNuisance or specify a target with more data support.")
            
            # Append IC values to the result list
            ic_data.append(pd.DataFrame({
                "ID": np.arange(len(ic_j_tau)),
                "Time": tau,
                "Event": event,
                "IC": ic_j_tau
            }))
    
    # Concatenate all IC values into a single DataFrame
    ic_df = pd.concat(ic_data, ignore_index=True)
    
    return ic_df



def get_g_comp(eval_times, hazards, total_survival, target_time):
    """
    Computes risk estimates for each event type at specified target times using hazard and survival data.

    Args:
        eval_times (np.ndarray): Array of evaluation times.
        hazards (list): List of 2D arrays representing hazards for each event type, where each array has
                        time on rows and individuals on columns.
        total_survival (np.ndarray): 2D array of survival probabilities with time on rows and individuals on columns.
        target_time (np.ndarray): Array of target times for risk estimation.

    Returns:
        pd.DataFrame: DataFrame with columns 'Event', 'Time', and 'Risk' summarizing risk estimates.
    """
    risks = []
    
    for hazard_matrix in hazards:
        event_type = hazard_matrix["event_type"]  # Assuming each hazard matrix has an event type attribute
        
        # Calculate cumulative risk for each time and individual
        risk_matrix = np.array([
            np.cumsum(total_survival[:, i] * hazard_matrix[:, i])
            for i in range(hazard_matrix.shape[1])
        ]).T  # Transpose to align with time x individual structure

        # Calculate mean risk across individuals at each target time
        time_mask = np.isin(eval_times, target_time)
        f_j_tau = np.mean(risk_matrix[time_mask, :], axis=1)
        
        # Create a risk summary for each target time and event type
        risk_df = pd.DataFrame({
            "Event": event_type,
            "Time": eval_times[time_mask],
            "F.j.tau": f_j_tau
        })
        risks.append(risk_df)

    # Concatenate all event-specific risks into a single DataFrame
    risks = pd.concat(risks, ignore_index=True)
    
    # Calculate the overall survival risk (Event = -1) at each time point
    overall_risk = (
    risks.groupby("Time", as_index=False)["F.j.tau"]
    .sum()
    .assign(Event=-1)
)
    
    # Modify F.j.tau to represent the overall survival risk (1 - sum(F.j.tau))
    overall_risk["F.j.tau"] = 1 - overall_risk["F.j.tau"]
    
    # Combine event-specific and overall risks
    risks = pd.concat([risks, overall_risk], ignore_index=True)
    
    # Rename columns for the final output
    risks = risks.rename(columns={"F.j.tau": "Risk"})
    
    return risks[["Event", "Time", "Risk"]]



def summarize_ic(ic_data):
    """
    Summarizes influence curve (IC) statistics, adding a synthetic row with Event = -1 to capture
    the sum of IC values across events for each (ID, Time) combination.
    
    Args:
        ic_data (pd.DataFrame): DataFrame with columns 'ID', 'Time', 'Event', and 'IC' representing 
                                influence curve values.
    
    Returns:
        pd.DataFrame: Summary DataFrame with columns 'Time', 'Event', 'PnEIC', 'seEIC', 
                      and 'seEIC/(sqrt(n)log(n))'.
    """
    # Add synthetic "Event = -1" rows representing sum of IC values across events
    synthetic_ic = (
        ic_data.groupby(["ID", "Time"], as_index=False)["IC"]
        .sum()
        .assign(Event=-1)
    )
    ic_data_extended = pd.concat([ic_data, synthetic_ic], ignore_index=True)
    
    # Calculate summary statistics for each (Time, Event) group
    summary_ic = (
        ic_data_extended.groupby(["Time", "Event"], as_index=False)
        .agg(
            PnEIC=("IC", "mean"),
            seEIC=("IC", lambda x: np.sqrt(np.mean(x**2))),
            seEIC_sqrt_n_log_n=("IC", lambda x: np.sqrt(np.mean(x**2)) / (np.sqrt(len(x)) * np.log(len(x))))
        )
    )
    summary_ic.rename(columns={"seEIC_sqrt_n_log_n": "seEIC/(sqrt(n)log(n))"}, inplace=True)
    
    return summary_ic


def get_norm_pn_eic(pn_eic, sigma=None):
    """
    Calculates the norm of the efficient influence curve (PnEIC), with optional weighting by the inverse
    of a covariance matrix Sigma.
    
    Args:
        pn_eic (np.ndarray): Array representing the efficient influence curve values.
        sigma (np.ndarray, optional): Covariance matrix. If provided, it is used to weight PnEIC.

    Returns:
        float: The norm of the efficient influence curve.
    """
    weighted_pn_eic = pn_eic
    
    if sigma is not None:
        try:
            sigma_inv = np.linalg.solve(sigma, np.eye(sigma.shape[0]))
        except np.linalg.LinAlgError:
            # Regularize Sigma by adding a small constant to the diagonal
            sigma += np.eye(sigma.shape[0]) * 1e-6
            sigma_inv = np.linalg.solve(sigma, np.eye(sigma.shape[0]))
            print("Warning: Regularization of Sigma needed for inversion.")
        
        weighted_pn_eic = pn_eic @ sigma_inv

    # Calculate and return the norm
    norm = np.sqrt(np.sum(pn_eic * weighted_pn_eic))
    return norm