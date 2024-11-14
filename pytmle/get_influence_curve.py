import numpy as np
import pandas as pd


def get_eic(estimates, data, regime, target_event, target_time, min_nuisance, g_comp=False):
    """
    Calculate the Efficient Influence Curve (EIC) and G-computation estimates if specified.

    Args:
        estimates (list): List of dictionaries containing estimates for each regime.
        data (dict): Dictionary with data attributes like event time and type.
        regime (str): Treatment or exposure regime to evaluate.
        target_event (int or list of int): Target event type(s) for which to calculate EIC.
        target_time (int or list of int): Target time point(s) for the EIC.
        min_nuisance (float): Minimum value for nuisance parameters to avoid overflow.
        g_comp (bool): If True, calculates the G-computation estimate. Defaults to False.

    Returns:
        list: Updated estimates with Efficient Influence Curve and, if specified, G-computation estimates.
    """
    eval_times = estimates.get("times")
    t_tilde = data.get("event_time")
    delta = data.get("event_indicator")

    for a, estimate in estimates["initial_estimates"].items():
        nuisance_weight = estimate.nuisance_weight
        g_star = estimate.g_star_obs
        hazards = estimate.hazards
        total_surv = estimate.event_free_survival_function

    # Call getIC function with the extracted parameters
        ic_a = get_ic(
            g_star=g_star,
            hazards=hazards,
            total_surv=total_surv,
            nuisance_weight=nuisance_weight,
            target_event=target_event,
            target_time=target_time,
            t_tilde=t_tilde,
            delta=delta,
            eval_times=eval_times,
            g_comp=g_comp
        )

    # Conditional assignment for g_compEst
    if g_comp:
        estimate["g_comp_est"] = get_g_comp(eval_times, hazards, total_surv, target_time)

    # Assign SummEIC and IC to the estimate
    estimate["summ_eic"] = summarize_ic(ic_a)
    estimate["ic_a"] = ic_a

        # Store G-computation estimate if requested
    if g_comp:
        estimate["g_comp_est"] = get_g_comp(eval_times, hazards, total_surv, target_time)

        # Store summarized EIC and raw IC in estimates
        estimate["summ_eic"] = summarize_ic(ic_a)
        estimate["ic"] = ic_a

    return estimates

def get_ic(g_star, hazards, total_surv, nuisance_weight, target_event, target_time, 
           t_tilde, delta, eval_times, g_comp):
    """
    Calculates the influence curve (IC) for a target cumulative incidence function (CIF) 
    based on given event hazards and survival functions over time.
    
    Args:
        g_star (array-like): Intervention vector, typically binary or numeric.
        hazards (dict): Dictionary where each key is an event type, and each value is a hazard 
            matrix with rows as time points and columns as instances.
        total_surv (array-like): Survival probabilities over time for each instance.
        nuisance_weight (array-like): Nuisance weights over evaluation times for each instance.
        target_event (list): List of target events to evaluate in the influence curve.
        target_time (list): List of target times to evaluate in the influence curve.
        t_tilde (array-like): Observed times for each instance.
        delta (array-like): Event indicators for each instance, with unique events specified as integers.
        eval_times (array-like): Evaluation times for the cumulative incidence functions.
        g_comp (array-like): G-computation estimate for each instance.
        
    Returns:
        DataFrame: The resulting influence curve (IC) as a DataFrame with columns for ID, time, event, and IC values.
    """
    #target = pd.DataFrame([(tau, j) for tau in target_time for j in target_event], columns=["Time", "Event"])
    #unique_events = sorted(set(delta)) - {0}
    g_star = np.array(g_star).flatten()

    # Initialize results list
    ic_results = []

    for j in target_event:
        # Calculate the cumulative incidence function for the current event
        print("hazards[j] \n", hazards[j].shape)
        print("total_surv \n", total_surv.shape)
        f_j_t = np.cumsum(hazards[j] * total_surv, axis=0)
        
        for tau in target_time:
            # The event-related (F(t) and S(t)) contributions to the clever covariate (h)
            h_fs = np.tile(f_j_t[eval_times == tau, :], (np.sum(eval_times <= tau), 1))
            h_fs = (h_fs - f_j_t[eval_times <= tau, :]) / total_surv[eval_times <= tau, :]
            
            # Calculate IC for this particular (j, tau) pair
            ic_j_tau = []

            for l, hazard_l in hazards.items():
                clev_cov = get_clever_covariate(
                    g_star=g_star,
                    nuisance_weight=nuisance_weight[eval_times <= tau, :],
                    h_fs=h_fs,
                    leq_j=int(l == j)
                )
                
                # Initialize the matrix for non-likelihood event indicators
                nlds = np.zeros_like(h_fs)
                for i, time in enumerate(t_tilde):
                    if delta[i] == l and time <= tau:
                        nlds[np.where(eval_times == time)[0][0], i] = 1
                
                haz_ls = get_haz_ls(
                    t_tilde=t_tilde,
                    eval_times=eval_times[eval_times <= tau],
                    haz_l=hazard_l[eval_times <= tau, :]
                )
                
                # Sum contributions for IC
                ic_j_tau.append(np.sum(clev_cov * (nlds - haz_ls), axis=0))
            
            ic_j_tau = np.sum(ic_j_tau, axis=0) + f_j_t[eval_times == tau, :] - np.mean(f_j_t[eval_times == tau, :])
            
            # Check for overflow (NaNs)
            if np.any(np.isnan(ic_j_tau)):
                raise ValueError(
                    "IC overflow: either increase min_nuisance or specify a target estimand "
                    "(Target Event, Target Time, & Intervention) with more support in the data."
                )

            # Store the results for this tau and j as dictionaries
            for idx, ic_val in enumerate(ic_j_tau):
                ic_results.append({"ID": idx + 1, "Time": tau, "Event": j, "IC": ic_val})

    # Convert results to DataFrame
    ic_a = pd.DataFrame(ic_results)
    
    return ic_a


def get_clever_covariate(g_star, nuisance_weight, h_fs, leq_j):
    """
    Computes the clever covariate for influence curve calculation.

    Args:
        g_star (numpy.ndarray): Intervention vector for each instance.
        nuisance_weight (numpy.ndarray): Nuisance weights matrix for each instance and time point.
        h_fs (numpy.ndarray): Clever covariate contributions matrix.
        leq_j (int): Indicator of whether the current event type equals the target event type.

    Returns:
        numpy.ndarray: Adjusted clever covariate matrix.
    """
    # Element-wise multiplication of each column of nuisance_weight by corresponding g_star values
    for i in range(nuisance_weight.shape[1]):
        nuisance_weight[:, i] *= g_star[i]
    
    # Element-wise multiplication with (LeqJ - h_fs)
    return nuisance_weight * (leq_j - h_fs)

def get_haz_ls(t_tilde, eval_times, haz_l):
    """
    Computes the adjusted hazard matrix for each instance and time, based on evaluation times.

    Args:
        t_tilde (numpy.ndarray): Observed times for each instance.
        eval_times (numpy.ndarray): Evaluation times for calculating hazards.
        HazL (numpy.ndarray): Hazard matrix for each time and instance.

    Returns:
        numpy.ndarray: Adjusted hazard matrix where HazL values are retained for times <= t_tilde.
    """
    for i in range(haz_l.shape[1]):
        haz_l[:, i] = np.where(eval_times <= t_tilde[i], haz_l[:, i], 0)
    
    return haz_l

def get_g_comp(eval_times, hazards, total_surv, target_time):
    """
    Calculates the G-computation estimate for a target cumulative incidence function (CIF)
    based on given event hazards and survival functions over time.
    
    Args:
        eval_times (array-like): Evaluation times for the cumulative incidence functions.
        hazards (dict): Dictionary where each key is an event type, and each value is a hazard 
            matrix with rows as time points and columns as instances.
        total_surv (array-like): Survival probabilities over time for each instance.
        target_time (array-like): List of target times to evaluate in the influence curve.
        
    Returns:
        DataFrame: DataFrame with columns 'Event', 'Time', and 'Risk' containing the cumulative incidence estimates.
    """
    risks = []

    for event, haz_j in hazards.items():
        # Calculate cumulative risk for each instance (column) at each time point
        risk_a = np.cumsum(total_surv * haz_j, axis=0)

        # Filter only the rows corresponding to target times
        target_rows = eval_times[np.isin(eval_times, target_time)]
        risk_a_target = risk_a[np.isin(eval_times, target_time), :]

        # Average over columns (instances) to get the mean cumulative incidence for each target time
        f_j_tau = np.mean(risk_a_target, axis=1)

        # Store results for each event type
        for t, risk in zip(target_rows, f_j_tau):
            risks.append({"Event": int(event), "Time": t, "F.j.tau": risk})

    # Convert to DataFrame
    risks_df = pd.DataFrame(risks)

    # Append row for overall survival (Event = -1)
    total_risk = risks_df.groupby("Time")["F.j.tau"].sum()
    total_risk_df = pd.DataFrame({
        "Event": -1,
        "Time": total_risk.index,
        "F.j.tau": 1 - total_risk.values
    })
    risks_df = pd.concat([risks_df, total_risk_df], ignore_index=True)

    # Rename 'F.j.tau' to 'Risk' in final DataFrame
    risks_df.rename(columns={"F.j.tau": "Risk"}, inplace=True)

    return risks_df[["Event", "Time", "Risk"]]

def summarize_ic(ic_a):
    """
    Summarizes the influence curve (IC) estimates for the target cumulative incidence function (CIF).

    Args:
        ic_a (DataFrame): DataFrame containing columns 'ID', 'Time', 'Event', and 'IC' 
                          representing influence curve estimates for each event and time.

    Returns:
        DataFrame: Summary DataFrame with columns 'Time', 'Event', 'PnEIC', 'seEIC', 
                   and 'seEIC/(sqrt(n)log(n))' containing mean and standard error estimates.
    """
    # Append overall influence calculation for 'Event = -1'
    overall_ic = ic_a.groupby(["ID", "Time"])["IC"].sum().reset_index()
    overall_ic["Event"] = -1
    overall_ic["IC"] = -overall_ic["IC"]
    ic_a = pd.concat([ic_a, overall_ic], ignore_index=True)

    # Calculate summary statistics
    summary = ic_a.groupby(["Time", "Event"]).agg(
        PnEIC=("IC", "mean"),
        seEIC=("IC", lambda x: np.sqrt(np.mean(x ** 2))),
        seEIC_sqrt_n_log_n=("IC", lambda x: np.sqrt(np.mean(x ** 2)) / (np.sqrt(len(x)) * np.log(len(x))))
    ).reset_index()

    # Rename columns to match the output format
    summary.rename(columns={"seEIC_sqrt_n_log_n": "seEIC/(sqrt(n)log(n))"}, inplace=True)

    return summary