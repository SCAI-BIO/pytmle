import pandas as pd
import numpy as np
from pytmle.clever_covariate import getCleverCovariate, getHazLS



def getEIC(Estimates, Data, Regime, TargetEvent, TargetTime, MinNuisance, GComp=False):
    """
    Calculate Efficient Influence Curves (EICs) for a list of estimates across different treatment regimes.

    Parameters:
    Estimates (list of dicts): Each dictionary contains nuisance weights, propensity scores, hazards, etc., for each regime.
    Data (pd.DataFrame): Data containing observed event times and event types.
    Regime (list): List defining the intervention regimes.
    TargetEvent (list or np.array): Target events to analyze.
    TargetTime (list or np.array): Time points for calculating the influence curve.
    MinNuisance (float): Minimum threshold for nuisance weights to avoid destabilization.
    GComp (bool): If True, calculates the G-Computation estimate.

    Returns:
    list of dicts: Updated Estimates with Efficient Influence Curves, G-computation estimates (if applicable), and summaries.
    """
    # Extract evaluation times from Estimates attribute
    EvalTimes = Estimates[0].get("Times", None)  # Assume all regimes have the same EvalTimes
    T_tilde = Data[Data.attrs.get("EventTime")]
    Delta = Data[Data.attrs.get("EventType")]

    # Iterate over each regime in Estimates
    for a, estimate in enumerate(Estimates):
        # Extract necessary components from the current regime in Estimates
        NuisanceWeight = estimate["NuisanceWeight"]
        GStar = estimate["PropScore"].get("g.star.obs", None)
        Hazards = estimate["Hazards"]
        TotalSurv = estimate["EvntFreeSurv"]

        # Calculate the Influence Curve (IC) for the current regime
        IC_a = getIC(
            GStar=GStar,
            Hazards=Hazards,
            TotalSurv=TotalSurv,
            NuisanceWeight=NuisanceWeight,
            TargetEvent=TargetEvent,
            TargetTime=TargetTime,
            T_tilde=T_tilde,
            Delta=Delta,
            EvalTimes=EvalTimes,
            GComp=GComp
        )

        # If GComputation is enabled, calculate and add it to Estimates
        if GComp:
            estimate["GCompEst"] = getGComp(EvalTimes, Hazards, TotalSurv, TargetTime)

        # Summarize the influence curve and add it to Estimates
        estimate["SummEIC"] = summarizeIC(IC_a)
        estimate["IC"] = IC_a

    return Estimates


    

def getIC(GStar, Hazards, TotalSurv, NuisanceWeight, TargetEvent, TargetTime, 
          T_tilde, Delta, EvalTimes, GComp):
    """
    Calculate the Influence Curve (IC) for given events and time points.

    Parameters:
    GStar (np.array): Modified propensity scores.
    Hazards (dict of np.array): Dictionary of hazard matrices for different events.
    TotalSurv (np.array): Overall survival probability matrix across time points.
    NuisanceWeight (np.array): Weight matrix for nuisance parameters.
    TargetEvent (list): List of specific events to target.
    TargetTime (list): List of target times.
    T_tilde (np.array): Observed event times.
    Delta (np.array): Event type indicator (e.g., censoring or event).
    EvalTimes (np.array): Array of all evaluation time points.
    GComp (bool): Flag indicating whether to use G-computation adjustments.

    Returns:
    pd.DataFrame: Influence curve values for each ID, time, and event.
    """
    IC_list = []

    for j in TargetEvent:
        # Calculate the cumulative incidence function for the event j
        F_j_t = np.cumsum(Hazards[str(j)] * TotalSurv, axis=0)

        for tau in TargetTime:
            # Clever covariate contributions for hFS
            h_FS = np.tile(F_j_t[EvalTimes == tau, :], (len(EvalTimes[EvalTimes <= tau]), 1))
            h_FS = (h_FS - F_j_t[EvalTimes <= tau, :]) / TotalSurv[EvalTimes <= tau, :]

            # Compute Influence Curve for each hazard event type
            IC_j_tau = np.zeros(h_FS.shape[1])
            for l, haz in Hazards.items():
                ClevCov = getCleverCovariate(GStar=GStar,
                                             NuisanceWeight=NuisanceWeight[EvalTimes <= tau, :],
                                             hFS=h_FS,
                                             LeqJ=int(l == str(j)))

                NLdS = np.zeros_like(h_FS)
                for i, (delta, t_tilde) in enumerate(zip(Delta, T_tilde)):
                    if delta == int(l) and t_tilde <= tau:
                        NLdS[EvalTimes == t_tilde, i] = 1

                HazLS = getHazLS(T_tilde, EvalTimes[EvalTimes <= tau], haz[EvalTimes <= tau, :])
                IC_j_tau += np.sum(ClevCov * (NLdS - HazLS), axis=0)

            IC_j_tau += F_j_t[EvalTimes == tau, :] - np.mean(F_j_t[EvalTimes == tau, :])
            
            # Check for any overflow issues
            if np.any(np.isnan(IC_j_tau)):
                raise ValueError("IC overflow: Adjust MinNuisance or specify a target with more support in the data.")
            
            # Collect results for each ID, Time, and Event
            IC_list.append(pd.DataFrame({
                "ID": np.arange(1, len(IC_j_tau) + 1),
                "Time": tau,
                "Event": j,
                "IC": IC_j_tau
            }))

    # Concatenate results from all events and times
    IC_df = pd.concat(IC_list, ignore_index=True)
    return IC_df



def getGComp(EvalTimes, Hazards, TotalSurv, TargetTime):
    """
    Calculate the G-Computation estimate for specific target times.

    Parameters:
    EvalTimes (list or np.array): Time points for evaluation.
    Hazards (list of np.array): List of hazard matrices, each representing hazard rates for an event type at different times.
    TotalSurv (np.array): Matrix of overall survival probabilities across time points.
    TargetTime (list or np.array): Specific time points for risk calculation.

    Returns:
    pd.DataFrame: DataFrame with 'Event', 'Time', and 'Risk' columns containing the computed risks.
    """
    risks_list = []

    # Calculate cumulative risk for each hazard in Hazards
    for haz in Hazards:
        event = int(haz.attrs.get("j", -1))  # Extract event type from attributes
        risk_a = np.array([np.cumsum(TotalSurv[:, i] * haz[:, i]) for i in range(haz.shape[1])])
        
        # Filter for TargetTime and average across columns (observations)
        target_indices = [i for i, time in enumerate(EvalTimes) if time in TargetTime]
        risk_a_filtered = risk_a[target_indices, :].mean(axis=1)
        
        # Create rows for each TargetTime point and append to risks_list
        for time, risk in zip(np.array(EvalTimes)[target_indices], risk_a_filtered):
            risks_list.append({"Event": event, "Time": time, "Tau": risk})

    # Convert risks_list to a DataFrame
    risks_df = pd.DataFrame(risks_list)

    # Calculate the cumulative risk for all events (Event = -1)
    total_risks = risks_df.groupby("Time")["Tau"].sum()
    overall_risks = pd.DataFrame({
        "Event": -1,
        "Time": total_risks.index,
        "Tau": 1 - total_risks.values
    })

    # Combine individual event risks and overall risks
    risks_df = pd.concat([risks_df, overall_risks], ignore_index=True)

    # Rename columns to match output format and return
    return risks_df.rename(columns={"Tau": "Risk"})



def summarizeIC(IC_a):
    """
    Summarizes the Influence Curve (IC) values with specific statistics.

    Parameters:
    IC_a (pd.DataFrame): DataFrame containing columns 'ID', 'Time', 'Event', and 'IC',
                         representing the influence curve values for each observation.

    Returns:
    pd.DataFrame: Summary DataFrame with mean IC (PnEIC), standard error (seEIC),
                  and normalized error (seEIC / (sqrt(n) * log(n))) for each 'Time' and 'Event' combination.
    """
    # Step 1: Add a summary row where Event = -1 and IC is the negative sum of IC by 'ID' and 'Time'
    sum_IC = IC_a.groupby(['ID', 'Time'])['IC'].sum()
    summary_rows = pd.DataFrame({
        'ID': sum_IC.index.get_level_values('ID'),
        'Time': sum_IC.index.get_level_values('Time'),
        'Event': -1,
        'IC': -sum_IC.values
    })
    IC_a = pd.concat([IC_a, summary_rows], ignore_index=True)
    
    # Step 2: Calculate summary statistics by 'Time' and 'Event'
    summary = IC_a.groupby(['Time', 'Event']).agg(
        PnEIC=('IC', 'mean'),
        seEIC=('IC', lambda x: np.sqrt(np.mean(x ** 2))),
        seEIC_sqrt_n_log_n=('IC', lambda x: np.sqrt(np.mean(x ** 2)) / (np.sqrt(len(x)) * np.log(len(x))))
    ).reset_index()

    return summary


