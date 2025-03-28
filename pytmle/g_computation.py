import numpy as np
import pandas as pd
from typing import List

def get_g_comp(
    eval_times: np.ndarray,
    hazards: np.ndarray,
    total_surv: np.ndarray,
    target_time: List[float],
    target_events: List[int],
) -> pd.DataFrame:
    """
    Calculates the G-computation estimate for a target cumulative incidence function (CIF)
    based on given event hazards and survival functions over time.

    Args:
        eval_times (numpy.ndarray): Evaluation times for the cumulative incidence functions.
        hazards (numpy.ndarray): Hazard matrix with rows as instances and columns as time points.
        total_surv (numpy.ndarray): Survival probabilities over time for each instance.
        target_time (numpy.ndarray): List of target times to evaluate in the influence curve.
        target_events (numpy.ndarray): List of target event types to evaluate in the influence curve.

    Returns:
        DataFrame: DataFrame with columns 'Event', 'Time', and 'Risk' containing the cumulative incidence estimates.
    """
    risks = []

    # survival function needs to lagged for CIF calculation
    lagged_total_surv = np.column_stack(
        [
            np.ones((total_surv.shape[0], 1)),
            total_surv[:, :-1],
        ],
    )

    for j in range(hazards.shape[-1]):
        if j + 1 not in target_events:
            continue
        # Calculate cumulative risk for each instance (row) at each time point
        risk_a = np.cumsum(lagged_total_surv * hazards[..., j], axis=1)

        # Filter only the columns corresponding to target times
        target_cols = eval_times[np.isin(eval_times, target_time)]
        risk_a_target = risk_a[:, np.isin(eval_times, target_time)]

        # Average over rows (instances) to get the mean cumulative incidence for each target time
        f_j_tau = np.mean(risk_a_target, axis=0)

        # Store results for each event type
        for t, risk in zip(target_cols, f_j_tau):
            risks.append({"Event": int(j + 1), "Time": t, "F.j.tau": risk})

    # Convert to DataFrame
    risks_df = pd.DataFrame(risks)

    # # Append row for overall survival (Event = -1)
    # total_risk = risks_df.groupby("Time")["F.j.tau"].sum()
    # total_risk_df = pd.DataFrame(
    #     {"Event": -1, "Time": total_risk.index, "F.j.tau": total_risk.values}
    # )
    # risks_df = pd.concat([risks_df, total_risk_df], ignore_index=True)

    # Rename 'F.j.tau' to 'Risk' in final DataFrame
    risks_df.rename(columns={"F.j.tau": "Risk"}, inplace=True)

    return risks_df[["Event", "Time", "Risk"]]
