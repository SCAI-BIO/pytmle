from typing import List, Dict
import numpy as np
import pandas as pd

from .estimates import UpdatedEstimates
from .g_computation import get_g_comp


def get_eic(
    estimates: Dict[int, UpdatedEstimates],
    event_times: np.ndarray,
    event_indicator: np.ndarray,
    g_comp: bool = False,
) -> Dict[int, UpdatedEstimates]:
    """
    Calculate the Efficient Influence Curve (EIC) and G-computation estimates if specified.

    Args:
        estimates (dict): Dictionary of UpdatedEstimates objects with current estimates.
        event_times (numpy.ndarray): Array of event times.
        event_indicator (numpy.ndarray): Array of event indicators (censoring is 0).
        g_comp (bool): Flag to calculate G-computation estimates in addition to EIC.

    Returns:
        dict: Updated estimates with Efficient Influence Curve and, if specified, G-computation estimates.
    """

    for a, estimate in estimates.items():
        nuisance_weight = estimate.nuisance_weight
        g_star = estimate.g_star_obs
        hazards = estimate.hazards
        total_surv = estimate.event_free_survival_function

        # Call getIC function with the extracted parameters
        ic_a = get_ic(
            g_star=g_star,
            hazards=hazards,
            total_surv=total_surv,
            nuisance_weight=nuisance_weight,  # type: ignore
            target_events=estimate.target_events,  # type: ignore
            target_time=estimate.target_times,  # type: ignore
            t_tilde=event_times,
            delta=event_indicator,
            eval_times=estimate.times,
        )

        # Store G-computation estimate if requested
        if g_comp:
            estimate.g_comp_est = get_g_comp(
                eval_times=estimate.times,
                hazards=hazards,
                total_surv=total_surv,
                target_time=estimate.target_times,  # type: ignore
                target_events=estimate.target_events,  # type: ignore
            )

        # Assign SummEIC and IC to the estimate
        estimate.summ_eic = summarize_ic(ic_a)
        estimate.ic = ic_a

    return estimates


def get_ic(
    g_star: np.ndarray,
    hazards: np.ndarray,
    total_surv: np.ndarray,
    nuisance_weight: np.ndarray,
    target_events: List[int],
    target_time: List[float],
    t_tilde: np.ndarray,
    delta: np.ndarray,
    eval_times: np.ndarray,
):
    """
    Calculates the influence curve (IC) for a target cumulative incidence function (CIF)
    based on given event hazards and survival functions over time.

    Args:
        g_star (numpy.ndarray): Intervention vector for each instance.
        hazards (numpy.ndarray): Hazard matrix with rows as instances and columns as time points.
        total_surv (numpy.ndarray): Survival probabilities over time for each instance and time point.
        nuisance_weight (numpy.ndarray): Nuisance weights matrix for each instance and time point.
        target_events (list): List of target event types to evaluate in the influence curve.
        target_time (list): List of target times to evaluate in the influence curve.
        t_tilde (numpy.ndarray): Observed times for each instance.
        delta (numpy.ndarray): Event indicators for each instance.
        eval_times (numpy.ndarray): Evaluation times for calculating the influence curve.

    Returns:
        DataFrame: The resulting influence curve (IC) as a DataFrame with columns for ID, time, event, and IC values.
    """
    # target = pd.DataFrame([(tau, j) for tau in target_time for j in target_events], columns=["Time", "Event"])
    # unique_events = sorted(set(delta)) - {0}
    g_star = np.array(g_star).flatten()

    # Initialize results list
    ic_results = []

    eval_times = np.array(eval_times)  # Ensure eval_times is a numpy array

    # survival function needs to lagged for CIF calculation
    lagged_total_surv = np.column_stack(
        [
            np.ones((total_surv.shape[0], 1)),
            total_surv[:, :-1],
        ],
    )
    for j in target_events:
        # Calculate the cumulative incidence function for the current event
        # print("hazards[j] \n", hazards[j].shape)
        # print("lagged_total_surv \n", lagged_total_surv.shape)
        f_j_t = np.cumsum(hazards[..., j - 1] * lagged_total_surv, axis=1)

        for tau in target_time:
            # The event-related (F(t) and S(t)) contributions to the clever covariate (h)
            tau_idx = np.where(eval_times == tau)[0][0]
            leq_tau_idx = np.where(eval_times <= tau)[0]
            h_fs = np.tile(f_j_t[:, tau_idx, np.newaxis], (1, len(leq_tau_idx)))
            h_fs = (h_fs - f_j_t[:, leq_tau_idx]) / total_surv[:, leq_tau_idx]

            # Calculate IC for this particular (j, tau) pair
            ic_j_tau = []

            for l in range(hazards.shape[-1]):
                clev_cov = get_clever_covariate(
                    g_star=g_star,
                    nuisance_weight=nuisance_weight[
                        :, : len(eval_times[eval_times <= tau])
                    ],
                    h_fs=h_fs,
                    leq_j=int(l == j - 1),
                )

                # Initialize the matrix for non-likelihood event indicators
                nlds = np.zeros_like(h_fs)
                for i, time in enumerate(t_tilde):
                    if delta[i] == l + 1 and time <= tau:
                        if i < nlds.shape[0]:
                            nlds[i, np.where(eval_times == time)[0][0]] = 1

                haz_ls = get_haz_ls(
                    t_tilde=t_tilde,
                    eval_times=eval_times[eval_times <= tau],
                    haz_l=hazards[:, : len(eval_times[eval_times <= tau]), l].copy(),
                )

                # Sum contributions for IC
                # print("nlds \n", nlds.shape)
                # print("haz_ls \n", haz_ls.shape)
                # print("clev_cov \n", clev_cov.shape)
                ic_j_tau.append(np.sum(clev_cov * (nlds - haz_ls), axis=1))

            ic_j_tau = (
                np.sum(ic_j_tau, axis=0)
                + f_j_t[:, tau_idx]
                - np.mean(f_j_t[:, tau_idx])
            )

            # Check for overflow (NaNs)
            if np.any(np.isnan(ic_j_tau)):
                raise RuntimeError(
                    "IC overflow: either increase min_nuisance or specify a target estimand "
                    "(Target Event, Target Time, & Intervention) with more support in the data."
                )

            # Store the results for this tau and j as dictionaries
            for idx, ic_val in enumerate(ic_j_tau):
                ic_results.append(
                    {"ID": idx + 1, "Time": tau, "Event": j, "IC": ic_val}
                )

    # Convert results to DataFrame
    ic_a = pd.DataFrame(ic_results)

    return ic_a


def get_clever_covariate(
    g_star: np.ndarray, nuisance_weight: np.ndarray, h_fs: np.ndarray, leq_j: int
) -> np.ndarray:
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
    # Element-wise multiplication of each row of nuisance_weight by corresponding g_star values
    nuisance_weight = g_star[:, np.newaxis] * nuisance_weight

    # Element-wise multiplication with (LeqJ - h_fs)
    return nuisance_weight * (leq_j - h_fs)


def get_haz_ls(
    t_tilde: np.ndarray, eval_times: np.ndarray, haz_l: np.ndarray
) -> np.ndarray:
    """
    Computes the adjusted hazard matrix for each instance and time, based on evaluation times.

    Args:
        t_tilde (numpy.ndarray): Observed times for each instance.
        eval_times (numpy.ndarray): Evaluation times for calculating hazards.
        HazL (numpy.ndarray): Hazard matrix for each time and instance.

    Returns:
        numpy.ndarray: Adjusted hazard matrix where HazL values are retained for times <= t_tilde.
    """
    for i in range(haz_l.shape[0]):
        haz_l[i, :] = np.where(eval_times <= t_tilde[i], haz_l[i, :], 0)

    return haz_l


def summarize_ic(ic_a: pd.DataFrame) -> pd.DataFrame:
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
    summary = (
        ic_a.groupby(["Time", "Event"])
        .agg(
            PnEIC=("IC", "mean"),
            seEIC=("IC", lambda x: np.sqrt(np.mean(x**2))),
            seEIC_sqrt_n_log_n=(
                "IC",
                lambda x: np.sqrt(np.mean(x**2)) / (np.sqrt(len(x)) * np.log(len(x))),
            ),
        )
        .reset_index()
    )

    # Rename columns to match the output format
    summary.rename(
        columns={"seEIC_sqrt_n_log_n": "seEIC/(sqrt(n)log(n))"}, inplace=True
    )

    return summary
