from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from pytmle.estimates import UpdatedEstimates
from scipy.stats import norm

def get_counterfactual_risks(updated_estimates: Dict[int, UpdatedEstimates], 
                         g_comp: bool = False,
                         alpha: float = 0.05,
                         key_1: int = 1,
                         key_0: int = 0) -> pd.DataFrame:
    """
    Get counterfactual risks for the treatment and control groups.
    """    

    if not key_1 in updated_estimates.keys() or not key_0 in updated_estimates.keys():
        raise ValueError("Both keys must be present in the updated estimates dictionary.")
    pred_risk_1 = updated_estimates[key_1].predict_mean_risks(g_comp=g_comp)
    pred_risk_1["Group"] = key_1
    pred_risk_0 = updated_estimates[key_0].predict_mean_risks(g_comp=g_comp)
    pred_risk_0["Group"] = key_0

    assert (pred_risk_1["Time"] == pred_risk_0["Time"]).all(), "Time values do not match between groups."
    assert (pred_risk_1["Event"] == pred_risk_0["Event"]).all(), "Event values do not match between groups."

    pred_risk = pd.concat([pred_risk_1, pred_risk_0], ignore_index=True)

    # confidence intervals
    if not g_comp:
        pred_risk["CI_lower"] = pred_risk["Pt Est"] - norm.ppf(1 - alpha / 2) * pred_risk["SE"]
        pred_risk["CI_upper"] = pred_risk["Pt Est"] + norm.ppf(1 - alpha / 2) * pred_risk["SE"]
    else:
        pred_risk["CI_lower"] = np.nan
        pred_risk["CI_upper"] = np.nan
    return pred_risk


def threshold(x) -> float:
    """
    Threshold function for E-values.

    Args:
        x (float): Value to threshold.
    """
    if x <= 1:
        x = 1 / x
    return x + (x * (x - 1)) ** 0.5


def get_evalues_rr(
    rr: pd.Series,
    ci_lower: Optional[pd.Series] = None,
    ci_upper: Optional[pd.Series] = None,
) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    Compute the E-values for the Risk Ratio (RR) estimate.

    Args:
        rr (pd.Series): Point estimate for the Risk Ratio.
        ci_lower (pd.Series): Lower bound of the confidence interval.
        ci_upper (pd.Series): Upper bound of the confidence interval.

    Returns:
        Tuple[pd.Series, Optional[pd.Series]]: E-values for the RR estimate and the confidence interval.
    """
    # compute E-values for the point estimate
    e = rr.apply(threshold)

    if ci_lower is not None and ci_upper is not None:
        # check if CI crosses the null
        null_ci = np.where(rr > 1, ci_lower < 1, ci_upper > 1)
        # compute E-values for the CI
        e_lower = np.where(null_ci, 1, ci_lower.apply(threshold))
        e_upper = np.where(null_ci, 1, ci_upper.apply(threshold))

        return e, pd.Series(np.where(rr > 1, e_lower, e_upper))
    return e, None


def ate_ratio(
    updated_estimates: Dict[int, UpdatedEstimates], 
    g_comp: bool = False,
    alpha: float = 0.05,
    key_1: int = 1,
    key_0: int = 0
) -> pd.DataFrame:
    """
    Calculate the Average Treatment Effect (ATE) based on ratio from the updated estimates.

    Args:
        updated_estimates (dict): Dictionary of UpdatedEstimates objects with current estimates.
        g_comp (bool): Flag to use G-computation estimates if available.
        alpha (float): Significance level for confidence intervals.
        key_1 (int): Key for the treatment group.
        key_0 (int): Key for the control group.

    Returns:
        pandas.DataFrame: DataFrame with ATE ratio estimates.
    """
    pred_risk = get_counterfactual_risks(updated_estimates, g_comp, alpha, key_1, key_0)
    pred_risk_1 = pred_risk[pred_risk["Group"] == key_1].reset_index(drop=True)
    pred_risk_0 = pred_risk[pred_risk["Group"] == key_0].reset_index(drop=True)

    pred_ratios = pd.DataFrame(columns=["Time", "Event", "Pt Est"])
    pred_ratios["Time"] = pred_risk_1["Time"]
    pred_ratios["Event"] = pred_risk_1["Event"]
    pred_ratios["Pt Est"] = pred_risk_1["Pt Est"] / pred_risk_0["Pt Est"]

    if not g_comp:
        ic_1 = updated_estimates[key_1].ic.set_index(["Event", "Time"])
        ic_0 = updated_estimates[key_0].ic.set_index(["Event", "Time"])
        pred_risk_1 = pred_risk_1.set_index(["Event", "Time"])
        pred_risk_0 = pred_risk_0.set_index(["Event", "Time"])

        if ic_1 is None or ic_0 is None:
            raise ValueError("IC is not available for one or both groups.")

        for idx, row in pred_ratios.iterrows():
            time = row["Time"]
            event = row["Event"]
            R1 = float(pred_risk_1.loc[(event, time), "Pt Est"])
            R0 = float(pred_risk_0.loc[(event, time), "Pt Est"])
            IC_1 = np.array(ic_1.loc[(event, time), "IC"], dtype=float)
            IC_0 = np.array(ic_0.loc[(event, time), "IC"], dtype=float)
            se = np.sqrt(np.mean((IC_1 / R0 - IC_0 * R1 / R0**2)**2) / len(IC_0))
            pred_ratios.at[idx, "SE"] = se
        # confidence intervals
        pred_ratios["CI_lower"] = pred_ratios["Pt Est"] - norm.ppf(1 - alpha / 2) * pred_ratios["SE"]
        pred_ratios["CI_upper"] = pred_ratios["Pt Est"] + norm.ppf(1 - alpha / 2) * pred_ratios["SE"]
        # p-values
        z_stat = (pred_ratios["Pt Est"] - 1) / pred_ratios["SE"]
        pred_ratios["p_value"] = 2 * (1 - norm.cdf(np.abs(z_stat)))
        evalues, evalues_ci = get_evalues_rr(
            pred_ratios["Pt Est"], pred_ratios["CI_lower"], pred_ratios["CI_upper"]
        )
        pred_ratios["E_value"] = evalues
        pred_ratios["E_value CI"] = evalues_ci
    else:
        pred_ratios["SE"] = np.nan
        pred_ratios["CI_lower"] = np.nan
        pred_ratios["CI_upper"] = np.nan
        pred_ratios["p_value"] = np.nan
        evalues, _ = get_evalues_rr(pred_ratios["Pt Est"])
        pred_ratios["E_value"] = evalues
        pred_ratios["E_value CI"] = np.nan

    return pred_ratios


def get_evalues_rd(
    rd: pd.Series, se: Optional[pd.Series] = None
) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    Compute the E-values for the Risk Difference (RD) estimate.
    Approximation as described in https://dash.harvard.edu/bitstream/handle/1/36874927/EValue_FinalSubmission.pdf, p. 37

    Args:
        rd (pd.Series): Point estimate for the Risk Difference.
        se (pd.Series): Standard error for the Risk Difference.

    Returns:
        Tuple[pd.Series, Optional[pd.Series]]: E-values for the RD estimate and the confidence interval.
    """
    #
    rr = pd.Series(np.exp(0.91 * rd))
    if se is not None:
        ci_lower = pd.Series(np.exp(0.91 * rd - 1.78 * se))
        ci_upper = pd.Series(np.exp(0.91 * rd + 1.78 * se))
        return get_evalues_rr(rr, ci_lower, ci_upper)
    return get_evalues_rr(rr)


def ate_diff(
    updated_estimates: Dict[int, UpdatedEstimates], 
    g_comp: bool = False,
    alpha: float = 0.05,
    key_1: int = 1,
    key_0: int = 0
) -> pd.DataFrame:
    """
    Calculate the Average Treatment Effect (ATE) based on difference from the updated estimates.

    Args:
        updated_estimates (dict): Dictionary of UpdatedEstimates objects with current estimates.
        g_comp (bool): Flag to use G-computation estimates if available.
        alpha (float): Significance level for confidence intervals.
        key_1 (int): Key for the treatment group.
        key_0 (int): Key for the control group.

    Returns:
        pandas.DataFrame: DataFrame with ATE difference estimates.
    """
    pred_risk = get_counterfactual_risks(updated_estimates, g_comp, alpha, key_1, key_0)
    pred_risk_1 = pred_risk[pred_risk["Group"] == key_1].reset_index(drop=True)
    pred_risk_0 = pred_risk[pred_risk["Group"] == key_0].reset_index(drop=True)

    pred_diffs = pd.DataFrame(columns=["Time", "Event", "Pt Est"])
    pred_diffs["Time"] = pred_risk_1["Time"]
    pred_diffs["Event"] = pred_risk_1["Event"]
    pred_diffs["Pt Est"] = pred_risk_1["Pt Est"] - pred_risk_0["Pt Est"]

    if not g_comp:
        ic_1 = updated_estimates[key_1].ic.set_index(["Event", "Time"])
        ic_0 = updated_estimates[key_0].ic.set_index(["Event", "Time"])

        if ic_1 is None or ic_0 is None:
            raise ValueError("IC is not available for one or both groups.")

        se = (
            (ic_1 - ic_0)
            .groupby(["Event", "Time"])
            .apply(lambda x: np.sqrt(np.mean(x**2) / len(x)))
            .reset_index(name="SE")
        )
        pred_diffs = pred_diffs.merge(se, on=["Event", "Time"])
        # confidence intervals
        pred_diffs["CI_lower"] = pred_diffs["Pt Est"] - norm.ppf(1 - alpha / 2) * pred_diffs["SE"]
        pred_diffs["CI_upper"] = pred_diffs["Pt Est"] + norm.ppf(1 - alpha / 2) * pred_diffs["SE"]
        # p-values
        z_stat = pred_diffs["Pt Est"] / pred_diffs["SE"]
        pred_diffs["p_value"] = 2 * (1 - norm.cdf(np.abs(z_stat)))
        evalues, evalues_ci = get_evalues_rd(pred_diffs["Pt Est"], pred_diffs["SE"])
        pred_diffs["E_value"] = evalues
        pred_diffs["E_value CI"] = evalues_ci
    else:
        pred_diffs["SE"] = np.nan
        pred_diffs["CI_lower"] = np.nan
        pred_diffs["CI_upper"] = np.nan
        pred_diffs["p_value"] = np.nan
        evalues, _ = get_evalues_rd(pred_diffs["Pt Est"])
        pred_diffs["E_value"] = evalues
        pred_diffs["E_value CI"] = np.nan

    return pred_diffs
