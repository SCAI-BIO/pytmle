import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from get_initial_estimates import *
from get_influence_curve import *
from tmle_update import *

def do_pytmle(data_table, target_time, target_event, regime, cv_folds, model, max_update_iter, 
              one_step_eps, min_nuisance, verbose=False, g_comp=False, return_models=False):
    """
    Executes the TMLE procedure with continuous-time one-step targeting for cause-specific absolute risks.
    
    Args:
        data_table (pd.DataFrame): Data containing event time and event type information.
        target_time (list): List of target time points for risk estimation.
        target_event (list): List of target events for TMLE targeting.
        regime (list): List specifying different intervention regimes of interest.
        cv_folds (list): Cross-validation folds for initial model estimation.
        model (dict): Dictionary of model configurations for covariate adjustment.
        max_update_iter (int): Maximum number of TMLE update iterations.
        one_step_eps (float): Step size for the TMLE update.
        min_nuisance (float): Minimum threshold for nuisance values to ensure stability.
        verbose (bool): If True, prints diagnostic information at each iteration.
        g_comp (bool): If True, performs G-computation as part of the initial EIC calculation.
        return_models (bool): If True, returns fitted models for further inspection.

    Returns:
        dict: Final TMLE estimates, with convergence status and diagnostics.
    """
    # Step 1: Initial Estimation
    estimates = get_initial_estimate(
        data=data_table, 
        model=model, 
        cv_folds=cv_folds, 
        min_nuisance=min_nuisance, 
        target_event=target_event, 
        target_time=target_time, 
        regime=regime, 
        return_models=return_models
    )

    # Step 2: Calculate Initial EIC (and possibly G-Comp estimates)
    estimates = get_eic(
        estimates=estimates, 
        data=data_table, 
        regime=regime, 
        target_event=target_event, 
        target_time=target_time, 
        min_nuisance=min_nuisance, 
        g_comp=g_comp
    )

    # Step 3: Check if EIC criteria are met and calculate NormPnEIC
    summ_eic = pd.concat(
        [pd.DataFrame({"Trt": name, **estimate["SummEIC"]}) for name, estimate in estimates.items()],
        ignore_index=True
    )
    norm_pn_eic = get_norm_pn_eic(
        summ_eic[(summ_eic["Time"].isin(target_time)) & (summ_eic["Event"].isin(target_event))]["PnEIC"]
    )
    one_step_stop = summ_eic.assign(
        check=lambda df: abs(df["PnEIC"]) <= df["seEIC/(sqrt(n)log(n))"],
        ratio=lambda df: abs(df["PnEIC"]) / df["seEIC/(sqrt(n)log(n))"]
    )

    # Step 4: Print Diagnostics
    if verbose:
        print_one_step_diagnostics(one_step_stop, norm_pn_eic)

    # Step 5: TMLE Update Loop (if necessary)
    if not all(one_step_stop["check"]):
        estimates = do_tmle_update(
            estimates=estimates, 
            summ_eic=summ_eic, 
            data=data_table, 
            target_event=target_event, 
            target_time=target_time, 
            max_update_iter=max_update_iter, 
            one_step_eps=one_step_eps, 
            norm_pn_eic=norm_pn_eic, 
            verbose=verbose
        )
    else:
        estimates["TmleConverged"] = {"converged": True, "step": 0}
        estimates["NormPnEICs"] = norm_pn_eic

    # Step 6: Set Additional Attributes
    estimates["TargetTime"] = target_time
    estimates["T.tilde"] = data_table[data_table.attrs.get("EventTime")]
    estimates["TargetEvent"] = target_event
    estimates["Delta"] = data_table[data_table.attrs.get("EventType")]
    estimates["GComp"] = g_comp
    estimates["class"] = ["ConcreteEst"] + list(estimates.get("class", []))

    return estimates

def print_concrete_est(estimates):
    """
    Custom print method for ConcreteEst-like object, summarizing TMLE results, convergence, and diagnostics.
    
    Args:
        estimates (dict): A dictionary containing TMLE results with metadata and initial fit information.
    """
    # Print intervention, target event, and target time information
    interventions = ", ".join(f'"{name}"' for name in estimates.keys())
    target_events = ", ".join(map(str, estimates.get("TargetEvent", [])))
    target_times = estimates.get("TargetTime", [])
    target_time_str = (
        f"{', '.join(map(str, target_times[:3]))}, ..., {', '.join(map(str, target_times[-3:]))}"
        if len(target_times) > 6
        else ", ".join(map(str, target_times))
    )

    print("Continuous-Time One-Step TMLE targeting the Cause-Specific Absolute Risks for:")
    print(f"Intervention{'s' if len(estimates) > 1 else ''}: {interventions}  |  ", end="")
    print(f"Target Event{'s' if len(target_events) > 1 else ''}: {target_events}  |  ", end="")
    print(f"Target Time{'s' if len(target_times) > 1 else ''}: {target_time_str}\n")

    # Convergence status
    tmle_converged = estimates.get("TmleConverged", {}).get("converged", False)
    step = estimates.get("TmleConverged", {}).get("step", "unknown")
    print(f"{'TMLE converged at step ' + str(step) if tmle_converged else '**TMLE did not converge!!**'}\n")

    if not tmle_converged:
        # Diagnostic information if TMLE did not converge
        pn_eics = []
        for name, estimate in estimates.items():
            if "SummEIC" in estimate:
                summ_eic = estimate["SummEIC"]
                summ_eic["Intervention"] = name
                pn_eics.append(summ_eic)

        pn_eics_df = pd.concat(pn_eics, ignore_index=True) if pn_eics else pd.DataFrame()
        
        if not pn_eics_df.empty:
            pn_eics_df["|Pn EIC| / Stop Criteria"] = pn_eics_df["PnEIC"] / pn_eics_df["seEIC/(sqrt(n)log(n))"]
            pn_eics_df = pn_eics_df[pn_eics_df["|Pn EIC| / Stop Criteria"] > 1]
            print(pn_eics_df.sort_values(by="|Pn EIC| / Stop Criteria", ascending=False).head(3).to_string(index=False))
            print("\n")

    # Print messages for each Nuisance Weight
    for name, estimate in estimates.items():
        if "NuisanceWeight" in estimate and hasattr(estimate["NuisanceWeight"], "message"):
            print(f'{estimate["NuisanceWeight"].message}\n')

    # Initial Estimators
    print("Initial Estimators:\n")
    initial_fits = estimates.get("InitFits", {})
    delta_values = set(estimates.get("Delta", []))

    for treatment, fit in initial_fits.items():
        if treatment not in delta_values:
            print(f'Treatment "{treatment}":')
            if "SuperLearner" in fit:
                if isinstance(fit, pd.DataFrame):
                    fit.columns = fit.columns.str.replace("Coef", "SL Weight")
                    print(fit)
                else:
                    print(pd.DataFrame({"Risk": fit["cvRisk"], "SL Weight": fit["coef"]}))
                print("\n")
            else:
                print(f'Treatment "{treatment}": Printing for non-"SuperLearner" learners not yet enabled\n')

    # Event-specific and censoring information
    for delta in sorted(delta_values):
        fit = initial_fits.get(str(delta))
        print(f"{'Cens. ' if delta <= 0 else 'Event '}{delta}:")
        if fit:
            print(pd.DataFrame({"Risk": fit["SupLrnCVRisks"], "Coef": fit["SLCoef"]}))
            print("\n")

def plot_concrete_est(estimates, convergence=False, gweights=True, ask=False):
    """
    Generates diagnostic plots for a ConcreteEst-like object containing TMLE results.
    
    Args:
        estimates (dict): A dictionary-like object containing TMLE results, 
                          including "NormPnEICs" for convergence and "NuisanceWeight" for weights.
        convergence (bool): If True, plots the TMLE convergence plot.
        gweights (bool): If True, plots the distribution of intervention-related nuisance weights.
        ask (bool): If True, prompts before showing each plot.
    """
    figs = {}

    # Convergence Plot
    if convergence:
        norm_pn_eic = estimates.get("NormPnEICs", [])
        if norm_pn_eic:
            fig_conv, ax = plt.subplots()
            ax.plot(range(len(norm_pn_eic)), norm_pn_eic, marker='o', linestyle='-')
            ax.set_title("TMLE Convergence")
            ax.set_xlabel("TMLE step")
            ax.set_ylabel("PnEIC Norm")
            figs["TMLEConvergence"] = fig_conv
            plt.show() if ask else plt.close(fig_conv)

    # Distribution of Nuisance Weights
    if gweights:
        weight_data = []
        for intervention, est in estimates.items():
            if "NuisanceWeight" in est:
                nuisance_weight = np.array(est["NuisanceWeight"])
                original_weight = nuisance_weight if nuisance_weight.ndim == 1 else nuisance_weight.mean(axis=0)
                weight_data.extend([{"Intervention": intervention, "gDenomWeight": w} for w in original_weight])

        if weight_data:
            weight_df = pd.DataFrame(weight_data)
            fig_gweights, ax = plt.subplots()
            sns.kdeplot(data=weight_df, x="gDenomWeight", hue="Intervention", ax=ax, cut=0)
            ax.axvline(5 / (np.sqrt(len(weight_df)) * np.log(len(weight_df))), color="red", linestyle="--")
            ax.set_title("Distribution of Intervention-Related Nuisance Weights")
            ax.set_xlabel(r"$\pi(a|w) S_c(t|a,w)$")
            ax.set_ylabel("Density")
            figs["PropScores"] = fig_gweights
            plt.show() if ask else plt.close(fig_gweights)

    return figs
