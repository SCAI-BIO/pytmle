import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, Generator, Tuple, List

def initialize_subplots(target_events: np.ndarray) -> tuple:
    num_events = len(target_events)
    num_cols = min(3, num_events)
    num_rows = (num_events + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 7 * num_rows))
    axes = axes.flatten() if num_events > 1 else [axes]
    # leave space for legend
    fig.subplots_adjust(right=0.88)
    for ax in axes[num_events:]:
        # remove superfluous axes
        fig.delaxes(ax)

    return fig, axes


def plot_risks(
    tmle_est: pd.DataFrame,
    g_comp_est: Optional[pd.DataFrame] = None,
    color_1: Optional[str] = None,
    color_0: Optional[str] = None,
    use_bootstrap: bool = False,
) -> tuple:
    target_events = np.unique(tmle_est["Event"])
    fig, axes = initialize_subplots(target_events)

    mean_key = "mean_bootstrap" if use_bootstrap else "Pt Est"
    ci_lower_key = "CI_lower_bootstrap" if use_bootstrap else "CI_lower"
    ci_upper_key = "CI_upper_bootstrap" if use_bootstrap else "CI_upper"

    fig.suptitle("Risk Estimates Over Time", fontsize=16)

    all_ci_upper = []

    groups = np.unique(tmle_est["Group"])
    assert len(groups) == 2, "Only two groups are supported for risk plotting."

    for i, event in enumerate(target_events):
        ax = axes[i]
        used_colors = []
        for group, color in zip(groups, [color_0, color_1]):
            time = tmle_est[(tmle_est["Event"] == event) & (tmle_est["Group"] == group)]["Time"].values
            pt_est = tmle_est[
                (tmle_est["Event"] == event) & (tmle_est["Group"] == group)
            ]["Pt Est"].values
            mean = tmle_est[
                (tmle_est["Event"] == event) & (tmle_est["Group"] == group)
            ][mean_key].values
            ci_lower = tmle_est[
                (tmle_est["Event"] == event) & (tmle_est["Group"] == group)
            ][ci_lower_key].values
            ci_upper = tmle_est[
                (tmle_est["Event"] == event) & (tmle_est["Group"] == group)
            ][ci_upper_key].values
            all_ci_upper.append(ci_upper)

            yerr = [mean - ci_lower, ci_upper - mean]

            container = ax.plot(time, pt_est, linestyle="--", color=color, marker="o")
            used_colors.append(container[0].get_color())
            ax.errorbar(
                time, mean, yerr=yerr, capsize=13, color=used_colors[-1], linestyle=""
            )

            if g_comp_est is not None:
                assert all(time == g_comp_est[(g_comp_est["Event"] == event)  & (tmle_est["Group"] == group)]["Time"].values), "Target times do not match for TMLE and g-computation."
                ate_estimates_g_comp = g_comp_est[(g_comp_est["Event"] == event)  & (tmle_est["Group"] == group)]["Pt Est"].values
                ax.scatter(
                    time, ate_estimates_g_comp, color=used_colors[-1], marker="x", s=100
                )

        ax.set_title(f"Event {event}")
        ax.set_xlabel("Time")
        ax.set_xlim(0, None)   
        ax.set_ylabel("Predicted Risk")

    # add legend
    if g_comp_est is not None:
        l1_handle = [Line2D([], [], marker='o', color='black', markersize=6, linestyle='--', markerfacecolor='black', markeredgewidth=1.5),
            Line2D([], [], marker='x', color="black", markersize=6, linestyle='')]
        l1 = fig.legend(l1_handle, ["TMLE", "G-computation"], loc="upper right", title="Estimator", bbox_to_anchor=(1, 0.8))
    l2_handle = [Line2D([], [], marker='', color=used_colors[0], markersize=6, linestyle='--'),
        Line2D([], [], marker='', color=used_colors[1], markersize=6, linestyle='--')]
    l2 = fig.legend(l2_handle, groups, loc='upper right', title='Group', bbox_to_anchor=(1, 0.9))
    if g_comp_est is not None:    
        fig.add_artist(l1)

    # unify y-axis limits across all subplots
    for ax in axes:
        ax.set_ylim(0, max(np.concat(all_ci_upper)) * 1.1)

    return fig, axes


def plot_ate(
    tmle_est: pd.DataFrame,
    g_comp_est: Optional[pd.DataFrame] = None,
    type="rr",
    use_bootstrap: bool = False,
) -> tuple:
    target_events = tmle_est["Event"].unique()
    fig, axes = initialize_subplots(target_events)

    mean_key = "mean_bootstrap" if use_bootstrap else "Pt Est"
    ci_lower_key = "CI_lower_bootstrap" if use_bootstrap else "CI_lower"
    ci_upper_key = "CI_upper_bootstrap" if use_bootstrap else "CI_upper"

    if type == "rr" or type == "rd":
        fig.suptitle("Average Treatment Effect (ATE) Estimates Over Time", fontsize=16)
    else:
        raise ValueError(f"type must be either 'rr' or 'rd', got {type}.")

    all_ci_lower = []
    all_ci_upper = []

    for i, event in enumerate(target_events):
        ax = axes[i]

        time = tmle_est[tmle_est["Event"] == event]["Time"].values
        mean = tmle_est[tmle_est["Event"] == event][mean_key].values
        pt_est = tmle_est[tmle_est["Event"] == event]["Pt Est"].values
        ci_lower = tmle_est[tmle_est["Event"] == event][ci_lower_key].values
        ci_upper = tmle_est[tmle_est["Event"] == event][ci_upper_key].values
        all_ci_lower.append(ci_lower)
        all_ci_upper.append(ci_upper)

        yerr = [mean - ci_lower, ci_upper - mean]

        ax.plot(time, pt_est, linestyle="--", color="black", marker="o")
        ax.errorbar(time, mean, yerr=yerr, capsize=13, color="black", linestyle="")

        if g_comp_est is not None:
            assert all(time == g_comp_est[g_comp_est["Event"] == event]["Time"].values), "Target times do not match for TMLE and g-computation."
            ate_estimates_g_comp = g_comp_est[g_comp_est["Event"] == event]["Pt Est"].values
            ax.scatter(time, 
                ate_estimates_g_comp, 
                color="black",
                marker="x",
                s=100)

        ax.set_title(f"Event {event}")
        ax.set_xlabel("Time")
        ax.set_xlim(0, None)
        if type == "rr":
            ax.set_ylabel("ATE (RR)")
            ax.axhline(y=1, linestyle="--", color="gray", alpha=0.7)
        elif type == "rd":
            ax.set_ylabel("ATE (RD)")
            ax.axhline(y=0, linestyle="--", color="gray", alpha=0.7)

        # add legend
        if g_comp_est is not None:
            l1_handle = [Line2D([], [], marker='o', color='black', markersize=6, linestyle='--', markerfacecolor='black', markeredgewidth=1.5),
                 Line2D([], [], marker='x', color="black", markersize=6, linestyle='')]
            l1 = fig.legend(l1_handle, ["TMLE", "G-computation"], loc="upper right", title="Estimator", bbox_to_anchor=(1, 0.8))

    # unify y-axis limits across all subplots
    min_y = min(np.concat(all_ci_lower))
    max_y = max(np.concat(all_ci_upper))
    if max_y < 0:
        max_y *= 0.9
    else:
        max_y *= 1.1
    if min_y < 0:
        min_y *= 1.1
    else:
        min_y *= 0.9
    for ax in axes:
        ax.set_ylim(min_y, max_y)

    return fig, axes


def plot_nuisance_weights(
    target_times: List[float],
    times: np.ndarray,
    min_nuisance: float,
    nuisance_weights: np.ndarray,
    g_star_obs: np.ndarray,
    plot_size: Tuple[float, float],
    color_1: Optional[str] = None,
    color_0: Optional[str] = None,
) -> Generator[tuple, None, None]:

    times_idx = [i for i, time in enumerate(times) if time in target_times]
    for t_idx, t in zip(times_idx, target_times):
        nuisance_weight = 1 / nuisance_weights[:, t_idx]

        # Filter the data
        weights_g0 = nuisance_weight[g_star_obs == 0]
        weights_g1 = nuisance_weight[g_star_obs == 1]

        # Plot the density functions
        fig, ax = plt.subplots(figsize=plot_size)
        sns.kdeplot(weights_g0, label="0", fill=True, color=color_0)
        sns.kdeplot(weights_g1, label="1", fill=True, color=color_1)

        # vertical line for min_nuisance
        plt.axvline(x=min_nuisance, color="gray", linestyle="--", label="Min. Nuisance")

        plt.suptitle(f'Nuisance weights at time t={t} for positivity check', fontsize=15)
        if t==0:
            plt.title('Weights close to 0 or 1 warn of possible positivity violations', fontsize=13)
        else:
            plt.title('Weights close to 0 warn of possible positivity violations', fontsize=13)
        plt.xlabel(r"$\pi(a|w) \, G(t|a,w)$", fontsize=13)
        plt.xlim(0,1)
        plt.ylabel('Density', fontsize=13)
        plt.legend(title="Group")

        yield fig, ax, t
