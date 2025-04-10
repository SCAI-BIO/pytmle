import warnings
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Generator, List, Tuple

class EvaluesBenchmark:
    """
    Class to compute observed covariate E-values like proposed by McGowan and Greevy (2020) (https://arxiv.org/pdf/2011.07030).
    """

    def __init__(
        self,
        model=None,
        verbose=2,
    ):
        if model is not None:
            self.model = deepcopy(model)
            # set to None to avoid infinite recursion
            self.model.evalues_benchmark = None
            self.benchmark_features = model._feature_names
            self.skip_benchmark = False 
        else:
            self.skip_benchmark = True
        self.benchmarking_results = None
        self.verbose = verbose

    def benchmark(
        self,
        full_model,
        max_updates: int = 100,
        alpha: float = 0.05,
        **kwargs,
    ):
        self.rr_full = full_model.predict("rr", alpha=alpha)
        self.rd_full = full_model.predict("rd", alpha=alpha)
        self.rr_full["Limiting bound"] = np.where(self.rr_full["E_value CI limit"]=="lower", self.rr_full["CI_lower"], self.rr_full["CI_upper"])
        # transformed RR and CIs proposed by VanderWeele and Ding (2017)
        self.rd_full["RR"] = np.exp(0.91 * self.rd_full["Pt Est"])
        self.rd_full["Limiting bound"] = np.where(
            self.rd_full["E_value CI limit"] == "lower",
            np.exp(
                0.91 * self.rd_full["Pt Est"]
                - 0.91 * norm.ppf(1 - alpha / 2) * self.rd_full["SE"]
            ),
            np.exp(
                0.91 * self.rd_full["Pt Est"]
                + 0.91 * norm.ppf(1 - alpha / 2) * self.rd_full["SE"]
            ),
        )
        if full_model._bootstrap_results is not None:
            self.rr_full["Limiting bound (bootstrap)"] = np.where(
                self.rr_full["E_value CI limit (bootstrap)"] == "lower",
                self.rr_full["CI_lower_bootstrap"],
                self.rr_full["CI_upper_bootstrap"],
            )
            self.rd_full["Limiting bound (bootstrap)"] = np.where(
                self.rd_full["E_value CI limit (bootstrap)"] == "lower",
                # transformed bootstrapped quantile-based CIs
                np.exp(0.91 * self.rd_full["CI_lower_bootstrap"]),
                np.exp(0.91 * self.rd_full["CI_upper_bootstrap"]),
            )
        if self.skip_benchmark:
            return
        if max_updates > 100 and self.verbose >= 1:
            warnings.warn(
                f"Running E-values benchmark can take a long time because a PyTMLE model is fitted with up to {max_updates} for each of {len(self.benchmark_features)} features. Consider reducing the max_updates.",
                RuntimeWarning,
            )
        evalues_df_list = []
        for i, f in enumerate(self.benchmark_features):
            if self.verbose >= 2:
                print(f"Computing E-Value benchmark for {f}...")
            tmle = deepcopy(self.model)
            # print less for each E-value benchmark model
            tmle.verbose = self.verbose - 1 if self.verbose < 4 else self.verbose
            # feature is not dropped but set to np.nan for model-specific handling
            tmle._X[:, i] = np.nan
            tmle.fit(max_updates=max_updates, **kwargs)
            # get ratio estimates for the benchmark model
            rr = tmle.predict("rr")
            rr["type"] = "rr"
            rr["benchmark_feature"] = f
            ci_rr = np.where(self.rr_full["E_value CI limit"]=="lower", rr["CI_lower"], rr["CI_upper"])
            rr["E_value measured"] = [
                self._observed_covariate_evalue(ci, ci_new)
                for ci, ci_new in zip(self.rr_full["Limiting bound"], ci_rr)
            ]
            # get diff estimates for the benchmark model
            rd = tmle.predict("rd")
            rd["type"] = "rd"
            rd["benchmark_feature"] = f
            ci_rd = np.where(
                self.rd_full["E_value CI limit"] == "lower",
                # transformation for SE-based CIs
                np.exp(0.91 * rd["Pt Est"] - 0.91 * norm.ppf(1 - alpha / 2) * rd["SE"]),
                np.exp(0.91 * rd["Pt Est"] + 0.91 * norm.ppf(1 - alpha / 2) * rd["SE"]),
            )
            rd["E_value measured"] = [
                self._observed_covariate_evalue(ci, ci_new)
                for ci, ci_new in zip(self.rd_full["Limiting bound"], ci_rd)
            ]
            if (
                full_model._bootstrap_results is not None
                and tmle._bootstrap_results is not None
            ):
                ci_rr_bs = np.where(
                    self.rr_full["E_value CI limit (bootstrap)"] == "lower",
                    rr["CI_lower_bootstrap"],
                    rr["CI_upper_bootstrap"],
                )
                rr["E_value measured (bootstrap)"] = [
                    self._observed_covariate_evalue(ci, ci_new)
                    for ci, ci_new in zip(
                        self.rr_full["Limiting bound (bootstrap)"], ci_rr_bs
                    )
                ]
                ci_rd_bs = np.where(
                    self.rd_full["E_value CI limit (bootstrap)"] == "lower",
                    # transformation for bootstrapped quantile-based CIs
                    np.exp(0.91 * rd["CI_lower_bootstrap"]),
                    np.exp(0.91 * rd["CI_upper_bootstrap"]),
                )
                rd["E_value measured (bootstrap)"] = [
                    self._observed_covariate_evalue(ci, ci_new)
                    for ci, ci_new in zip(
                        self.rd_full["Limiting bound (bootstrap)"], ci_rd_bs
                    )
                ]
            else:
                rr["E_value measured (bootstrap)"] = np.nan
                rd["E_value measured (bootstrap)"] = np.nan
            evalues_df_list.append(
                rr[
                    [
                        "benchmark_feature",
                        "type",
                        "Time",
                        "Event",
                        "E_value measured",
                        "E_value measured (bootstrap)",
                    ]
                ]
            )
            evalues_df_list.append(
                rd[
                    [
                        "benchmark_feature",
                        "type",
                        "Time",
                        "Event",
                        "E_value measured",
                        "E_value measured (bootstrap)",
                    ]
                ]
            )
        self.benchmarking_results = pd.concat(evalues_df_list, ignore_index=True)

    def _observed_covariate_evalue(self, ci, new_ci):
        """
        Compute the E-value for the observed covariate as proposed
        by McGowan and Greevy (2020).

        Args:
            ci (float): Confidence interval for the original model.
            new_ci (float): Confidence interval for the benchmark model.
        """
        # lower CIs < 0 can occur but should be ignored for the E-value calculation
        if ci <= 0 or new_ci <= 0:
            if self.verbose >= 1:
                warnings.warn(
                    "Observed E-values are not defined for non-positive limiting bounds.",
                    RuntimeWarning,
                )
            return np.nan
        if ci < 1:
            ci = 1 / ci
            new_ci = 1 / new_ci
        if ci < new_ci:
            ratio = new_ci / ci
        else:
            ratio = ci / new_ci

        return ratio + (ratio * (ratio - 1))**0.5

    def plot(
        self,
        target_times: List[float],
        target_events: List[int],
        ate_type: str,
        num_points_per_contour: int,
        color_point_estimate: str,
        color_ci: str,
        color_benchmarking: str,
        plot_size: Tuple[float, float],
        use_bootstrap: bool = False,
    ) -> Generator[tuple, None, None]:
        for ev in target_events:
            for t in target_times:
                yield self._plot(
                    num_points_per_contour=num_points_per_contour,
                    color_point_estimate=color_point_estimate,
                    color_ci=color_ci,
                    color_benchmarking=color_benchmarking,
                    plot_size=plot_size,
                    target_event=ev,
                    target_time=t,
                    ate_type=ate_type,
                    use_bootstrap=use_bootstrap,
                ) + (t, ev)

    def _plot(
        self,
        target_time: float,
        target_event: int,
        ate_type: str,
        num_points_per_contour: int,
        color_point_estimate: str,
        color_ci: str,
        color_benchmarking: str,
        plot_size: tuple,
        use_bootstrap: bool,
        **kwargs,
    ):
        fig, ax = plt.subplots(1, 1, figsize=plot_size)

        evalue_ci_key = "E_value CI (bootstrap)" if use_bootstrap else "E_value CI"
        limiting_bound_key = (
            "Limiting bound (bootstrap)" if use_bootstrap else "Limiting bound"
        )
        evalue_measured_key = (
            "E_value measured (bootstrap)" if use_bootstrap else "E_value measured"
        )
        if ate_type == "rr":
            full_df = self.rr_full[(self.rr_full["Time"] == target_time) & 
                                 (self.rr_full["Event"] == target_event)]
            rr = full_df["Pt Est"].item()
            if self.benchmarking_results is not None:
                benchmark_df = self.benchmarking_results[
                    (self.benchmarking_results["Time"] == target_time)
                    & (self.benchmarking_results["Event"] == target_event)
                    & (self.benchmarking_results["type"] == "rr")
                ]
                benchmark_df = benchmark_df.sort_values(
                    by=evalue_measured_key, ascending=False
                )
        elif ate_type == "rd":
            full_df = self.rd_full[(self.rd_full["Time"] == target_time) & 
                                 (self.rd_full["Event"] == target_event)]
            # load the RD transformed to RR
            rr = full_df["RR"].item()
            if self.benchmarking_results is not None:
                benchmark_df = self.benchmarking_results[
                    (self.benchmarking_results["Time"] == target_time)
                    & (self.benchmarking_results["Event"] == target_event)
                    & (self.benchmarking_results["type"] == "rd")
                ]
                benchmark_df = benchmark_df.sort_values(
                    by=evalue_measured_key, ascending=False
                )
        else:
            raise ValueError(
                f"ate_type must be either 'ratio' or 'diff', got {ate_type}."
            )
        if (
            not evalue_ci_key in full_df.columns
            or not limiting_bound_key in full_df.columns
        ):
            raise ValueError(
                "Requested E-value confidence intervals are not available."
            )

        eval_est = full_df["E_value"].item()
        if rr < 1:
            rr = 1 / rr

        if self.benchmarking_results is not None:
            is_na = benchmark_df[evalue_measured_key].isna()
            if all(is_na) and self.verbose >= 1:
                warnings.warn(
                    "No observed E-values available for the benchmarking features.",
                    RuntimeWarning,
                )
            elif sum(is_na) > 0 and self.verbose >= 1:
                warnings.warn(
                    f"Observed E-values are not available for {sum(is_na)} out of {len(is_na)} features.",
                    RuntimeWarning,
                )
            benchmark_df = benchmark_df.fillna(0)
            xy_limit = max(eval_est, np.max(benchmark_df[evalue_measured_key])) * 2
        else:
            xy_limit = eval_est * 2

        self._plot_contour(ax, 
                           rr, 
                           eval_est, 
                           num_points_per_contour, 
                           color_point_estimate, 
                           xy_limit)

        eval_ci = full_df[evalue_ci_key].item()
        if eval_ci is None and self.verbose >= 2:
            print(
                "Plotting contour for point estimate only. Confidence interval is not available."
            )
        elif eval_ci == 1 and self.verbose >= 2:
            print(
                "Plotting contour for point estimate only. Confidence interval is already tipped."
            )
        else:
            rr_ci = full_df[limiting_bound_key].item()
            if rr_ci < 1:
                rr_ci = 1 / rr_ci

            self._plot_contour(
                ax,
                rr_ci,
                eval_ci,
                num_points_per_contour,
                color_ci,
                xy_limit,
                point_est=False,
            )

        if self.benchmarking_results is not None and any(
            benchmark_df[evalue_measured_key] > 1
        ):
            ax.scatter(
                benchmark_df[evalue_measured_key],
                benchmark_df[evalue_measured_key],
                label="Observed covariate E-values",
                color=color_benchmarking,
            )
            example_var = benchmark_df.iloc[0]
            obs_evalue = example_var[evalue_measured_key]
            ax.text(obs_evalue, obs_evalue, example_var["benchmark_feature"], fontsize=8)

        ax.set(xlabel="$RR_{treatment-confounder}$", ylabel="$RR_{confounder-outcome}$")
        plt.ylim(1, xy_limit)
        plt.xlim(1, xy_limit)
        plt.legend()
        plt.title(f"E-value contours for event {target_event} at time {target_time}")

        return fig, ax

    def _plot_contour(self, ax, rr, evalue, n_pts, color, xy_limit, point_est=True):
        """
        Plots a single contour line. Copied from https://www.pywhy.org/dowhy/v0.12/_modules/dowhy/causal_refuters/evalue_sensitivity_analyzer.html#EValueSensitivityAnalyzer.

        Args:
            ax: Matplotlib axis object
            rr: Point estimate for the Risk Ratio.
            evalue: E-value for the point estimate.
            n_pts: Number of points to plot.
            color: Color of the contour line.
            xy_limit: Limit for the x and y axis.
            point_est: Whether to plot the point estimate or the confidence interval.

        """

        step = (xy_limit - rr) / n_pts
        x_est = np.linspace(rr + step, xy_limit, num=n_pts)
        y_est = rr * (rr - 1) / (x_est - rr) + rr

        est_string = "point estimate" if point_est else "confidence interval"
        ax.scatter(
            evalue,
            evalue,
            label=f"E-value for {est_string}: {np.round(evalue, 2)}",
            color=color,
        )
        ax.fill_between(
            x_est,
            y_est,
            xy_limit,
            color=color,
            alpha=0.2,
            label=f"Tips {est_string}",
        )
        ax.plot(x_est, y_est, color=color)
