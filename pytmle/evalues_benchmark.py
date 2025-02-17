import logging
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Generator, List, Tuple

logger = logging.getLogger(__name__)

class EvaluesBenchmark:
    """
    Class to compute observed covariate E-values like proposed by McGowan and Greevy (2020) (https://arxiv.org/pdf/2011.07030).
    """

    def __init__(self, model=None):
        if model is not None:
            self.model = deepcopy(model)
            # set to None to avoid infinite recursion
            self.model.evalues_benchmark = None
            self.benchmark_features = model._feature_names
            self.skip_benchmark = False 
        else:
            self.skip_benchmark = True
        self.benchmarking_results = None

    def benchmark(self, full_model, max_updates=100, **kwargs):
        self.rr_full = full_model.predict("ratio")
        self.rd_full = full_model.predict("diff")
        self.rr_full["Limiting bound"] = np.where(self.rr_full["E_value CI limit"]=="lower", self.rr_full["CI_lower"], self.rr_full["CI_upper"])
        # transformed RR and CIs proposed by VanderWeele and Ding (2017)
        self.rd_full["RR"] = np.exp(0.91*self.rd_full["Pt Est"])
        self.rd_full["Limiting bound"]  = np.where(self.rd_full["E_value CI limit"]=="lower", 
                              np.exp(0.91*self.rd_full["Pt Est"]-1.78* self.rd_full["SE"]), 
                              np.exp(0.91*self.rd_full["Pt Est"]+1.78* self.rd_full["SE"]))
        if self.skip_benchmark:
            return
        if max_updates > 100:
            logger.warning(
                f"Running E-values benchmark can take a long time because a PyTMLE model is fitted with up to {max_updates} for each of {len(self.benchmark_features)} features. Consider reducing the max_updates."
            )
        evalues_df_list = []
        for i, f in enumerate(self.benchmark_features):
            logger.info(f"Computing E-Value benchmark for {f}...")
            tmle = deepcopy(self.model)
            tmle._X = np.delete(tmle._X, i, axis=1)
            tmle.fit(max_updates=max_updates, **kwargs)
            # get ratio estimates for the benchmark model
            rr = tmle.predict("ratio")
            rr["type"] = "ratio"
            rr["benchmark_feature"] = f
            ci_rr = np.where(self.rr_full["E_value CI limit"]=="lower", rr["CI_lower"], rr["CI_upper"])
            rr["E_value measured"] = [
                self._observed_covariate_evalue(ci, ci_new) 
                for ci, ci_new in zip( self.rr_full["Limiting bound"], ci_rr)
            ]
            # get diff estimates for the benchmark model
            rd = tmle.predict("diff")
            rd["type"] = "diff"
            rd["benchmark_feature"] = f
            ci_rd = np.where(self.rd_full["E_value CI limit"]=="lower", 
                            np.exp(0.91*rd["Pt Est"]-1.78* rd["SE"]), 
                            np.exp(0.91*rd["Pt Est"]+1.78* rd["SE"]))
            rd["E_value measured"] = [
                self._observed_covariate_evalue(ci, ci_new) 
                for ci, ci_new in zip(self.rd_full["Limiting bound"], ci_rd)
            ]
            evalues_df_list.append(rr[["benchmark_feature", "type", "Time", "Event", "E_value measured"]])
            evalues_df_list.append(rd[["benchmark_feature", "type", "Time", "Event", "E_value measured"]])
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
            logger.warning("Observed E-values are not defined for non-positive limiting bounds.")
            return np.nan
        if ci < 1:
            ci = 1 / ci
            new_ci = 1 / new_ci
        if ci < new_ci:
            ratio = new_ci / ci
        else:
            ratio = ci / new_ci

        return ratio + (ratio * (ratio - 1))**0.5
    
    def plot(self, 
            target_times: List[float], 
            target_events: List[int], 
            ate_type: str,
            num_points_per_contour: int,
            color_point_estimate: str, 
            color_ci: str, 
            color_benchmarking: str,
            plot_size: Tuple[float, float]) -> Generator[tuple, None, None]:
        for ev in target_events:
            for t in target_times:
                yield self._plot(num_points_per_contour=num_points_per_contour, 
                        color_point_estimate=color_point_estimate, 
                        color_ci=color_ci, 
                        color_benchmarking=color_benchmarking, 
                        plot_size=plot_size, 
                        target_event=ev,
                        target_time=t,
                        ate_type=ate_type) + (t, ev)


    def _plot(self, 
              target_time: float,
              target_event: int,
              ate_type: str,
              num_points_per_contour: int, 
              color_point_estimate: str, 
              color_ci: str, 
              color_benchmarking: str, 
              plot_size: tuple, 
              **kwargs):
        fig, ax = plt.subplots(1, 1, figsize=plot_size)

        if ate_type == "ratio":
            full_df = self.rr_full[(self.rr_full["Time"] == target_time) & 
                                 (self.rr_full["Event"] == target_event)]
            rr = full_df["Pt Est"].item()
            if self.benchmarking_results is not None:
                benchmark_df = self.benchmarking_results[(self.benchmarking_results["Time"] == target_time) &
                                                    (self.benchmarking_results["Event"] == target_event) &
                                                    (self.benchmarking_results["type"] == "ratio")]
                benchmark_df = benchmark_df.sort_values(by="E_value measured", ascending=False).fillna(0)
        elif ate_type == "diff":
            full_df = self.rd_full[(self.rd_full["Time"] == target_time) & 
                                 (self.rd_full["Event"] == target_event)]
            # load the RD transformed to RR
            rr = full_df["RR"].item()
            if self.benchmarking_results is not None:
                benchmark_df = self.benchmarking_results[(self.benchmarking_results["Time"] == target_time) &
                                                    (self.benchmarking_results["Event"] == target_event) &
                                                    (self.benchmarking_results["type"] == "diff")]
                benchmark_df = benchmark_df.sort_values(by="E_value measured", ascending=False).fillna(0)  
        else:
            raise ValueError(f"ate_type must be either 'ratio' or 'diff', got {ate_type}.")      
        eval_est = full_df["E_value"].item()
        if rr < 1:
            rr = 1 / rr

        xy_limit = eval_est * 2

        self._plot_contour(ax, 
                           rr, 
                           eval_est, 
                           num_points_per_contour, 
                           color_point_estimate, 
                           xy_limit)

        eval_ci = full_df["E_value CI"].item()
        if eval_ci is None:
            logger.info("Plotting contour for point estimate only. Confidence interval is not available.")
        elif eval_ci==1:
            logger.info("Plotting contour for point estimate only. Confidence interval is already tipped.")
        else:
            rr_ci = full_df["Limiting bound"].item()
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

        if self.benchmarking_results is not None and any(benchmark_df["E_value measured"] > 1):
            ax.scatter(
                benchmark_df["E_value measured"],
                benchmark_df["E_value measured"],
                label="Observed covariate E-values",
                color=color_benchmarking,
            )
            example_var = benchmark_df.iloc[0]
            obs_evalue = example_var["E_value measured"]
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