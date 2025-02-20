from .estimates import InitialEstimates
from .get_initial_estimates import fit_default_propensity_model, fit_default_risk_model, fit_default_censoring_model
from .tmle_update import tmle_update
from .predict_ate import (
    get_counterfactual_risks,
    ate_ratio,
    ate_diff,
)
from .evalues_benchmark import EvaluesBenchmark
from .plotting import plot_risks, plot_ate, plot_nuisance_weights

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PyTMLE:

    def __init__(
        self,
        data: pd.DataFrame,
        col_event_times: str = "event_time",
        col_event_indicator: str = "event_indicator",
        col_group: str = "group",
        target_times: Optional[List[float]] = None,
        target_events: List[int] = [1],
        g_comp: bool = True,
        evalues_benchmark: bool = False,
        key_1: int = 1,
        key_0: int = 0,
        initial_estimates: Optional[Dict[int, InitialEstimates]] = None,
        verbose: bool = True,
    ):
        """
        Initialize the PyTMLE model.

        Parameters
        ----------
        data : pd.DataFrame
            The input data containing event times, event indicators, group information, and predictors.
        col_event_times : str, optional
            The column name in the data that contains event times. Default is "event_time".
        col_event_indicator : str, optional
            The column name in the data that contains event indicators. Default is "event_indicator".
        col_group : str, optional
            The column name in the data that contains group information. Default is "group".
        target_times : Optional[List[float]], optional
            Specific times at which to estimate the target parameter. If None, estimates for the last observed event time are used. Default is None.
        target_events : List[int], optional
            List of event types to target. Default is [1].
        g_comp : bool, optional
            Whether to use g-computation for initial estimates. Default is True.
        evalues_benchmark : bool, optional
            Whether to compute E-values for measured confounders. Default is False.
        key_1 : int, optional
            The key representing the treatment group. Default is 1.
        key_0 : int, optional
            The key representing the control group. Default is 0.
        initial_estimates : Optional[Dict[int, InitialEstimates]], optional
            Dict with pre-computed initial estimates for the two potential outcomes. Default is None.
        verbose : bool, optional
            Whether to print verbose output. Default is True.
        """
        self._check_inputs(data, 
                           col_event_times, 
                           col_event_indicator, 
                           col_group, 
                           target_times, 
                           target_events, 
                           key_1, 
                           key_0, 
                           initial_estimates)
        self._initial_estimates = initial_estimates
        self._updated_estimates = None
        self._X = data.drop(
            columns=[col_event_times, col_event_indicator, col_group]
        ).to_numpy()
        self._feature_names = data.drop(
            columns=[col_event_times, col_event_indicator, col_group]
        ).columns
        self._event_times = data[col_event_times].to_numpy()
        self._event_indicator = data[col_event_indicator].to_numpy()
        self._group = data[col_group].to_numpy()
        if target_times is None:
            # default behavior: Estimates for last observed event time
            self.target_times = [max(self._event_times)]
        else:
            self.target_times = target_times
        self.target_events = target_events
        self.g_comp = g_comp
        self.key_1 = key_1
        self.key_0 = key_0
        self.verbose = verbose
        self._fitted = False
        self.has_converged = False
        self.step_num = 0
        self.norm_pn_eics = []
        self.models = {}
        if evalues_benchmark:
            if initial_estimates is not None:
                logger.warning(
                     "E-values benchmark for measured covariates may be incorrect if pre-computed initial estimates are provided because the measured covariates need to be dropped during model fitting."
                )
            self.evalues_benchmark = EvaluesBenchmark(self)
        else:
            self.evalues_benchmark = EvaluesBenchmark()

    def _check_inputs(
                self, 
                data: pd.DataFrame, 
                col_event_times: str, 
                col_event_indicator: str, 
                col_group: str, 
                target_times: Optional[List[float]],
                target_events: List[int],
                key_1: int,
                key_0: int,
                initial_estimates: Optional[Dict[int, InitialEstimates]]):
        if col_event_times not in data.columns:
            raise ValueError(f"Column {col_event_times} not found in the given data.")
        if col_event_indicator not in data.columns:
            raise ValueError(f"Column {col_event_indicator} not found in the given data.")
        if col_group not in data.columns:
            raise ValueError(f"Column {col_group} not found in the given data.")
        if len(data[col_group].unique()) != 2:
            raise ValueError("Only two groups are supported.")
        if initial_estimates is not None:
            if (
                key_1 not in initial_estimates.keys()
                or key_0 not in initial_estimates.keys()
            ):
                raise ValueError(
                    "key_1 and key_0 have to be in line with the keys of the given initial estimates."
                )
        if not (data[col_event_indicator] == 0).any():
            raise ValueError("Censoring has to be indicated by 0 in the event_indicator column.")
        unique_events = np.unique(data[col_event_indicator])
        if not unique_events[-1] - unique_events[0] == len(unique_events) - 1:
            raise ValueError(f"Event indicators have to be consecutive integers. Got {unique_events}.")
        if not all(np.isin(target_events, data[col_event_indicator])):
            raise ValueError("All target events have to be in the event_indicator column.")
        if target_times is not None and not max(target_times) <= max(data[col_event_times]):
            raise ValueError("All target times have to be smaller or equal to the maximum event time in the data.")
        if target_times is not None and min(target_times) < 0:
            raise ValueError("All target times have to be positive.")
        # TODO: Either make the selection of target events more flexible or remove the option altogether (such that all non-zero events are targeted).
        if not (np.array_equal(target_events, np.arange(1, len(target_events) + 1))):
            raise ValueError(
                "target_events must be consecutive integers starting from 1."
            )

    def _get_initial_estimates(self, cv_folds: int, save_models: bool):
        if self._initial_estimates is None:
            self._initial_estimates = {self.key_1:
                                       InitialEstimates(g_star_obs = self._group,
                                                        times=np.unique(self._event_times)),
                                        self.key_0:
                                        InitialEstimates(g_star_obs = 1 - self._group,
                                                        times=np.unique(self._event_times)),
                                        }

        if (self._initial_estimates[self.key_1].propensity_scores is None or 
            self._initial_estimates[self.key_0].propensity_scores is None):
            logger.info("Estimating propensity scores...")
            propensity_scores_1, propensity_scores_0, model_dict = (
                fit_default_propensity_model(
                    X=self._X,
                    y=self._group,
                    cv_folds=cv_folds,
                    return_model=save_models,
                )
            )
            self.models.update(model_dict)
            self._initial_estimates[self.key_1].propensity_scores = propensity_scores_1
            self._initial_estimates[self.key_0].propensity_scores = propensity_scores_0
        else:
            logger.info("Using given propensity score estimates")
        if (self._initial_estimates[self.key_1].hazards is None or 
            self._initial_estimates[self.key_0].hazards is None or
            self._initial_estimates[self.key_1].event_free_survival_function is None or 
            self._initial_estimates[self.key_0].event_free_survival_function is None):
            logger.info("Estimating hazards and event-free survival...")
            hazards_1, hazards_0, surv_1, surv_0, model_dict = fit_default_risk_model(
                X=self._X,
                trt=self._group,
                event_times=self._event_times,
                event_indicator=self._event_indicator,
                target_events=self.target_events,
                cv_folds=cv_folds,
                return_model=save_models,
            )
            self.models.update(model_dict)
            self._initial_estimates[self.key_1].hazards = hazards_1
            self._initial_estimates[self.key_1].event_free_survival_function = surv_1
            self._initial_estimates[self.key_0].hazards = hazards_0
            self._initial_estimates[self.key_0].event_free_survival_function = surv_0
        else:
            logger.info("Using given hazard and event-free survival estimates")
        if (self._initial_estimates[self.key_1].censoring_survival_function is None or 
            self._initial_estimates[self.key_0].censoring_survival_function is None):
            logger.info("Estimating censoring survival...")
            cens_surv_1, cens_surv_0, model_dict = fit_default_censoring_model(
                X=self._X,
                trt=self._group,
                event_times=self._event_times,
                event_indicator=self._event_indicator,
                cv_folds=cv_folds,
                return_model=save_models,
            )
            self.models.update(model_dict)
            self._initial_estimates[self.key_1].censoring_survival_function = cens_surv_1
            self._initial_estimates[self.key_0].censoring_survival_function = cens_surv_0
        else:
            logger.info("Using given censoring survival estimates")

    def _update_estimates(
        self, max_updates: int, min_nuisance: Optional[float], one_step_eps: float
    ):
        assert self._initial_estimates is not None, "Initial estimates have to be available before calling _update_estimates()."
        for k in self._initial_estimates:
            assert self._initial_estimates[k] is not None, "Initial estimates have to be available before calling _update_estimates()."
        logger.info("Starting TMLE update loop...")
        (
            self._updated_estimates,
            self.norm_pn_eics,
            self.has_converged,
            self.step_num,
        ) = tmle_update(
            self._initial_estimates,
            event_times=self._event_times,
            event_indicator=self._event_indicator,
            target_times=self.target_times,
            target_events=self.target_events,
            max_updates=max_updates,
            min_nuisance=min_nuisance,
            one_step_eps=one_step_eps,
            g_comp=self.g_comp,
            verbose=self.verbose,
        )  # type: ignore

    def fit(
        self,
        cv_folds: int = 10,
        max_updates: int = 500,
        min_nuisance: Optional[float] = None,
        one_step_eps: float = 0.1,
        save_models: bool = False,
        alpha: float = 0.05,
    ):
        """
        Fit the TMLE model.

        Parameters
        ----------
        cv_folds : int, optional
            Number of cross-validation folds for the initial estimate models.
            The number is the same for the inner and outer loop used by the super learners. Default is 10.
        max_updates : int
            Maximum number of updates to the estimates in the TMLE loop. Default is 500.
        min_nuisance : Optional[float], optional
            Value between 0 and 1 for truncating the g-related denomiator of the clever covariate. Default is None.
        one_step_eps : float
            Initial epsilon for the one-step update. Default is 0.1.
        save_models : bool, optional
            Whether to save the models used for the initial estimates. Default is False.
        alpha : float, optional 
            The alpha level for confidence intervals (relevant only for E-value benchmark). Default is 0.05.
        """
        if self._fitted:
            raise RuntimeError("Model has already been fitted. fit() can only be called once.")
        self._get_initial_estimates(cv_folds, save_models)
        self._update_estimates(max_updates, min_nuisance, one_step_eps)
        self._fitted = True

        # running E-value benchmark
        if self.evalues_benchmark is not None:
            self.evalues_benchmark.benchmark(
                full_model=self,
                cv_folds=cv_folds,
                max_updates=max_updates,
                min_nuisance=min_nuisance,
                one_step_eps=one_step_eps,
                alpha=alpha,
            )

    def predict(self, type: str = "risks", alpha: float = 0.05, g_comp: bool = False) -> pd.DataFrame:
        """
        Predict the counterfactual risks or average treatment effect.

        Parameters
        ----------
        type : str, optional
            The type of prediction. "risks", "ratio" and "diff" are supported. Default is "risks".
        alpha : float, optional 
            The alpha level for confidence intervals. Default is 0.05.
        g_comp : bool, optional
            Whether to return the g-computation estimates instead of the updated estimates. Default is False.
        """
        if not self._fitted or self._updated_estimates is None:
            raise RuntimeError("Model has to be fitted before calling predict().")
        if type == "risks":
            return get_counterfactual_risks(self._updated_estimates, 
                                            g_comp=g_comp,
                                            alpha=alpha,
                                            key_1=self.key_1,
                                            key_0=self.key_0)
        elif type == "ratio":
            return ate_ratio(self._updated_estimates, 
                            g_comp=g_comp,
                            alpha=alpha,
                            key_1=self.key_1,
                            key_0=self.key_0)
        elif type == "diff":
            return ate_diff(self._updated_estimates, 
                            g_comp=g_comp,
                            alpha=alpha,
                            key_1=self.key_1,
                            key_0=self.key_0)
        else: 
            raise ValueError(
                f"Only 'risks', 'ratio' and 'diff' are supported as type, got {type}."
            )

    def plot(self, 
             save_path: Optional[str] = None,
             type: str = "risks", 
             alpha: float = 0.05, 
             g_comp: bool = False,
             color_1: Optional[str] = None, 
             color_0: Optional[str] = None) -> tuple:
        """
        Plot the counterfactual risks or average treatment effect.

        Parameters
        ----------
        save_path : Optional[str], optional
            Path to save the plot. Default is None.
        type : str, optional
            The type of prediction. "risks", "ratio" and "diff" are supported. Default is "risks".
        alpha : float, optional
            The alpha level for confidence intervals. Default is 0.05.
        g_comp : bool, optional
            Whether to return the g-computation estimates instead of the updated estimates. Default is False.
        color_1 : Optional[str], optional
            Color for the treatment group. Default is None.
        color_0 : Optional[str], optional
            Color for the control group. Default is None.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : np.ndarray
            The axes objects.
        """
        if type == "risks":
            pred = self.predict(type=type, alpha=alpha)
            if g_comp:
                pred_g_comp = self.predict(type=type, alpha=alpha, g_comp=True)
            fig, axes = plot_risks(pred,  
                    pred_g_comp if g_comp else None,
                    color_1=color_1, 
                    color_0=color_0)
        elif type == "ratio" or type == "diff":
            pred = self.predict(type=type, alpha=alpha)
            if g_comp:
                pred_g_comp = self.predict(type=type, alpha=alpha, g_comp=True)
            fig, axes= plot_ate(pred,
                                pred_g_comp if g_comp else None,
                                type=type)
        else: 
            raise ValueError(f"Only 'risks', 'ratio' and 'diff' are supported as type, got {type}.")

        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        return fig, axes

    def plot_nuisance_weights(
        self,
        time: Optional[float] = None,
        save_dir_path: Optional[str] = None,
        color_1: Optional[str] = None,
        color_0: Optional[str] = None,
        plot_size: Tuple[float, float] = (6.4, 4.8),
    ):
        """
        Plot the nuisance weights.

        Parameters
        ----------
        time : Optional[float], optional
            Time at which to plot the nuisance weights. If None, all target times are plotted. Default is None.
        save_dir_path : Optional[str], optional
            Path to directory to save the plots. Default is None.
        color_1 : Optional[str], optional
            Color for the treatment group. Default is None.
        color_0 : Optional[str], optional
            Color for the control group. Default is None.
        """
        if self._updated_estimates is None: 
            raise RuntimeError("Updated estimates must have been initialized before calling plot_nuisance_weights().")
        if save_dir_path is not None and not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
        if time is not None:
            assert (
                time in self.target_times or time == 0
            ), f"Time has to be 0 or one of the target times {self.target_times}."
            target_times = [time]
        else:
            target_times = [0.0] + list(self.target_times)
        for _, _, time in plot_nuisance_weights(
            target_times=target_times,
            times=self._updated_estimates[self.key_1].times,  # type: ignore
            min_nuisance=self._updated_estimates[self.key_1].min_nuisance,  # type: ignore
            nuisance_weights=self._updated_estimates[
                self.key_1
            ].nuisance_weight,  # type: ignore
            g_star_obs=self._updated_estimates[self.key_1].g_star_obs,
            plot_size=plot_size,
            color_1=color_1,
            color_0=color_0,
        ):
            if save_dir_path is not None:
                plt.savefig(f'{save_dir_path}/nuisance_weights_t{time}.png', bbox_inches="tight")
            else:
                plt.show()
            plt.close()

    def plot_norm_pn_eic(
        self,
        save_dir_path: Optional[str] = None,
        plot_size: Tuple[float, float] = (6.4, 4.8),
    ):
        _, ax = plt.subplots(figsize=plot_size)
        ax.plot(self.norm_pn_eics, marker="o")
        ax.set_title("Norm of the Pointwise Nuisance Function", fontsize=16)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("||PnEIC||")
        if save_dir_path is not None:
            plt.savefig(save_dir_path, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def plot_evalue_contours(
        self,
        save_dir_path: Optional[str] = None,
        time: Optional[float] = None,
        event: Optional[int] = None,
        type: str = "ratio",
        num_points_per_contour: int = 200,
        color_point_estimate: str = "blue",
        color_ci: str = "red",
        color_benchmarking: str = "green",
        plot_size: Tuple[float, float] = (6.4, 4.8),
    ):
        if not self._fitted:
            raise RuntimeError(
                "Model has to be fitted before calling plot_evalue_contours()."
            )
        if save_dir_path is not None and not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
        if time is not None:
            assert (
                time in self.target_times
            ), f"Time has to be one of the target times {self.target_times}."
            target_times = [time]
        else:
            target_times = self.target_times
        if event is not None:
            assert (
                event in self.target_events
            ), f"Event has to be one of the target events {self.target_events}."
            target_events = [event]
        else:
            target_events = self.target_events
        for _, _, time, event in self.evalues_benchmark.plot(
            target_times=target_times,
            target_events=target_events,
            ate_type=type,
            num_points_per_contour=num_points_per_contour,
            color_point_estimate=color_point_estimate,
            color_ci=color_ci,
            color_benchmarking=color_benchmarking,
            plot_size=plot_size,
        ):
            if save_dir_path is not None:
                plt.savefig(f"{save_dir_path}/evalue_contours_{event}_t{time}.png", bbox_inches="tight")
            else:
                plt.show()
            plt.close()
