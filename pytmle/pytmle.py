import os
import warnings
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple

from .estimates import InitialEstimates
from .get_initial_estimates import fit_propensity_super_learner, fit_state_learner
from .tmle_update import tmle_update
from .predict_ate import (
    get_counterfactual_risks,
    ate_ratio,
    ate_diff,
)
from .evalues_benchmark import EvaluesBenchmark
from .plotting import plot_risks, plot_ate, plot_nuisance_weights
from .bootstrap import bootstrap_tmle_loop


class PyTMLE:

    def __init__(
        self,
        data: pd.DataFrame,
        col_event_times: str = "event_time",
        col_event_indicator: str = "event_indicator",
        col_group: str = "group",
        target_times: Optional[List[float]] = None,
        g_comp: bool = True,
        evalues_benchmark: bool = False,
        key_1: int = 1,
        key_0: int = 0,
        initial_estimates: Optional[Dict[int, InitialEstimates]] = None,
        verbose: int = 2,
    ):
        """
        Initialize the PyTMLE class.

        Parameters
        ----------
        data : pd.DataFrame
            The input data containing event times, event indicators, treatment group information, and predictors.
        col_event_times : str, optional
            The column name in the data that contains event times. Default is "event_time".
        col_event_indicator : str, optional
            The column name in the data that contains event indicators. Needs to contain consecutive integers starting at 0 (where 0 is for censored). Default is "event_indicator".
        col_group : str, optional
            The column name in the data that contains treatment group information. Needs to be binary. Default is "group".
        target_times : Optional[List[float]], optional
            Specific times at which to estimate the target parameter. If None, estimates for the last observed event time are used. Default is None.
        g_comp : bool, optional
            Whether to store g-computation for initial estimates. Default is True.
        evalues_benchmark : bool, optional
            Whether to compute E-values for measured confounders. Default is False.
        key_1 : int, optional
            The key representing the treatment group. Default is 1.
        key_0 : int, optional
            The key representing the control group. Default is 0.
        initial_estimates : Optional[Dict[int, InitialEstimates]], optional
            Dict with pre-computed initial estimates for the two potential outcomes, which can be passed right to the second TMLE stage. Default is None.
        verbose : int, optional
            Verbosity level. 0: Absolutely so logging at all, 1: only warnings, 2: major execution steps, 3: execution steps, 4: everything for debugging. Default is 2.
        """
        self._check_inputs(
            data,
            col_event_times,
            col_event_indicator,
            col_group,
            target_times,
            key_1,
            key_0,
            initial_estimates,
        )
        self._initial_estimates = initial_estimates
        self._updated_estimates = None
        self._X = data.drop(
            columns=[col_event_times, col_event_indicator, col_group]
        ).to_numpy(dtype=float)
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
        self.target_events = np.unique(
            data.loc[data[col_event_indicator] != 0, col_event_indicator]
        )
        self.g_comp = g_comp
        self.key_1 = key_1
        self.key_0 = key_0
        self.verbose = verbose
        self._bootstrap_results = None
        self._fitted = False
        self.has_converged = False
        self.step_num = 0
        self.norm_pn_eics = []
        self.models = {}
        self.state_learner_cv_fit = None
        if evalues_benchmark:
            if initial_estimates is not None and self.verbose >= 1:
                warnings.warn(
                    "E-values benchmark for measured covariates may be incorrect "
                    "if pre-computed initial estimates are provided because "
                    "the measured covariates need to be dropped during model fitting.",
                    RuntimeWarning,
                )
            self.evalues_benchmark = EvaluesBenchmark(self, verbose=self.verbose)
        else:
            self.evalues_benchmark = EvaluesBenchmark(verbose=self.verbose)

    def _check_inputs(
        self,
        data: pd.DataFrame,
        col_event_times: str,
        col_event_indicator: str,
        col_group: str,
        target_times: Optional[List[float]],
        key_1: int,
        key_0: int,
        initial_estimates: Optional[Dict[int, InitialEstimates]],
    ):
        if col_event_times not in data.columns:
            raise ValueError(f"Column {col_event_times} not found in the given data.")
        if col_event_indicator not in data.columns:
            raise ValueError(
                f"Column {col_event_indicator} not found in the given data."
            )
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
            if not np.array_equal(
                np.unique(data[col_event_times]),
                np.unique(initial_estimates[key_1].times),
            ) or not np.array_equal(
                np.unique(data[col_event_times]),
                np.unique(initial_estimates[key_0].times),
            ):
                raise ValueError(
                    "All values in data[col_event_times] must be present in initial_estimates[key_1].times and initial_estimates[key_0] and vice versa."
                )
        unique_events = np.unique(data[col_event_indicator])
        if unique_events.dtype != int or not (
            unique_events[-1] == len(unique_events) - 1
        ):
            raise ValueError(
                f"Event indicators have to be consecutive integers starting from 0. Got {unique_events}."
            )
        if target_times is not None and not max(target_times) <= max(
            data[col_event_times]
        ):
            raise ValueError(
                "All target times have to be smaller or equal to the maximum event time in the data."
            )
        if target_times is not None and min(target_times) < 0:
            raise ValueError("All target times have to be positive.")

    def _get_initial_estimates(
        self,
        cv_folds: int,
        models,
        labtrans,
        propensity_score_models,
        additional_inputs: Optional[Tuple],
        n_epochs: int,
        batch_size: int,
        save_models: bool,
    ):
        if self._initial_estimates is None:
            self._initial_estimates = {
                self.key_1: InitialEstimates(
                    g_star_obs=self._group, times=np.unique(self._event_times)
                ),
                self.key_0: InitialEstimates(
                    g_star_obs=1 - self._group, times=np.unique(self._event_times)
                ),
            }

        if (
            self._initial_estimates[self.key_1].propensity_scores is None
            or self._initial_estimates[self.key_0].propensity_scores is None
        ):
            if self.verbose >= 2:
                print("Estimating propensity scores...")
            propensity_scores_1, propensity_scores_0, model_dict = (
                fit_propensity_super_learner(
                    X=self._X,
                    y=self._group,
                    cv_folds=cv_folds,
                    return_model=save_models,
                    base_learners=propensity_score_models,
                    verbose=self.verbose >= 4,
                )
            )
            self.models.update(model_dict)
            self._initial_estimates[self.key_1].propensity_scores = propensity_scores_1
            self._initial_estimates[self.key_0].propensity_scores = propensity_scores_0
        else:
            if self.verbose >= 2:
                print("Using given propensity score estimates")

        hazards_missing = (
            self._initial_estimates[self.key_1].hazards is None
            or self._initial_estimates[self.key_0].hazards is None
            or self._initial_estimates[self.key_1].event_free_survival_function is None
            or self._initial_estimates[self.key_0].event_free_survival_function is None
        )
        if hazards_missing:
            if self.verbose >= 2:
                print("Estimating hazards and event-free survival...")
            factual_event_free_survival = None
            fit_risks_model = True
        else:
            if self.verbose >= 2:
                print("Using given hazard and event-free survival estimates")
            factual_event_free_survival = np.where(
                np.expand_dims(self._group, 1) == 1,
                self._initial_estimates[self.key_1].event_free_survival_function,  # type: ignore
                self._initial_estimates[self.key_0].event_free_survival_function,  # type: ignore
            )
            fit_risks_model = False

        cens_missing = (
            self._initial_estimates[self.key_1].censoring_survival_function is None
            or self._initial_estimates[self.key_0].censoring_survival_function is None
        )
        if cens_missing:
            if self.verbose >= 2:
                print("Estimating censoring survival...")
            factual_censoring_survival = None
            fit_censoring_model = True
        else:
            if self.verbose >= 2:
                print("Using given censoring survival estimates")
            factual_censoring_survival = np.where(
                np.expand_dims(self._group, 1) == 1,
                self._initial_estimates[self.key_1].censoring_survival_function,  # type: ignore
                self._initial_estimates[self.key_0].censoring_survival_function,  # type: ignore
            )
            fit_censoring_model = False

        if fit_risks_model or fit_censoring_model:
            (
                hazards_1,
                hazards_0,
                surv_1,
                surv_0,
                cens_surv_1,
                cens_surv_0,
                model_dict,
                labtrans,
                self.state_learner_cv_fit,
            ) = fit_state_learner(
                X=self._X,
                trt=self._group,
                event_times=self._event_times,
                event_indicator=self._event_indicator,
                additional_inputs=additional_inputs,
                cv_folds=cv_folds,
                return_model=save_models,
                models=models,
                labtrans=labtrans,
                max_time=max(self.target_times),
                n_epochs=n_epochs,
                batch_size=batch_size,
                precomputed_event_free_survival=factual_event_free_survival,
                precomputed_censoring_survival=factual_censoring_survival,
                verbose=self.verbose,
            )
            self.models.update(model_dict)
            # update times if they were tranformed in the risk model
            if labtrans is not None:
                self._initial_estimates[self.key_1].times = labtrans.cuts
                self._initial_estimates[self.key_0].times = labtrans.cuts
            if hazards_1 is not None and hazards_0 is not None:
                self._initial_estimates[self.key_1].hazards = hazards_1
                self._initial_estimates[self.key_0].hazards = hazards_0
            if surv_1 is not None and surv_0 is not None:
                self._initial_estimates[self.key_1].event_free_survival_function = (
                    surv_1
                )
                self._initial_estimates[self.key_0].event_free_survival_function = (
                    surv_0
                )
            if cens_surv_1 is not None and cens_surv_0 is not None:
                self._initial_estimates[self.key_1].censoring_survival_function = (
                    cens_surv_1
                )
                self._initial_estimates[self.key_0].censoring_survival_function = (
                    cens_surv_0
                )

        # there may be changes if times were transformed
        if labtrans is not None:
            binned_times, _ = labtrans.transform(
                self._event_times, self._event_indicator
            )
            self._event_times = labtrans.cuts[binned_times]

    def _update_estimates(
        self,
        max_updates: int,
        min_nuisance: Optional[float],
        one_step_eps: float,
        bootstrap: bool = False,
        n_bootstrap: int = 100,
        n_jobs: int = 4,
        stratified_bootstrap: bool = False,
    ):
        assert (
            self._initial_estimates is not None
        ), "Initial estimates have to be available before calling _update_estimates()."
        for k in self._initial_estimates:
            assert (
                self._initial_estimates[k] is not None
            ), "Initial estimates have to be available before calling _update_estimates()."
        if self.verbose >= 2:
            print("Starting TMLE update loop...")
        if bootstrap:
            self._bootstrap_results = bootstrap_tmle_loop(
                self._initial_estimates,
                event_times=self._event_times,
                event_indicator=self._event_indicator,
                target_times=self.target_times,
                target_events=self.target_events,
                n_bootstrap=n_bootstrap,
                n_jobs=n_jobs,
                stratify_by_event=stratified_bootstrap,
                max_updates=max_updates,
                min_nuisance=min_nuisance,
                one_step_eps=one_step_eps,
                key_1=self.key_1,
                key_0=self.key_0,
                verbose=self.verbose,
            )
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
        bootstrap: bool = False,
        n_bootstrap: int = 100,
        n_jobs: int = 4,
        stratified_bootstrap: bool = False,
        models=None,
        labtrans=None,
        propensity_score_models=None,
        additional_inputs: Optional[Tuple] = None,
        n_epochs: int = 100,
        batch_size: int = 128,
    ):
        """
        Fit the TMLE model.

        Parameters
        ----------
        cv_folds : int, optional
            Number of cross-validation folds for the initial estimate models. Default is 10.
        max_updates : int
            Maximum number of updates to the estimates in the TMLE loop. Default is 500.
        min_nuisance : Optional[float], optional
            Value between 0 and 1 for truncating the g-related denominator of the clever covariate. Default is None, which means no truncation at all.
        one_step_eps : float
            Initial epsilon for the one-step update. Default is 0.1.
        save_models : bool, optional
            Whether to save the models used for the initial estimates. Default is False.
        alpha : float, optional
            The alpha level for confidence intervals (relevant only for E-value benchmark). Default is 0.05.
        bootstrap : bool, optional
            Whether to perform bootstrapping of the second TMLE stage for confidence intervals. Default is False.
        n_bootstrap : int, optional
            Number of bootstrap samples. Has no effect if bootstrap is False. Default is 100.
        n_jobs : int, optional
            Number of parallel jobs for bootstrapping. Has no effect if bootstrap is False. Default is 4.
        stratified_bootstrap : bool, optional
            Whether to perform bootstrapping stratified by event indicator. Has no effect if bootstrap is False. Default is False.
        models : Optional, optional
            A list of models to use for the state learner. If None, use the default library. Default is None.
        labtrans : Optional, optional
            A list of labtrans objects to use for the risk model (if required; e.g., discretizer for DeepHit). If not None, needs to be one object for all models, or one object per model. Default is None.
        propensity_score_models : Optional, optional
            A list of models to use for the propensity score stacking classifier. If None, use the default library. Default is None.
        additional_inputs : Optional[Tuple], optional
            Additional inputs for the risk and censoring models. Can be tuple of torch.Tensors or np.ndarray, but has to be compatible with torchtuples. Default is None.
        n_epochs : int, optional
            Number of epochs for training models in each cross fitting fold (if applicable). Default is 100.
        batch_size : int, optional
            Batch size for training models in each cross fitting fold (if applicable). Default is 128.
        """
        if self._fitted:
            raise RuntimeError(
                "Model has already been fitted. fit() can only be called once."
            )
        self._get_initial_estimates(
            cv_folds,
            save_models=save_models,
            models=models,
            labtrans=labtrans,
            propensity_score_models=propensity_score_models,
            additional_inputs=additional_inputs,
            n_epochs=n_epochs,
            batch_size=batch_size,
        )
        self._update_estimates(
            max_updates,
            min_nuisance,
            one_step_eps,
            bootstrap,
            n_bootstrap,
            n_jobs,
            stratified_bootstrap,
        )
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
                bootstrap=bootstrap,
                n_bootstrap=n_bootstrap,
                n_jobs=n_jobs,
                stratified_bootstrap=stratified_bootstrap,
                models=models,
                labtrans=labtrans,
                n_epochs=n_epochs,
                batch_size=batch_size,
            )

    def predict(
        self, type: str = "risks", alpha: float = 0.05, g_comp: bool = False
    ) -> pd.DataFrame:
        """
        Predict the counterfactual risks or average treatment effect.

        Parameters
        ----------
        type : str, optional
            The type of prediction. "risks", "rr" and "rd" are supported. Default is "risks".
        alpha : float, optional
            The alpha level for confidence intervals. Default is 0.05.
        g_comp : bool, optional
            Whether to return the g-computation estimates instead of the updated estimates. Default is False.

        Returns
        -------
        pd.DataFrame
            The predicted counterfactual risks or average treatment effect.
        """
        if not self._fitted or self._updated_estimates is None:
            raise RuntimeError("Model has to be fitted before calling predict().")
        if type == "risks":
            return get_counterfactual_risks(
                self._updated_estimates,
                g_comp=g_comp,
                alpha=alpha,
                key_1=self.key_1,
                key_0=self.key_0,
                bootstrap_results=self._bootstrap_results,
            )
        elif type == "rr":
            return ate_ratio(
                self._updated_estimates,
                g_comp=g_comp,
                alpha=alpha,
                key_1=self.key_1,
                key_0=self.key_0,
                bootstrap_results=self._bootstrap_results,
            )
        elif type == "rd":
            return ate_diff(
                self._updated_estimates,
                g_comp=g_comp,
                alpha=alpha,
                key_1=self.key_1,
                key_0=self.key_0,
                bootstrap_results=self._bootstrap_results,
            )
        else:
            raise ValueError(
                f"Only 'risks', 'rr' and 'rd' are supported as type, got {type}."
            )

    def plot(
        self,
        save_path: Optional[str] = None,
        type: str = "risks",
        alpha: float = 0.05,
        g_comp: bool = False,
        color_1: Optional[str] = None,
        color_0: Optional[str] = None,
        use_bootstrap: bool = False,
    ) -> Optional[Tuple[Figure, np.ndarray]]:
        """
        Plot the counterfactual risks or average treatment effect (in terms of RR or RD).

        Parameters
        ----------
        save_path : Optional[str], optional
            Path to save the plot. If None, will return figure and axes. Default is None.
        type : str, optional
            The type of prediction. "risks", "rr" and "rd" are supported. Default is "risks".
        alpha : float, optional
            The alpha level for confidence intervals. Default is 0.05.
        g_comp : bool, optional
            Whether to include the g-computation estimates in the plot. Default is False.
        color_1 : Optional[str], optional
            Color for the potential outcome for "treated". Pick None for standard matplotlib colors. Default is None.
        color_0 : Optional[str], optional
            Color for the potential outcome for "untreated". Pick None for standard matplotlib colors. Default is None.
        use_bootstrap : bool, optional
            Whether to use the bootstrapped bounds instead of the theoretical bounds. Default is False.

        Returns
        -------
        Optional[Tuple[Figure, np.ndarray]]
            The figure and axes of the plot. Only returned if save_path is None.
        """
        if use_bootstrap and self._bootstrap_results is None:
            raise RuntimeError(
                "Bootstrapping has to be performed before plotting with bootstrap estimates."
            )
        if type == "risks":
            pred = self.predict(type=type, alpha=alpha)
            if g_comp:
                pred_g_comp = self.predict(type=type, alpha=alpha, g_comp=True)
            fig, axes = plot_risks(
                pred,
                pred_g_comp if g_comp else None,
                color_1=color_1,
                color_0=color_0,
                use_bootstrap=use_bootstrap,
            )
        elif type in ("rr", "rd"):
            pred = self.predict(type=type, alpha=alpha)
            if g_comp:
                pred_g_comp = self.predict(type=type, alpha=alpha, g_comp=True)
            fig, axes = plot_ate(
                pred,
                pred_g_comp if g_comp else None,
                type=type,
                use_bootstrap=use_bootstrap,
            )
        else:
            raise ValueError(
                f"Only 'risks', 'rr' and 'rd' are supported as type, got {type}."
            )

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
    ) -> None:
        """
        Plot the nuisance weights.

        Parameters
        ----------
        time : Optional[float], optional
            Time at which to plot the nuisance weights. If None, all target times are plotted. Default is None.
        save_dir_path : Optional[str], optional
            Path to directory to save the plots. If None, will simply display the plots. Default is None.
        color_1 : Optional[str], optional
            Color for the treatment group. Pick None for standard matplotlib colors. Default is None.
        color_0 : Optional[str], optional
            Color for the control group. Pick None for standard matplotlib colors. Default is None.
        """
        if self._updated_estimates is None:
            raise RuntimeError(
                "Updated estimates must have been initialized before calling plot_nuisance_weights()."
            )
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
            nuisance_weights=self._updated_estimates[self.key_1].nuisance_weight,  # type: ignore
            g_star_obs=self._updated_estimates[self.key_1].g_star_obs,
            plot_size=plot_size,
            color_1=color_1,
            color_0=color_0,
        ):
            if save_dir_path is not None:
                plt.savefig(
                    f"{save_dir_path}/nuisance_weights_t{time}.svg", bbox_inches="tight"
                )
            else:
                plt.show()
            plt.close()

    def plot_norm_pn_eic(
        self,
        save_dir_path: Optional[str] = None,
        plot_size: Tuple[float, float] = (6.4, 4.8),
    ):
        """
        Plot the norm of the empirical measure of the EIC.

        Parameters
        ----------
        save_dir_path : Optional[str], optional
            Path to directory to save the plot. If None, will simply display the plot. Default is None.
        plot_size : Tuple[float, float], optional
            Size of the plot. Default is (6.4, 4.8).
        """

        _, ax = plt.subplots(figsize=plot_size)
        ax.plot(self.norm_pn_eics, marker="o")
        ax.set_title("Norm of the empirical measure of the EIC", fontsize=16)
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
        type: str = "rr",
        use_bootstrap: bool = False,
        num_points_per_contour: int = 200,
        color_point_estimate: str = "blue",
        color_ci: str = "red",
        color_benchmarking: str = "green",
        plot_size: Tuple[float, float] = (6.4, 4.8),
    ):
        """
        Plot the E-value contours for the estimated average treatment effect.

        Parameters
        ----------
        save_dir_path : Optional[str], optional
            Path to directory to save the plot. If None, will simply display the plot. Default is None.
        time : Optional[float], optional
            Time at which to plot the E-value contours. If None, will plot for all target times. Default is None.
        event : Optional[int], optional
            Event at which to plot the E-value contours. If None, will plot for all target events. Default is None.
        type : str, optional
            The type of prediction. "rr" and "rd" are supported. Default is "rr".
        use_bootstrap : bool, optional
            Whether to use the bootstrapped bounds instead of the theoretical bounds. Default is False.
        num_points_per_contour : int, optional
            Number of points per contour. Default is 200.
        color_point_estimate : str, optional
            Color for the point estimate. Default is "blue".
        color_ci : str, optional
            Color for the confidence interval. Default is "red".
        color_benchmarking : str, optional
            Color for the benchmarking. Default is "green".
        plot_size : Tuple[float, float], optional
            Size of the plot. Default is (6.4, 4.8).
        """

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
            use_bootstrap=use_bootstrap,
        ):
            if save_dir_path is not None:
                plt.savefig(
                    f"{save_dir_path}/evalue_contours_{event}_t{time}.svg",
                    bbox_inches="tight",
                )
            else:
                plt.show()
            plt.close()
