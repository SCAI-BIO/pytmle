import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from typing import Tuple, Optional, Any, List, Literal, Union
from copy import deepcopy
import time
import warnings

from .pycox_wrapper import PycoxWrapper, PycoxWrapperCauseSpecific, wrap_model
from .initial_estimates_default_models import get_default_models


def fit_propensity_super_learner(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    base_learners: Optional[List],
    return_model: bool,
    verbose: int,
    calibration_method: Optional[Literal["isotonic", "sigmoid"]] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Fit a stacking classifier to estimate the propensity scores.

    Parameters
    ----------
    X : np.ndarray
        The feature matrix.
    y : np.ndarray
        The treatment vector.
    cv_folds : int
        The number of cross-validation folds.
    return_model : bool
        Whether to return the fitted model.
    base_learners : List
        The base learners to use in the stacking classifier. If None, will use the default base learners.
    verbose: bool
        If true, will set verbosity to maximum level.
    calibration_method : Optional[Literal["isotonic", "sigmoid"]]
        The calibration method to use. If None, no calibration is performed. Default is None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, dict]
        The estimated propensity scores, the estimated inverse propensity scores, the fitted model.
    """
    # all-nan columns are removed (relevant for E-value benchmark)
    X = X[:, ~np.isnan(X).all(axis=0)]
    if base_learners is None:
        # defaults
        base_learners = [
            ("rf", RandomForestClassifier()),
            ("gb", GradientBoostingClassifier()),
        ]
    else:
        base_learners = [
            (str(i), deepcopy(model)) for i, model in enumerate(base_learners)
        ]
    # ('lr', LogisticRegression(max_iter=200))] # don't use for now because of convergence issues
    super_learner = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=cv_folds,
        verbose=10 if verbose else 0,
    )

    if calibration_method is not None:
        # Calibrate the super learner if requested (using "inner" cross-validation with same number of folds)
        super_learner = CalibratedClassifierCV(
            super_learner, cv=cv_folds, method=calibration_method
        )

    # Use cross_val_predict to generate out-of-fold predictions
    pred = cross_val_predict(
        super_learner,
        X,
        y,
        cv=cv_folds,
        method="predict_proba",
        verbose=10 if verbose else 0,
        n_jobs=-1,
    )
    if return_model:
        return pred[:, 1], pred[:, 0], {"propensity_model": super_learner}
    return pred[:, 1], pred[:, 0], {}


def abs_risk_integrated_brier_score(
    chf: np.ndarray, counting_processes: np.ndarray, time_grid: np.ndarray
) -> float:
    """
    Compute the integrated Brier score for the absolute risks
    and cause-specific counting processes used as loss in the state learner.

    Parameters
    ----------
    chf : np.ndarray
        The cumulative hazard functions.
    counting_processes : np.ndarray
        The counting processes.
    """
    s = np.exp(-np.sum(chf, axis=-1))
    s_ = np.column_stack([np.ones((s.shape[0], 1)), s[:, :-1]])
    hf = np.diff(chf, axis=1, prepend=0)
    abs_risk = np.cumsum(hf * s_[..., np.newaxis], axis=1)

    # add event-free state (0)
    abs_risk = np.concatenate([s[..., np.newaxis], abs_risk], axis=-1)
    counting_processes = np.concatenate([(~np.sum(counting_processes, axis=-1, dtype=bool)[..., np.newaxis]).astype(int), 
                                         counting_processes], axis=-1)

    brier = (abs_risk - counting_processes) ** 2
    # TODO: Add weights?
    ibs = np.mean(
        np.sum(brier * np.diff(time_grid, prepend=0)[np.newaxis, :, np.newaxis], axis=1)
    )
    return ibs


def fit_state_learner(
    X: np.ndarray,
    trt: np.ndarray,
    event_times: np.ndarray,
    event_indicator: np.ndarray,
    cv_folds: int,
    return_model: bool,
    models,
    labtrans,
    additional_inputs: Optional[Tuple],
    max_time: float,
    n_epochs: int = 100,
    batch_size: int = 128,
    precomputed_event_free_survival: Optional[np.ndarray] = None,
    precomputed_censoring_survival: Optional[np.ndarray] = None,
    verbose: int = 2,
    mlflow_logging: bool = False,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    dict,
    Optional[Any],
    pd.DataFrame,
]:
    """
    Fit a version of the state learner (Munch & Gerds, 2024) to estimate the hazard functions for each event type.
    https://arxiv.org/abs/2405.17259

    Parameters
    ----------
    X : np.ndarray
        The feature matrix.
    trt : np.ndarray
        The treatment vector.
    event_times : np.ndarray
        The event times.
    event_indicator : np.ndarray
        The event indicator.
    target_events : List[int]
        The list of event types.
    cv_folds : int
        The number of cross-validation folds..
    return_model : bool
        Whether to return the fitted model.
    models :
        The risk / censoring models.
    labtrans :
        The label transformer.
    additional_inputs :
        Additional inputs to the models (for pycox models; need to be compatible with torchtuples). Default is None.
    max_time :
        The time up to which the hazard functions are evaluated.
    risk_models :
        The risk models
    n_epochs : int
        The number of epochs.
    batch_size : int
        The batch size.
    precomputed_event_free_survival : np.ndarray
        Precomputed event-free survival functions. Default is None.
    precomputed_censoring_survival : np.ndarray
        Precomputed censoring survival functions. Default is None.
    verbose : int
        Verbosity level. 0: Absolutely so logging at all, 1: only warnings, 2: major execution steps, 3: execution steps, 4: everything for debugging. Default is 2.
    mlflow_logging : bool
        Whether to log the results to mlflow. Default is False.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]
        The estimated counterfactual hazard functions for each event type,
        the estimated counterfactual survival functions for each event type,
        the estimated counterfactual censoring survival functions for each event type,
        the fitted models, the label transformer, a dataframe with the losses per model tuple.
    """
    fit_risks_model = precomputed_event_free_survival is None
    fit_censoring_model = precomputed_censoring_survival is None
    # A time grid for the integrated Brier scores
    time_grid = np.linspace(0, max_time, 100)
    # If no models are provided, use the default models
    if models is None:
        risks_models, censoring_models, labtrans = get_default_models(
            event_times=event_times,
            event_indicator=event_indicator,
            input_size=X.shape[1] + 1,  # number of covariateses + treatment
            verbose=verbose,
        )
        if not fit_risks_model:
            risks_models = [None]
        if not fit_censoring_model:
            censoring_models = [None]
    else:
        if not isinstance(models, list):
            risks_models = [models] if fit_risks_model else [None]
            censoring_models = [models] if fit_censoring_model else [None]
        else:
            risks_models = models if fit_risks_model else [None]
            censoring_models = models if fit_censoring_model else [None]
    if not isinstance(labtrans, list):
        if labtrans is None:
            labtrans = [None] * len(models)
        else:
            labtrans = [labtrans]
    risks_label_transformers = labtrans if fit_risks_model else [None]
    censoring_label_transformers = labtrans if fit_censoring_model else [None]
    if (fit_risks_model and len(risks_models) != len(labtrans)) or (
        fit_censoring_model and len(censoring_models) != len(labtrans)
    ):
        raise ValueError(
            "The number of models and label transformers must be the same."
        )

    loss_list = []
    risk_models_cache_dict = {}
    censoring_models_cache_dict = {}
    best_pairs_dict = {}
    final_labtrans = None
    min_loss = np.inf

    # one object for consistent stratified k-fold cross fitting
    skf = StratifiedKFold(n_splits=cv_folds)

    for risks_model, risks_labtrans in zip(risks_models, risks_label_transformers):
        for censoring_model, censoring_labtrans in zip(
            censoring_models, censoring_label_transformers
        ):  
            # keep track of the time taken for each model combination
            start_time = time.time()
            # Combine the label transformers
            if risks_labtrans == censoring_labtrans:
                combined_labtrans = risks_labtrans
            elif risks_labtrans is None and censoring_labtrans is not None:
                combined_labtrans = censoring_labtrans
            elif risks_labtrans is not None and censoring_labtrans is None:
                combined_labtrans = risks_labtrans
            else:
                # chain the transformations if they are different
                class CombinedLabtrans:
                    def __init__(self, risks_lt, censoring_lt):
                        self.risks_lt = risks_lt
                        self.censoring_lt = censoring_lt

                    def transform(self, durations, events):
                        return self.risks_lt.transform(
                            self.censoring_lt.transform(durations, events)
                        )

                combined_labtrans = CombinedLabtrans(risks_labtrans, censoring_labtrans)

            # Get the counting processes per event type and stack all cumulative hazards
            grid_mat = np.tile(time_grid, (X.shape[0], 1))
            event_mat = event_times[:, np.newaxis] <= grid_mat
            chfs_list = []
            events_by_cause_list = []

            # cross-fit the risk and censoring models
            try:
                if fit_risks_model:
                    if (risks_model, combined_labtrans) in risk_models_cache_dict:
                        # if risk model has already been fitted with the given label transformer, just load it from the dictionary
                        risks_models_fitted = risk_models_cache_dict[(risks_model, combined_labtrans)]
                    else:
                        risks_models_fitted = cross_fit_model(
                            skf=skf,
                            X=X,
                            trt=trt,
                            event_times=event_times,
                            event_indicator=event_indicator,
                            labtrans=combined_labtrans,
                            model=risks_model,
                            additional_inputs=additional_inputs,
                            n_epochs=n_epochs,
                            batch_size=batch_size,
                            verbose=verbose >= 4,
                        )
                        risk_models_cache_dict[(risks_model, combined_labtrans)] = risks_models_fitted
                    # get out-of-fold predictions
                    (
                        surv_1_i,
                        surv_0_i,
                        cumhaz_1_i,
                        cumhaz_0_i,
                        cumhaz_f_i,
                        jumps,
                    ) = predict_oof(
                        models_per_fold=risks_models_fitted,
                        skf=skf,
                        X=X,
                        trt=trt,
                        event_times=event_times,
                        event_indicator=event_indicator,
                        labtrans=combined_labtrans,
                        additional_inputs=additional_inputs,
                    )
                    causes = np.unique(event_indicator[event_indicator != 0])
                    for c in causes:
                        events_by_cause_list.append(
                            event_mat * (event_indicator == c)[:, np.newaxis]
                        )
                    for i in range(len(events_by_cause_list)):
                        chfs_list.append(cumhaz_f_i[..., i])  # type: ignore
                else:
                    # precomputed event-free survival enters the loss if given
                    events_by_cause_list.append(
                        event_mat * (event_indicator > 0)[:, np.newaxis]
                    )
                    chfs_list.append(-np.log(precomputed_event_free_survival))

                if fit_censoring_model:
                    if (censoring_model, combined_labtrans) in censoring_models_cache_dict:
                        # if censoring model has already been fitted with the given label transformer, just load it from the dictionary
                        censoring_models_fitted = censoring_models_cache_dict[(censoring_model, combined_labtrans)]
                    else:
                        censoring_models_fitted = cross_fit_model(
                            skf=skf,
                            X=X,
                            trt=trt,
                            event_times=event_times,
                            event_indicator=event_indicator==0,
                            labtrans=combined_labtrans,
                            model=censoring_model,
                            additional_inputs=additional_inputs,
                            n_epochs=n_epochs,
                            batch_size=batch_size,
                            verbose=verbose >= 4,
                        )
                        censoring_models_cache_dict[(censoring_model, combined_labtrans)] = censoring_models_fitted
                    # get out-of-fold predictions
                    (
                        cens_surv_1_i,
                        cens_surv_0_i,
                        _,
                        _,
                        cens_cumhaz_f_i,
                        jumps,
                    ) = predict_oof(
                        models_per_fold=censoring_models_fitted,
                        skf=skf,
                        X=X,
                        trt=trt,
                        event_times=event_times,
                        event_indicator=event_indicator==0,
                        labtrans=combined_labtrans,
                        additional_inputs=additional_inputs,
                    )
                    chfs_list.append(cens_cumhaz_f_i[..., 0])
                else:
                    # precomputed censoring survival enters the loss if given
                    chfs_list.append(-np.log(precomputed_censoring_survival))

                events_by_cause_list.append(
                    event_mat * (event_indicator == 0)[:, np.newaxis]
                )
                try:
                    chfs = np.stack(chfs_list, axis=-1)
                except ValueError:
                    raise ValueError(
                        "The cumulative hazards have different shapes. This could be due to label transformation in one of the models."
                    )
                events_by_cause = np.stack(events_by_cause_list, axis=-1)
                # Get cumulative hazards for the time_grid
                time_grid_indices = np.searchsorted(jumps, time_grid, side="right") - 1
                time_grid_indices = np.clip(time_grid_indices, 0, len(jumps) - 1)
                chfs = chfs[:, time_grid_indices, :]

                # compute the integrated Brier score including absolute risks
                loss = abs_risk_integrated_brier_score(chfs, events_by_cause, time_grid)

                end_time = time.time()

                # Add loss and model info to the list
                loss_list.append(
                    {
                        "risks_model": risks_model.__class__.__name__,
                        "censoring_model": censoring_model.__class__.__name__,
                        "loss": loss,
                        "time": end_time - start_time,
                    }
                )
                # Log to mlflow using a child run for each configuration (like in hyperparameter optimization)
                if mlflow_logging:
                    with mlflow.start_run(
                        nested=True,
                        run_name=f"{risks_model.__class__.__name__},{censoring_model.__class__.__name__}",
                    ):
                        mlflow.log_params(
                            {
                                "risks_model": risks_model.__class__.__name__,
                                "censoring_model": censoring_model.__class__.__name__,
                            }
                        )
                        mlflow.log_metric("loss", loss)

                if verbose >= 3:
                    print(
                        f"({risks_model.__class__.__name__} | {censoring_model.__class__.__name__}): {loss}"
                    )

                if loss < min_loss:
                    min_loss = loss
                    if fit_risks_model:
                        haz_1 = np.diff(cumhaz_1_i, prepend=0, axis=1)
                        haz_0 = np.diff(cumhaz_0_i, prepend=0, axis=1)
                        surv_1 = surv_1_i
                        surv_0 = surv_0_i
                    else:
                        haz_1 = None
                        haz_0 = None
                        surv_1 = None
                        surv_0 = None
                    if fit_censoring_model:
                        cens_surv_1 = cens_surv_1_i
                        cens_surv_0 = cens_surv_0_i
                    else:
                        cens_surv_1 = None
                        cens_surv_0 = None
                    final_labtrans = combined_labtrans
                    if return_model:
                        if fit_risks_model:
                            best_pairs_dict.update({f"risks_model_{k}": v for k, v in risks_models_fitted.items()})
                        if fit_censoring_model:
                            best_pairs_dict.update({f"censoring_model_{k}": v for k, v in censoring_models_fitted.items()})

            except Exception as e:
                # If one of the model fails, skip it
                if verbose >= 1:
                    warnings.warn(
                        f"({risks_model.__class__.__name__} | {censoring_model.__class__.__name__}) failed: {e}",
                        RuntimeWarning,
                    )
                continue
    loss_df = pd.DataFrame(loss_list).sort_values(by="loss")
    if return_model:
        return (
            haz_1,
            haz_0,
            surv_1,
            surv_0,
            cens_surv_1,
            cens_surv_0,
            best_pairs_dict,
            final_labtrans,
            loss_df,
        )
    return (
        haz_1,
        haz_0,
        surv_1,
        surv_0,
        cens_surv_1,
        cens_surv_0,
        {},
        final_labtrans,
        loss_df,
    )


def cross_fit_model(
    skf: StratifiedKFold,
    X: np.ndarray,
    trt: np.ndarray,
    event_times: np.ndarray,
    event_indicator: np.ndarray,
    labtrans,
    model,
    additional_inputs: Optional[Tuple],
    n_epochs: int,
    batch_size: int,
    verbose: bool,
):

    if additional_inputs is not None:
        if not isinstance(additional_inputs, tuple):
            additional_inputs = (additional_inputs,)
        for a in additional_inputs:
            if X.shape[0] != a.shape[0]:
                raise ValueError(
                    "The first dimension of all additional inputs must match the number of samples in X."
                )

    models = {}
    
    for i, (train_indices, _) in enumerate(skf.split(X, trt)):
        X_train = X[train_indices]
        trt_train = trt[train_indices]
        event_times_train = event_times[train_indices]
        event_indicator_train = event_indicator[train_indices]
        if additional_inputs is not None:
            additional_inputs_train = ()
            for a in additional_inputs:
                additional_inputs_train += (a[train_indices],)
        else:
            additional_inputs_train = None
        input = np.column_stack((trt_train, X_train)).astype(np.float32)


        model_i = wrap_model(
            deepcopy(model),
            labtrans=labtrans,
            all_times=event_times,
            all_events=event_indicator,
            input_size=X.shape[1] + 1,
            verbose=verbose,
        )
        labels = (
            event_times_train.astype(np.float32),
            event_indicator_train.astype(int),
        )

        model_i.fit(
            input,
            labels,
            batch_size=batch_size,
            epochs=n_epochs,
            verbose=verbose,
            additional_inputs=additional_inputs_train,
        )  # type: ignore
        models[f"fold_{i}"] = model_i


    return models


def predict_oof(
    models_per_fold: dict[str, Union[PycoxWrapperCauseSpecific, PycoxWrapper]],
    skf: StratifiedKFold,
    X: np.ndarray,
    trt: np.ndarray,
    event_times: np.ndarray,
    event_indicator: np.ndarray,
    labtrans,
    additional_inputs: Optional[Tuple],
):
    num_risks = len(np.unique(event_indicator)) - 1  # subtract 1 for censoring

    if labtrans is not None:
        jumps = labtrans.cuts
    else:
        jumps = np.unique(event_times)

    surv_1 = np.empty((X.shape[0], len(jumps)))
    surv_0 = np.empty((X.shape[0], len(jumps)))
    cumhaz_1 = np.empty((X.shape[0], len(jumps), num_risks))
    cumhaz_0 = np.empty((X.shape[0], len(jumps), num_risks))
    cumhaz_f = np.empty((X.shape[0], len(jumps), num_risks))

    for i, (_, val_indices) in enumerate(skf.split(X, trt)):
        model_i = models_per_fold[f"fold_{i}"]

        X_val = X[val_indices]
        trt_val = trt[val_indices]
        if additional_inputs is not None:
            additional_inputs_val = ()
            for a in additional_inputs:
                additional_inputs_val += (a[val_indices],)
        else:
            additional_inputs_val = None

        # counterfactual
        X_val_1 = np.column_stack((np.ones_like(trt_val), X_val)).astype(np.float32)
        X_val_0 = np.column_stack((np.zeros_like(trt_val), X_val)).astype(np.float32)
        # factual
        X_val_f = np.column_stack((trt_val, X_val)).astype(np.float32)

        surv_1[val_indices] = model_i.predict_surv(X_val_1, additional_inputs_val)
        surv_0[val_indices] = model_i.predict_surv(X_val_0, additional_inputs_val)
        cumhaz_1[val_indices] = model_i.predict_cumhaz(
            X_val_1, additional_inputs_val
        )
        cumhaz_0[val_indices] = model_i.predict_cumhaz(
            X_val_0, additional_inputs_val
        )
        cumhaz_f[val_indices] = model_i.predict_cumhaz(
            X_val_f, additional_inputs_val
        )
        
    return (
        surv_1,
        surv_0,
        cumhaz_1,
        cumhaz_0,
        cumhaz_f,
        jumps,
    )