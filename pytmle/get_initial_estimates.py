import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from typing import Tuple, Optional, List, Any
from copy import deepcopy

from .pycox_wrapper import PycoxWrapper

def fit_default_propensity_model(
    X: np.ndarray, y: np.ndarray, cv_folds: int, return_model: bool
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

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, dict]
        The estimated propensity scores, the estimated inverse propensity scores, the fitted model.
    """
    # all-nan columns are removed (relevant for E-value benchmark)
    X = X[:, ~np.isnan(X).all(axis=0)]
    base_learners = [
        ("rf", RandomForestClassifier()),
        ("gb", GradientBoostingClassifier()),
    ]
    # ('lr', LogisticRegression(max_iter=200))] # don't use for now because of convergence issues
    super_learner = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=cv_folds,
    )

    # This may overfit on the training data -> use cross_val_predict instead
    # super_learner.fit(X, y)
    # pred = super_learner.predict(X)

    # Use cross_val_predict to generate out-of-fold predictions
    pred = cross_val_predict(super_learner, X, y, cv=cv_folds, method='predict_proba')
    if return_model:
        return pred[:, 1], pred[:, 0], {"propensity_model": super_learner}
    return pred[:, 1], pred[:, 0], {}


def fit_default_risk_model(
    X: np.ndarray,
    trt: np.ndarray,
    event_times: np.ndarray,
    event_indicator: np.ndarray,
    target_events: List[int],
    cv_folds: int,
    return_model: bool,
    model,
    labtrans,
    use_cox_superlearner: bool = False,
    n_epochs: int = 100,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, Optional[Any]]:
    """
    Fit a superlearner model to estimate the hazard functions for each event type.

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
    use_cox_superlearner : bool
        Whether to use the Cox PH superlearner instead of cross-fitting an ML model.
    labtrans :
        The label transformer. Will be ignored if use_cox_superlearner is True.
    model :
        The risk model. Will be ignored if use_cox_superlearner is True.
    n_epochs : int
        The number of epochs. Will be ignored if use_cox_superlearner is True.
    batch_size : int
        The batch size. Will be ignored if use_cox_superlearner is True.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]
        The estimated counterfactual hazard functions for each event type,
        the estimated counterfactual survival functions for each event type,
        the fitted models, the label transformer.
    """

    if use_cox_superlearner:
        cum_hazards_1, cum_hazards_0 = [], []
        hazards_1, hazards_0 = [], []
        models = {}
        for event in target_events:
            _, _, cum_haz_1, cum_haz_0, model = fit_haz_superlearner(
                X=X,
                trt=trt,
                event_times=event_times,
                event_indicator=event_indicator == event,
                cv_folds=cv_folds,
            )
            hazards_1.append(np.diff(cum_haz_1, prepend=0))
            hazards_0.append(np.diff(cum_haz_0, prepend=0))
            cum_hazards_1.append(cum_haz_1)
            cum_hazards_0.append(cum_haz_0)
            models[f"event_{event}_model"] = model
        haz_1 = np.stack(hazards_1, axis=-1)
        haz_0 = np.stack(hazards_0, axis=-1)
        cum_haz_1 = np.stack(cum_hazards_1, axis=-1)
        cum_haz_0 = np.stack(cum_hazards_0, axis=-1)
        surv_1 = np.exp(-cum_haz_1.sum(axis=-1))
        surv_0 = np.exp(-cum_haz_0.sum(axis=-1))
    else:
        surv_1, surv_0, haz_1, haz_0, models, labtrans = cross_fit_risk_model(
            X=X,
            trt=trt,
            event_times=event_times,
            event_indicator=event_indicator,
            cv_folds=cv_folds,
            labtrans=labtrans,
            model=model,
            n_epochs=n_epochs,
            batch_size=batch_size,
        )
        models = {"risk_model_" + k: v for k, v in models.items()}

    if return_model:
        return haz_1, haz_0, surv_1, surv_0, models, labtrans
    return haz_1, haz_0, surv_1, surv_0, {}, labtrans


def fit_default_censoring_model(
    X: np.ndarray,
    trt: np.ndarray,
    event_times: np.ndarray,
    event_indicator: np.ndarray,
    cv_folds: int,
    return_model: bool,
    labtrans,
    model,
    use_cox_superlearner: bool = False,
    n_epochs: int = 100,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, dict, Optional[Any]]:
    """
    Fit a superlearner model to estimate the censoring hazard function.

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
    cv_folds : int
        The number of cross-validation folds.
    return_model : bool
        Whether to return the fitted model.
    labtrans : Optional[LabTrans]
        The label transformer. Will be ignored if use_cox_superlearner is True.
    model :
        The risk model. Will be ignored if use_cox_superlearner is True.
    use_cox_superlearner : bool
        Whether to use the Cox PH superlearner instead of cross-fitting an ML model.
    n_epochs : int
        The number of epochs. Will be ignored if use_cox_superlearner is True.
    batch_size : int
        The batch size. Will be ignored if use_cox_superlearner is True.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, dict]
        The estimated counterfactual survival functions for censoring, the fitted model, the label transformer.
    """
    if use_cox_superlearner:
        cens_surv_1, cens_surv_0, _, _, model = fit_haz_superlearner(
            X=X,
            trt=trt,
            event_times=event_times,
            event_indicator=event_indicator == 0,
            cv_folds=cv_folds,
        )
        if return_model:
            return cens_surv_1, cens_surv_0, {"censoring_model": model}, labtrans
    else:
        cens_surv_1, cens_surv_0, _, _, models, labtrans = cross_fit_risk_model(
            X=X,
            trt=trt,
            event_times=event_times,
            event_indicator=event_indicator == 0,
            cv_folds=cv_folds,
            labtrans=labtrans,
            model=model,
            n_epochs=n_epochs,
            batch_size=batch_size,
        )
        if return_model:
            models = {"censoring_model_" + k: v for k, v in models.items()}
            return cens_surv_1, cens_surv_0, models, labtrans

    return cens_surv_1, cens_surv_0, {}, labtrans


def cross_fit_risk_model(
    X: np.ndarray,
    trt: np.ndarray,
    event_times: np.ndarray,
    event_indicator: np.ndarray,
    cv_folds: int,
    labtrans,
    model,
    n_epochs: int,
    batch_size: int,
):
    num_risks = len(np.unique(event_indicator)) - 1  # subtract 1 for censoring
    model = PycoxWrapper(
        model,
        labtrans=labtrans,
        all_times=event_times,
        all_events=event_indicator,
        input_size=X.shape[1] + 1,
    )

    models = {}
    skf = StratifiedKFold(n_splits=cv_folds)
    surv_1 = np.empty((X.shape[0], len(model.jumps)))
    surv_0 = np.empty((X.shape[0], len(model.jumps)))
    haz_1 = np.empty((X.shape[0], len(model.jumps), num_risks))
    haz_0 = np.empty((X.shape[0], len(model.jumps), num_risks))
    for i, (train_indices, val_indices) in enumerate(skf.split(X, trt)):
        model_i = deepcopy(model)
        X_train, X_val = X[train_indices], X[val_indices]
        trt_train, trt_val = trt[train_indices], trt[val_indices]
        event_times_train = event_times[train_indices]
        event_indicator_train = event_indicator[train_indices]

        input = np.column_stack((trt_train, X_train)).astype(np.float32)
        labels = (
            event_times_train.astype(np.float32),
            event_indicator_train.astype(int),
        )

        model_i.fit(
            input,
            labels,
            batch_size=batch_size,
            epochs=n_epochs,
            verbose=False,
        )  # type: ignore
        models[f"fold_{i}"] = model_i
        X_val_1 = np.column_stack((np.ones_like(trt_val), X_val)).astype(np.float32)
        X_val_0 = np.column_stack((np.zeros_like(trt_val), X_val)).astype(np.float32)

        surv_1[val_indices] = model_i.predict_surv(X_val_1)
        surv_0[val_indices] = model_i.predict_surv(X_val_0)
        haz_1[val_indices] = model_i.predict_haz(X_val_1)
        haz_0[val_indices] = model_i.predict_haz(X_val_0)

    return surv_1, surv_0, haz_1, haz_0, models, model.labtrans


def fit_haz_superlearner(
    X: np.ndarray,
    trt: np.ndarray,
    event_times: np.ndarray,
    event_indicator: np.ndarray,
    cv_folds: int,
    epsilon: float = 1e-10,
) -> tuple:
    """
    Fit a superlearner model to estimate the hazard function.
    Allows selection between Cox PH and DeepHit models.

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
    cv_folds : int
        The number of cross-validation folds.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional]
        The estimated counterfactual survival and cumulative hazard functions.
    """
    models_candidates = ["treatment_only", "main_terms"]
    skf = StratifiedKFold(n_splits=cv_folds)
    sup_lrn_lib_risk = pd.DataFrame(
        np.full((X.shape[0], 2), np.nan),
        columns=models_candidates,
    )

    # all-nan columns are removed (relevant for E-value benchmark)
    X = X[:, ~np.isnan(X).all(axis=0)]

    for train_indices, val_indices in skf.split(X, trt):
        X_train, X_val = X[train_indices], X[val_indices]
        trt_train, trt_val = trt[train_indices], trt[val_indices]
        event_times_train, event_times_val = event_times[train_indices], event_times[val_indices]
        event_indicator_train, event_indicator_val = event_indicator[train_indices], event_indicator[val_indices]

        val_df = pd.DataFrame(
            np.column_stack((event_indicator_val, event_times_val, trt_val)),
            columns=["event_indicator", "event_times", "trt"],
        )
        val_df.sort_values(by="event_times", ascending=False, inplace=True)

        for model_j in models_candidates:
            model = CoxPHSurvivalAnalysis()
            y_train = Surv.from_arrays(event_indicator_train, event_times_train)
            inputs_train = (
                trt_train.reshape(-1, 1)
                if model_j == "treatment_only"
                else np.column_stack((trt_train, X_train))
            )
            inputs_val = (
                trt_val.reshape(-1, 1)
                if model_j == "treatment_only"
                else np.column_stack((trt_val, X_val))
            )
            model.fit(inputs_train, y_train)
            val_df[model_j] = -model.score(
                inputs_val, Surv.from_arrays(event_indicator_val, event_times_val)
            )

        sup_lrn_lib_risk.loc[val_indices] = val_df[models_candidates]

    sl_cv_risk = sup_lrn_lib_risk.sum()
    sl_chosen_model = sl_cv_risk.idxmin()

    final_model = CoxPHSurvivalAnalysis()
    y = Surv.from_arrays(event_indicator, event_times)
    inputs = (
        trt.reshape(-1, 1)
        if sl_chosen_model == "treatment_only"
        else np.column_stack((trt, X))
    )
    final_model.fit(inputs, y)

    inputs_copy = inputs.copy()
    inputs_copy[:, 0] = 1
    surv_1 = final_model.predict_survival_function(inputs_copy, return_array=True)
    cum_haz_1 = final_model.predict_cumulative_hazard_function(
        inputs_copy, return_array=True
    )

    inputs_copy[:, 0] = 0
    surv_0 = final_model.predict_survival_function(inputs_copy, return_array=True)
    cum_haz_0 = final_model.predict_cumulative_hazard_function(
        inputs_copy, return_array=True
    )

    return surv_1, surv_0, cum_haz_1, cum_haz_0, final_model
