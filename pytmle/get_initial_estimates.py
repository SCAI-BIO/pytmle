import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from typing import Tuple, Optional, List


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
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
        The number of cross-validation folds.
    return_model : bool
        Whether to return the fitted models.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]
        The estimated counterfactual hazard functions for each event type,
        the estimated counterfactual survival functions for each event type,
        the fitted models.
    """
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
        models[f"event_{event}"] = model
    haz_1 = np.stack(hazards_1, axis=-1)
    haz_0 = np.stack(hazards_0, axis=-1)
    cum_haz_1 = np.stack(cum_hazards_1, axis=-1)
    cum_haz_0 = np.stack(cum_hazards_0, axis=-1)
    surv_1 = np.exp(-cum_haz_1.sum(axis=-1))
    surv_0 = np.exp(-cum_haz_0.sum(axis=-1))

    if return_model:
        return haz_1, haz_0, surv_1, surv_0, models
    return haz_1, haz_0, surv_1, surv_0, {}


def fit_default_censoring_model(
    X: np.ndarray,
    trt: np.ndarray,
    event_times: np.ndarray,
    event_indicator: np.ndarray,
    cv_folds: int,
    return_model: bool,
) -> Tuple[np.ndarray, np.ndarray, dict]:
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

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, dict]
        The estimated counterfactual survival functions for censoring, the fitted model.
    """
    cens_surv_1, cens_surv_0, _, _, model = fit_haz_superlearner(
        X=X,
        trt=trt,
        event_times=event_times,
        event_indicator=event_indicator == 0,
        cv_folds=cv_folds,
    )
    if return_model:
        return cens_surv_1, cens_surv_0, {"censoring_model": model}
    return cens_surv_1, cens_surv_0, {}


def fit_haz_superlearner(
    X: np.ndarray,
    trt: np.ndarray,
    event_times: np.ndarray,
    event_indicator: np.ndarray,
    cv_folds: int,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[CoxPHSurvivalAnalysis]
]:
    """
    Fit a superlearner model to estimate the hazard function.
    The superlearner is a discrete selector between two Cox PH models: treatment-only and main terms.
    The main terms model includes the treatment and the features as inputs.
    The superlearner is trained using cross-validation.
    The model with the lowest cross-validated loss is selected.
    The selected model is then refit on the whole data.

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
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[CoxPHSurvivalAnalysis]]
        The estimated counterfactual survival functions for treatment and control,
        the estimated counterfactual cumulative hazard functions for treatment and control,
        the fitted model.
    """
    models_candidates = ["treatment_only", "main_terms"]

    skf = StratifiedKFold(n_splits=cv_folds)

    sup_lrn_lib_risk = pd.DataFrame(
        np.full((X.shape[0], 2), np.nan),
        columns=models_candidates,
    )

    for train_indices, val_indices in skf.split(X, trt):
        X_train, X_val = X[train_indices], X[val_indices]
        trt_train, trt_val = trt[train_indices], trt[val_indices]
        event_times_train, event_times_val = (
            event_times[train_indices],
            event_times[val_indices],
        )
        event_indicator_train, event_indicator_val = (
            event_indicator[train_indices],
            event_indicator[val_indices],
        )
        val_df = pd.DataFrame(
            np.column_stack((event_indicator_val, event_times_val, trt_val)),
            columns=["event_indicator", "event_times", "trt"],
        )
        val_df.sort_values(by="event_times", ascending=False, inplace=True)

        for model_j in models_candidates:
            # train model
            cph = CoxPHSurvivalAnalysis()
            y_train = Surv.from_arrays(event_indicator_train, event_times_train)
            if model_j == "treatment_only":
                inputs_train = trt_train.reshape(-1, 1)
                inputs_val = trt_val.reshape(-1, 1)
            else:
                inputs_train = np.column_stack((trt_train, X_train))
                inputs_val = np.column_stack((trt_val, X_val))
            cph.fit(inputs_train, y_train)

            # validation loss (-log partial likelihood)
            val_df["lp"] = cph.predict(inputs_val)
            val_df["at_risk"] = np.exp(val_df["lp"]).cumsum()
            val_df.loc[val_df["at_risk"] == 0, "at_risk"] = 1
            val_df[model_j] = (val_df["event_indicator"] == 1) * (
                val_df["lp"] - np.log(val_df["at_risk"])
            )

        sup_lrn_lib_risk.loc[val_indices] = val_df[models_candidates]

    # metalearner (discrete selector)
    sl_cv_risk = -sup_lrn_lib_risk.sum()
    sl_chosen_model = sl_cv_risk.idxmin()

    # refit chosen model on the whole data
    cph = CoxPHSurvivalAnalysis()
    y = Surv.from_arrays(event_indicator, event_times)
    if sl_chosen_model == "treatment_only":
        inputs = trt.reshape(-1, 1)
    else:
        inputs = np.column_stack((trt, X))
    cph.fit(inputs, y)

    # get survival predictions
    inputs_copy = inputs.copy()
    inputs_copy[:, 0] = 1
    surv_1 = cph.predict_survival_function(inputs_copy, return_array=True)
    cum_haz_1 = cph.predict_cumulative_hazard_function(inputs_copy, return_array=True)

    inputs_copy[:, 0] = 0
    surv_0 = cph.predict_survival_function(inputs_copy, return_array=True)
    cum_haz_0 = cph.predict_cumulative_hazard_function(inputs_copy, return_array=True)

    return surv_1, surv_0, cum_haz_1, cum_haz_0, cph
