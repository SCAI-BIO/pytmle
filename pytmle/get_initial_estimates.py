import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from typing import Tuple, Optional, List
from pycox.models import DeepHit
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
import torchtuples as tt
import torch


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
    model_type: str = "coxph",
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
            model_type=model_type,
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
    model_type: str = "coxph",
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
        model_type=model_type,
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
    model_type: str = "coxph",
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
    model_type : str
        The type of survival model to use: "coxph" (default) or "deephit".

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
    
    # Label transformation for DeepHit
    labtrans = LabTransDiscreteTime(10)  # Set a fixed number of discretization bins
    
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
            if model_type == "coxph":
                model = CoxPHSurvivalAnalysis()
                y_train = Surv.from_arrays(event_indicator_train, event_times_train)
                inputs_train = trt_train.reshape(-1, 1) if model_j == "treatment_only" else np.column_stack((trt_train, X_train))
                inputs_val = trt_val.reshape(-1, 1) if model_j == "treatment_only" else np.column_stack((trt_val, X_val))
                model.fit(inputs_train, y_train)
                val_df[model_j] = -model.score(inputs_val, Surv.from_arrays(event_indicator_val, event_times_val))
            
            elif model_type == "deephit":
                net = tt.practical.MLPVanilla(X.shape[1] + 1, [32, 32], 1, batch_norm=True, dropout=0.1)
                model = DeepHit(net, tt.optim.Adam, duration_index=np.arange(event_times.max()))
                idx_durations_train, events_train = labtrans.fit_transform(event_times_train, event_indicator_train)
                train_data = (torch.tensor(np.column_stack((trt_train, X_train)), dtype=torch.float32), 
                              (torch.tensor(idx_durations_train, dtype=torch.long), 
                               torch.tensor(events_train, dtype=torch.long)))
                model.fit(
                    train_data,
                    batch_size=128,
                    epochs=100,
                    verbose=False,
                )
                inputs_val = np.column_stack((trt_val, X_val))
                val_df[model_j] = model.interpolate(10).predict_surv(inputs_val)

        sup_lrn_lib_risk.loc[val_indices] = val_df[models_candidates]

    sl_cv_risk = sup_lrn_lib_risk.sum()
    sl_chosen_model = sl_cv_risk.idxmin()

    if model_type == "coxph":
        final_model = CoxPHSurvivalAnalysis()
        y = Surv.from_arrays(event_indicator, event_times)
        inputs = trt.reshape(-1, 1) if sl_chosen_model == "treatment_only" else np.column_stack((trt, X))
        final_model.fit(inputs, y)
        
        inputs_copy = inputs.copy()
        inputs_copy[:, 0] = 1
        surv_1 = final_model.predict_survival_function(inputs_copy, return_array=True)
        cum_haz_1 = final_model.predict_cumulative_hazard_function(inputs_copy, return_array=True)

        inputs_copy[:, 0] = 0
        surv_0 = final_model.predict_survival_function(inputs_copy, return_array=True)
        cum_haz_0 = final_model.predict_cumulative_hazard_function(inputs_copy, return_array=True)

    elif model_type == "deephit":
        idx_durations, events = labtrans.fit_transform(event_times, event_indicator)
        train_data = (torch.tensor(np.column_stack((trt, X)), dtype=torch.float32), 
                      (torch.tensor(idx_durations, dtype=torch.long), 
                       torch.tensor(events, dtype=torch.long)))
        final_model.fit(
            train_data,
            batch_size=128,
            epochs=100,
            verbose=False,
        )
        
        inputs_copy = np.column_stack((np.ones_like(trt), X))
        surv_1 = final_model.interpolate(10).predict_surv(inputs_copy)
        cum_haz_1 = -np.log(surv_1 + epsilon)  # Convert survival to cumulative hazard
        
        inputs_copy[:, 0] = 0
        surv_0 = final_model.interpolate(10).predict_surv(inputs_copy)
        cum_haz_0 = -np.log(surv_0 + epsilon)

    return surv_1, surv_0, cum_haz_1, cum_haz_0, final_model
