import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from typing import Tuple, Optional, List, Any
from copy import deepcopy

from .pycox_wrapper import PycoxWrapper
from .initial_estimates_default_models import get_default_models


def fit_propensity_super_learner(
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

    # Use cross_val_predict to generate out-of-fold predictions
    pred = cross_val_predict(super_learner, X, y, cv=cv_folds, method='predict_proba')
    if return_model:
        return pred[:, 1], pred[:, 0], {"propensity_model": super_learner}
    return pred[:, 1], pred[:, 0], {}


def fit_state_learner(
    X: np.ndarray,
    trt: np.ndarray,
    event_times: np.ndarray,
    event_indicator: np.ndarray,
    target_events: List[int],
    cv_folds: int,
    return_model: bool,
    models,
    labtrans,
    n_epochs: int = 100,
    batch_size: int = 128,
    fit_risks_model: bool = True,
    fit_censoring_model: bool = True,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    dict,
    Optional[Any],
]:
    """
    Fit a state learner to estimate the hazard functions for each event type.

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
    labtrans :
        The label transformer. Will be ignored if use_cox_superlearner is True.
    models :
        The risk model. Will be ignored if use_cox_superlearner is True.
    n_epochs : int
        The number of epochs. Will be ignored if use_cox_superlearner is True.
    batch_size : int
        The batch size. Will be ignored if use_cox_superlearner is True.
    fit_risks_model : bool
        Whether to fit the risk model.
    fit_censoring_model : bool
        Whether to fit the censoring model.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]
        The estimated counterfactual hazard functions for each event type,
        the estimated counterfactual survival functions for each event type,
        the estimated counterfactual censoring survival functions for each event type,
        the fitted models, the label transformer.
    """
    if models is None:
        models, labtrans = get_default_models(
            event_times=event_times,
            event_indicator=event_indicator,
            input_size=X.shape[1] + 1,  # number of covariateses + treatment
            labtrans=labtrans,
        )
    elif not isinstance(models, list):
        models = [models]
    if not isinstance(labtrans, list):
        if labtrans is None:
            labtrans = [None] * len(models)
        else:
            labtrans = [labtrans]
    if len(models) != len(labtrans):
        raise ValueError(
            "The number of models and label transformers must be the same."
        )
    fitted_models_dict = {}
    for risk_model, risk_labtrans in zip(models, labtrans):
        for censoring_model, censoring_labtrans in zip(models, labtrans):

            # TODO: Combine risk_labtrans and censoring_labtrans
            if risk_labtrans == censoring_labtrans:
                combined_labtrans = risk_labtrans
            (
                surv_1,
                surv_0,
                cens_surv_1,
                cens_surv_0,
                haz_1,
                haz_0,
                fitted_models,
            ) = cross_fit_risk_model(
                X=X,
                trt=trt,
                event_times=event_times,
                event_indicator=event_indicator,
                cv_folds=cv_folds,
                labtrans=combined_labtrans,
                risks_model=risk_model if fit_risks_model else None,
                censoring_model=censoring_model if fit_censoring_model else None,
                n_epochs=n_epochs,
                batch_size=batch_size,
            )
            # TODO: evaluate and keep only the best combination of models
            labtrans = labtrans[0]
            fitted_models_dict.update(fitted_models)

    if return_model:
        return (
            haz_1,
            haz_0,
            surv_1,
            surv_0,
            cens_surv_1,
            cens_surv_0,
            fitted_models_dict,
            labtrans,
        )
    return haz_1, haz_0, surv_1, surv_0, cens_surv_1, cens_surv_0, {}, labtrans


def cross_fit_risk_model(
    X: np.ndarray,
    trt: np.ndarray,
    event_times: np.ndarray,
    event_indicator: np.ndarray,
    cv_folds: int,
    labtrans,
    risks_model,
    censoring_model,
    n_epochs: int,
    batch_size: int,
):
    num_risks = len(np.unique(event_indicator)) - 1  # subtract 1 for censoring

    if labtrans is not None:
        jumps = labtrans.cuts
    else:
        jumps = np.unique(event_times)

    models = {}
    skf = StratifiedKFold(n_splits=cv_folds)
    surv_1 = np.empty((X.shape[0], len(jumps)))
    surv_0 = np.empty((X.shape[0], len(jumps)))
    cens_surv_1 = np.empty((X.shape[0], len(jumps)))
    cens_surv_0 = np.empty((X.shape[0], len(jumps)))
    haz_1 = np.empty((X.shape[0], len(jumps), num_risks))
    haz_0 = np.empty((X.shape[0], len(jumps), num_risks))
    for i, (train_indices, val_indices) in enumerate(skf.split(X, trt)):
        X_train, X_val = X[train_indices], X[val_indices]
        trt_train, trt_val = trt[train_indices], trt[val_indices]
        event_times_train = event_times[train_indices]
        event_indicator_train = event_indicator[train_indices]

        input = np.column_stack((trt_train, X_train)).astype(np.float32)

        X_val_1 = np.column_stack((np.ones_like(trt_val), X_val)).astype(np.float32)
        X_val_0 = np.column_stack((np.zeros_like(trt_val), X_val)).astype(np.float32)

        if risks_model is not None:
            model_i = PycoxWrapper(
                deepcopy(risks_model),
                labtrans=labtrans,
                all_times=event_times,
                all_events=event_indicator,
                input_size=X.shape[1] + 1,
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
                verbose=False,
            )  # type: ignore
            models[f"risks_model_fold_{i}"] = model_i

            surv_1[val_indices] = model_i.predict_surv(X_val_1)
            surv_0[val_indices] = model_i.predict_surv(X_val_0)
            haz_1[val_indices] = model_i.predict_haz(X_val_1)
            haz_0[val_indices] = model_i.predict_haz(X_val_0)
        if censoring_model is not None:
            model_i_censoring = PycoxWrapper(
                deepcopy(censoring_model),
                labtrans=labtrans,
                all_times=event_times,
                all_events=event_indicator,
                input_size=X.shape[1] + 1,
            )
            labels = (
                event_times_train.astype(np.float32),
                (event_indicator_train == 0).astype(int),
            )

            model_i_censoring.fit(
                input,
                labels,
                batch_size=batch_size,
                epochs=n_epochs,
                verbose=False,
            )  # type: ignore
            models[f"censoring_model_fold_{i}"] = model_i_censoring

            cens_surv_1[val_indices] = model_i_censoring.predict_surv(X_val_1)
            cens_surv_0[val_indices] = model_i_censoring.predict_surv(X_val_0)

    return (
        surv_1 if risks_model is not None else None,
        surv_0 if risks_model is not None else None,
        cens_surv_1 if censoring_model is not None else None,
        cens_surv_0 if censoring_model is not None else None,
        haz_1 if risks_model is not None else None,
        haz_0 if risks_model is not None else None,
        models,
    )
