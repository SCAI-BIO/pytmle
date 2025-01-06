import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_predict
from typing import Tuple

def fit_default_propensity_model(X: np.ndarray,
                                 y: np.ndarray,
                                 cv_folds: int) -> Tuple[np.ndarray, np.ndarray]:
    base_learners = [('rf', RandomForestClassifier()),
                    ('gb', GradientBoostingClassifier()),
                    ('lr', LogisticRegression(max_iter=200))]
    super_learner = StackingClassifier(estimators=base_learners,
                                       final_estimator=LogisticRegression(max_iter=200),
                                       cv=cv_folds)

    # This may overfit on the training data -> use cross_val_predict instead
    #super_learner.fit(X, y)
    #pred = super_learner.predict(X)

    # Use cross_val_predict to generate out-of-fold predictions
    pred = cross_val_predict(super_learner, X, y, cv=cv_folds, method='predict_proba')
    return pred[:, 1], pred[:, 0]

def fit_default_risk_model() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raise NotImplementedError("Method is currently not implemented.")

def fit_default_censoring_model() -> Tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("Method is currently not implemented.")