import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import log_loss

def get_prop_score(treatment_values, covariate_data, treatment_model, min_nuisance, regime, cv_folds, treatment_loss=None, return_models=False):
    """
    Calculates propensity scores for each regime using specified treatment models and covariates.
    
    Args:
        treatment_values (np.ndarray): Array of observed treatment values.
        covariate_data (pd.DataFrame): DataFrame of covariates.
        treatment_model (list or dict): List or dictionary of model configurations or fitted models.
        min_nuisance (float): Minimum threshold for nuisance values to ensure stability.
        regime (dict): Dictionary specifying different intervention regimes of interest.
        cv_folds (list): Cross-validation folds for model fitting.
        treatment_loss (str or callable): Loss function for treatment model (optional).
        return_models (bool): If True, returns fitted models for further inspection.

    Returns:
        dict: A dictionary of propensity scores for each regime, with attributes for saved models.
    """
    # Initialize treatment model fits
    treatment_fits = []
    
    # Fit models for each treatment column
    for i in range(treatment_values.shape[1]):
        if treatment_model[i].get("Backend") == "SuperLearner":
            outcome_variable = treatment_values[:, i]
            predictor_variables = covariate_data if i == 0 else np.column_stack((treatment_values[:, :i], covariate_data))

            if len(np.unique(outcome_variable)) == 2:
                model = LogisticRegression()  # Binary outcome model
            else:
                model = RandomForestClassifier()  # Continuous outcome model
                
            # Cross-validated prediction
            cv_predictions = cross_val_predict(model, predictor_variables, outcome_variable, cv=len(cv_folds), method="predict_proba")[:, 1]
            model.fit(predictor_variables, outcome_variable)  # Fit model on entire data if return_models is True
            
            treatment_fits.append(model if return_models else None)
        else:
            raise NotImplementedError("Only 'SuperLearner'-like models are implemented.")

    # Calculate propensity scores for each regime
    propensity_scores = []
    for intervention in regime:
        if intervention.shape != treatment_values.shape:
            raise ValueError("Regime dimensions don't match observed treatment dimensions.")

        propensity_score = np.ones(treatment_values.shape[0])

        for i in range(treatment_values.shape[1]):
            intervention_column = intervention[:, i]
            if not np.all(np.isin(intervention_column, [0, 1])):
                raise ValueError("Non-binary intervention variables are not supported.")

            # Predict propensity scores for each treatment model and apply intervention
            if treatment_model[i].get("Backend") == "SuperLearner":
                predictor_variables = covariate_data if i == 0 else np.column_stack((treatment_values[:, :i], covariate_data))
                g_a = treatment_fits[i].predict_proba(predictor_variables)[:, 1] if return_models else cv_predictions
                g_a = np.where(intervention_column == 0, 1 - g_a, g_a)  # Adjust for 0/1 intervention
                propensity_score *= g_a

        # Apply minimum nuisance threshold for stability
        propensity_score = np.maximum(propensity_score, min_nuisance)
        
        # Store attributes for intervention and observed propensities
        propensity_scores.append(propensity_score)
    
    # Return models if requested
    if return_models:
        return {"PropensityScores": propensity_scores, "TreatmentFits": treatment_fits}
    else:
        return {"PropensityScores": propensity_scores, "TreatmentFits": "Treatment fits not saved because `return_models` was set to False"}