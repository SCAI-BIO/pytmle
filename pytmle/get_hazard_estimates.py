import numpy as np
import pandas as pd
from functools import reduce
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.util import Surv

def get_haz_fit(data, model, cv_folds, hazards, return_models):
    """
    Fits hazard models for survival analysis using cross-validation and model selection.

    Args:
        data (pd.DataFrame): Input dataset with attributes for event time, event type, treatment, and covariates.
        model (dict): Dictionary of model configurations for different hazard types.
        cv_folds (list): List of dictionaries containing training and validation indices for cross-validation.
        hazards (pd.DataFrame): Hazard values for specific time points.
        return_models (bool): If True, returns fitted models for further inspection.

    Returns:
        list: A list of fitted hazard models with baseline hazards for each event type.
    """
    event_time_col = data.attrs.get("EventTime")
    event_type_col = data.attrs.get("EventType")
    treatment_col = data.attrs.get("Treatment")
    covariate_cols = data.attrs.get("CovNames", [])
    
    sup_learner_models = []

    for hazard_type, hazard_model in model.items():
        sup_learner_lib_risk = pd.DataFrame(np.nan, index=range(len(data)), columns=hazard_model.keys())
        model_fits = []

        for fold in cv_folds:
            train_indices = fold["training_set"]
            valid_indices = fold["validation_set"]

            train_data = data.iloc[train_indices].copy()
            valid_data = data.iloc[valid_indices].copy()

            for model_name, model_spec in hazard_model.items():
                covariate_cols = [treatment_col] + [col for col in train_data.columns if col not in {event_time_col, event_type_col, treatment_col}]
                x_train = train_data[covariate_cols].values
                y_train = Surv.from_dataframe(event_type_col, event_time_col, train_data)

                if model_spec["type"] == "coxnet":
                    # Penalized Cox model with scikit-survival CoxnetSurvivalAnalysis
                    model_fit = CoxnetSurvivalAnalysis()
                    model_fit.fit(x_train, y_train)
                    x_valid = valid_data[covariate_cols].values
                    exp_coef = np.exp(model_fit.predict(x_valid))

                else:
                    # Unpenalized Cox model with scikit-survival CoxPHSurvivalAnalysis
                    model_fit = CoxPHSurvivalAnalysis()
                    model_fit.fit(x_train, y_train)
                    x_valid = valid_data[covariate_cols].values
                    exp_coef = np.exp(model_fit.predict(x_valid))

                # Storing model fit if return_models is True
                if return_models:
                    model_fits.append(model_fit)

                # Compute validation risk
                valid_data["fit_lp"] = exp_coef
                valid_data["at_risk"] = valid_data["fit_lp"].cumsum()[::-1]
                valid_data["risk"] = (valid_data[event_type_col] == hazard_type) * (valid_data["fit_lp"] - np.log(valid_data["at_risk"].replace(0, 1)))
                sup_learner_lib_risk.loc[valid_indices, model_name] = -valid_data["risk"].sum()

        # Selecting best model based on cross-validation risk
        best_model_name = sup_learner_lib_risk.mean().idxmin()
        best_model = hazard_model[best_model_name]

        # Fit best model on full data
        x_data = data[covariate_cols].values
        y_data = Surv.from_dataframe(event_type_col, event_time_col, data)
        
        if best_model["type"] == "coxnet":
            final_model_fit = CoxnetSurvivalAnalysis()
            final_model_fit.fit(x_data, y_data)
        else:
            final_model_fit = CoxPHSurvivalAnalysis()
            final_model_fit.fit(x_data, y_data)

        # Compute baseline hazards
        baseline_hazards = pd.DataFrame({"Time": hazards["Time"], "BaseHaz": 0})
        baseline_hazards["BaseHaz"] = np.cumsum(np.maximum(0, baseline_hazards["BaseHaz"].ffill().diff()))

        haz_fit_out = {
            "HazFit": final_model_fit,
            "BaseHaz": baseline_hazards,
            "hazard_type": hazard_type,
            "SupLearnerCVRisks": sup_learner_lib_risk.mean(),
            "best_model": best_model_name
        }

        if return_models:
            haz_fit_out["ModelFits"] = model_fits

        sup_learner_models.append(haz_fit_out)

    return sup_learner_models


def get_haz_surv_pred(data, haz_fits, min_nuisance, target_event, target_time, regime):
    """
    Computes hazard and survival predictions for each regime at specified target times and events
    using CoxnetSurvivalAnalysis or CoxPHSurvivalAnalysis models from scikit-survival.
    
    Args:
        data (pd.DataFrame): Input dataset containing event and censoring information, with relevant 
                             attributes like 'EventTime', 'EventType', 'ID', 'Treatment', and 'CovNames' 
                             stored in `data.attrs`.
        haz_fits (list): List of fitted hazard models (CoxnetSurvivalAnalysis or CoxPHSurvivalAnalysis).
        min_nuisance (float): Minimum threshold for nuisance weights to avoid instability.
        target_event (list): List of target events for survival estimation.
        target_time (list): List of target times for hazard estimation.
        regime (list): List of intervention regimes.

    Returns:
        list: Hazard and survival predictions, including total survival and lagged censoring survival.
    """
    # Accessing attributes from data.attrs
    event_time_col = data.attrs.get("EventTime")
    event_type_col = data.attrs.get("EventType")
    treatment_col = data.attrs.get("Treatment")
    covariate_cols = data.attrs.get("CovNames", [])
    
    # Check for censoring based on event type column
    censored = any(data[event_type_col] <= 0)
    
    # Generate target combinations for time and event
    target = pd.DataFrame({"Time": target_time, "Event": target_event})
    
    # Select columns for predictions (excluding event time/type and ID)
    covariate_cols = [treatment_col] + [col for col in data.columns if col not in {event_time_col, event_type_col, treatment_col}]
    
    pred_haz_surv = []
    
    for reg in regime:
        # Create prediction data with adjusted treatment values from regime
        pred_data = data[covariate_cols].copy()
        trt_names = reg.columns
        pred_data[trt_names] = reg[trt_names].values
        
        pred_haz = []
        for haz_fit in haz_fits:
            # Check the type of hazard model and make predictions accordingly
            if isinstance(haz_fit["HazFit"], CoxnetSurvivalAnalysis):  # Penalized Cox model
                exp_coef = np.exp(haz_fit["HazFit"].predict(pred_data))
                haz = np.array([haz_fit["BaseHaz"]["BaseHaz"] * exp_lp for exp_lp in exp_coef])
            
            elif isinstance(haz_fit["HazFit"], CoxPHSurvivalAnalysis):  # Unpenalized Cox model
                exp_coef = np.exp(haz_fit["HazFit"].predict(pred_data))
                haz = np.array([haz_fit["BaseHaz"]["BaseHaz"] * exp_lp for exp_lp in exp_coef])
                
            haz = np.array(haz)
            haz['j'] = haz_fit.get("j")
            pred_haz.append(haz)
        
        # Separate censoring and hazard components
        cens_ind = [i for i, haz in enumerate(pred_haz) if haz["j"] <= 0]
        haz_ind = [i for i in range(len(pred_haz)) if i not in cens_ind]
        
        # Calculate cumulative survival function
        total_surv = np.exp(-np.cumsum(reduce(np.add, [pred_haz[i] for i in haz_ind]), axis=1))
        total_surv[total_surv < 1e-12] = 1e-12  # Avoid values too close to zero
        
        # Handle censoring if needed
        if censored:
            lagged_cens_surv = np.concatenate(([1], np.exp(-np.cumsum(pred_haz[cens_ind[0]][:-1]))))
        else:
            lagged_cens_surv = np.array([1])
        
        # Collect hazard and survival information
        pred_haz = [pred_haz[i] for i in haz_ind]
        
        survival = {"TotalSurv": total_surv, "LaggedCensSurv": lagged_cens_surv}
        pred_haz_surv.append({"Hazards": pred_haz, "Survival": survival})
    
    return pred_haz_surv
