import numpy as np
import pandas as pd
from pytmle.nuisance import trunc_nuisance_weight
from pytmle.get_hazard_estimates import get_haz_fit, get_haz_surv_pred
from pytmle.get_propensity_score import get_prop_score

def get_initial_estimate(data, model, cv_folds, min_nuisance, target_event, target_time, regime, return_models):
    """
    Calculates initial estimates for treatment propensity scores, hazards, survival, and nuisance weights.
    
    Args:
        data (pd.DataFrame): Input dataset, expected to include attributes for event time, event type, 
                             treatment, and covariate names.
        model (dict): A dictionary containing model specifications for different variables in `data`.
        cv_folds (list): Cross-validation fold information for fitting models.
        min_nuisance (float): Minimum threshold for nuisance values to ensure stability.
        target_event (list): Numeric vector defining the target event.
        target_time (list): Numeric vector defining the target time points.
        regime (dict): Dictionary specifying different intervention regimes of interest.
        return_models (bool): If True, returns fitted models for further inspection.
    
    Returns:
        dict: A dictionary containing initial estimates for each regime, including propensity scores, 
              hazards, event-free survival estimates, and nuisance weights.
    """
    # Accessing custom attributes using data.attrs.get
    time_val = data[data.attrs.get("EventTime")]
    censored = any(data[data.attrs.get("EventType")] <= 0)
    
    # Propensity Scores for Regimes of Interest
    print("\nEstimating Treatment Propensity:\n")
    prop_scores = get_prop_score(
        trt_val=data[data.attrs.get("Treatment")],
        cov_dt=data[data.attrs.get("CovNames")],
        trt_model={k: model[k] for k in data[data.attrs.get("Treatment")]},
        min_nuisance=min_nuisance,
        regime=regime,
        cv_folds=cv_folds,
        trt_loss=None,
        return_models=return_models
    )
    init_fits = prop_scores.get("TrtFit")
    
    # Hazard times and event baseline
    haz_times = np.unique(np.concatenate([target_time, time_val]))
    haz_times = haz_times[haz_times <= max(target_time)]
    hazards = pd.DataFrame({"Time": np.insert(haz_times, 0, 0)})
    
    print("\nEstimating Hazards:\n")
    haz_fits = get_haz_fit(data=data, model=model, cv_folds=cv_folds, hazards=hazards, return_models=return_models)
    init_fits.extend([hf.get("HazSL") for hf in haz_fits])
    
    haz_surv_preds = get_haz_surv_pred(
        data=data,
        haz_fits=haz_fits,
        min_nuisance=min_nuisance,
        target_event=target_event,
        target_time=target_time,
        regime=regime
    )
    
    initial_estimates = {}
    for a, prop_score in enumerate(prop_scores):
        if censored:
            nuisance_denom = np.array([
                prop_score[i] * haz_surv_preds[a]["Survival"]["LaggedCensSurv"][:, i] 
                for i in range(len(prop_score))
            ])
        else:
            srv = haz_surv_preds[a]["Survival"]["TotalSurv"]
            nuisance_denom = np.tile(prop_score, (srv.shape[0], srv.shape[1]))

        nuisance_weight = 1 / trunc_nuisance_weight(
            nuisance_denom=nuisance_denom,
            min_nuisance=min_nuisance,
            regime_name=list(prop_scores.keys())[a]
        )
        
        initial_estimates[regime[a]] = {
            "PropScore": prop_score,
            "Hazards": haz_surv_preds[a]["Hazards"],
            "EvntFreeSurv": haz_surv_preds[a]["Survival"]["TotalSurv"],
            "NuisanceWeight": nuisance_weight
        }
    
    initial_estimates["Times"] = hazards["Time"]
    initial_estimates["InitFits"] = init_fits
    
    return initial_estimates
