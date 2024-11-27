# fixtures for synthetic model inputs can be defined here
import pytest
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from pytmle.estimates import InitialEstimates, UpdatedEstimates
from pytmle.get_influence_curve import get_eic

def get_mock_input_data(n_samples: int = 1000) -> pd.DataFrame:
    np.random.seed(42)

    data = {
        "group": np.random.binomial(1, 0.5, n_samples),
        "event_indicator": np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.2, 0.1]),
        "event_time": np.round(np.random.exponential(scale=10, size=n_samples), 2),
        "x1": np.random.normal(0, 1, n_samples),
        "x2": np.random.normal(0, 1, n_samples),
        "x3": np.random.normal(0, 1, n_samples),
    }

    df = pd.DataFrame(data)
    return df


def get_mock_initial_estimates(df: pd.DataFrame) -> Dict[int, InitialEstimates]:
    # Columns of 0s and 1s for potential outcomes
    df["group_1"] = 1
    df["group_0"] = 0

    # Estimate propensity scores
    treatment_model = LogisticRegression()
    treatment_model.fit(df[["x1", "x2", "x3"]], df["group"])
    propensity_scores = treatment_model.predict_proba(df[["x1", "x2", "x3"]])
    propensity_scores[df["group"] == 0] = 1 - propensity_scores[df["group"] == 0]

    # Estimate censoring survival function using Cox regression from scikit-survival
    # Create structured arrays for survival analysis
    df["censored"] = df["event_indicator"] == 0
    df["event_1"] = df["event_indicator"] == 1
    df["event_2"] = df["event_indicator"] == 2
    survival_data_censoring = Surv.from_dataframe("censored", "event_time", df)
    survival_data_event_1 = Surv.from_dataframe("event_1", "event_time", df)
    survival_data_event_2 = Surv.from_dataframe("event_2", "event_time", df)

    # Fit Cox proportional hazards model to estimate censoring survival function
    cph_censoring = CoxPHSurvivalAnalysis()
    cph_censoring.fit(df[["group", "x1", "x2", "x3"]].values, survival_data_censoring)
    # Predict censoring survival function for both potential outcomes
    censoring_survival_function_1 = cph_censoring.predict_survival_function(
        df[["group_1", "x1", "x2", "x3"]].values, return_array=True
    )
    censoring_survival_function_0 = cph_censoring.predict_survival_function(
        df[["group_0", "x1", "x2", "x3"]].values, return_array=True
    )

    # Fit cause-specific Cox proportional hazards model to estimate event 1 hazards
    cph_event_1 = CoxPHSurvivalAnalysis()
    cph_event_1.fit(df[["group", "x1", "x2", "x3"]].values, survival_data_event_1)
    # Predict event 1 hazards for both potential outcomes
    event_1_cum_hazards_1 = cph_event_1.predict_cumulative_hazard_function(
        df[["group_1", "x1", "x2", "x3"]].values, return_array=True
    )
    event_1_cum_hazards_0 = cph_event_1.predict_cumulative_hazard_function(
        df[["group_0", "x1", "x2", "x3"]].values, return_array=True
    )
    event_1_hazards_1 = np.diff(event_1_cum_hazards_1, prepend=0)
    event_1_hazards_0 = np.diff(event_1_cum_hazards_0, prepend=0)

    # Fit cause-specific Cox proportional hazards model to estimate event 2 hazards
    cph_event_2 = CoxPHSurvivalAnalysis()
    cph_event_2.fit(df[["group", "x1", "x2", "x3"]].values, survival_data_event_2)
    # Predict event 1 hazards for both potential outcomes
    event_2_cum_hazards_1 = cph_event_2.predict_cumulative_hazard_function(
        df[["group_1", "x1", "x2", "x3"]].values, return_array=True
    )
    event_2_cum_hazards_0 = cph_event_2.predict_cumulative_hazard_function(
        df[["group_0", "x1", "x2", "x3"]].values, return_array=True
    )
    event_2_hazards_1 = np.diff(event_2_cum_hazards_1, prepend=0)
    event_2_hazards_0 = np.diff(event_2_cum_hazards_0, prepend=0)

    initial_estimates_1 = InitialEstimates(
        propensity_scores=propensity_scores[:, 1],
        censoring_survival_function=censoring_survival_function_1,
        event_free_survival_function=np.exp(
            -(event_1_cum_hazards_1 + event_2_cum_hazards_1)
        ),
        hazards=np.stack((event_1_hazards_1, event_2_hazards_1), axis=-1),
        g_star_obs=df["group"].to_numpy(),
        times=cph_censoring.unique_times_,
    )
    initial_estimates_0 = InitialEstimates(
        propensity_scores=propensity_scores[:, 0],
        censoring_survival_function=censoring_survival_function_0,
        event_free_survival_function=np.exp(
            -(event_1_cum_hazards_0 + event_2_cum_hazards_0)
        ),
        hazards=np.stack((event_1_hazards_0, event_2_hazards_0), axis=-1),
        g_star_obs=1 - df["group"].to_numpy(),
        times=cph_censoring.unique_times_,
    )

    initial_estimates = {0: initial_estimates_0, 1: initial_estimates_1}
    return initial_estimates


@pytest.fixture()
def mock_tmle_update_inputs() -> Dict[str, Any]:
    # create mock initial estimates for testing, using default n_samples of 1000
    df = get_mock_input_data()
    mock_inputs = {
        "initial_estimates": get_mock_initial_estimates(df),
        "target_times": [1.0, 2.0, 3.0, 10.0, 20.0],
        "event_times": df["event_time"].values,
        "event_indicator": df["event_indicator"].values,
    }
    return mock_inputs

@pytest.fixture()
def mock_updated_estimates(mock_tmle_update_inputs) -> Dict[int, UpdatedEstimates]:
    updated_estimates = {
        i: UpdatedEstimates.from_initial_estimates(
            mock_tmle_update_inputs["initial_estimates"][i],
            target_events=[1],
            target_times=mock_tmle_update_inputs["target_times"],
        )
        for i in mock_tmle_update_inputs["initial_estimates"].keys()
    }
    # TODO: Change to actual TMLE update function when implemented
    updated_estimates = get_eic(
        estimates=updated_estimates,
        event_times=mock_tmle_update_inputs["event_times"],
        event_indicator=mock_tmle_update_inputs["event_indicator"],
        g_comp=True
    )
    return updated_estimates

if __name__ == "__main__":
    mock_inputs = get_mock_input_data()
    mock_initial_estimates = get_mock_initial_estimates(mock_inputs)
