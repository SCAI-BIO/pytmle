import pytest
import pandas as pd

from pytmle import PyTMLE
from pytmle.bootstrap import BOOTSTRAP_METHODS


@pytest.mark.parametrize(
    "precomputed_initial_est_mask",
    [
        ([True, True, True]),
        ([True, False, True]),
        ([True, True, False]),
        ([False, True, True]),
        ([False, False, False]),
    ],
)
@pytest.mark.slow
def test_fit(mock_main_class_inputs, precomputed_initial_est_mask):
    df = mock_main_class_inputs["data"][["event_time", "event_indicator", "group", "x1", "x2", "x3"]]

    initial_estimates = mock_main_class_inputs["initial_estimates"]
    # test if method works for different sets of pre-computed initial estimates
    if not any(precomputed_initial_est_mask):
        initial_estimates = None
    else:
        if not precomputed_initial_est_mask[0]:
            initial_estimates[1].propensity_scores = None
            initial_estimates[0].propensity_scores = None
        if not precomputed_initial_est_mask[1]:
            initial_estimates[1].hazards = None
            initial_estimates[0].hazards = None
            initial_estimates[1].event_free_survival_function = None
            initial_estimates[0].event_free_survival_function = None
        if not precomputed_initial_est_mask[2]:
            initial_estimates[1].censoring_survival_function = None
            initial_estimates[0].censoring_survival_function = None
    tmle = PyTMLE(
        data=df, target_times=[1.0, 2.0, 3.0], initial_estimates=initial_estimates
    )

    tmle.fit(
        max_updates=100,
        bootstrap=True,
        n_bootstrap=8,
        stratified_bootstrap=True,
        cv_folds=2,
    )
    assert tmle._fitted
    # TMLE should converge easily on the simple mock data
    assert tmle.has_converged, "TMLE update did not converge."
    # check if the bootstrap results are stored
    assert tmle._bootstrap_results is not None

    # One bootstrap run keeps *both* interval constructions, in long format:
    # one row per estimand and per method, so that comparing percentile against
    # bias-corrected never costs a second run.
    boot = tmle._bootstrap_results
    assert "bootstrap_method" in boot.columns
    assert set(boot["bootstrap_method"]) == set(BOOTSTRAP_METHODS)
    assert not boot.duplicated(
        subset=["type", "Event", "Time", "Group", "bootstrap_method"]
    ).any()
    counts = boot["bootstrap_method"].value_counts()
    assert counts["percentile"] == counts["bc"]

    # ... and predict() hands both of them out, leaving the analytic bounds be.
    for type_ in ("risks", "rr", "rd"):
        pred = tmle.predict(type=type_)
        for suffix in ("pct", "bc"):
            for side in ("lower", "upper"):
                col = f"CI_{side}_bootstrap_{suffix}"
                assert col in pred.columns, f"{type_}: {col} missing"
                assert pred[col].notna().all(), f"{type_}: {col} is all missing"
            assert f"mean_bootstrap_{suffix}" in pred.columns
        assert "CI_lower" in pred.columns and "CI_upper" in pred.columns
