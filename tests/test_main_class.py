import pytest
import pandas as pd

from pytmle import PyTMLE


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

    tmle.fit(max_updates=100, bootstrap=True, n_bootstrap=4, use_cox_superlearner=True)
    assert tmle._fitted
    # TMLE should converge easily on the simple mock data
    assert tmle.has_converged, "TMLE update did not converge."
    # check if the bootstrap results are stored
    assert tmle._bootstrap_results is not None

    # tmle.plot(
    #     "/home/jguski/COMMUTE/tmle/plot.png",
    #     type="risks",
    #     g_comp=True,
    #     color_1="#c00000",
    #     color_0="#699aaf",
    # )
    # tmle.plot_nuisance_weights(
    #     "/home/jguski/COMMUTE/tmle/nuisance_weights_plots",
    #     color_1="#c00000",
    #     color_0="#699aaf",
    # )
    # tmle.plot_norm_pn_eic("/home/jguski/COMMUTE/tmle/norm_pn_eic.png")
