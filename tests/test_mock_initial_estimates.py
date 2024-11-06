import pytest
from .conftest import get_mock_input_data, get_mock_initial_estimates
from pytmle.estimates import InitialEstimates


@pytest.mark.parametrize("n_samples", [100, 500, 1000])
def test_get_mock_initial_estimates(n_samples):
    """
    Test that the function get_mock_initial_estimates returns a dictionary of correct InitialEstimates objects.
    """
    mock_inputs = get_mock_input_data(n_samples)
    initial_estimates = get_mock_initial_estimates(mock_inputs)

    assert isinstance(initial_estimates, dict)
    assert 0 in initial_estimates
    assert 1 in initial_estimates

    for est in initial_estimates.values():
        assert isinstance(est, InitialEstimates)
        assert len(est) == n_samples
        assert est.propensity_scores.ndim == 1
        assert est.censoring_survival_function.ndim == 2
        assert est.event_free_survival_function.ndim == 2
        assert est.hazards.ndim == 3
        assert est.hazards.shape[-1] == 2  # two competing risks
