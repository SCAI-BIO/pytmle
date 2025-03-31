import pytest
from pytmle.tmle_update import tmle_update
from pytmle.estimates import UpdatedEstimates

import numpy as np


@pytest.mark.parametrize(
    "target_times, target_events",
    [
        ([1.0, 2.0, 3.0], [1]),
        ([1.0, 2.0, 3.0], [1, 2]),
        ([1.111, 2.222, 3.333], [1]),
        ([1.111, 2.222, 3.333], [1, 2]),
    ],
)
# (None, [1]),
# (None, [1, 2])])
def test_tmle_update(mock_tmle_update_inputs, target_times, target_events):
    mock_tmle_update_inputs["event_indicator"] *= np.isin(
        mock_tmle_update_inputs["event_indicator"], [0] + target_events
    ).astype(int)
    mock_tmle_update_inputs["target_times"] = target_times
    updated_estimates, _, has_converged, _ = tmle_update(
        **mock_tmle_update_inputs, g_comp=True, target_events=target_events
    )

    # TMLE should converge easily on the simple mock data
    assert has_converged, "TMLE update did not converge."
    assert isinstance(updated_estimates, dict)
    for _, estimate in updated_estimates.items():
        assert isinstance(estimate, UpdatedEstimates)

        # concerning EIC
        assert not estimate.g_comp_est is None
        if target_times is not None:
            assert all(estimate.g_comp_est["Time"].unique() == np.array(target_times))
        else:
            # if no targt_times given: Last available time point should be used
            assert all(estimate.g_comp_est["Time"].unique() == estimate.times[-1])
        assert not estimate.ic is None
        if target_times is not None:
            assert all(estimate.ic["Time"].unique() == np.array(target_times))
            assert len(estimate.ic) == len(target_times) * len(target_events) * len(estimate)
        else:
            # if no targt_times given: Last available time point should be used
            assert all(estimate.ic["Time"].unique() == estimate.times[-1])
            assert len(estimate.ic) == len(target_events) * len(estimate)
