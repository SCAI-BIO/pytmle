from typing import Dict, List, Optional
import numpy as np

from pytmle.estimates import InitialEstimates, UpdatedEstimates
from pytmle.get_influence_curve import get_eic


def tmle_update(
    initial_estimates: Dict[int, InitialEstimates],
    event_times: np.ndarray,
    event_indicator: np.ndarray,
    target_times: List[float],
    target_events: List[int] = [1],
    max_updates: int = 500,
    min_nuisance: Optional[float] = None,
    gcomp: bool = False,
) -> Dict[int, UpdatedEstimates]:
    """
    Function to update the initial estimates using the TMLE algorithm.

    Parameters
    ----------
    initial_estimates : Dict[int, InitialEstimates]
        Dictionary of initial estimates.
    target_times : List[float]
        List of target times for which effects are estimated.
    target_events : List[int]
        List of target events for which effects are estimated. Default is [1].
    event_times : np.ndarray
        Array of event times.
    event_indicator : np.ndarray
        Array of event indicators (censoring is 0).
    max_updates : int
        Maximum number of updates to the estimates in the TMLE loop.
    min_nuisance : Optional[float]
        Value between 0 and 1 for truncating the g-related denomiator of the clever covariate.
    gcomp : bool
        Whether to return the g-computation estimates. Default is False.

    Returns
    -------
    updated_estimates : Dict[int, UpdatedEstimates]
        Dictionary of updated estimates.
    """

    updated_estimates = {
        i: UpdatedEstimates.from_initial_estimates(
            initial_estimates[i], target_events, target_times, min_nuisance
        )
        for i in initial_estimates.keys()
    }
    updated_estimates = get_eic(
        estimates=updated_estimates,
        event_times=event_times,
        event_indicator=event_indicator,
        g_comp=gcomp,
    )
    # TODO: Implement TMLE update loop
    return updated_estimates
