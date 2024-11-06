from typing import Dict, List, Optional
import numpy as np

from pytmle.estimates import InitialEstimates, UpdatedEstimates


def tmle_update(
    initial_estimates: Dict[int, InitialEstimates],
    target_times: List[float],
    event_times: np.ndarray,
    event_indicator: np.ndarray,
    max_updates: int = 500,
    min_nuisance: Optional[float] = None,
) -> Dict[int, UpdatedEstimates]:
    """
    Function to update the initial estimates using the TMLE algorithm.

    Parameters
    ----------
    initial_estimates : Dict[int, InitialEstimates]
        Dictionary of initial estimates.
    target_times : List[float]
        List of target times for which effects are estimated.
    event_times : np.ndarray
        Array of event times.
    event_indicator : np.ndarray
        Array of event indicators (censoring is 0).
    max_updates : int
        Maximum number of updates to the estimates in the TMLE loop.
    min_nuisance : Optional[float]
        Value between 0 and 1 for truncating the g-related denomiator of the clever covariate.

    Returns
    -------
    updated_estimates : Dict[int, UpdatedEstimates]
        Dictionary of updated estimates.
    """
    raise NotImplementedError("TMLE update algorithm not yet implemented.")

    updated_estimates = {
        i: UpdatedEstimates.from_initial_estimates(initial_estimates[i], min_nuisance)
        for i in initial_estimates.keys()
    }
    # TODO: Implement TMLE algorithm
    return updated_estimates
