import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class InitialEstimates:
    propensity_scores: np.ndarray
    hazards: np.ndarray
    event_free_survival_function: np.ndarray
    censoring_survival_function: np.ndarray
    g_star_obs: np.ndarray
    times: np.ndarray

    def __post_init__(self):
        if not (
            len(self.propensity_scores)
            == len(self.hazards)
            == len(self.event_free_survival_function)
            == len(self.censoring_survival_function)
            == len(self.g_star_obs)
        ):
            raise RuntimeError(
                f"All initial estimates must have the same first dimension, got ({len(self.propensity_scores)}, {len(self.hazards)}, {len(self.event_free_survival_function)}, {len(self.censoring_survival_function)})"
            )

    def __len__(self):
        return len(self.propensity_scores)


@dataclass
class UpdatedEstimates(InitialEstimates):
    min_nuisance: Optional[float] = None
    has_converged: bool = False
    nuisance_weight: Optional[np.ndarray] = None
    target_events: Optional[List[int]] = None
    target_times: Optional[List[float]] = None
    g_comp_est: Optional[pd.DataFrame] = None
    ic: Optional[pd.DataFrame] = None
    summ_eic: Optional[pd.DataFrame] = None

    def __post_init__(self):
        super().__post_init__()
        if self.min_nuisance is None:
            self.min_nuisance = (
                5
                / (len(self.propensity_scores) ** 0.5)
                / (np.log(len(self.propensity_scores)))
            )
        self._set_nuisance_weight()
        if self.target_times is None:
            # default if not target_times are given: only target the last time point
            self.target_times = [self.times[-1]]
        else:
            self._update_for_target_times()

    def _set_nuisance_weight(self):
        nuisance_denominator = (
            self.propensity_scores[:, np.newaxis] * self.censoring_survival_function
        )
        # TODO: Add positivity check as in https://github.com/imbroglio-dc/concrete/blob/main/R/getInitialEstimate.R#L64?
        self.nuisance_weight = 1 / np.maximum(nuisance_denominator, self.min_nuisance)  # type: ignore

    @classmethod
    def from_initial_estimates(
        cls,
        initial_estimates: InitialEstimates,
        target_events: Optional[List[int]] = None,
        target_times: Optional[List[float]] = None,
        min_nuisance: Optional[float] = None,
    ) -> "UpdatedEstimates":
        return cls(
            propensity_scores=initial_estimates.propensity_scores,
            hazards=initial_estimates.hazards,
            event_free_survival_function=initial_estimates.event_free_survival_function,
            censoring_survival_function=initial_estimates.censoring_survival_function,
            min_nuisance=min_nuisance,
            target_events=target_events,
            target_times=target_times,
            g_star_obs=initial_estimates.g_star_obs,
            times=initial_estimates.times,
        )

    def _update_for_target_times(self):
        """
        Updates the time-related attributes of the object to include target times.
        This method performs the following steps:
        1. Combines and sorts the existing times and target times.
        2. Finds the indices where the target times should be inserted.
        3. Updates the `hazards`, `event_free_survival_function`, and `censoring_survival_function`
           attributes to account for the new target times by inserting appropriate values.
        4. Trims the `hazards`, `event_free_survival_function`, and `censoring_survival_function`
           attributes to only include times up to the maximum target time.
        5. Updates the `times` attribute to include the target times up to the maximum target time.
        Attributes:
            times (np.ndarray): Array of existing times.
            target_times (np.ndarray): Array of target times to be included.
            hazards (np.ndarray): Array of hazard values.
            event_free_survival_function (np.ndarray): Array of event-free survival function values.
            censoring_survival_function (np.ndarray): Array of censoring survival function values.
        """

        # Combine and sort the times
        all_times = np.sort(np.unique(np.concatenate((self.times, self.target_times))))  # type: ignore
        # get the maximum target time
        max_target_time = max(self.target_times)  # type: ignore

        # Find the indices where the new times should be inserted
        insert_indices = np.searchsorted(all_times, self.target_times)  # type: ignore

        # Update hazards, event_free_survival_function, and censoring_survival_function
        self.hazards = np.insert(self.hazards, insert_indices, 0, axis=1)
        self.event_free_survival_function = np.insert(
            self.event_free_survival_function,
            insert_indices,
            self.event_free_survival_function[:, insert_indices - 1],
            axis=1,
        )
        self.censoring_survival_function = np.insert(
            self.censoring_survival_function,
            insert_indices,
            self.censoring_survival_function[:, insert_indices - 1],
            axis=1,
        )

        # Find the index of the maximum target time and keep only times up to this index
        max_index = np.searchsorted(all_times, max_target_time)
        self.hazards = self.hazards[:, : max_index + 1, :]
        self.event_free_survival_function = self.event_free_survival_function[
            :, : max_index + 1
        ]
        self.censoring_survival_function = self.censoring_survival_function[
            :, : max_index + 1
        ]

        # Update times
        self.times = all_times[: max_index + 1]
