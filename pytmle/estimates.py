import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Union

from pytmle.g_computation import get_g_comp

@dataclass
class InitialEstimates:
    # these fields must be filled on instatiation
    times: np.ndarray
    g_star_obs: np.ndarray
    # these fields are optional and can be filled later
    propensity_scores: Optional[np.ndarray] = field(default=None)
    hazards: Optional[np.ndarray] = field(default=None)
    event_free_survival_function: Optional[np.ndarray] = field(default=None)
    censoring_survival_function: Optional[np.ndarray] = field(default=None)
    _length: Optional[int] = field(default=None, init=False)
    _run_checks: bool = field(default=True, init=False)

    def __setattr__(self, name, value):
        if value is not None and self._run_checks:
            if name in ["propensity_scores", 
                        "g_star_obs"]:
                self._check_compatibility(value, check_width=False)
            elif name in ["hazards", 
                        "event_free_survival_function", 
                        "censoring_survival_function"]:
                self._check_compatibility(value, check_width=True)
        super().__setattr__(name, value)

    def _check_compatibility(self, new_element, check_width):
        # check that all given estimates have the same length (first dimension size)
        if self._length is None:
            self._length = len(new_element)
        elif self._length != len(new_element):
            raise ValueError(
                f"All initial estimates must have the same first dimension, got elements with sizes {self._length} and {len(new_element)}."
            )
        if check_width and ((len(new_element.shape) < 2) or (new_element.shape[1] != len(self.times))):
            raise ValueError(
                f"The second dimension of all initial estimates must be in line with the given times, got {len(self.times)} times and element of shape {new_element.shape}."
            )

    def __getitem__(self, key: Union[np.ndarray, List[int]]) -> "InitialEstimates":
        """
        Enable subsetting of an InitialEstimates object (needed for bootstrapping)

        Args:
            key (Union[np.ndarray, List[int]]): The indices of the subset.

        Returns:
            InitialEstimates: A new InitialEstimates object containing the subset.
        """
        return InitialEstimates(
            times=self.times,
            g_star_obs=self.g_star_obs[key],
            propensity_scores=(
                self.propensity_scores[key]
                if self.propensity_scores is not None
                else None
            ),
            hazards=self.hazards[key] if self.hazards is not None else None,
            event_free_survival_function=(
                self.event_free_survival_function[key]
                if self.event_free_survival_function is not None
                else None
            ),
            censoring_survival_function=(
                self.censoring_survival_function[key]
                if self.censoring_survival_function is not None
                else None
            ),
        )

    def __len__(self):
        return self._length


@dataclass
class UpdatedEstimates(InitialEstimates):
    # all have to be given
    propensity_scores: np.ndarray # type: ignore
    hazards: np.ndarray # type: ignore
    event_free_survival_function: np.ndarray # type: ignore
    censoring_survival_function: np.ndarray # type: ignore

    # is set on initialization
    nuisance_weight: Optional[np.ndarray] = field(default=None, init=False)

    min_nuisance: Optional[float] = field(default=None)
    target_events: Optional[List[int]] = field(default=None)
    target_times: Optional[List[float]] = field(default=None)
    g_comp_est: Optional[pd.DataFrame] = field(default=None)
    ic: Optional[pd.DataFrame] = field(default=None)
    summ_eic: Optional[pd.DataFrame] = field(default=None)

    def __post_init__(self):
        if self.min_nuisance is None:
            self.min_nuisance = (
                5
                / (len(self.propensity_scores) ** 0.5)
                / (np.log(len(self.propensity_scores)))
            )
        if self.target_times is None:
            # default if not target_times are given: only target the last time point
            self.target_times = [self.times[-1]]
        else:
            self._update_for_target_times()
        self._set_nuisance_weight()

    def _set_nuisance_weight(self):
        lagged_censoring_survival_function = np.column_stack(
            [
                np.ones((self.censoring_survival_function.shape[0], 1)),
                self.censoring_survival_function[:, :-1],
            ],
        )
        nuisance_denominator = (
            self.propensity_scores[:, np.newaxis] * lagged_censoring_survival_function
        )
        # TODO: Add positivity check as in https://github.com/imbroglio-dc/concrete/blob/main/R/getInitialEstimate.R#L64?
        self.nuisance_weight = 1 / np.maximum(nuisance_denominator, self.min_nuisance)  # type: ignore
        self._check_compatibility(self.nuisance_weight, check_width=True)

    @classmethod
    def from_initial_estimates(
        cls,
        initial_estimates: InitialEstimates,
        target_events: Optional[List[int]] = None,
        target_times: Optional[List[float]] = None,
        min_nuisance: Optional[float] = None,
    ) -> "UpdatedEstimates":
        assert (initial_estimates.propensity_scores is not None and 
                initial_estimates.hazards is not None and 
                initial_estimates.event_free_survival_function is not None and
                initial_estimates.censoring_survival_function is not None), "All initial estimates have to be provided prior to an instatiation of UpdatedEstimates."
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
        Updates the time-related attributes of the object to include target times (plus 0).
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
        all_times = np.sort(np.unique(np.concatenate((self.times, [0] + self.target_times))))  # type: ignore

        if len(all_times) > len(self.times):

            # Update hazards, event_free_survival_function, and censoring_survival_function
            if 0 not in self.times:
                self.times = np.insert(self.times, 0, 0)
                self.hazards = np.insert(self.hazards, 0, 0, axis=1)
                self.event_free_survival_function = np.insert(self.event_free_survival_function, 0, 1, axis=1)
                self.censoring_survival_function = np.insert(self.censoring_survival_function, 0, 1, axis=1)

            # Find the indices where the new times should be inserted
            insert_times = [t for t in self.target_times if t not in self.times]
            insert_indices = np.searchsorted(all_times, insert_times)

            self.times = all_times

            hazards_new = self.hazards
            event_free_survival_function_new = self.event_free_survival_function
            censoring_survival_function_new = self.censoring_survival_function
            for idx in insert_indices:
                hazards_new = np.insert(hazards_new, idx, 0, axis=1)
                event_free_survival_function_new = np.insert(
                    event_free_survival_function_new,
                    idx,
                    event_free_survival_function_new[:, idx - 1],
                    axis=1,
                )
                censoring_survival_function_new = np.insert(
                    censoring_survival_function_new,
                    idx,
                    censoring_survival_function_new[:, idx - 1],
                    axis=1,
                )
            self.hazards = hazards_new
            self.event_free_survival_function = event_free_survival_function_new
            self.censoring_survival_function = censoring_survival_function_new

        # Find the index of the maximum target time
        max_target_time = max(self.target_times)  # type: ignore
        max_index = np.searchsorted(all_times, max_target_time)
        # Keep only times up to the maximum index
        self.times = all_times[: max_index + 1]
        self.hazards = self.hazards[:, : max_index + 1, :]
        self.event_free_survival_function = self.event_free_survival_function[
            :, : max_index + 1
        ]
        self.censoring_survival_function = self.censoring_survival_function[
            :, : max_index + 1
        ]

    def predict_mean_risks(self, g_comp: bool = False) -> pd.DataFrame:
        """
        Predict the mean risks for the target events and times.
        Args:           
            g_comp (bool): Flag to return the G-computation estimate instead of the TMLE estimate.
        Returns:
            pd.DataFrame: DataFrame with columns 'Event', 'Time', 'Pt Est', and 'SE' containing the mean counterfactual risks.
        """
        if g_comp:
            if self.g_comp_est is None:
                raise ValueError(
                    "g_comp_est is not available."
                )
            # return g_comp_estimate from BEFORE the TMLE update loop (standard error not available)
            pred_risk = self.g_comp_est
            pred_risk["SE"] = np.nan
        else:
            # return g_comp_estimate from AFTER the TMLE update loop
            if self.summ_eic is None or self.ic is None:
                raise ValueError(
                    "ic or summ_eic is not available."
                )
            pred_risk =  get_g_comp(
                eval_times=self.times,
                hazards=self.hazards,
                total_surv=self.event_free_survival_function,
                target_time=self.target_times,  # type: ignore
                target_events=self.target_events,  # type: ignore
            )
            pred_risk = pred_risk.merge(self.summ_eic, on=["Event", "Time"])
            pred_risk["SE"] = pred_risk["seEIC"] / len(self)**0.5
            pred_risk = pred_risk[["Event", "Time", "Risk", "SE"]]
        pred_risk.rename(columns={"Risk": "Pt Est"}, inplace=True)
        return pred_risk
