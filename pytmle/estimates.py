import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class InitialEstimates:
    propensity_scores: np.ndarray
    hazards: np.ndarray
    event_free_survival_function: np.ndarray
    censoring_survival_function: np.ndarray
    g_star_obs: np.ndarray

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

    def __post_init__(self):
        super().__post_init__()
        if self.min_nuisance is None:
            self.min_nuisance = (
                5
                / (len(self.propensity_scores) ** 0.5)
                / (np.log(len(self.propensity_scores)))
            )
        self._set_nuisance_weight()

    def _set_nuisance_weight(self):
        nuisance_denominator = (
            self.propensity_scores[:, np.newaxis] * self.censoring_survival_function
        )
        # TODO: Add positivity check as in https://github.com/imbroglio-dc/concrete/blob/main/R/getInitialEstimate.R#L64?
        self.nuisance_weight = 1 / np.maximum(nuisance_denominator, self.min_nuisance)  # type: ignore

    @classmethod
    def from_initial_estimates(
        cls, initial_estimates: InitialEstimates, min_nuisance: Optional[float] = None
    ) -> "UpdatedEstimates":
        return cls(
            propensity_scores=initial_estimates.propensity_scores,
            hazards=initial_estimates.hazards,
            event_free_survival_function=initial_estimates.event_free_survival_function,
            censoring_survival_function=initial_estimates.censoring_survival_function,
            min_nuisance=min_nuisance,
        )
