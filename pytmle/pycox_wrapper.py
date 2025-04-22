import numpy as np
from copy import deepcopy
from sksurv.util import Surv
from typing import Optional, Tuple
import warnings


class PycoxWrapper:
    """
    A wrapper class to unify the interface of different survival analysis libraries (pycox, scikit-survival)
    """

    def __init__(
        self,
        wrapped_model,
        labtrans,
        all_times: np.ndarray,
        all_events: np.ndarray,
        input_size: int = 0,
        verbose: int = 2,
    ):
        self.labtrans = labtrans
        self.all_times = all_times
        self.all_events = all_events
        self.input_size = input_size
        self.wrapped_model = wrapped_model

        if self.labtrans is not None:
            self.all_times, _ = self.labtrans.transform(self.all_times, self.all_events)
        self.fitted = False
        self.verbose = verbose

    def __str__(self):
        # Return the name of the wrapped model
        return type(self.wrapped_model).__name__

    def __repr__(self):
        # Return the name of the wrapped model
        return type(self.wrapped_model).__name__

    def _handle_all_missing_columns(self, input: np.ndarray, mode: str) -> np.ndarray:
        """Handle all missing columns in the input data.
        Some models cannot handle all-zero columns (e.g., Cox PH), others cannot handle changes in input dimension (neural network models).
        This is relevant for the E-value benchmark."""
        if mode == "remove":
            return input[:, ~np.isnan(input).all(axis=0)]
        elif mode == "zero":
            input[:, np.isnan(input).all(axis=0)] = 0
            return input
        else:
            raise ValueError("Invalid mode. Choose 'remove' or 'zero'.")

    def _update_times(
        self, predictions: np.ndarray, value: int, ffill: bool
    ) -> np.ndarray:
        """Update the predictions to include all given times with jumps in the dataset"""
        if len(predictions.shape) == 2:
            pred_updated = np.full((predictions.shape[0], len(self.jumps)), np.nan)
        else:
            pred_updated = np.full(
                (predictions.shape[0], len(self.jumps), predictions.shape[-1]), np.nan
            )

        if self.labtrans is None:
            fit_times_indices = (
                np.searchsorted(np.unique(self.all_times), np.unique(self.fit_times))
                - 1
            )
        else:
            if len(self.jumps) != predictions.shape[1]:
                fit_times_indices = np.searchsorted(
                    np.arange(len(self.jumps)), np.unique(self.fit_times)
                )
            else:
                fit_times_indices = np.arange(len(self.jumps))
        pred_updated[:, fit_times_indices] = predictions

        mask = np.isnan(pred_updated)
        if ffill:
            row, col = pred_updated.shape[:2]
            for i in range(row):
                for j in range(col):
                    if mask[i][j].all():
                        if j == 0:
                            pred_updated[i][j] = value
                        else:
                            pred_updated[i][j] = pred_updated[i][j - 1]
        else:
            pred_updated[mask] = value
        return pred_updated

    def fit(
        self,
        input: np.ndarray,
        target: np.ndarray,
        additional_inputs: Optional[Tuple[np.ndarray]] = None,
        **kwargs,
    ):
        if self.labtrans is not None:
            target = self.labtrans.transform(*target)
        self.fit_times = target[0]
        if "sksurv" in type(self.wrapped_model).__module__:
            # scikit-survival-based model
            target = Surv.from_arrays(target[1], target[0])
            input = self._handle_all_missing_columns(input, "remove")
            if additional_inputs is not None and self.verbose >= 1:
                warnings.warn(
                    "Additional inputs are not supported for scikit-survival models and will be ignored.",
                    RuntimeWarning,
                )
            self.wrapped_model.fit(input, target)
        else:
            # pycox-based model
            target = (target[0], target[1].astype(int))
            input = self._handle_all_missing_columns(input, "zero")
            if additional_inputs is not None:
                input = (input,) + additional_inputs  # type: ignore
            self.wrapped_model.fit(input, target, **kwargs)
            # Cox-like models in pycox require the baseline hazard to be computed after fitting
            if hasattr(self.wrapped_model, "compute_baseline_hazards"):
                self.wrapped_model.compute_baseline_hazards()
        self.fitted = True

    def predict_surv(
        self,
        input: np.ndarray,
        additional_inputs: Optional[Tuple[np.ndarray]],
        **kwargs,
    ) -> np.ndarray:
        """Predict survival function for a given input" """
        if not self.fitted:
            raise ValueError("Model has not been fitted")
        if hasattr(self.wrapped_model, "predict_surv"):
            # pycox
            input = self._handle_all_missing_columns(input, "zero")
            if additional_inputs is not None:
                input = (input,) + additional_inputs  # type: ignore
            surv = self.wrapped_model.predict_surv(input, **kwargs)
        elif hasattr(self.wrapped_model, "predict_survival_function"):
            # scikit-survival
            input = self._handle_all_missing_columns(input, "remove")
            surv = self.wrapped_model.predict_survival_function(
                input, return_array=True
            )
        else:
            raise ValueError("Model does not have a predict_surv method")
        if surv.shape[1] == len(input):
            surv = surv.T
        surv = self._update_times(surv, 1, ffill=True)
        return surv

    def predict_cumhaz(
        self,
        input: np.ndarray,
        additional_inputs: Optional[Tuple[np.ndarray]],
        **kwargs,
    ) -> np.ndarray:
        """Predict cumulative hazard function for a given input" """
        if not self.fitted:
            raise ValueError("Model has not been fitted")

        if hasattr(self.wrapped_model, "predict_cif"):
            # pycox with competing risks (e.g., DeepHit)
            input = self._handle_all_missing_columns(input, "zero")
            surv = self.predict_surv(input, additional_inputs)
            lagged_surv = np.column_stack([np.ones((surv.shape[0], 1)), surv[:, :-1]])
            if additional_inputs is not None:
                input = (input,) + additional_inputs  # type: ignore
            cif = self.wrapped_model.predict_cif(input).swapaxes(0, 2)
            lagged_surv_expanded = lagged_surv[..., np.newaxis]
            lagged_surv_expanded = np.repeat(
                lagged_surv_expanded, cif.shape[-1], axis=-1
            )
            if np.any(lagged_surv_expanded == 0):
                raise ValueError(
                    "Zero values found in estimate of survival function, cannot derive hazards from CIF."
                )
            haz = np.diff(cif, prepend=0, axis=1) / lagged_surv_expanded
            cum_haz = np.cumsum(haz, axis=1)
            if cum_haz.shape[2] > len(np.unique(self.all_events)) - 1:
                raise RuntimeError(
                    f"CIF output has {cum_haz.shape[2]} causes of failure, but only {len(np.unique(self.all_events)) - 1} are present in the data."
                )
        elif hasattr(self.wrapped_model, "predict_cumulative_hazards"):
            # pycox without competing risks (e.g., DeepSurv)
            input = self._handle_all_missing_columns(input, "zero")
            if additional_inputs is not None:
                input = (input,) + additional_inputs  # type: ignore
            cum_haz = self.wrapped_model.predict_cumulative_hazards(input).values
            if cum_haz.shape[1] == len(input):
                cum_haz = cum_haz.T
            if len(cum_haz.shape) == 2:
                cum_haz = cum_haz[..., np.newaxis]
        elif hasattr(self.wrapped_model, "predict_cumulative_hazard_function"):
            # scikit-survival
            input = self._handle_all_missing_columns(input, "remove")
            cum_haz = self.wrapped_model.predict_cumulative_hazard_function(
                input, return_array=True
            )
            if cum_haz.shape[1] == len(input):
                cum_haz = cum_haz.T
            if len(cum_haz.shape) == 2:
                cum_haz = cum_haz[..., np.newaxis]
        else:
            raise ValueError(
                "Model has no method to predict cumulative hazards or CIF."
            )

        cum_haz = self._update_times(cum_haz, 0, ffill=True)
        return cum_haz

    @property
    def jumps(self) -> np.ndarray:
        if self.labtrans is not None:
            return self.labtrans.cuts
        else:
            return np.unique(self.all_times)


class PycoxWrapperCauseSpecific(PycoxWrapper):
    """
    A version of the PycoxWrapper that fits one model per cause of failure.
    """

    def __init__(self, wrapped_model, all_events: np.ndarray, **kwargs):
        super().__init__(wrapped_model=wrapped_model, all_events=all_events, **kwargs)
        self.wrapped_model = {}
        for i in np.unique(all_events):
            if i != 0:
                self.wrapped_model[int(i)] = deepcopy(wrapped_model)

    def __str__(self):
        return f"CauseSpecific{type(self.wrapped_model[1]).__name__}"

    def __repr__(self):
        return f"CauseSpecific{type(self.wrapped_model[1]).__name__}"

    def fit(
        self,
        input: np.ndarray,
        target: np.ndarray,
        additional_inputs: Optional[Tuple[np.ndarray]] = None,
        **kwargs,
    ):
        if self.labtrans is not None:
            event_indicator = target[1]
            target = self.labtrans.transform(*target)
            target = (target[0], target[1] * event_indicator)
        self.fit_times = target[0]
        for i, model in self.wrapped_model.items():
            if "sksurv" in type(model).__module__:
                # scikit-survival-based model
                target_i = Surv.from_arrays(target[1] == i, target[0])
                input = self._handle_all_missing_columns(input, "remove")
                if additional_inputs is not None and self.verbose >= 1:
                    warnings.warn(
                        "Additional inputs are not supported for scikit-survival models and will be ignored.",
                        RuntimeWarning,
                    )
                model.fit(input, target_i)
            else:
                # pycox-based model
                target_i = (target[0], (target[1].astype(int) == i).astype(int))
                input = self._handle_all_missing_columns(input, "zero")
                if additional_inputs is not None:
                    input = (input,) + additional_inputs  # type: ignore
                model.fit(input, target_i, **kwargs)
                # Cox-like models in pycox require the baseline hazard to be computed after fitting
                if hasattr(model, "compute_baseline_hazards"):
                    model.compute_baseline_hazards()
        self.fitted = True

    def predict_surv(
        self,
        input: np.ndarray,
        additional_inputs: Optional[Tuple[np.ndarray]],
        **kwargs,
    ) -> np.ndarray:
        cum_haz = self.predict_cumhaz(input, additional_inputs)
        return np.exp(-np.sum(cum_haz, axis=-1))

    def predict_cumhaz(
        self,
        input: np.ndarray,
        additional_inputs: Optional[Tuple[np.ndarray]],
        **kwargs,
    ) -> np.ndarray:
        cum_haz = np.zeros((input.shape[0], len(self.jumps), len(self.wrapped_model)))
        for i, model in self.wrapped_model.items():
            if hasattr(model, "predict_cumulative_hazards"):
                # pycox without competing risks (e.g., DeepSurv), for cause i
                input = self._handle_all_missing_columns(input, "zero")
                if additional_inputs is not None:
                    input = (input,) + additional_inputs  # type: ignore
                cum_haz_i = model.predict_cumulative_hazards(input).values
                if cum_haz_i.shape[1] == len(input):
                    cum_haz_i = cum_haz_i.T
                cum_haz_i = self._update_times(cum_haz_i, 0, ffill=True)
                cum_haz[..., i - 1] = cum_haz_i
            elif hasattr(model, "predict_cumulative_hazard_function"):
                # scikit-survival
                input = self._handle_all_missing_columns(input, "remove")
                cum_haz_i = model.predict_cumulative_hazard_function(
                    input, return_array=True
                )
                if cum_haz.shape[1] == len(input):
                    cum_haz_i = cum_haz.T
                cum_haz_i = self._update_times(cum_haz_i, 0, ffill=True)
                cum_haz[..., i - 1] = cum_haz_i

        return cum_haz
