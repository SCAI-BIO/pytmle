import pandas as pd
import numpy as np

from sksurv.util import Surv

class PycoxWrapper:
    """
    A wrapper class to unify the interface of different survival analysis libraries (pycox, scikit-survival)
    """
    def __init__(self, 
                 wrapped_model, 
                 labtrans, 
                 all_times: np.ndarray, 
                 all_events: np.ndarray, 
                 input_size: int = 0):
        self.labtrans = labtrans
        self.all_times = all_times
        self.all_events = all_events
        self.input_size = input_size
        if wrapped_model is not None:
            if not hasattr(wrapped_model, "predict_cif") and len(np.unique(all_events)) > 2:
                raise ValueError(f"It seems like {type(wrapped_model).__name__} does not support competing risks because it does not have a predict_cif method.") 
            self.wrapped_model = wrapped_model
        else:
            self.wrapped_model = self._deephit()
        if self.labtrans is not None:
            self.all_times, _ = self.labtrans.transform(self.all_times, self.all_events)
        self.fitted = False

    def __str__(self):
        # Return the name of the wrapped model
        return type(self.wrapped_model).__name__
    
    def __repr__(self):
        # Return the name of the wrapped model
        return type(self.wrapped_model).__name__

    def _deephit(self):
        """
        A simplified version of the DeepHit model from the pycox library as default if no model is provided."""
        import torch
        import torchtuples as tt
        from pycox.models import DeepHit
        from pycox.preprocessing.label_transforms import LabTransDiscreteTime
        class CauseSpecificNet(torch.nn.Module):
            """Network structure similar to the DeepHit paper, but without the residual
            connections (for simplicity).
            """

            def __init__(
                self,
                in_features,
                num_nodes_shared,
                num_nodes_indiv,
                num_risks,
                out_features,
                batch_norm=True,
                dropout=None,
            ):
                super().__init__()
                self.shared_net = tt.practical.MLPVanilla(
                    in_features,
                    num_nodes_shared[:-1],
                    num_nodes_shared[-1],
                    batch_norm,
                    dropout,
                )
                self.risk_nets = torch.nn.ModuleList()
                for _ in range(num_risks):
                    net = tt.practical.MLPVanilla(
                        num_nodes_shared[-1],
                        num_nodes_indiv,
                        out_features,
                        batch_norm,
                        dropout,
                    )
                    self.risk_nets.append(net)

            def forward(self, input):
                out = self.shared_net(input)
                out = [net(out) for net in self.risk_nets]
                out = torch.stack(out, dim=1)
                return out
        
        if self.labtrans is None:
            self.labtrans = LabTransDiscreteTime(40, scheme="quantiles")
            self.labtrans.fit(self.all_times, self.all_events)
        net = CauseSpecificNet(
            self.input_size,
            num_nodes_shared=[64, 64],
            num_nodes_indiv=[32],
            num_risks=len(np.unique(self.all_events)) -1,
            out_features=len(self.jumps),
            batch_norm=True,
            dropout=0.1,
        )
        return DeepHit(net, tt.optim.Adam, duration_index=self.jumps)


    def _update_times(
        self, predictions: np.ndarray, value: int, ffill: bool
    ) -> np.ndarray:
        """ Update the predictions to include all given times with jumps in the dataset"""
        if len(predictions.shape) == 2:
            pred_updated = np.full((predictions.shape[0], len(self.jumps)), np.nan)
        else:
            pred_updated = np.full((predictions.shape[0], len(self.jumps), predictions.shape[-1]), np.nan)

        if self.labtrans is None:
            fit_times_indices = np.searchsorted(np.unique(self.all_times), np.unique(self.fit_times))
        else:
            fit_times_indices = np.arange(len(self.jumps))
        pred_updated[:, fit_times_indices] = predictions

        mask = np.isnan(pred_updated)
        if ffill:
            row,col = pred_updated.shape
            for i in range(row):
                for j in range(col):
                    if mask[i][j]:
                        if j == 0:
                            pred_updated[i][j] = value
                        else:
                            pred_updated[i][j] =pred_updated[i][j-1]
        else:
            pred_updated[mask] = value 
        return pred_updated
        

    def fit(self, input, target, **kwargs):
        if self.labtrans is not None:
            target = self.labtrans.transform(*target)
        self.fit_times = target[0]
        if "sksurv" in type(self.wrapped_model).__module__:
            # scikit-survival-based model
            target = Surv.from_arrays(target[1], target[0])
            self.wrapped_model.fit(input, target)
        else:
            # pycox-based model
            target = (target[0], target[1].astype(int))
            self.wrapped_model.fit(input, target, **kwargs)
            # Cox-like models in pycox require the baseline hazard to be computed after fitting
            if hasattr(self.wrapped_model, "compute_baseline_hazards"):
                self.wrapped_model.compute_baseline_hazards()
        self.fitted = True

    def predict_surv (self, input, **kwargs) -> np.ndarray:
        """ Predict survival function for a given input"
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted")
        if hasattr(self.wrapped_model, 'predict_surv'):
            # pycox
            surv = self.wrapped_model.predict_surv(input, **kwargs)
        elif hasattr(self.wrapped_model, "predict_survival_function"):
            #scikit-survival
            surv = self.wrapped_model.predict_survival_function(input, return_array=True)
        else:
            raise ValueError("Model does not have a predict_surv method")
        if surv.shape[1] == len(input):
            surv = surv.T
        surv = self._update_times(surv, 1, ffill=True)
        return surv
    
    def predict_haz(self, input, **kwargs) -> np.ndarray:
        """ Predict hazard function for a given input"
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted")

        if hasattr(self.wrapped_model, 'predict_cif'):
            # pycox with competing risks (e.g., DeepHit)
            surv = self.predict_surv(input)
            cif = self.wrapped_model.predict_cif(input).swapaxes(0, 2)
            surv_expanded = np.expand_dims(surv, axis=-1)
            surv_expanded = np.repeat(surv_expanded, cif.shape[-1], axis=-1)
            haz = np.diff(cif, prepend=0, axis=1) / surv_expanded
        elif hasattr(self.wrapped_model, 'predict_cumulative_hazards'):
            # pycox without competing risks (e.g., DeepSurv)
            cum_haz = self.wrapped_model.predict_cumulative_hazards(input).values
            if cum_haz.shape[1] == len(input):
                cum_haz = cum_haz.T
            haz = np.diff(cum_haz, prepend=0, axis=1)
            if len(haz.shape) == 2:
                haz = np.expand_dims(haz, -1)
        elif hasattr(self.wrapped_model, 'predict_cumulative_hazard_function'):
            # scikit-survival
            cum_haz = self.wrapped_model.predict_cumulative_hazard_function(input, return_array=True)
            if cum_haz.shape[1] == len(input):
                cum_haz = cum_haz.T
            haz = np.diff(cum_haz, prepend=0, axis=1)
            if len(haz.shape) == 2:
                haz = np.expand_dims(haz, -1)
        else:
            raise ValueError("Model has no method to predict cumulative hazards or CIF.")

        haz = self._update_times(haz, 0, ffill=False)
        return haz

    @property
    def jumps(self) -> np.ndarray:
        if self.labtrans is not None:
            return self.labtrans.cuts
        else:
            return np.unique(self.all_times)