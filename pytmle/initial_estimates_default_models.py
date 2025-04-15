import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from typing import List, Tuple
import warnings


def get_default_models(
    event_times,
    event_indicator,
    input_size,
    labtrans=None,
    verbose=2,
) -> Tuple[List, List, List]:
    """
    Get the default models for the initial estimates of the hazard functions: Vanilla DeepHit, CoxPH, and RandomSurvivalForest.
    """
    risk_models = []
    censoring_models = []
    label_transformers = []
    # DeepHit model
    try:
        deephit, label_discretizer = vanilla_deephit(
            labtrans, event_indicator, event_times, input_size
        )
        deephit_censoring, _ = vanilla_deephit(
            label_discretizer,
            (event_indicator == 0).astype(int),
            event_times,
            input_size,
        )
        risk_models.append(deephit)
        censoring_models.append(deephit_censoring)
        label_transformers.append(label_discretizer)
    except ImportError as e:
        if verbose >= 1:
            warnings.warn(
                f"Default DeepHit model not available: {e}. Will only cross-fit Cox PH and random survival forest.",
                UserWarning,
            )

    # CoxPH model
    risk_models.append(CoxPHSurvivalAnalysis())
    censoring_models.append(CoxPHSurvivalAnalysis())
    label_transformers.append(None)

    # Random survival forest
    risk_models.append(RandomSurvivalForest(n_jobs=-1))
    censoring_models.append(RandomSurvivalForest(n_jobs=-1))
    label_transformers.append(None)

    return risk_models, censoring_models, label_transformers


def vanilla_deephit(labtrans, event_indicator, event_times, input_size):
    """
    A simplified version of the DeepHit model from the pycox library as default if no model is provided.
    """
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

    if labtrans is None:
        labtrans = LabTransDiscreteTime(15, scheme="quantiles")
        labtrans.fit(event_times, event_indicator)

    net = CauseSpecificNet(
        input_size,
        num_nodes_shared=[64, 64],
        num_nodes_indiv=[32],
        num_risks=len(np.unique(event_indicator)) - 1,
        out_features=len(labtrans.cuts),
        batch_norm=True,
        dropout=0.1,
    )
    return DeepHit(net, tt.optim.Adam, duration_index=labtrans.cuts), labtrans
