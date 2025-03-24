import torch
import torchtuples as tt
from pycox.models import DeepHit
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
import numpy as np
from typing import List, Tuple

def get_default_models(event_times, event_indicator, input_size, labtrans=None) -> Tuple[List, List]:
    """
    Get the default models for the initial estimates of the hazard functions.
    """
    deephit, label_discretizer = default_deephit(labtrans, event_indicator, event_times, input_size)

    risk_models = [deephit]
    label_discretizer = [label_discretizer]

    return risk_models, label_discretizer

def default_deephit(labtrans, event_indicator, event_times, input_size) -> Tuple[DeepHit, LabTransDiscreteTime]:
    """
    A simplified version of the DeepHit model from the pycox library as default if no model is provided."""
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
        labtrans = LabTransDiscreteTime(40, scheme="quantiles")
        labtrans.fit(event_times, event_indicator)
    net = CauseSpecificNet(
        input_size,
        num_nodes_shared=[64, 64],
        num_nodes_indiv=[32],
        num_risks=len(np.unique(event_indicator)) -1,
        out_features=len(labtrans.cuts),
        batch_norm=True,
        dropout=0.1,
    )
    return DeepHit(net, tt.optim.Adam, duration_index=labtrans.cuts), labtrans