from pytmle.pycox_wrapper import PycoxWrapper, PycoxWrapperCauseSpecific
from .conftest import mock_main_class_inputs

import pytest
import numpy as np

import torchtuples as tt
from pycox.models import CoxPH
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis


def deepsurv(**kwargs):
    in_features = 4 # x1, x2, x3, group
    num_nodes = [32, 32]
    out_features = 1
    batch_norm = True
    dropout = 0.1
    output_bias = False

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                dropout, output_bias=output_bias)
    return CoxPH(net, tt.optim.Adam)


def coxph(**kwargs):
    return CoxPHSurvivalAnalysis()


def rsf(**kwargs):
    return RandomSurvivalForest(n_estimators=10, random_state=42)


def gb(**kwargs):
    return GradientBoostingSurvivalAnalysis(n_estimators=10, random_state=42)


@pytest.mark.parametrize("get_model", ["deepsurv", "coxph", "rsf", "gb"])
def test_fit(mock_main_class_inputs, get_model):
    model = eval(get_model)()
    df = mock_main_class_inputs["data"]
    X = df[["group", "x1", "x2", "x3"]].astype(np.float32)
    y = df[["event_time", "event_indicator"]].astype(np.float32)

    if not hasattr(model, "predict_cif") and len(np.unique(y["event_indicator"])) > 2:
        wrapper = PycoxWrapperCauseSpecific(
            model,
            labtrans=None,
            all_times=y["event_time"].values,
            all_events=y["event_indicator"].values,
            input_size=4,
        )
    else:
        wrapper = PycoxWrapper(
            model,
            labtrans=None,
            all_times=y["event_time"].values,
            all_events=y["event_indicator"].values,
            input_size=4,
        )
    wrapper.fit(X.values, (y["event_time"].values, y["event_indicator"].values))
    assert wrapper.fitted is True
    assert wrapper.fit_times is not None


@pytest.mark.parametrize("get_model", ["deepsurv", "coxph", "rsf", "gb"])
def test_predict(mock_main_class_inputs, get_model):
    model = eval(get_model)()
    df = mock_main_class_inputs["data"]

    # only use a subset to check that the output times are correct
    df = df[:100]

    X = df[["group", "x1", "x2", "x3"]].astype(np.float32)
    y = df[["event_time", "event_indicator"]].astype(np.float32)
    if not hasattr(model, "predict_cif") and len(np.unique(y["event_indicator"])) > 2:
        wrapper = PycoxWrapperCauseSpecific(
            model,
            labtrans=None,
            all_times=y["event_time"].values,
            all_events=y["event_indicator"].values,
            input_size=4,
        )
    else:
        wrapper = PycoxWrapper(
            model,
            labtrans=None,
            all_times=y["event_time"].values,
            all_events=y["event_indicator"].values,
            input_size=4,
        )
    wrapper.fit(X.values, (y["event_time"].values, y["event_indicator"].values))

    # predict survival function
    surv = wrapper.predict_surv(X[:25].values, additional_inputs=None)
    assert surv.shape[0] == 25
    assert surv.shape[1] == len(wrapper.jumps)

    # predict hazard function
    haz = wrapper.predict_cumhaz(X[:25].values, additional_inputs=None)
    assert haz.shape[0] == 25
    assert haz.shape[1] == len(wrapper.jumps)
