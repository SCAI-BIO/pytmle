from pytmle.pycox_wrapper import PycoxWrapper
from .conftest import mock_main_class_inputs

import pytest
import numpy as np

import torchtuples as tt
from pycox.models import CoxPH
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis

def deephit():
    # DeepHit is the default model
    return None

def deepsurv():
    in_features = 4 # x1, x2, x3, group
    num_nodes = [32, 32]
    out_features = 1
    batch_norm = True
    dropout = 0.1
    output_bias = False

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                dropout, output_bias=output_bias)
    return CoxPH(net, tt.optim.Adam)

def coxph():
    return CoxPHSurvivalAnalysis()

def rsf():
    return RandomSurvivalForest(n_estimators=10, random_state=42)

def gb():
    return GradientBoostingSurvivalAnalysis(n_estimators=10, random_state=42)

@pytest.mark.parametrize("get_model", ["deephit", 
                                       "deepsurv", 
                                       "coxph", 
                                       "rsf",
                                       "gb"])
def test_fit(mock_main_class_inputs, get_model):
    model = eval(get_model)()
    df = mock_main_class_inputs["data"]
    X = df[["group", "x1", "x2", "x3"]].astype(np.float32)
    y = df[["event_time", "event_indicator"]].astype(np.float32)

    if get_model != "default":  
        # No support for competing risks for any model except DeepHit
        y["event_indicator"] = (y["event_indicator"] == 1).astype(int)

    wrapper = PycoxWrapper(model, 
                           labtrans=None, 
                           all_times=y["event_time"].values, 
                           all_events=y["event_indicator"].values, 
                           input_size=4)
    wrapper.fit(X.values, (y["event_time"].values, y["event_indicator"].values))
    assert wrapper.fitted is True
    assert wrapper.fit_times is not None

@pytest.mark.parametrize("get_model", ["deephit", 
                                       "deepsurv", 
                                       "coxph", 
                                       "rsf",
                                       "gb"])
def test_predict(mock_main_class_inputs, get_model):
    model = eval(get_model)()
    df = mock_main_class_inputs["data"]

    if get_model != "default":
        # No support for competing risks for any model except DeepHit
        df["event_indicator"] = (df["event_indicator"] == 1).astype(int)

    # only use a subset to check that the output times are correct
    all_times = df["event_time"].values
    all_events = df["event_indicator"].values
    df = df[:100]

    X = df[["group", "x1", "x2", "x3"]].astype(np.float32)
    y = df[["event_time", "event_indicator"]].astype(np.float32)
    wrapper = PycoxWrapper(model, 
                           labtrans=None, 
                           all_times=all_times, 
                           all_events=all_events, 
                           input_size=4)
    wrapper.fit(X.values, (y["event_time"].values, y["event_indicator"].values))
    
    # predict survival function
    surv = wrapper.predict_surv(X[:25].values)
    assert surv.shape[0] == 25
    assert surv.shape[1] == len(wrapper.jumps)

    # predict hazard function
    haz = wrapper.predict_haz(X[:25].values)
    assert haz.shape[0] == 25
    assert haz.shape[1] == len(wrapper.jumps)
