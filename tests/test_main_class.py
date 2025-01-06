import pytest
import pandas as pd

from pytmle import PyTMLE

# TODO: Parametrize different sets of given pre-computed initial estimates
@pytest.mark.parametrize("target_times, target_events", 
                         [([1.0, 2.0, 3.0], [1]), 
                          ([1.0, 2.0, 3.0], [1, 2]), 
                          ([1.111, 2.222, 3.333], [1]), 
                          ([1.111, 2.222, 3.333], [1, 2]), 
                          (None, [1]), 
                          (None, [1, 2])])
def test_fit(mock_main_class_inputs, target_times, target_events):
    df = pd.DataFrame({"event_time": mock_main_class_inputs.pop("event_times"),
                       "event_indicator": mock_main_class_inputs.pop("event_indicator"),
                       "group": mock_main_class_inputs.pop("group"),})

    initial_estimates = mock_main_class_inputs["initial_estimates"]
    est = PyTMLE(data=df,
                 target_times=target_times,
                 target_events=target_events,
                 initial_estimates=initial_estimates)
    
    est.fit()
    assert est._fitted

#     est.plot("/home/jguski/COMMUTE/tmle/plot.png", 
#             type="risks", 
#             g_comp=True,
#             color_1="#c00000", 
#             color_0="#699aaf")
#     est.plot_nuisance_weights("/home/jguski/COMMUTE/tmle/nuisance_weights_plots",
#                               color_1="#c00000", 
#                               color_0="#699aaf")