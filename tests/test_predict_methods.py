import pandas as pd
from pytmle.estimates import UpdatedEstimates
from pytmle.predict_ate import ate_ratio, ate_diff

def test_predict_mean_risks(mock_updated_estimates):   
    g_comp = False
    for k in mock_updated_estimates.keys():
        assert isinstance(mock_updated_estimates[k], UpdatedEstimates)
        result = mock_updated_estimates[0].predict_mean_risks(g_comp=g_comp)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"Time", "Event", "Pt Est", "SE"}
        assert result["SE"].isna().all() == g_comp # should be all NA for g_comp=True
        assert len(result) == len(mock_updated_estimates[k].target_times) * len(mock_updated_estimates[k].target_events)
        g_comp = not g_comp # invert g_comp flag for next iteration to check both behaviors

def test_ate_ratio(mock_updated_estimates):
    g_comp = False
    for k in mock_updated_estimates.keys():
        assert isinstance(mock_updated_estimates[k], UpdatedEstimates)
        result = ate_ratio(mock_updated_estimates, gcomp=g_comp, key_1=1, key_0=0)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"Time", "Event", "Pt Est", "SE", "CI_lower", "CI_upper", "p_value"}
        assert len(result) == len(mock_updated_estimates[k].target_times) * len(mock_updated_estimates[k].target_events)
        g_comp = not g_comp # invert g_comp flag for next iteration to check both behaviors

def test_ate_diff(mock_updated_estimates):
    g_comp = False
    for k in mock_updated_estimates.keys():
        assert isinstance(mock_updated_estimates[k], UpdatedEstimates)
        result = ate_diff(mock_updated_estimates, gcomp=g_comp, key_1=1, key_0=0)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"Time", "Event", "Pt Est", "SE", "CI_lower", "CI_upper", "p_value"}
        assert len(result) == len(mock_updated_estimates[k].target_times) * len(mock_updated_estimates[k].target_events)
        g_comp = not g_comp # invert g_comp flag for next iteration to check both behaviors