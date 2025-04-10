import pandas as pd
import matplotlib.pyplot as plt
import os

from pytmle.estimates import UpdatedEstimates
from pytmle.predict_ate import get_counterfactual_risks, ate_ratio, ate_diff
from pytmle.plotting import plot_risks, plot_ate, plot_nuisance_weights

def test_predict_mean_risks(mock_updated_estimates):   
    g_comp = False
    for k in mock_updated_estimates.keys():
        assert isinstance(mock_updated_estimates[k], UpdatedEstimates)
        result = mock_updated_estimates[k].predict_mean_risks(g_comp=g_comp)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"Time", "Event", "Pt Est", "SE"}
        assert result["SE"].isna().all() == g_comp # should be all NA for g_comp=True
        assert len(result) == len(mock_updated_estimates[k].target_times) * len(mock_updated_estimates[k].target_events)
        g_comp = not g_comp # invert g_comp flag for next iteration to check both behaviors

        # test plotting (of nuisance weights)
        for _, _, time in plot_nuisance_weights(target_times=mock_updated_estimates[k].target_times,
                                                times=mock_updated_estimates[k].times,
                                                min_nuisance=mock_updated_estimates[k].min_nuisance,
                                                nuisance_weights=mock_updated_estimates[k].nuisance_weight,
                                                g_star_obs=mock_updated_estimates[k].g_star_obs,
                                                plot_size=(14, 7),
                                                color_1="#c00000", 
                                                color_0="#699aaf"):
            plt.savefig(f'/tmp/test_nuisance_weights_plot_t{time}.png')  # Save the plot to a file
            assert os.path.exists(f'/tmp/test_nuisance_weights_plot_t{time}.png')  # Check if the plot file exists
            os.remove(f'/tmp/test_nuisance_weights_plot_t{time}.png')  # Clean up the file after the test
            plt.close()  # Close the plot


def test_get_counterfactual_risks(mock_updated_estimates):
    results = dict.fromkeys(["tmle", "g_comp"])
    for estimator in ["tmle", "g_comp"]:
        g_comp = estimator == "g_comp"
        result = get_counterfactual_risks(mock_updated_estimates, g_comp=g_comp, key_1=1, key_0=0)
        results[estimator] = result
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"Time", "Event", "Pt Est", "SE", "CI_lower", "CI_upper", "Group"}
        assert result["SE"].isna().all() == g_comp # should be all NA for g_comp=True
        assert result["CI_lower"].isna().all() == g_comp # should be all NA for g_comp=True
        assert result["CI_upper"].isna().all() == g_comp # should be all NA for g_comp=True
        assert len(result) == len(mock_updated_estimates) * len(mock_updated_estimates[0].target_times) * len(mock_updated_estimates[1].target_events)

    # test plotting
    plot_risks(results["tmle"], results["g_comp"], color_1="#c00000", color_0="#699aaf")
    assert len(plt.gcf().axes) == len(mock_updated_estimates[0].target_events)  # Check if the plot has one axis for each target event
    plt.savefig('/tmp/test_risk_plot.png')  # Save the plot to a file
    assert os.path.exists('/tmp/test_risk_plot.png')  # Check if the plot file exists
    os.remove('/tmp/test_risk_plot.png')  # Clean up the file after the test
    plt.close()  # Close the plot


def test_ate_ratio(mock_updated_estimates):
    results = dict.fromkeys(["tmle", "g_comp"])
    for estimator in ["tmle", "g_comp"]:
        g_comp = estimator == "g_comp"
        result = ate_ratio(mock_updated_estimates, g_comp=g_comp, key_1=1, key_0=0)
        results[estimator] = result
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {
            "Time",
            "Event",
            "Pt Est",
            "SE",
            "CI_lower",
            "CI_upper",
            "p_value",
            "E_value",
            "E_value CI",
            "E_value CI limit",
            "E_value CI (bootstrap)",
            "E_value CI limit (bootstrap)",
        }
        assert result["SE"].isna().all() == g_comp # should be all NA for g_comp=True
        assert result["CI_lower"].isna().all() == g_comp # should be all NA for g_comp=True
        assert result["CI_upper"].isna().all() == g_comp # should be all NA for g_comp=True
        assert result["p_value"].isna().all() == g_comp # should be all NA for g_comp=True
        assert result["E_value CI"].isna().all() == g_comp # should be all NA for g_comp=True
        assert result["E_value CI limit"].isna().all() == g_comp # should be all NA for g_comp=True
        assert len(result) == len(mock_updated_estimates[0].target_times) * len(mock_updated_estimates[1].target_events)

    # test plotting
    plot_ate(results["tmle"], results["g_comp"], type="rr")
    assert len(plt.gcf().axes) == len(mock_updated_estimates[0].target_events)  # Check if the plot has one axis for each target event
    plt.savefig('/tmp/test_ate_ratio_plot.png')  # Save the plot to a file
    assert os.path.exists('/tmp/test_ate_ratio_plot.png')  # Check if the plot file exists
    os.remove('/tmp/test_ate_ratio_plot.png')  # Clean up the file after the test
    plt.close()  # Close the plot


def test_ate_diff(mock_updated_estimates):
    results = dict.fromkeys(["tmle", "g_comp"])
    for estimator in ["tmle", "g_comp"]:
        g_comp = estimator == "g_comp"
        result = ate_diff(mock_updated_estimates, g_comp=g_comp, key_1=1, key_0=0)
        results[estimator] = result
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {
            "Time",
            "Event",
            "Pt Est",
            "SE",
            "CI_lower",
            "CI_upper",
            "p_value",
            "E_value",
            "E_value CI",
            "E_value CI limit",
            "E_value CI (bootstrap)",
            "E_value CI limit (bootstrap)",
        }
        assert result["SE"].isna().all() == g_comp # should be all NA for g_comp=True
        assert result["CI_lower"].isna().all() == g_comp # should be all NA for g_comp=True
        assert result["CI_upper"].isna().all() == g_comp # should be all NA for g_comp=True
        assert result["p_value"].isna().all() == g_comp # should be all NA for g_comp=True
        assert result["E_value CI"].isna().all() == g_comp # should be all NA for g_comp=True
        assert result["E_value CI limit"].isna().all() == g_comp # should be all NA for g_comp=True
        assert len(result) == len(mock_updated_estimates[0].target_times) * len(mock_updated_estimates[1].target_events)

    # test plotting
    plot_ate(results["tmle"], results["g_comp"], type="rd")
    assert len(plt.gcf().axes) == len(mock_updated_estimates[0].target_events)  # Check if the plot has one axis for each target event
    plt.savefig('/tmp/test_ate_diff_plot.png')  # Save the plot to a file
    assert os.path.exists('/tmp/test_ate_diff_plot.png')  # Check if the plot file exists
    os.remove('/tmp/test_ate_diff_plot.png')  # Clean up the file after the test
    plt.close()  # Close the plot
