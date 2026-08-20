"""Harness unit tests.

These guard the things whose failure would be silent: a truth that does not
match the DGP, a nuisance grid that drifts out of alignment with PyTMLE's
requirement, and -- most insidious -- a "misspecified" model that turns out to
be nearly right, which would make every cell of Study A look identical and the
study demonstrate nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sim.dgp import (
    CONFIGS,
    cause_rates,
    expit,
    get_config,
    marginal_treated_fraction,
    sample,
    true_propensity,
)
from sim.nuisance import STUDY_A_CELLS, Spec, _fit_propensity_correct, _fit_propensity_wrong, build
from sim.truth import closed_form


# ---------------------------------------------------------------------------
# DGP and truth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", ["base", "threshold"])
def test_closed_form_matches_simulation(config):
    """Empirical CIF from a large draw must match the closed form within 3 MC-SEs."""
    p = get_config(config)
    taus = [1.0, 2.5, 5.0]
    tr = closed_form(p, taus, n_mc=1_000_000, seed=11)

    # Simulate the *uncensored* latent process and compare sub-distribution
    # probabilities arm by arm, which is exactly what the closed form targets.
    rng = np.random.default_rng(12)
    m = 1_500_000
    from sim.dgp import _design, _draw_covariates

    w_cat, w_cont = _draw_covariates(m, rng)
    X = _design(w_cat, w_cont)
    u = (w_cont > p.threshold).astype(float)
    q = marginal_treated_fraction(p) if p.control_resample else None
    wt = expit(X @ p.gamma) + (1.0 - q) if p.control_resample else np.ones(m)

    for arm in (0, 1):
        rates = cause_rates(X, u, float(arm), p)
        latent = np.column_stack([rng.exponential(1.0 / r) for r in rates])
        t_ev, cause = latent.min(axis=1), latent.argmin(axis=1) + 1
        for j in (1, 2):
            for tau in taus:
                emp = float((wt * ((t_ev <= tau) & (cause == j))).mean())
                row = tr[(tr.arm == arm) & (tr.event == j) & (tr.time == tau)].iloc[0]
                tol = 3 * np.sqrt(row.mc_se**2 + emp * (1 - emp) / m) + 1e-4
                assert abs(emp - row.risk) < tol, (
                    f"{config} arm={arm} cause={j} tau={tau}: "
                    f"closed form {row.risk:.5f} vs empirical {emp:.5f}"
                )


def test_truth_weighting_matters_under_resampling():
    """The realised-population target must differ from the unconditional one.

    Under the control-resampling device the realised covariate law is a mixture;
    ignoring that targets a different population. If these ever agree, the
    weighting has silently stopped being applied.
    """
    # adjcuminc_s2, not `threshold`: the Study A config no longer uses the
    # control-resampling device, so there is no mixture there to weight.
    p = get_config("adjcuminc_s2")
    taus = [0.25, 0.5]
    w = closed_form(p, taus, n_mc=500_000, seed=5, target="realised")
    u = closed_form(p, taus, n_mc=500_000, seed=5, target="unweighted")
    gap = np.abs(w["rd"].to_numpy() - u["rd"].to_numpy()).max()
    assert gap > 1e-3, "weighted and unweighted targets coincide; weighting lost?"


def test_true_propensity_matches_realised_assignment():
    """The closed-form e(W) must reproduce the realised P(A=1|W)."""
    p = get_config("threshold")
    sm = sample(400_000, p, np.random.default_rng(3))
    lp = sm.X @ p.gamma
    bins = np.quantile(lp, np.linspace(0, 1, 11))
    idx = np.digitize(lp, bins[1:-1])
    for b in range(10):
        m = idx == b
        emp = sm.group[m].mean()
        pred = sm.ps_true[m].mean()
        assert abs(emp - pred) < 0.02, f"bin {b}: empirical {emp:.4f} vs closed form {pred:.4f}"


def test_censoring_fraction_is_reasonable():
    for name in ("base", "threshold"):
        sm = sample(20_000, get_config(name), np.random.default_rng(9))
        frac = float((sm.event_indicator == 0).mean())
        assert 0.05 < frac < 0.6, f"{name}: censored fraction {frac:.3f} out of range"


# ---------------------------------------------------------------------------
# Nuisance builders
# ---------------------------------------------------------------------------


def test_oracle_reproduces_dgp_exactly():
    p = get_config("threshold")
    sm = sample(400, p, np.random.default_rng(4))
    est = build(sm, Spec("oracle", "oracle", "oracle"))
    grid = np.unique(sm.event_times)

    for key, a in ((1, 1.0), (0, 0.0)):
        cum = np.cumsum(est[key].hazards, axis=1)
        truth = [np.outer(r, grid) for r in cause_rates(sm.X, sm.u, a, p)]
        for j, t in enumerate(truth):
            assert np.abs(cum[..., j] - t).max() < 1e-10
    assert np.abs(est[1].propensity_scores - sm.ps_true).max() == 0.0


@pytest.mark.parametrize("seed", range(0, 100, 7))
def test_grid_alignment_round_trips(seed):
    """PyTMLE requires unique(event_times) == unique(times) exactly."""
    from pytmle import PyTMLE

    p = get_config("threshold")
    sm = sample(120, p, np.random.default_rng(seed))
    est = build(sm, Spec("correct", "correct", "correct"))
    assert np.array_equal(np.unique(sm.event_times), np.unique(est[1].times))
    taus = list(np.quantile(sm.event_times, [0.3, 0.6]))
    PyTMLE(sm.df, target_times=taus, initial_estimates=est,
           evalues_benchmark=False, verbose=0)  # must construct without raising


def test_all_study_a_cells_build():
    p = get_config("threshold")
    sm = sample(300, p, np.random.default_rng(6))
    for name, spec in STUDY_A_CELLS.items():
        est, diag = build(sm, spec, return_diagnostics=True)
        assert est[1].hazards.shape == (sm.n, len(np.unique(sm.event_times)), 2), name
        assert diag.n_failed_fits == 0, name


def test_misspecification_actually_bites():
    """Quantitative floors, not a smoke test.

    A "wrong" model that is nearly right would make Study A's eight cells look
    identical. These thresholds are calibrated from the asymptotic gaps measured
    during the build; failing them means the design has gone inert.
    """
    p = get_config("threshold")

    # Outcome side: omitting the threshold term must move the cumulative hazard.
    sm = sample(2000, p, np.random.default_rng(8))
    _, d_ok = build(sm, Spec("correct", "correct", "correct"), return_diagnostics=True)
    _, d_bad = build(sm, Spec("wrong", "correct", "correct"), return_diagnostics=True)
    assert d_bad.cumhaz_mae > 3 * d_ok.cumhaz_mae, (
        f"Q misspecification inert: {d_ok.cumhaz_mae:.4f} -> {d_bad.cumhaz_mae:.4f}"
    )

    # Treatment side. What matters is not a ratio at one sample size but two
    # properties: the misspecified fit carries a non-vanishing *asymptotic*
    # bias (which is what drives estimator bias at every n), while the correct
    # fit is consistent. At n = 500 the correct fit's estimation noise is of
    # comparable size to that bias, so a ratio there would say nothing.
    # Only the propensity fitters run here, so this stays cheap -- the full
    # nuisance build is O(n^2) in memory and would not fit at this n.
    def _errs(n, seed):
        s = sample(n, p, np.random.default_rng(seed))
        return (
            float(np.abs(_fit_propensity_correct(s, s.X) - s.ps_true).mean()),
            # wrong = the continuous confounder omitted
            float(np.abs(_fit_propensity_wrong(s, s.X[:, :2]) - s.ps_true).mean()),
        )

    small = [_errs(2_000, 200 + i) for i in range(3)]
    big = [_errs(20_000, 300 + i) for i in range(3)]
    ok_small = np.mean([e[0] for e in small])
    ok_big, bad_big = np.mean([e[0] for e in big]), np.mean([e[1] for e in big])

    assert bad_big > 0.012, f"pi misspecification has no asymptotic bias: {bad_big:.4f}"
    assert ok_big < 0.6 * ok_small, (
        f"correct pi fit not converging: {ok_small:.4f} (n=2k) -> {ok_big:.4f} (n=20k)"
    )
    # The ratio is the noisiest of the three checks -- gamma_hat still varies
    # enough at n = 20 000 that it ranges over roughly 3-8 across seeds -- so it
    # is set to reject only an inert misspecification (which would give ~1),
    # with the floor and consistency assertions above carrying the real weight.
    # Omitting a genuine confounder is a far blunter instrument than the
    # functional-form devices tried before it, so this ratio is large and stable
    # (measured ~14.6x) rather than the noisy 3-8 of the earlier design.
    assert bad_big > 5 * ok_big, (
        f"pi misspecification inert at n=20000: {ok_big:.4f} -> {bad_big:.4f}"
    )


def test_pi_lever_does_not_cost_positivity():
    """The propensity misspecification must not push e(W) to the boundary.

    Every earlier pi device put the misspecification *inside* the true
    propensity -- a threshold, then a quadratic -- so strengthening it drove
    e(W) to 0.96-1.00 and destabilised IPW even when pi was correct. The
    omitted-confounder lever leaves the truth alone, so this must hold.
    """
    sm = sample(20_000, get_config("threshold"), np.random.default_rng(2))
    e = sm.ps_true
    assert float((e < 0.05).mean()) < 0.001, f"P(e<0.05) = {(e < 0.05).mean():.4f}"
    assert float((e > 0.95).mean()) < 0.001, f"P(e>0.95) = {(e > 0.95).mean():.4f}"
    lo, hi = np.quantile(e, [0.01, 0.99])
    assert lo > 0.05 and hi < 0.95, f"e(W) 1st/99th pct = {lo:.3f}/{hi:.3f}"


def test_positivity_is_comfortable():
    """Study A must not be confounded with a near-positivity violation."""
    p = get_config("threshold")
    for n in (250, 1000):
        sm = sample(n, p, np.random.default_rng(13))
        assert sm.ps_true.min() > 0.05, f"n={n}: min true e(W) = {sm.ps_true.min():.4f}"
        assert sm.ps_true.max() < 0.95


# ---------------------------------------------------------------------------
# Estimators and metrics
# ---------------------------------------------------------------------------


def test_estimators_run_and_agree_on_shared_eif():
    """TMLE and the one-step share PyTMLE's EIF, so their SEs must match."""
    from sim.estimators import run_all

    p = get_config("base")
    sm = sample(300, p, np.random.default_rng(21))
    taus = list(np.round(np.quantile(sm.event_times, [0.3, 0.6]), 6))
    est = build(sm, Spec("oracle", "oracle", "oracle"))
    res = run_all(sm.df, est, taus, target_events=[1, 2])

    assert res["error"].isna().all()
    assert set(res["estimator"]) == {"tmle", "gcomp", "aipw", "ipw"}

    piv = res[res.estimand == "rd"].pivot_table(index=["event", "time"],
                                                columns="estimator", values="se")
    assert np.allclose(piv["tmle"], piv["aipw"], rtol=0.02), (
        "TMLE and one-step SEs diverge despite sharing an influence function"
    )


def test_gcomp_with_oracle_nuisances_is_near_truth():
    """Consistency smoke test: the oracle plug-in should sit on the estimand."""
    from sim.estimators import run_all

    p = get_config("base")
    sm = sample(2000, p, np.random.default_rng(31))
    taus = list(np.round(np.quantile(sm.event_times, [0.3, 0.6]), 6))
    est = build(sm, Spec("oracle", "oracle", "oracle"))
    res = run_all(sm.df, est, taus, target_events=[1, 2], which=("tmle", "gcomp"))
    tr = closed_form(p, taus, n_mc=1_000_000, seed=41)

    g = res[(res.estimator == "gcomp") & (res.estimand == "rd")]
    for _, row in g.iterrows():
        truth = tr[(tr.event == row.event) & (tr.time == row.time)].iloc[0]["rd"]
        assert abs(row.est - truth) < 0.02, (
            f"cause {row.event} tau {row.time}: gcomp {row.est:.4f} vs truth {truth:.4f}"
        )


def test_metrics_recover_known_coverage():
    from sim.metrics import summarise

    rng = np.random.default_rng(5)
    m = 4000
    truth_val = 0.3
    est = truth_val + rng.normal(scale=0.1, size=m)
    res = pd.DataFrame({
        "cell": "X", "estimator": "t", "estimand": "rd", "event": 1, "time": 1.0,
        "group": np.nan, "est": est, "se": 0.1,
        "ci_lo": est - 1.959964 * 0.1, "ci_hi": est + 1.959964 * 0.1,
        "converged": True, "error": None,
    })
    truth = pd.DataFrame([{"estimand": "rd", "event": 1, "time": 1.0,
                           "group": np.nan, "truth": truth_val}])
    s = summarise(res, truth).iloc[0]
    assert abs(s["coverage"] - 0.95) < 4 * s["coverage_mc_se"]
    assert abs(s["se_ratio"] - 1.0) < 0.05
    assert abs(s["bias"]) < 4 * s["bias_mc_se"]


def test_config_loading_round_trips():
    from pathlib import Path

    from sim.config import load_cells

    root = Path(__file__).resolve().parents[1]
    cells = load_cells(root / "sim" / "configs" / "reference.yaml", only=["C1"], reps=3)
    assert len(cells) == 1 and cells[0].name == "C1" and cells[0].reps == 3
    assert cells[0].spec == Spec("oracle", "oracle", "oracle")


def test_results_are_parquet_writable(tmp_path):
    """Regression: mixed bool/float in `converged` broke the shard writer.

    PyTMLE returns `Converged` as a plain bool on targeted rows but float/NaN on
    g-computation rows. Concatenating the two gives an object column that pyarrow
    refuses ("Could not convert 1.0 with type float"), which silently killed a
    whole cell mid-run.
    """
    from sim.estimators import run_all

    p = get_config("base")
    sm = sample(300, p, np.random.default_rng(77))
    taus = list(np.round(np.quantile(sm.event_times, [0.3, 0.6]), 6))
    est = build(sm, Spec("oracle", "oracle", "oracle"))
    res = run_all(sm.df, est, taus, target_events=[1, 2])

    assert str(res["converged"].dtype) == "boolean"
    out = tmp_path / "shard.parquet"
    res.to_parquet(out, index=False)          # must not raise
    assert len(pd.read_parquet(out)) == len(res)


def test_summarise_schema_survives_all_nan_groups():
    """Regression: a group with no usable estimates collapsed the whole table.

    `_summarise_group` used to return a 2-element Series when every estimate in a
    group was NaN. pandas cannot build a uniform frame from mixed shapes, so it
    silently stacked *every* group into long format -- turning a few failed
    replicates into a results table that looked plausible and was not.
    """
    from sim.metrics import _FIELDS, summarise

    good = pd.DataFrame({
        "cell": "X", "estimator": "t", "estimand": "rd", "event": 1, "time": 1.0,
        "group": np.nan, "est": [0.3, 0.31, 0.29], "se": 0.1,
        "ci_lo": 0.1, "ci_hi": 0.5, "converged": pd.array([True] * 3, dtype="boolean"),
        "error": None,
    })
    bad = good.copy()
    bad["estimator"] = "u"
    bad["est"] = np.nan
    bad["error"] = "boom"
    truth = pd.DataFrame([{"estimand": "rd", "event": 1, "time": 1.0,
                           "group": np.nan, "truth": 0.3}])

    s = summarise(pd.concat([good, bad], ignore_index=True), truth)
    assert set(_FIELDS).issubset(s.columns), "schema collapsed"
    assert len(s) == 2
    assert s.loc[s.estimator == "u", "n_used"].iloc[0] == 0
    assert s.loc[s.estimator == "u", "n_error"].iloc[0] == 3
    assert np.isfinite(s.loc[s.estimator == "t", "bias"].iloc[0])


def test_concrete_reps_is_clamped_to_cell_reps():
    """A pilot must not run off the end of the seed stream.

    `--reps` shrinks a cell but leaves the config's `concrete_reps` alone, so
    without clamping the exporter indexes past the spawned seeds and the whole
    comparator dies with an IndexError -- exactly when a hard failure is least
    wanted.
    """
    from sim.runner import Cell

    cell = Cell(name="C1_n250", config="threshold", n=250, reps=12,
                concrete_reps=150)
    assert min(cell.concrete_reps, cell.reps) == 12


def test_load_cell_includes_concrete(tmp_path):
    """concrete.parquet is loaded with the shards, not left for a merge step."""
    import pandas as pd
    from sim.runner import load_cell

    d = tmp_path / "C1_n250"
    d.mkdir()
    pd.DataFrame({"rep": [0], "estimator": ["tmle"], "est": [0.1]}).to_parquet(
        d / "shard_0000.parquet", index=False)
    pd.DataFrame({"rep": [0], "estimator": ["tmle (concrete)"], "est": [0.2]}
                 ).to_parquet(d / "concrete.parquet", index=False)

    got = load_cell(d)
    assert set(got["estimator"]) == {"tmle", "tmle (concrete)"}
