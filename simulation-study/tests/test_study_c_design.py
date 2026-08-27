"""Study C gates: the score, the standard error, and a runtime worth quoting.

Study C compared point estimates and nothing else. Two implementations can agree
on `Psi` while solving different estimating equations -- FINDINGS 9 is exactly
that -- so the score `Pn D*` was added, along with the standard error, and a
benchmark that is actually matched.

Every test here pins something that has already gone wrong once, or that would
go wrong silently if it did:

    the score      extracted *post*-update on both sides; the pre-update
                   summary sits in scope one line away in the R bridge and
                   would look entirely plausible
    the SE         compared as a ratio; an absolute tolerance on a quantity
                   that scales as n^-1/2 is a gate that loosens with n while
                   appearing not to
    skips          a comparison that cannot be made is reported, not dropped;
                   the old code fell through `if diff.empty: continue`, so a
                   missing quantity read as a pass
    the benchmark  serialised and single-threaded, and *verified* to have been

Tests needing R are skipped when `Rscript` or `concrete` is unavailable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sim.dgp import get_config, sample
from sim.estimators import COLUMNS, run_all
from sim.nuisance import Spec, build as build_nuisance
from sim.report import _runtime_agg
from sim.study_c_report import QUANTITIES, agreement, score_panel

CORRECT = Spec(Q="correct", pi="correct", G="correct")


def _fit(n=150, taus=(0.5, 2.0), seed=5):
    sm = sample(n, get_config("base"), np.random.default_rng(seed))
    ie = build_nuisance(sm, CORRECT)
    return run_all(sm.df, ie, list(taus), target_events=[1, 2], return_eic=True)


# ---------------------------------------------------------------------------
# the score
# ---------------------------------------------------------------------------


def test_score_reaches_the_estimate_rows_and_the_score_frame():
    out, eic = _fit()
    t = out[(out["estimator"] == "tmle") & (out["estimand"] == "risk")]
    assert t["pn_eic"].notna().all()
    assert np.isfinite(t["se_eic"]).all()
    # the plug-in does no targeting, so it has no score of its own
    assert out[out["estimator"] == "gcomp"]["pn_eic"].isna().all()
    assert {"pn_eic", "se_eic", "eic_crit", "group", "n_times"} <= set(eic.columns)


def test_score_frame_keeps_the_event_free_row():
    """`event = -1` is part of the stopping criterion in *both* packages.

    It has no estimate row to attach to, which is why the score needs its own
    frame; dropping it would compare a weaker gate than either package applies.
    """
    _, eic = _fit()
    assert -1 in set(eic["event"].astype(int))
    assert set(eic["group"].astype(int)) == {0, 1}


def test_a_converged_fit_solved_its_own_score_equation():
    """The check that catches a pre-update extraction.

    Before targeting the score is large by construction; after it, it is below
    the criterion the loop stops on. Reading `SummEIC` one line too early in the
    R bridge produces numbers that look fine but fail this.
    """
    out, eic = _fit()
    if not bool(out["tmle_converged"].iloc[0]):
        pytest.skip("fit did not converge; the criterion does not apply")
    real = eic[eic["event"] > 0]
    assert (real["pn_eic"].abs() <= real["eic_crit"]).all()


def test_score_reduction_is_recorded_and_positive():
    out, _ = _fit()
    first = float(out["norm_pn_eic_first"].iloc[0])
    last = float(out["norm_pn_eic_last"].iloc[0])
    assert np.isfinite(first) and np.isfinite(last)
    assert last <= first, "the update should not increase ||Pn D*||"


# ---------------------------------------------------------------------------
# diagnostics survive to disk
# ---------------------------------------------------------------------------


def test_steps_and_grid_size_are_columns_not_attrs():
    """`res.attrs` is dropped by `concat` and by parquet.

    Study C recorded a runtime with no step count to normalise it by for exactly
    this reason, which made `report.runtimes()` inapplicable to it.
    """
    out, _ = _fit()
    for col in ("tmle_steps", "n_times", "tmle_converged"):
        assert col in COLUMNS and col in out.columns
    t = out[out["estimator"] == "tmle"]
    assert t["tmle_steps"].notna().all() and t["n_times"].notna().all()


def test_estimator_frame_survives_a_parquet_round_trip(tmp_path):
    out, _ = _fit()
    f = tmp_path / "est.parquet"
    out.to_parquet(f, index=False)
    assert len(pd.read_parquet(f)) == len(out)


# ---------------------------------------------------------------------------
# agreement over the three quantities
# ---------------------------------------------------------------------------


def _synthetic(se_scale=1.0):
    rows = []
    for rep in range(6):
        for est, off in (("tmle", 0.0), ("tmle (concrete)", 0.004),
                         ("gcomp", 0.02), ("gcomp (concrete)", 0.024)):
            rows.append({"n": 500, "rep": rep, "event": 1, "time": 2.0,
                         "group": np.nan, "estimator": est, "estimand": "rd",
                         "est": -0.15 + off,
                         # the plug-in has no standard error in either package
                         "se": (np.nan if est.startswith("gcomp")
                                else (0.03 + 0.001 * off) * se_scale)})
    return pd.DataFrame(rows)


def test_se_is_compared_as_a_ratio_not_a_difference():
    """Scaling both sides must leave the statistic unchanged.

    A standard error scales as n^-1/2, so an absolute tolerance silently
    loosens at small n and tightens at large n while looking like one gate.
    """
    a = agreement(_synthetic(1.0), event=1, quantity="se")
    b = agreement(_synthetic(10.0), event=1, quantity="se")
    key = ["implementation", "mean_abs_diff"]
    x = a[~a["skipped"]][key].reset_index(drop=True)
    y = b[~b["skipped"]][key].reset_index(drop=True)
    pd.testing.assert_frame_equal(x, y)
    assert (a["statistic"] == "log ratio").all()


def test_the_targeting_increment_is_only_used_for_the_point_estimate():
    """It cancels the FINDINGS 8 grid offset, which is a property of the level.

    For the SE and the score it is undefined -- `gcomp` has no SE in either
    package -- so those must fall through to a level comparison rather than
    being skipped, or the study loses its most valuable SE comparison.
    """
    est = agreement(_synthetic(), event=1, quantity="est")
    se = agreement(_synthetic(), event=1, quantity="se")
    row = est[est["implementation"] == "tmle (concrete)"].iloc[0]
    assert row["compared"] == "targeting increment"
    row2 = se[se["implementation"] == "tmle (concrete)"].iloc[0]
    assert row2["compared"] == "log ratio" and not row2["skipped"]
    assert row2["pairs"] > 0


def test_an_impossible_comparison_is_reported_not_dropped():
    """The old code skipped it silently, which read as a pass."""
    se = agreement(_synthetic(), event=1, quantity="se")
    gc = se[se["implementation"] == "gcomp (concrete)"]
    assert len(gc) == 1
    assert bool(gc["skipped"].iloc[0]) and gc["pairs"].iloc[0] == 0
    assert gc["skip_reason"].iloc[0]


def test_contrast_rows_are_not_lost_to_a_nan_group():
    """`pivot_table` silently drops index rows containing NaN.

    `group` is NaN on every contrast row, so adding it to the index without a
    sentinel would empty the table rather than raise.
    """
    d = _synthetic()
    assert d["group"].isna().all()
    a = agreement(d, event=1, quantity="est")
    assert a[~a["skipped"]]["pairs"].max() == 6


def test_unknown_quantity_raises():
    with pytest.raises(ValueError, match="unknown quantity"):
        agreement(_synthetic(), quantity="not_a_column")
    assert set(QUANTITIES) == {"est", "se", "pn_eic"}


def test_score_panel_measures_each_package_against_its_own_criterion():
    """A paired difference alone is uninformative: if both solve their score,
    it goes to zero and any tolerance passes for the wrong reason."""
    rows = []
    for src, ratio in (("pytmle", 0.4), ("concrete", 0.15)):
        for rep in range(5):
            rows.append({"n": 500, "rep": rep, "source": src, "event": 1,
                         "time": 2.0, "pn_eic": ratio * 0.01, "se_eic": 0.5,
                         "eic_crit": 0.01, "norm_pn_eic_first": 0.05,
                         "norm_pn_eic_last": 0.005})
    sp = score_panel(pd.DataFrame(rows))
    got = sp.set_index("source")["median_ratio"].round(3).to_dict()
    assert got == {"pytmle": 0.4, "concrete": 0.15}
    assert (sp["frac_solved"] == 1.0).all()
    assert (sp["median_score_reduction"].round(1) == 10.0).all()


# ---------------------------------------------------------------------------
# the benchmark
# ---------------------------------------------------------------------------


def test_runtime_aggregation_refuses_to_mix_contended_and_matched():
    """Two columns named `stage2_seconds` now exist with different validity.

    One is measured 8-way parallel and multi-threaded, one alone and pinned.
    Their mean is neither, so a mixed group must be an error, not an average.
    """
    d = pd.DataFrame({"implementation": ["tmle"] * 4, "n": [500] * 4,
                      "stage2_seconds": [1.5, 1.6, 4.0, 4.1],
                      "tmle_steps": [26] * 4, "n_times": [500] * 4,
                      "contended": [False, False, True, True]})
    with pytest.raises(ValueError, match="contended"):
        _runtime_agg(d, ["n", "implementation"])
    ok = _runtime_agg(d[~d["contended"]], ["n", "implementation"])
    assert int(ok["n_runs"].iloc[0]) == 2


def test_each_implementation_is_normalised_by_its_own_grid():
    """concrete builds a coarser grid than PyTMLE (209 against 300 at n = 300).

    Normalising both by PyTMLE's would hand concrete that difference as free
    speed, so the per-cell figure must use each one's own `n_times`.
    """
    d = pd.DataFrame({"implementation": ["tmle"] * 2 + ["tmle (concrete)"] * 2,
                      "n": [300] * 4, "stage2_seconds": [1.0, 1.0, 1.0, 1.0],
                      "tmle_steps": [10] * 4, "n_times": [300, 300, 209, 209],
                      "contended": [False] * 4})
    r = _runtime_agg(d, ["n", "implementation"]).set_index("implementation")
    assert r.loc["tmle", "median_n_times"] == 300
    assert r.loc["tmle (concrete)", "median_n_times"] == 209
    # same wall time, coarser grid -> a *higher* per-cell cost, not a lower one
    assert (r.loc["tmle (concrete)", "median_ns_per_step_cell"]
            > r.loc["tmle", "median_ns_per_step_cell"])


def test_thread_env_pins_every_pool_the_two_stacks_read():
    from sim.bench_stage2 import THREAD_VARS, single_thread_env

    env = single_thread_env({"PATH": "/usr/bin", "OMP_NUM_THREADS": "16"})
    assert all(env[v] == "1" for v in THREAD_VARS)
    assert "OPENBLAS_NUM_THREADS" in THREAD_VARS   # R's BLAS is the pthreads build
    assert env["PATH"] == "/usr/bin"


def test_benchmark_refuses_a_busy_machine():
    from sim.bench_stage2 import check_idle

    with pytest.raises(RuntimeError, match="load average"):
        check_idle(max_load=-1.0)
    assert check_idle(max_load=-1.0, allow_busy=True) >= 0.0


def test_a_multithreaded_run_fails_the_fairness_band():
    """The gate has to bite, not just exist.

    `wall/cpu` is the real check: one thread working alone spends its wall time
    computing (~1.0); several threads drive CPU above wall; a competing job
    drives wall above CPU. A live thread *count* cannot distinguish these --
    R showed 7 idle pool threads while doing one thread of work.
    """
    from sim.bench_stage2 import WALL_CPU_BAND

    lo, hi = WALL_CPU_BAND
    assert lo < 1.0 < hi
    for ratio, ok in ((1.0, True), (0.25, False), (4.0, False)):
        assert (lo <= ratio <= hi) is ok


def test_concrete_bridge_pins_data_table_threads():
    """data.table defaults to half the cores and no env var reaches it."""
    src = Path(__file__).resolve().parents[1] / "R" / "run_concrete_injected.R"
    text = src.read_text()
    assert "setDTthreads(1L)" in text
    # the post-update summary, not the one built before doTmleUpdate
    assert 'est[[z]][["SummEIC"]]' in text



def test_the_two_packages_summarise_the_influence_curve_identically(tmp_path):
    """Pins the claim the whole cross-package score comparison rests on.

    `summarize_ic` and concrete's `summarizeIC` are asserted elsewhere to use
    "byte-identical formulas"; this evaluates both on one shared input instead
    of taking that on trust.
    """
    from pytmle.get_influence_curve import summarize_ic

    from sim.concrete_bridge import RSCRIPT   # the env's R, not the system one

    if not Path(RSCRIPT).exists():
        pytest.skip("Rscript not available")
    rng = np.random.default_rng(0)
    ic = pd.DataFrame({"ID": np.tile(np.arange(1, 41), 2),
                       "Time": np.repeat([1.0, 2.0], 40),
                       "Event": 1,
                       "IC": rng.normal(size=80)})
    py = summarize_ic(ic.copy()).sort_values(["Time", "Event"])

    csv_path = tmp_path / "ic.csv"
    ic.to_csv(csv_path, index=False)
    script = (
        'suppressMessages(library(data.table));'
        f'ic <- fread("{csv_path}");'
        'r <- concrete:::summarizeIC(ic);'
        'setorder(r, Time, Event);'
        'cat(sprintf("%.10f", r$PnEIC), sep=",");cat("|");'
        'cat(sprintf("%.10f", r$seEIC), sep=",")'
    )
    proc = subprocess.run([RSCRIPT, "-e", script], text=True,
                          capture_output=True)
    if proc.returncode != 0:
        pytest.skip(f"concrete not loadable: {proc.stderr.strip()[-200:]}")
    pn_s, se_s = proc.stdout.split("|")
    r_pn = np.array([float(x) for x in pn_s.split(",")])
    r_se = np.array([float(x) for x in se_s.split(",")])
    assert np.allclose(py["PnEIC"].to_numpy(), r_pn, atol=1e-9)
    assert np.allclose(py["seEIC"].to_numpy(), r_se, atol=1e-9)
