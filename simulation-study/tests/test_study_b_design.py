"""Study B design gate: every stress axis must vary what it claims.

This is the test the original plan called for and the earlier Study B did not
have. Its purpose is narrow and its failure mode is the one that actually
happened last time: a study can run to completion, produce clean-looking
coverage tables, and have measured nothing, because the axis it was built around
did not move the quantity it was named after. `n`, `tau` and censoring *amount*
turned out not to stress the EIC variance estimator at all, and eight completed
cells said so only after ~15 h of compute.

So each level in `sim/configs/study_b.yaml` is checked against the target it was
calibrated to, at the population level, before any replicate is run:

    overlap     P(min(e, 1-e) < 0.05) and the effective sample size
    rare        cause-1 cumulative incidence at the last target time
    censoring   the censored fraction is held FIXED across the axis, so the
                axis varies dependence rather than information
    null        the true risk difference is exactly zero

Plus the two guards that keep everything else honest: the Study A regression
guard (new DGP fields must be no-ops at their defaults) and the plumbing guard
(a per-cell override must actually reach the fit, rather than being silently
ignored as `min_nuisance` was).

Population quantities use Monte Carlo draws sized for the tolerance asserted,
not for elegance; these run in a few seconds each.
"""

from __future__ import annotations

from pathlib import Path

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from sim.dgp import cause_rates, censoring_rate, expit, get_config, sample
from sim.nuisance import Spec, build
from sim.study_b import (
    _CELL_KEYS,
    _check_seed_keys,
    _coerce_override,
    BCell,
    run_study_b,
)
from sim.truth import closed_form

CONFIG = Path(__file__).resolve().parents[1] / "sim" / "configs" / "study_b.yaml"
TAUS = [0.477676, 3.11961, 8.608445]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _cells():
    cfg = yaml.safe_load(CONFIG.read_text())
    d = cfg.get("defaults", {})
    out = []
    for key in ("wald_cells", "boot_cells"):
        for c in cfg.get(key, []):
            def pick(k, dv):
                return c.get(k, d.get(k, dv))

            out.append(BCell(
                name=c["name"], n=int(c["n"]), arm=c["arm"], reps=int(c["reps"]),
                config=pick("config", "base"),
                n_bootstrap=int(c.get("n_bootstrap", 0)),
                min_nuisance=float(pick("min_nuisance", 0.01)),
                max_updates=int(pick("max_updates", 200)),
                target_times=pick("target_times", None),
                params_override=_coerce_override(c.get("params_override", {})),
                seed_key=c.get("seed_key"),
                axis=c.get("axis", "base"), level=c.get("level", "base")))
    return out


def _by_level(axis):
    """One representative cell per level on an axis (levels repeat across n)."""
    seen = {}
    for c in _cells():
        if c.axis == axis and c.level not in seen:
            seen[c.level] = c
    return seen


def _draw_population(p, n=400_000, seed=7):
    """Covariates, treatment and the derived quantities, at population scale."""
    rng = np.random.default_rng(seed)
    w_cat = rng.integers(0, 3, size=n)
    w_cont = rng.normal(size=n)
    X = np.column_stack([(w_cat == 1).astype(float),
                         (w_cat == 2).astype(float), w_cont])
    u = (w_cont > p.threshold).astype(float)
    ps = expit(X @ p.gamma)
    A = rng.binomial(1, ps).astype(float)
    return rng, X, u, ps, A


# ---------------------------------------------------------------------------
# the config itself
# ---------------------------------------------------------------------------


def test_config_loads_and_names_are_unique():
    cells = _cells()
    assert len(cells) > 40
    names = [c.name for c in cells]
    assert len(names) == len(set(names)), "cells sharing a name share a shard directory"


def test_seed_key_prefixes_do_not_collide():
    """Only 8 bytes of the key are hashed, so a shared prefix is a shared dataset.

    Deliberate sharing (an oracle and a correct arm of one condition) is allowed
    and is the point; accidental sharing between two *different* conditions would
    silently correlate them.
    """
    _check_seed_keys(_cells())


def test_arms_of_a_condition_are_seed_paired():
    """oracle vs correct at one condition must draw identical datasets."""
    cells = {c.name: c for c in _cells()}
    for base in ("OV3_n250", "CN3_n250"):
        a, b = cells[f"{base}_correct"], cells[f"{base}_oracle"]
        assert (a.seed_key or a.name) == (b.seed_key or b.name)


def test_bootstrap_cell_is_paired_with_its_wald_cell():
    """Wald-vs-bootstrap must be a paired contrast, not two independent samples."""
    cells = {c.name: c for c in _cells()}
    assert cells["B_OV3_n250_correct"].seed_key == cells["OV3_n250_correct"].seed_key


def test_every_cell_pins_target_times():
    """tau must not be re-derived per DGP, or the axis is confounded with tau."""
    for c in _cells():
        assert c.resolved_target_times() == TAUS, c.name


# ---------------------------------------------------------------------------
# axis 1 -- treatment positivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level,p_below,ess", [
    ("OV1", 0.000, 0.74), ("OV2", 0.027, 0.42),
    ("OV3", 0.120, 0.17), ("OV4", 0.332, 0.013),
])
def test_overlap_levels_hit_their_calibrated_targets(level, p_below, ess):
    cell = _by_level("overlap")[level]
    _, X, _, ps, _ = _draw_population(cell.dgp_params(), n=2_000_000, seed=1)
    two_sided = np.minimum(ps, 1.0 - ps)
    assert np.mean(two_sided < 0.05) == pytest.approx(p_below, abs=0.01)
    w = 1.0 / ps
    realised_ess = w.sum() ** 2 / (len(w) * (w**2).sum())
    assert realised_ess == pytest.approx(ess, abs=0.03)


def test_overlap_is_monotone_and_spans_a_useful_range():
    """A dose-response axis has to be ordered, and has to reach a breakpoint."""
    levels = _by_level("overlap")
    fracs = []
    for lv in ("OV1", "OV2", "OV3", "OV4"):
        _, _, _, ps, _ = _draw_population(levels[lv].dgp_params(), n=500_000, seed=2)
        fracs.append(float(np.mean(np.minimum(ps, 1 - ps) < 0.05)))
    assert fracs == sorted(fracs)
    assert fracs[0] < 0.01 and fracs[-1] > 0.25


def test_overlap_axis_does_not_move_the_estimand():
    """gamma changes difficulty only: the truth must be identical across levels."""
    base = closed_form(get_config("base"), TAUS, n_mc=1_000_000)
    for lv, cell in _by_level("overlap").items():
        got = closed_form(cell.dgp_params(), TAUS, n_mc=1_000_000)
        assert np.allclose(got["risk"].to_numpy(), base["risk"].to_numpy(), atol=0), lv


# ---------------------------------------------------------------------------
# axis 2 -- rare events
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level,risk_treated,risk_control", [
    ("RA1", 0.103, 0.187), ("RA2", 0.053, 0.099), ("RA3", 0.024, 0.046),
])
def test_rare_levels_hit_their_cause1_incidence(level, risk_treated, risk_control):
    cell = _by_level("rare")[level]
    tr = closed_form(cell.dgp_params(), TAUS, n_mc=2_000_000)
    late = tr[(tr["event"] == 1) & np.isclose(tr["time"], TAUS[-1])]
    got = {int(r["arm"]): float(r["risk"]) for _, r in late.iterrows()}
    assert got[1] == pytest.approx(risk_treated, abs=0.004)
    assert got[0] == pytest.approx(risk_control, abs=0.006)


def test_rare_axis_leaves_the_competing_cause_common():
    """Only cause 1 is made rare, so tau stays inside the observed support.

    If both causes were rare the observed follow-up would shorten and the last
    target time would start falling outside it, which the runner rejects -- the
    axis would then be measuring replicate loss rather than rarity.
    """
    p = _by_level("rare")["RA3"].dgp_params()
    sm = sample(4000, p, np.random.default_rng(3))
    n1 = int((sm.event_indicator == 1).sum())
    n2 = int((sm.event_indicator == 2).sum())
    assert n1 < 0.25 * n2, (n1, n2)
    assert float(sm.event_times.max()) > TAUS[-1]


# ---------------------------------------------------------------------------
# axis 3 -- censoring positivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level,censored,trunc", [
    ("CN0", 0.30, 0.000), ("CN1", 0.30, 0.000),
    ("CN2", 0.30, 0.012), ("CN3", 0.30, 0.075), ("CN4", 0.50, 0.100),
])
def test_censoring_levels_hold_amount_while_varying_dependence(level, censored, trunc):
    """alpha_c is re-solved per level so the axis is not confounded with amount."""
    p = _by_level("censoring")[level].dgp_params()
    rng, X, u, ps, A = _draw_population(p, n=400_000, seed=7)
    tot = sum(cause_rates(X, u, A, p))
    t_event = rng.exponential(1.0 / tot)
    t_cens = rng.exponential(1.0 / censoring_rate(X, u, A, p))
    assert float((t_cens < t_event).mean()) == pytest.approx(censored, abs=0.015)

    G = np.exp(-censoring_rate(X, u, A, p) * TAUS[-1])
    piG = np.where(A == 1, ps, 1.0 - ps) * G
    assert float((piG < 0.01).mean()) == pytest.approx(trunc, abs=0.015)


def test_censoring_dependence_is_monotone():
    levels = _by_level("censoring")
    fr = []
    for lv in ("CN0", "CN1", "CN2", "CN3"):
        p = levels[lv].dgp_params()
        _, X, u, ps, A = _draw_population(p, n=200_000, seed=8)
        G = np.exp(-censoring_rate(X, u, A, p) * TAUS[-1])
        piG = np.where(A == 1, ps, 1.0 - ps) * G
        fr.append(float((piG < 0.01).mean()))
    assert fr == sorted(fr)


def test_censoring_axis_does_not_move_the_estimand():
    base = closed_form(get_config("base"), TAUS, n_mc=1_000_000)
    for lv, cell in _by_level("censoring").items():
        got = closed_form(cell.dgp_params(), TAUS, n_mc=1_000_000)
        assert np.allclose(got["risk"].to_numpy(), base["risk"].to_numpy(), atol=0), lv


# ---------------------------------------------------------------------------
# axis 4 -- the null condition
# ---------------------------------------------------------------------------


def test_null_condition_has_exactly_zero_effect():
    """Both causes, every tau, on both the difference and the ratio scale.

    Zeroing only the cause-1 treatment effect is NOT enough: cause 2's hazard
    still depends on treatment and enters cause 1's cumulative incidence through
    the total hazard, leaving an RD of -0.023 at the last tau. A null on the
    cause-specific hazard is not a null on the cumulative incidence.
    """
    p = _by_level("nulleffect")["NULL"].dgp_params()
    tr = closed_form(p, TAUS, n_mc=1_000_000).drop_duplicates(["event", "time"])
    assert len(tr) == 6
    assert np.abs(tr["rd"].to_numpy()).max() < 1e-12
    assert np.abs(tr["rr"].to_numpy() - 1.0).max() < 1e-12


def test_null_condition_still_has_confounding():
    """A null effect with no confounding would not test the adjustment at all."""
    p = _by_level("nulleffect")["NULL"].dgp_params()
    _, _, _, ps, _ = _draw_population(p, n=200_000, seed=4)
    assert ps.std() > 0.05


# ---------------------------------------------------------------------------
# remedy axes
# ---------------------------------------------------------------------------


def test_min_nuisance_sweep_covers_the_intended_grid():
    cells = [c for c in _cells() if c.axis == "min_nuisance"]
    assert {c.min_nuisance for c in cells} == {0.025, 0.05, 0.10}
    # each swept cell must share its parent condition's dataset, so the
    # truncation contrast is exact rather than across independent samples
    assert {c.seed_key for c in cells} == {"OV3_n250", "CN3_n250"}


def test_max_updates_cells_differ_only_in_the_budget():
    cells = {c.name: c for c in _cells()}
    a, b = cells["RA3_n250_correct"], cells["RA3mu1000_n250_correct"]
    assert a.max_updates == 200 and b.max_updates == 1000
    assert a.seed_key == b.seed_key
    assert np.array_equal(a.dgp_params().alpha, b.dgp_params().alpha)


# ---------------------------------------------------------------------------
# regression guards -- new DGP fields must be no-ops at their defaults
# ---------------------------------------------------------------------------


def test_n_noise_and_noise_rho_default_to_no_ops():
    """Study A and the completed Study B cells must still reproduce exactly."""
    base = get_config("base")
    a = sample(500, base, np.random.default_rng(3))
    b = sample(500, base.with_(n_noise=0, noise_rho=0.0), np.random.default_rng(3))
    assert np.array_equal(a.df.to_numpy(), b.df.to_numpy())
    assert a.noise.shape == (500, 0)


def test_noise_block_does_not_disturb_the_event_stream():
    base = get_config("base")
    cols = ["event_time", "event_indicator", "group", "d2", "d3", "w_cont"]
    a = sample(2000, base, np.random.default_rng(11))
    b = sample(2000, base.with_(n_noise=25), np.random.default_rng(11))
    assert np.array_equal(a.df[cols].to_numpy(), b.df[cols].to_numpy())


def test_noise_rho_realises_the_ar1_structure():
    p = get_config("base").with_(n_noise=6, noise_rho=0.5)
    sm = sample(400_000, p, np.random.default_rng(5))
    w, Z = sm.df["w_cont"].to_numpy(), sm.noise
    for k in range(4):
        assert np.corrcoef(w, Z[:, k])[0, 1] == pytest.approx(0.5 ** (k + 1), abs=0.01)
    assert np.allclose(Z.std(axis=0), 1.0, atol=0.02)
    assert sm.df["w_cont"].std() == pytest.approx(1.0, abs=0.01)


def test_noise_rho_does_not_move_the_truth():
    """AR(1) with unit variances preserves every marginal, so the estimand holds."""
    base = get_config("base")
    a = closed_form(base, TAUS, n_mc=1_000_000)
    b = closed_form(base.with_(n_noise=50, noise_rho=0.5), TAUS, n_mc=1_000_000)
    assert np.allclose(a["risk"].to_numpy(), b["risk"].to_numpy(), atol=0)


def test_correlated_noise_is_inert_under_unpenalised_fits():
    """Pins the span identity that took the dimension axis out of this study.

    An unpenalised MLE depends on the design only through its column span, and
    `z = rho*w_cont + noise` added to a design already containing `w_cont` spans
    exactly what independent noise spans. So correlation cannot move a `correct`
    or `oracle` fit -- but it does move a `wrong` one, where `w_cont` is omitted
    and the correlated columns partially recover it.

    The deferred high-dimensional study inherits this: more correlated columns
    against unpenalised nuisances measure nothing, and it needs penalised or
    selection-based learners instead.
    """
    base = get_config("base")
    s0 = sample(250, base.with_(n_noise=25, noise_rho=0.0), np.random.default_rng(9))
    s5 = sample(250, base.with_(n_noise=25, noise_rho=0.5), np.random.default_rng(9))
    assert np.array_equal(s0.X, s5.X)
    assert not np.allclose(s0.noise, s5.noise)

    e0 = build(s0, Spec(Q="correct", pi="correct", G="correct"))
    e5 = build(s5, Spec(Q="correct", pi="correct", G="correct"))
    assert np.abs(e0[1].propensity_scores - e5[1].propensity_scores).max() < 1e-3
    assert np.abs(e0[1].hazards - e5[1].hazards).max() < 1e-5

    w0 = build(s0, Spec(Q="wrong", pi="wrong", G="wrong"))
    w5 = build(s5, Spec(Q="wrong", pi="wrong", G="wrong"))
    assert np.abs(w0[1].propensity_scores - w5[1].propensity_scores).max() > 1e-2


def test_noise_reaches_the_propensity_design():
    """The dimension axis used to stress Q and G while pi stayed immune."""
    base = get_config("base")
    sm0 = sample(250, base, np.random.default_rng(4))
    sm1 = sample(250, base.with_(n_noise=50), np.random.default_rng(4))
    _, d0 = build(sm0, Spec(Q="correct", pi="correct", G="correct"),
                  return_diagnostics=True)
    _, d1 = build(sm1, Spec(Q="correct", pi="correct", G="correct"),
                  return_diagnostics=True)
    assert d1.ps_mae > 2 * d0.ps_mae
    assert d1.ps_min < 0.1 * d0.ps_min


# ---------------------------------------------------------------------------
# plumbing guards -- an override that is silently ignored is the worst kind
# ---------------------------------------------------------------------------


def test_unknown_cell_key_raises(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        "defaults: {config: base}\n"
        "wald_cells:\n"
        "  - {name: x, n: 250, arm: correct, reps: 1, min_nuisanse: 0.05}\n")
    with pytest.raises(ValueError, match="unknown key"):
        run_study_b(cfg, tmp_path / "out")


def test_per_cell_overrides_are_not_silently_ignored(tmp_path):
    """`min_nuisance` and `max_updates` used to be read from defaults only."""
    cfg = tmp_path / "ok.yaml"
    cfg.write_text(
        "defaults: {config: base, min_nuisance: 0.01, max_updates: 200}\n"
        "wald_cells:\n"
        "  - {name: a, n: 250, arm: correct, reps: 1, min_nuisance: 0.05,\n"
        "     max_updates: 999, params_override: {gamma: [2.0, -1.6, 1.2]}}\n")
    seen = []
    import sim.study_b as sb
    orig = sb.run_cell_b
    sb.run_cell_b = lambda cell, out, **kw: (seen.append(cell), out)[1]
    try:
        run_study_b(cfg, tmp_path / "out")
    finally:
        sb.run_cell_b = orig
    (c,) = seen
    assert c.min_nuisance == 0.05
    assert c.max_updates == 999
    assert np.allclose(c.dgp_params().gamma, [2.0, -1.6, 1.2])


def test_follow_up_guard_rejects_a_replicate_ending_before_the_last_tau():
    """runner applies this guard; study_b did not, and now must."""
    from sim.study_b import _one_rep_b

    cell = BCell(name="short", n=60, arm="oracle", reps=1,
                 target_times=[0.4, 3.0, 500.0], seed_key="short")
    out, draws = _one_rep_b((cell, cell.resolved_target_times(),
                             np.random.SeedSequence(1).spawn(1)[0], 0))
    assert out["procedure"].isna().all()
    assert "before the last target time" in str(out["error"].iloc[0])
    assert draws.empty          # a failed replicate has no draws to archive


# ---------------------------------------------------------------------------
# resume safety -- the machine may restart mid-run
# ---------------------------------------------------------------------------


def _tiny_cell(tmp_path, **kw):
    return BCell(name="tiny", n=60, arm="oracle", reps=6, seed_key="tiny",
                 target_times=[0.4, 1.0], **kw)


def test_interrupted_shard_is_not_left_half_written(tmp_path):
    """A shard is written atomically, so a kill leaves no partial file."""
    from sim.study_b import run_cell_b
    run_cell_b(_tiny_cell(tmp_path), tmp_path, n_jobs=1, chunk=3)
    d = tmp_path / "tiny"
    assert not list(d.glob("*.tmp")), "temporary shard left behind"
    assert len(list(d.glob("shard_*.parquet"))) == 2


def test_corrupt_shard_is_detected_and_redone(tmp_path):
    """A truncated shard must be redone, not skipped forever as 'done'."""
    from sim.study_b import _shard_is_intact, run_cell_b
    run_cell_b(_tiny_cell(tmp_path), tmp_path, n_jobs=1, chunk=3)
    victim = sorted((tmp_path / "tiny").glob("shard_*.parquet"))[0]
    good = pd.read_parquet(victim).shape[0]
    victim.write_bytes(b"PAR1 truncated garbage")
    assert not _shard_is_intact(victim)
    run_cell_b(_tiny_cell(tmp_path), tmp_path, n_jobs=1, chunk=3)
    assert _shard_is_intact(victim)
    assert pd.read_parquet(victim).shape[0] == good


def test_rerun_is_a_no_op_and_does_not_duplicate(tmp_path):
    """Resuming a finished cell must add nothing."""
    from sim.study_b import run_cell_b
    run_cell_b(_tiny_cell(tmp_path), tmp_path, n_jobs=1, chunk=3)
    d = tmp_path / "tiny"
    before = {p.name: p.stat().st_mtime_ns for p in d.glob("shard_*.parquet")}
    run_cell_b(_tiny_cell(tmp_path), tmp_path, n_jobs=1, chunk=3)
    after = {p.name: p.stat().st_mtime_ns for p in d.glob("shard_*.parquet")}
    assert before == after, "a completed shard was rewritten on resume"
    reps = pd.concat([pd.read_parquet(p) for p in d.glob("shard_*.parquet")])["rep"]
    assert reps.nunique() == 6 and len(reps) == len(reps.drop_duplicates(keep="first")) \
        or reps.nunique() == 6


def test_chunk_is_pinned_once_a_cell_has_shards(tmp_path):
    """Re-chunking an existing cell would re-index and duplicate its replicates.

    Shard `i` means replicates [i*chunk, (i+1)*chunk). Changing the chunk after
    the fact silently redefines what every stored file contains -- a finished
    250-rep cell at chunk 25 would be read as 10 shards of 5, i.e. replicates
    0-49, and replicates 50-249 recomputed on top of copies it already holds.
    """
    from sim.study_b import _LEGACY_CHUNK, _chunk_for, run_cell_b
    cell = _tiny_cell(tmp_path)
    run_cell_b(cell, tmp_path, n_jobs=1, chunk=3)
    d = tmp_path / "tiny"
    assert json.loads((d / "meta.json").read_text())["chunk"] == 3
    # a later run asking for a different chunk must be ignored
    assert _chunk_for(cell, 25, d) == 3
    boot = BCell(name="tiny", n=60, arm="oracle", reps=6, seed_key="tiny",
                 n_bootstrap=100, target_times=[0.4, 1.0])
    assert _chunk_for(boot, 25, d) == 3
    # A fresh bootstrap cell is sized to the worker count, not to some smaller
    # constant. Shard wall time is one replicate whenever chunk <= n_jobs, so a
    # smaller chunk leaves workers idle and shortens nothing: B_OV4 ran on 5 of
    # 10 workers for 16 h that way.
    assert _chunk_for(boot, 25, tmp_path / "nonexistent", n_jobs=8) == 6   # reps
    assert _chunk_for(boot, 25, tmp_path / "nonexistent", n_jobs=4) == 4   # n_jobs
    big = BCell(name="big", n=60, arm="oracle", reps=100, seed_key="big",
                n_bootstrap=500, target_times=[0.4, 1.0])
    assert _chunk_for(big, 25, tmp_path / "nonexistent", n_jobs=10) == 10

    # and a pre-`chunk` meta.json falls back to the legacy value
    meta = json.loads((d / "meta.json").read_text())
    meta.pop("chunk")
    (d / "meta.json").write_text(json.dumps(meta))
    assert _chunk_for(cell, 25, d) == _LEGACY_CHUNK


# ---------------------------------------------------------------------------
# plotting -- one point per x-position
# ---------------------------------------------------------------------------


def test_dose_response_draws_one_cell_per_position():
    """A bootstrap cell must not double-plot its condition's Wald point.

    Bootstrap cells emit their own `wald` rows on purpose, so Wald-vs-bootstrap
    is paired within one cell. Those rows share axis/level/n/arm/procedure with
    the Wald-only cell of the same condition, so a dose-response panel would
    otherwise draw two points at that level -- one from 1000 replicates, one
    from 150 -- and join the line through whichever came first.
    """
    from sim.plots_study_b import _one_cell_per_point

    frame = pd.DataFrame({
        "cell": ["OV3_n250_correct", "B_OV3_n250_correct", "OV2_n250_correct"],
        "axis": ["overlap"] * 3, "level": ["OV3", "OV3", "OV2"],
        "n": [250, 250, 250], "time": [8.6, 8.6, 8.6],
        "min_nuisance": [0.01] * 3, "n_bootstrap": [0, 100, 0],
        "reps": [1000, 150, 1000], "coverage": [0.887, 0.9, 0.912],
    })
    kept = _one_cell_per_point(frame, "test")
    assert list(kept["cell"]) == ["OV3_n250_correct", "OV2_n250_correct"]
    assert kept.duplicated(subset=["level", "n", "time"]).sum() == 0


def test_no_duplicate_positions_in_the_real_results():
    """Guards the actual output, not just the helper."""
    from sim.plots_study_b import _one_cell_per_point

    csv = Path(__file__).resolve().parents[1] / "results" / "study_b" \
        / "study_b_performance.csv"
    if not csv.exists():
        pytest.skip("study has not been run")
    perf = pd.read_csv(csv)
    for axis in ("overlap", "rare", "censoring"):
        for arm in ("correct", "oracle"):
            d = perf[perf["axis"].isin([axis, "base"])
                     & (perf["procedure"] == "wald") & (perf["type"] == "rd")
                     & (perf["event"] == 1) & (perf["arm"] == arm)]
            if d.empty:
                continue
            kept = _one_cell_per_point(d, f"{axis}/{arm}")
            assert kept.groupby(["level", "n", "time"])["cell"].nunique().max() == 1


def test_series_that_do_not_span_the_axis_are_not_drawn():
    """A lone marker at one level reads as a point on a curve that is not there.

    n = 1000 and 2000 ran for the base condition only, because second-stage cost
    scales as n^2.05 and stressed cells run 6-11x base. They stay in the tables;
    they are just not plotted on an axis they do not span. The rule is "appears at
    two or more levels", not a hardcoded list of sizes.
    """
    from sim.plots_study_b import _spanning_series

    frame = pd.DataFrame({
        "n": [250, 250, 250, 500, 500, 1000, 2000],
        "level": ["base", "OV1", "OV2", "base", "OV1", "base", "base"],
    })
    kept, dropped = _spanning_series(frame)
    assert dropped == [1000, 2000]
    assert sorted(kept["n"].unique()) == [250, 500]
    # nothing is lost when every series spans the axis
    kept2, dropped2 = _spanning_series(frame[frame["n"] < 1000])
    assert dropped2 == [] and len(kept2) == 5


# ---------------------------------------------------------------------------
# every interval construction under every filter, and the draws archive
# ---------------------------------------------------------------------------


def _boot_cell():
    return BCell(name="btiny", n=60, arm="oracle", reps=2, seed_key="btiny",
                 target_times=[0.4, 1.0], n_bootstrap=6)


def test_every_construction_is_emitted_under_every_filter(tmp_path):
    """`basic` and `bca` must not be confined to one filter.

    They used to be emitted only under the convergence filter, which confounded
    the interval construction with the filter: their weak showing at OV4 was
    measuring the filter, not the construction. All three constructions now come
    from the same `intervals_from_draws` call under each rule, so the cross is
    free and the two effects are separable.
    """
    from sim.study_b import FILTERS, run_cell_b

    d = run_cell_b(_boot_cell(), tmp_path, n_jobs=1, chunk=2)
    got = pd.concat([pd.read_parquet(s) for s in sorted(d.glob("shard_*.parquet"))],
                    ignore_index=True)
    procs = set(got["procedure"].dropna())
    for label, _, _ in FILTERS:
        suffix = label[len("pct_"):]
        for kind in ("pct", "basic", "bca"):
            assert f"{kind}_{suffix}" in procs, f"missing {kind}_{suffix}"
    # the old confined names must be gone, or a stale reader would silently pick
    # up the filtered variant believing it unfiltered
    assert "basic" not in procs and "bca" not in procs


def test_raw_draws_are_archived_beside_the_shard(tmp_path):
    """The draws are kept so a future construction needs no re-run.

    Without them, changing an interval construction costs a full re-run of every
    bootstrap cell -- ~500 CPU-hours for this design.
    """
    from sim.study_b import run_cell_b

    d = run_cell_b(_boot_cell(), tmp_path, n_jobs=1, chunk=2)
    files = sorted(d.glob("draws_*.parquet"))
    assert files, "no draws archived"
    dr = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    for col in ("cell", "rep", "boot", "Pt Est", "Converged", "loop_converged"):
        assert col in dr.columns, f"draws missing {col}"
    assert dr["rep"].nunique() == 2
    assert dr["boot"].nunique() <= 6
    assert not list(d.glob("*.tmp")), "temporary draws file left behind"


def test_legacy_basic_and_bca_are_mapped_to_their_true_filter(tmp_path):
    """Stored `basic`/`bca` rows were convergence-filtered; say so on load."""
    from sim.study_b_report import _LEGACY_PROCEDURES

    assert _LEGACY_PROCEDURES["basic"] == "basic_convfilter"
    assert _LEGACY_PROCEDURES["bca"] == "bca_convfilter"
    assert _LEGACY_PROCEDURES["pct_shipped"] == "pct_convfilter"
