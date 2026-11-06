"""Design gate for the Study A-OV3 sub-study.

Study A measured double robustness at the OV1 overlap rung; this sub-study
repeats a focused subset of its specification cells under Study B's OV3 rung.
The two things that can go silently wrong are the ones pinned here:

* the new ``threshold_ov3`` config must differ from ``threshold`` **only** in
  ``gamma``, so the estimand and its truth are unchanged and the overlap level
  is the only new axis;
* the cell names must keep the leading ``C<k>`` / trailing ``_n<size>`` shape,
  because the Study A tables and figures reconstruct the specification and
  sample size from the name rather than from stored columns.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sim.config import load_cells
from sim.dgp import _design, _draw_covariates, expit, get_config
from sim.nuisance import STUDY_A_CELLS
from sim.plots import _cells, _ns
from sim.plots import _prep as plot_prep
from sim.tables import _CELL_SPEC
from sim.tables import _prep as table_prep
from sim.truth import closed_form

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "sim" / "configs" / "study_a_ov3.yaml"
TAUS = [0.513946, 1.149562, 2.66982]
KEEP = ("C1", "C2", "C3", "C5", "C7")
NAMES = [f"{c}_OV3_n{n}" for n in (250, 500) for c in KEEP]


# ---------------------------------------------------------------------------
# the design file
# ---------------------------------------------------------------------------


def test_config_loads_the_subset_at_both_sizes():
    cells = load_cells(CONFIG)
    assert [c.name for c in cells] == NAMES
    for c in cells:
        token = c.name.split("_")[0]
        assert c.config == "threshold_ov3"
        assert c.n in (250, 500)
        assert c.reps == 500
        assert c.concrete_reps == 0
        assert c.params_override == {}
        assert tuple(c.estimators) == ("tmle", "gcomp", "aipw", "ipw")
        assert list(c.target_times) == TAUS
        assert c.spec == STUDY_A_CELLS[token]


def test_c3_and_c5_are_present_at_both_sizes():
    names = {c.name for c in load_cells(CONFIG)}
    for n in (250, 500):
        assert f"C3_OV3_n{n}" in names
        assert f"C5_OV3_n{n}" in names


# ---------------------------------------------------------------------------
# the DGP composition
# ---------------------------------------------------------------------------


def test_named_config_differs_from_threshold_only_in_gamma():
    p, base = get_config("threshold_ov3"), get_config("threshold")
    assert np.allclose(p.gamma, [2.0, -1.6, 1.2])
    for f in fields(p):
        if f.name == "gamma":
            continue
        got, want = getattr(p, f.name), getattr(base, f.name)
        if isinstance(got, np.ndarray):
            assert np.array_equal(got, want), f.name
        else:
            assert got == want, f.name


def test_truth_is_identical_to_study_a_at_the_pinned_taus():
    """gamma enters treatment assignment only, so the estimand cannot move.

    The unweighted branch of ``closed_form`` does not read gamma at all; using
    a small Monte Carlo draw keeps the check cheap while still exercising the
    full truth path.
    """
    a = closed_form(get_config("threshold"), TAUS, n_mc=200_000, seed=7)
    b = closed_form(get_config("threshold_ov3"), TAUS, n_mc=200_000, seed=7)
    assert np.array_equal(a["risk"].to_numpy(), b["risk"].to_numpy())
    assert np.array_equal(a["rd"].to_numpy(), b["rd"].to_numpy())
    assert np.array_equal(a["rr"].to_numpy(), b["rr"].to_numpy())
    assert np.allclose(sorted(b["time"].unique()), TAUS)


def test_ov3_overlap_matches_the_study_b_calibration():
    """The composition must sit on OV3, not merely somewhere stressful.

    Study B calibrated OV3 at P(min(e, 1-e) < 0.05) = 0.120 and an effective
    sample-size fraction of 0.17. Those depend on gamma and the covariate law
    only, both of which are identical here to Study B's OV3.
    """
    p = get_config("threshold_ov3")
    w_cat, w_cont = _draw_covariates(2_000_000, np.random.default_rng(1))
    ps = expit(_design(w_cat, w_cont) @ p.gamma)
    two_sided = np.minimum(ps, 1.0 - ps)
    assert float(np.mean(two_sided < 0.05)) == pytest.approx(0.120, abs=0.01)

    w = 1.0 / ps
    ess = float(w.sum() ** 2 / (len(w) * (w**2).sum()))
    assert ess == pytest.approx(0.17, abs=0.03)


# ---------------------------------------------------------------------------
# the naming convention the Study A reporters depend on
# ---------------------------------------------------------------------------


def test_prefixed_cell_names_parse_in_the_study_a_reporters():
    """A leading OV3_ token would make every figure raise; this pins the shape."""
    df = pd.DataFrame({
        "cell": NAMES,
        "estimand": "rd",
        "event": 1,
        "estimator": "tmle",
    })

    t = table_prep(df, "rd", 1)
    assert set(t["cellid"]) == set(KEEP)
    assert set(t["n"]) == {250, 500}
    assert t["spec"].notna().all()
    assert set(t["spec"]) == {_CELL_SPEC[c] for c in KEEP}

    d = plot_prep(df)
    assert _cells(d) == list(KEEP)
    assert _ns(d) == [250, 500]
