"""The CLI surface: where outputs land, what gets filtered, what gets reported.

Every test here exists because the corresponding bug shipped. The estimation code
is well covered and the suite passed at 94 tests throughout the day; all four
defects found while writing the reproduction guide were in **output-producing**
paths, which nothing exercised:

    report_study_a  `--fig-dir` / `--table-dir` / `summary_path` defaulted to flat
                    `results/...` and ignored `--output-dir`, so the documented
                    one-liner wrote figures where nobody was looking and two
                    output directories would overwrite one summary with the
                    other's.
    study_b         `--progress` ignored `--only` and always reported all 53 cells.
    validate        the rung-4 gate looked for a cell name that stopped existing
                    when the rung became a 2x2x2, so a rung that **had** run was
                    silently reported as "not run".
    validate        `q_model` was a pivot key but was left out of the printed
                    columns, so every rung-4 row appeared twice, indistinguishable
                    -- and it is the column the rung turns on.

The shared shape of all four: a command ran, exited zero, printed something
plausible, and did the wrong thing. So these assert on *destinations, filters and
completeness of reported columns*, not on numerics.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# outputs must land under --output-dir
# ---------------------------------------------------------------------------


def test_report_study_a_derives_every_path_from_output_dir():
    """`--output-dir` must move *all* of them, not just the shards it reads.

    The three used to default to `results/figures`, `results/tables` and
    `results/study_a_summary.parquet`, none of which contains `output_dir`. That
    is silent: the command prints a destination and exits zero.
    """
    from sim.report_study_a import build

    sig = inspect.signature(build)
    for name in ("fig_dir", "table_dir", "summary_path"):
        assert sig.parameters[name].default is None, (
            f"{name} has a hard-coded default; it must derive from output_dir")


def test_report_study_a_and_run_agree_on_where_the_report_goes(tmp_path):
    """Two entry points, one study, one destination.

    `sim.run --report` derived the paths and `sim.report_study_a` did not, so
    reporting the same study two ways wrote to two different places.
    """
    from sim.report_study_a import build

    src = inspect.getsource(build)
    assert 'output_dir / "figures"' in src
    assert 'output_dir / "tables"' in src
    assert 'output_dir / "summary.parquet"' in src

    run_src = inspect.getsource(__import__("sim.run", fromlist=["main"]).main)
    assert '"figures"' in run_src and '"tables"' in run_src


# ---------------------------------------------------------------------------
# a filter passed on the command line must actually filter
# ---------------------------------------------------------------------------


CONFIG = Path(__file__).resolve().parents[1] / "sim" / "configs" / "study_b.yaml"


def test_progress_honours_only(tmp_path):
    """`--progress --only X` must report X, not every cell in the config."""
    from sim.study_b import progress_b

    everything = progress_b(CONFIG, tmp_path)
    assert len(everything) > 40, "fixture assumption: the config has many cells"

    picked = ["OV3_n250_correct", "B_OV3_n250_correct"]
    subset = progress_b(CONFIG, tmp_path, only=picked)
    assert sorted(subset["cell"]) == sorted(picked)


def test_progress_only_accepts_an_unknown_name_without_inventing_rows():
    """A typo'd cell name must yield nothing, not silently fall back to all."""
    from sim.study_b import progress_b

    got = progress_b(CONFIG, Path("/nonexistent"), only=["no_such_cell"])
    assert len(got) == 0


# ---------------------------------------------------------------------------
# a completed rung must be detected as completed
# ---------------------------------------------------------------------------


def _make_rung4(tmp_path):
    """The current 2x2x2 cell names, with nothing else on disk."""
    rung4 = tmp_path / "rung4"
    for regime in ("none", "info"):
        for q in ("ok", "bad"):
            for g in ("ok", "bad"):
                d = rung4 / f"R4_{regime}_Q{q}_G{g}"
                d.mkdir(parents=True)
                (d / "meta.json").write_text("{}")
    return rung4


def test_rung4_gate_fires_on_the_current_cell_names(tmp_path, capsys):
    """A rung whose cells exist must not be reported as "not run".

    The gate named `R4_info_Gbad`, which stopped existing when the rung became a
    2x2x2 (`R4_{none,info}_Q{ok,bad}_G{ok,bad}`), so a rung that *had* run was
    silently reported as absent -- the failure mode that hides a whole validation
    rung behind a clean exit code.

    Asserted on behaviour rather than on source text: an earlier version of this
    test scanned `main` for the stale name and matched the *comment* explaining
    the bug.
    """
    import sim.validate as V

    _make_rung4(tmp_path)
    V.main(["--dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "rung 4: not run" not in out, "a present rung reported as absent"
    assert "rung 4" in out


def test_rung4_gate_reports_not_run_when_it_really_has_not(tmp_path, capsys):
    """The complement: the gate must not fire on an empty directory."""
    import sim.validate as V

    (tmp_path / "rung4").mkdir()
    V.main(["--dir", str(tmp_path)])
    assert "rung 4: not run" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# a printed table must carry every column its rows are keyed on
# ---------------------------------------------------------------------------


def test_rung4_prints_every_column_its_rows_are_keyed_on():
    """`q_model` is a pivot key, so omitting it duplicated every row.

    Worse than cosmetic: the rung's whole claim lives in that column -- double
    robustness holds in the `correct` row and can only fail in the `wrong` one.
    Without it the table reads as duplicated noise.
    """
    from sim.validate import RUNG4_COLUMNS, RUNG4_KEYS

    for key in RUNG4_KEYS:
        assert key in RUNG4_COLUMNS, f"pivot key {key} is not printed"
    assert "q_model" in RUNG4_KEYS
    assert len(RUNG4_COLUMNS) == len(set(RUNG4_COLUMNS))


def test_agreement_summary_keeps_every_pivot_key_as_a_column():
    """The same failure shape, in Study C's summary table.

    A key held in the index but dropped from the output makes rows collide or
    duplicate depending on which way the pivot goes.
    """
    from sim.study_c_report import agreement_summary

    agr = pd.DataFrame({
        "implementation": ["A", "A", "A", "A"],
        "vs": ["b", "b", "b", "b"],
        "tier": [1, 1, 1, 1],
        "quantity": ["est", "est", "se", "se"],
        "tolerance": [0.01, 0.01, 0.01, 0.01],
        "expect": ["agree"] * 4,
        "n": [500, 1000, 500, 1000],
        "mean_abs_diff": [0.4, 0.1, 0.02, 0.01],
        "as_expected": [True, True, True, True],
        "skipped": [False] * 4,
    })
    out = agreement_summary(agr)
    assert len(out) == 2, "one row per (comparison, quantity)"
    for col in ("comparison", "tier", "quantity_label", "unit", "tolerance",
                "expect", "as_expected"):
        assert col in out.columns, f"{col} dropped from the summary"
    assert "n=500" in out.columns and "n=1000" in out.columns
    # the trend column is the one that separates numerical from structural
    est = out[out["quantity_label"] == "point estimate"].iloc[0]
    assert est["shrink_ratio"] == pytest.approx(0.25)


def test_agreement_summary_does_not_expand_the_cartesian_product():
    """`pivot_table(dropna=False)` over many keys invents empty rows.

    Seven index keys turned nine real rows into hundreds of all-NaN ones, which
    renders as a plausible-looking table of nothing.
    """
    from sim.study_c_report import agreement_summary

    agr = pd.DataFrame({
        "implementation": ["A", "B"],
        "vs": ["a", "b"],
        "tier": [1, 2],
        "quantity": ["est", "se"],
        "tolerance": [0.002, 0.01],
        "expect": ["agree", "diverge"],
        "n": [500, 500],
        "mean_abs_diff": [0.001, 0.2],
        "as_expected": [True, True],
        "skipped": [False, False],
    })
    out = agreement_summary(agr)
    assert len(out) == 2, f"expanded to {len(out)} rows"
    assert out["n=500"].notna().all()


def test_agreement_summary_survives_an_all_skipped_table():
    """Every comparison skipped is a legitimate state, not a crash."""
    from sim.study_c_report import agreement_summary

    agr = pd.DataFrame({
        "implementation": ["A"], "vs": ["a"], "tier": [1], "quantity": ["se"],
        "tolerance": [float("nan")], "expect": ["agree"], "n": [float("nan")],
        "mean_abs_diff": [float("nan")], "as_expected": [None],
        "skipped": [True],
    })
    assert agreement_summary(agr).empty
    assert agreement_summary(pd.DataFrame()).empty
