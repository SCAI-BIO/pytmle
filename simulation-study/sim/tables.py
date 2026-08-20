"""Human-readable result tables.

Every table is written twice: a Markdown file to read or paste into a report, and
a CSV alongside it for anything downstream. The parquet summaries stay the
machine-readable source of truth -- these are the artefacts a person opens.

Numbers are rounded at the point of writing, not in the stored summary, and each
table carries its Monte Carlo standard errors next to the estimates rather than
in a separate file, because a bias without its MC-SE cannot be read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["write_tables"]

_CELL_SPEC = {
    "C1": "Q✓ π✓ G✓", "C2": "Q✓ π✓ G✗", "C3": "Q✓ π✗ G✓", "C4": "Q✓ π✗ G✗",
    "C5": "Q✗ π✓ G✓", "C6": "Q✗ π✓ G✗", "C7": "Q✗ π✗ G✓", "C8": "Q✗ π✗ G✗",
}
_ORDER = ["gcomp", "ipw", "aipw", "tmle", "tmle (concrete)"]


def _emit(df: pd.DataFrame, stem: Path, title: str, note: str = "") -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv = stem.with_suffix(".csv")
    md = stem.with_suffix(".md")
    df.to_csv(csv, index=False)
    body = df.to_markdown(index=False, floatfmt=".4f")
    md.write_text(f"# {title}\n\n" + (f"{note}\n\n" if note else "") + body + "\n")
    return [md, csv]


def _prep(summary: pd.DataFrame, estimand: str, event: Optional[int]) -> pd.DataFrame:
    d = summary[summary["estimand"] == estimand].copy()
    if event is not None:
        d = d[d["event"] == event]
    d["cell"] = d["cell"].astype(str)
    d["cellid"] = d["cell"].str.split("_").str[0]
    d["n"] = d["cell"].str.split("_n").str[1].astype(int)
    d["spec"] = d["cellid"].map(_CELL_SPEC)
    return d


def write_tables(
    summary: pd.DataFrame,
    out_dir: Path,
    estimand: str = "rd",
    event: Optional[int] = 1,
    diagnostics: Optional[pd.DataFrame] = None,
    runtimes: Optional[pd.DataFrame] = None,
) -> list[Path]:
    """Write the Study A tables as Markdown + CSV."""
    out_dir = Path(out_dir)
    d = _prep(summary, estimand, event)
    ev = f", cause {event}" if event is not None else ""
    written: list[Path] = []

    # --- per target: the full detail, nothing aggregated away ---------------
    detail = (d[["cellid", "spec", "n", "estimator", "time", "truth", "bias",
                 "bias_mc_se", "emp_sd", "mean_se", "se_ratio", "rmse",
                 "coverage", "coverage_mc_se", "ci_width", "n_used"]]
              .rename(columns={"cellid": "cell", "time": "target_time"})
              .sort_values(["n", "cell", "estimator", "target_time"]))
    written += _emit(detail, out_dir / f"study_a_detail_{estimand}",
                     f"Study A — per-target results ({estimand}{ev})",
                     "One row per cell x sample size x estimator x target time. "
                     "`bias_mc_se` and `coverage_mc_se` are Monte Carlo standard "
                     "errors; a difference smaller than ~2 MC-SE is not resolved.")

    # --- bias summary: the headline table -----------------------------------
    def _agg(g):
        return pd.Series({
            "bias": g["bias"].mean(),
            "mc_se": float(np.sqrt((g["bias_mc_se"] ** 2).sum()) / len(g)),
            "abs_bias": float(g["bias"].abs().mean()),
            "rmse": g["rmse"].mean(),
        })

    bias = (d.groupby(["n", "cellid", "spec", "estimator"])
              .apply(_agg, include_groups=False).reset_index())
    bias["z"] = bias["bias"] / bias["mc_se"]
    wide = (bias.pivot_table(index=["n", "cellid", "spec"], columns="estimator",
                             values="bias")
                .reindex(columns=[c for c in _ORDER if c in set(bias.estimator)])
                .reset_index().rename(columns={"cellid": "cell"}))
    written += _emit(wide, out_dir / f"study_a_bias_{estimand}",
                     f"Study A — bias by cell ({estimand}{ev})",
                     "Averaged over target times within a cell. Theory: `gcomp` "
                     "biased iff Q wrong; `ipw` iff π or G wrong; `aipw`/`tmle` "
                     "only when Q wrong *and* some g component wrong.")

    cov = (d.groupby(["n", "cellid", "spec", "estimator"])["coverage"].mean()
             .reset_index()
             .pivot_table(index=["n", "cellid", "spec"], columns="estimator",
                          values="coverage")
             .reindex(columns=[c for c in _ORDER if c in set(d.estimator)])
             .reset_index().rename(columns={"cellid": "cell"}))
    written += _emit(cov, out_dir / f"study_a_coverage_{estimand}",
                     f"Study A — 95% Wald coverage by cell ({estimand}{ev})",
                     "Nominal is 0.95. Double robustness buys consistency, not "
                     "inference: under-coverage where a nuisance is wrong is "
                     "expected, not a defect.")

    if diagnostics is not None and len(diagnostics):
        written += _emit(diagnostics, out_dir / "study_a_diagnostics",
                         "Study A — run diagnostics",
                         "Replicate failures, TMLE non-convergence, fitted "
                         "propensity range and nuisance discrepancy per cell.")
    if runtimes is not None and len(runtimes):
        written += _emit(runtimes, out_dir / "study_a_runtimes",
                         "Study A — second-stage runtime",
                         "Targeted update only; initial estimates are injected, so "
                         "this excludes all nuisance fitting.")
    return written
