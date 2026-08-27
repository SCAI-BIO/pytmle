"""Pre-fix against post-fix, on the same cells, seeds and truth.

The two runs use identical DGP configurations and the same master seed, so
replicate *r* of a cell is the same dataset in both directories. That makes the
comparison paired: what differs is only the targeted update.

Reported per cell and estimator: bias with its Monte Carlo SE, Wald coverage,
and the run-health columns that say whether a bias moved because the estimator
improved or because replicates dropped out.

    python -m sim.compare_fix --before results/study_a --after results/study_a_postfix
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .report import diagnostics, summarise_dir

__all__ = ["compare"]

#: `gcomp` and `ipw` are untouched by the targeted update -- gcomp is the
#: initial plug-in and ipw never uses it -- so they double as a control: if they
#: move between the two runs, something other than the fix changed.
CONTROL = ("gcomp", "ipw")


def _perf(d: Path, cells: Optional[Sequence[str]], estimand: str, event: int,
          n_mc: int) -> pd.DataFrame:
    s = summarise_dir(d, cells=cells, n_mc=n_mc)
    s = s[(s["estimand"] == estimand) & (s["event"] == event)].copy()
    s["cellid"] = s["cell"].str.split("_").str[0]
    return s


def compare(before: Path | str, after: Path | str,
            cells: Optional[Sequence[str]] = None, estimand: str = "rd",
            event: int = 1, n_mc: int = 4_000_000):
    before, after = Path(before), Path(after)
    if cells is None:
        cells = sorted(p.name for p in after.iterdir()
                       if p.is_dir() and not p.name.startswith("_"))

    rows: List[pd.DataFrame] = []
    for label, d in (("before", before), ("after", after)):
        s = _perf(d, cells, estimand, event, n_mc)
        s["run"] = label
        rows.append(s)
    s = pd.concat(rows, ignore_index=True)

    def _agg(g):
        return pd.Series({
            "truth": g["truth"].mean(),
            "bias": g["bias"].mean(),
            "mc_se": float(np.sqrt((g["bias_mc_se"] ** 2).sum()) / len(g)),
            "abs_bias": g["bias"].abs().mean(),
            "rmse": g["rmse"].mean(),
            "coverage": g["coverage"].mean(),
            "cov_mc_se": float(np.sqrt((g["coverage_mc_se"] ** 2).sum()) / len(g)),
            "n_used": g["n_used"].min(),
        })

    a = (s.groupby(["cellid", "estimator", "run"])
           .apply(_agg, include_groups=False).reset_index())

    w = a.pivot_table(index=["cellid", "estimator"], columns="run",
                      values=["bias", "mc_se", "coverage", "cov_mc_se", "rmse",
                              "n_used"])
    out = pd.DataFrame({
        "cell": w.index.get_level_values("cellid"),
        "estimator": w.index.get_level_values("estimator"),
        "bias_before": w[("bias", "before")].to_numpy(),
        "bias_after": w[("bias", "after")].to_numpy(),
        # paired only in the datasets, not in the estimates, so the difference's
        # SE is bounded by the independent combination -- conservative
        "diff_se": np.sqrt(w[("mc_se", "before")].to_numpy() ** 2
                           + w[("mc_se", "after")].to_numpy() ** 2),
        "cov_before": w[("coverage", "before")].to_numpy(),
        "cov_after": w[("coverage", "after")].to_numpy(),
        "rmse_before": w[("rmse", "before")].to_numpy(),
        "rmse_after": w[("rmse", "after")].to_numpy(),
        "reps_before": w[("n_used", "before")].to_numpy(),
        "reps_after": w[("n_used", "after")].to_numpy(),
    })
    out["bias_change"] = out["bias_after"] - out["bias_before"]
    out["z_change"] = out["bias_change"] / out["diff_se"]

    diag = pd.concat([
        diagnostics(d, cells=cells).assign(run=label)
        for label, d in (("before", before), ("after", after))
    ], ignore_index=True)
    return out.sort_values(["cell", "estimator"]).reset_index(drop=True), diag


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="sim.compare_fix", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", default="results/study_a")
    ap.add_argument("--after", default="results/study_a_postfix")
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--event", type=int, default=1)
    ap.add_argument("--estimand", default="rd")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)

    out, diag = compare(a.before, a.after, cells=a.cells, estimand=a.estimand,
                        event=a.event)
    cols = ["cell", "estimator", "bias_before", "bias_after", "bias_change",
            "z_change", "cov_before", "cov_after", "rmse_before", "rmse_after",
            "reps_before", "reps_after"]
    print(f"\nStudy A, {a.estimand} cause {a.event}: before vs after the "
          f"targeted-update fix\n")
    print(out[cols].round(4).to_string(index=False))

    print("\nRun health (a bias that moved because replicates dropped out is "
          "not an improvement)\n")
    dcols = ["cell", "run", "reps", "frac_rep_error", "frac_tmle_nonconverged",
             "median_tmle_steps"]
    print(diag[dcols].sort_values(["cell", "run"]).round(4).to_string(index=False))

    ctrl = out[out["estimator"].isin(CONTROL)]
    worst = ctrl["bias_change"].abs().max() if len(ctrl) else np.nan
    print(f"\nControl check: gcomp/ipw are untouched by the update; largest "
          f"|change| = {worst:.5f} (should be ~0).")

    if a.out_dir:
        p = Path(a.out_dir); p.mkdir(parents=True, exist_ok=True)
        out.to_csv(p / "fix_comparison.csv", index=False)
        (p / "fix_comparison.md").write_text(out[cols].round(4).to_markdown(index=False))
        diag.to_csv(p / "fix_diagnostics.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
