"""Run concrete's second stage on Study A's replicates.

PyTMLE's targeted update is a port of concrete's, so running both on
byte-identical injected nuisances turns "does the port behave like its source?"
into a measurement that sits inside the study rather than beside it.

Two things make the comparison paired rather than merely parallel:

* **the same seeds.** ``cell_seeds`` reproduces ``runner.run_cell``'s seeding
  exactly, so replicate *r* of a cell is the same dataset in both languages. If
  that drifts, the two columns stop being comparable and nothing else here would
  notice.
* **the same nuisances.** Nothing is refitted on the R side; concrete's
  ``getInitialEstimate`` runs once per replicate only to obtain its scaffolding
  (time grid, object shape) and every component is overwritten.

Nuisance arrays are ``(n, K)`` with ``K ~ n``, so a cell's exports are deleted as
soon as its results are read -- keeping all of Study A's would run to ~150 GB.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .dgp import get_config, sample
from .nuisance import Spec, build
from .runner import Cell, target_times_for

__all__ = ["cell_seeds", "run_cell_concrete"]

RSCRIPT = "/home/jguski/.conda/envs/pytmle-sim/bin/Rscript"


def cell_seeds(cell_name: str, reps: int, master_seed: int = 20250301):
    """The exact seed stream ``runner.run_cell`` uses, so replicates line up."""
    cell_seed = np.random.SeedSequence(
        [master_seed,
         int.from_bytes(cell_name.encode()[:8].ljust(8, b"\0"), "little") % (2**31)]
    )
    return cell_seed.spawn(reps)


def _export(cell: Cell, taus: Sequence[float], reps: int, out: Path,
            master_seed: int = 20250301) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time": list(taus)}).to_parquet(out / "taus.parquet", index=False)
    children = cell_seeds(cell.name, cell.reps, master_seed)
    p = cell.dgp_params()
    for i in range(reps):
        sm = sample(cell.n, p, np.random.default_rng(children[i]))
        ie = build(sm, cell.spec)
        stub = out / f"rep{i:03d}"
        sm.df.to_parquet(f"{stub}_data.parquet", index=False)
        pd.DataFrame({"time": ie[1].times}).to_parquet(f"{stub}_grid.parquet", index=False)
        pd.DataFrame({"ps1": ie[1].propensity_scores}).to_parquet(
            f"{stub}_ps.parquet", index=False)
        for key, tag in ((1, "1"), (0, "0")):
            e = ie[key]
            cum = np.cumsum(e.hazards, axis=1)
            for j in (0, 1):
                pd.DataFrame(cum[:, :, j]).to_parquet(f"{stub}_H{j+1}_{tag}.parquet",
                                                      index=False)
            pd.DataFrame(-np.log(np.maximum(e.censoring_survival_function, 1e-300))
                         ).to_parquet(f"{stub}_HC_{tag}.parquet", index=False)


def run_cell_concrete(
    cell: Cell,
    taus: Optional[Sequence[float]] = None,
    reps: int = 150,
    workdir: Path = Path("results/_concrete_tmp"),
    master_seed: int = 20250301,
    min_nuisance: float = 0.01,
    keep: bool = False,
) -> pd.DataFrame:
    """Export one cell, run concrete over it, return tidy RD estimates.

    ``taus`` must be the target times the cell was actually run at. Recomputing
    them here from ``tau_quantiles`` would silently ignore a cell's explicit
    ``target_times`` override and score the two implementations at different
    times -- which is exactly the mistake rung 4 was built to avoid.
    """
    if taus is None:
        taus = (list(cell.target_times) if cell.target_times is not None
                else target_times_for(cell.dgp_params(), cell.tau_quantiles))
    wd = Path(workdir) / cell.name
    if wd.exists():
        shutil.rmtree(wd)
    _export(cell, taus, reps, wd, master_seed)

    proc = subprocess.run(
        [RSCRIPT, "R/run_concrete_injected.R", "--dir", str(wd),
         "--reps", str(reps), "--min-nuisance", str(min_nuisance)],
        capture_output=True, text=True,
    )
    est_path = wd / "concrete_estimates.parquet"
    if not est_path.exists():
        raise RuntimeError(
            f"concrete produced no output for {cell.name}:\n{proc.stderr[-2000:]}"
        )
    raw = pd.read_parquet(est_path)
    if not keep:
        shutil.rmtree(wd, ignore_errors=True)

    out = raw[raw["Estimator"] == "tmle"].rename(
        columns={"Event": "event", "Time": "time", "Pt Est": "est", "se": "se"}
    )
    keep = ["rep", "event", "time", "est", "se", "converged", "steps"]
    if "stage2_seconds" in out.columns:
        keep.append("stage2_seconds")
    out = out[keep].copy()
    out["tmle_steps"] = out.pop("steps")
    out["estimator"] = "tmle (concrete)"
    out["estimand"] = "rd"
    out["cell"] = cell.name
    out["n"] = cell.n
    out["group"] = np.nan
    out["ci_lo"] = out["est"] - 1.959964 * out["se"]
    out["ci_hi"] = out["est"] + 1.959964 * out["se"]
    out["error"] = None
    out["n_times"] = np.nan
    return out
