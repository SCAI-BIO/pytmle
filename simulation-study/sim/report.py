"""Aggregate parquet shards into summary tables.

Extended alongside each study rather than at the end, so that a metric can never
turn out not to have been logged after an overnight run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .dgp import get_config
from .metrics import summarise, truth_long
from .runner import load_cell
from .truth import cached_truth

__all__ = ["collect", "summarise_dir", "diagnostics", "runtimes"]


def collect(output_dir: Path | str, cells: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Load every cell's shards under ``output_dir`` into one frame."""
    output_dir = Path(output_dir)
    frames: List[pd.DataFrame] = []
    for meta_path in sorted(output_dir.glob("*/meta.json")):
        name = meta_path.parent.name
        if cells and name not in cells:
            continue
        df = load_cell(meta_path.parent)
        meta = json.loads(meta_path.read_text())
        df["config"] = meta["config"]
        df["target_times"] = [tuple(meta["target_times"])] * len(df)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no completed cells under {output_dir}")
    return pd.concat(frames, ignore_index=True)


def summarise_dir(
    output_dir: Path | str,
    cache_dir: Path | str | None = None,
    cells: Optional[Sequence[str]] = None,
    n_mc: int = 10_000_000,
) -> pd.DataFrame:
    """Per-cell performance table, joining each cell to its own truth."""
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir) if cache_dir else output_dir / "_truth"
    raw = collect(output_dir, cells=cells)

    out: List[pd.DataFrame] = []
    for (cfg_name, taus), grp in raw.groupby(["config", "target_times"], sort=False):
        p = get_config(cfg_name)
        tr = cached_truth(p, list(taus), cache_dir, n_mc=n_mc)
        out.append(summarise(grp[grp["estimator"].notna()], truth_long(tr)))
    return pd.concat(out, ignore_index=True)


def runtimes(
    output_dir: Path | str,
    cells: Optional[Sequence[str]] = None,
    implementations: Sequence[str] = ("tmle", "concrete"),
) -> pd.DataFrame:
    """Second-stage runtime, aggregated over replicates.

    Only the *targeted update* is timed. Initial estimates are injected, so this
    excludes all nuisance fitting and is the like-for-like number to compare
    PyTMLE against ``concrete`` -- the two implementations of the same
    algorithm. Anything else (g-computation, IPW, the one-step) rides along on
    work already done and has no separately meaningful second-stage cost.

    Reported alongside ``n``, the grid size and the iteration count, because
    cost is O(n * n_times) per update step and a bare wall-clock number is not
    interpretable without them.
    """
    raw = collect(output_dir, cells=cells)
    sub = raw[raw["estimator"].isin(implementations) & raw["stage2_seconds"].notna()]
    if sub.empty:
        return pd.DataFrame(
            columns=["cell", "implementation", "n", "n_runs", "median_s", "mean_s"]
        )
    # one row per (cell, implementation, replicate)
    per_rep = sub.drop_duplicates(["cell", "estimator", "rep"])

    def _agg(g: pd.DataFrame) -> pd.Series:
        s = g["stage2_seconds"].to_numpy(dtype=float)
        steps = g["tmle_steps"].to_numpy(dtype=float) if "tmle_steps" in g else np.array([np.nan])
        with np.errstate(invalid="ignore", divide="ignore"):
            per_step = s / steps
        return pd.Series(
            {
                "n": int(g["n"].iloc[0]),
                "median_n_times": float(np.median(g["n_times"])) if "n_times" in g else np.nan,
                "n_runs": int(len(g)),
                "mean_s": float(np.mean(s)),
                "sd_s": float(np.std(s, ddof=1)) if len(s) > 1 else np.nan,
                "median_s": float(np.median(s)),
                "q05_s": float(np.quantile(s, 0.05)),
                "q95_s": float(np.quantile(s, 0.95)),
                "min_s": float(np.min(s)),
                "max_s": float(np.max(s)),
                "total_s": float(np.sum(s)),
                "median_steps": float(np.nanmedian(steps)),
                "median_s_per_step": float(np.nanmedian(per_step)),
            }
        )

    out = (
        per_rep.groupby(["cell", "estimator"], dropna=False)
        .apply(_agg, include_groups=False)
        .reset_index()
        .rename(columns={"estimator": "implementation"})
    )
    return out.sort_values(["cell", "implementation"]).reset_index(drop=True)


def diagnostics(output_dir: Path | str, cells: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Per-cell run health: failures, non-convergence, nuisance discrepancy, positivity."""
    raw = collect(output_dir, cells=cells)
    per_rep = raw.drop_duplicates(["cell", "rep"])
    g = per_rep.groupby("cell", dropna=False)
    return pd.DataFrame(
        {
            "n": g["n"].first(),
            "reps": g["rep"].nunique(),
            "frac_rep_error": g["error"].apply(lambda s: float(s.notna().mean())),
            "frac_tmle_nonconverged": g["tmle_converged"].apply(
                lambda s: float((~s.dropna().astype(bool)).mean()) if s.notna().any() else np.nan
            ),
            "median_tmle_steps": g["tmle_steps"].median(),
            "mean_censored_frac": g["censored_frac"].mean(),
            "mean_ps_mae": g["ps_mae"].mean(),
            "min_ps": g["ps_min"].min(),
            "max_ps": g["ps_max"].max(),
            "mean_cumhaz_mae": g["cumhaz_mae"].mean(),
            "n_failed_fits": g["n_failed_fits"].sum(),
        }
    ).reset_index()
