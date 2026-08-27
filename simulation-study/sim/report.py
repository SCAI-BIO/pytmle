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

    return _runtime_agg(per_rep.rename(columns={"estimator": "implementation"}),
                        ["cell", "implementation"])


def _runtime_agg(per_rep: pd.DataFrame, group_keys) -> pd.DataFrame:
    """Aggregate one row per (group) from one row per replicate.

    Shared by Study A's `runtimes()` and the matched benchmark so the two tables
    cannot drift apart. The frame must carry `stage2_seconds`, `n`, and ideally
    `tmle_steps` and `n_times`.

    **Refuses to mix contended and uncontended timings.** After the matched
    benchmark exists, the repo holds two columns called `stage2_seconds` with
    very different validity -- one measured 8-way parallel and multi-threaded,
    one measured alone and pinned. Averaging them would produce a number that is
    neither, so a group containing both is an error rather than a mean.
    """
    if "contended" in per_rep and per_rep["contended"].nunique(dropna=False) > 1:
        raise ValueError(
            "refusing to aggregate contended and uncontended runtimes together: "
            "they are not the same measurement. Filter to one before calling.")

    emit_n = "n" not in group_keys   # else reset_index collides with the key

    def _agg(g: pd.DataFrame) -> pd.Series:
        s = g["stage2_seconds"].to_numpy(dtype=float)
        steps = (g["tmle_steps"].to_numpy(dtype=float) if "tmle_steps" in g
                 else np.full(len(g), np.nan))
        nt = (g["n_times"].to_numpy(dtype=float) if "n_times" in g
              else np.full(len(g), np.nan))
        n_val = float(g["_n"].iloc[0])
        with np.errstate(invalid="ignore", divide="ignore"):
            per_step = s / steps
            # per step and per grid cell: cost is O(n * n_times) per update, and
            # each implementation is normalised by *its own* grid, since concrete
            # builds a coarser one and would otherwise bank that as free speed
            per_cell = per_step / (n_val * nt)
        vals = {"n": int(n_val)} if emit_n else {}
        return pd.Series({
            **vals,
            "median_n_times": float(np.nanmedian(nt)),
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
            "median_ns_per_step_cell": float(np.nanmedian(per_cell) * 1e9),
        })

    # `n` is a grouping key for the benchmark and `include_groups=False` would
    # consume it, so it is carried through under an alias the group cannot claim
    per_rep = per_rep.copy()
    per_rep["_n"] = per_rep["n"]
    out = (per_rep.groupby(list(group_keys), dropna=False)
           .apply(_agg, include_groups=False).reset_index())
    return out.sort_values(list(group_keys)).reset_index(drop=True)


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
