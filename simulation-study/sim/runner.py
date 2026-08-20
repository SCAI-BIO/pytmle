"""Replicate loop: parallel execution, parquet shards, resume-on-restart.

Two design points that matter more than they look:

**Target times are frozen per configuration**, computed once from a large draw of
the DGP rather than from each replicate's own quantiles. Data-dependent target
times would move the estimand from replicate to replicate and make "bias"
meaningless.

**Seeds come from ``SeedSequence(master).spawn``**, keyed by (cell, replicate),
so every estimator and every comparator sees byte-identical data within a
replicate, and any single replicate can be re-run in isolation for debugging.
"""

from __future__ import annotations

import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .dgp import DGPParams, get_config, sample
from .nuisance import Spec, build
from .estimators import run_all

__all__ = ["Cell", "target_times_for", "run_cell", "run_cells", "default_n_jobs"]


def default_n_jobs(n: int, cap: Optional[int] = None) -> int:
    """Worker count that maximises throughput, not core count.

    The second-stage update is elementwise arithmetic over ``(n, K, .)`` arrays
    with ``K ~ n``, so it is memory-bandwidth bound rather than compute bound and
    throughput saturates well below the core count. Measured on a 20-core box:

        n = 500   n_jobs  2 ->  49 reps/min ( 49 s CPU per 24 reps)
                          4 ->  77          ( 58 s)
                          8 ->  98          ( 84 s)   <- ceiling
                         12 ->  96          (122 s)
                         20 ->  85          (169 s)   worse on both axes

        n = 1000  n_jobs  4 ->  22 reps/min (107 s CPU per 12 reps)
                          8 ->  25          (164 s)   <- ceiling
                         12 ->  25          (251 s)

    Running 20 workers costs ~2.4x the CPU of 8 for *lower* throughput. These
    defaults sit at or just below the ceiling, where the marginal worker still
    pays for itself.
    """
    if n <= 500:
        j = 8
    elif n <= 1000:
        j = 6
    else:
        j = 4
    return min(j, cap) if cap else j


@dataclass
class Cell:
    """One design cell: a DGP, a nuisance specification, and a sample size."""

    name: str
    config: str
    spec: Spec = field(default_factory=Spec)
    n: int = 500
    reps: int = 500
    estimators: Sequence[str] = ("tmle", "gcomp", "aipw", "ipw")
    min_nuisance: Optional[float] = 0.01
    max_updates: int = 200
    tau_quantiles: Sequence[float] = (0.3, 0.5, 0.7)
    #: Explicit target times, overriding ``tau_quantiles``. Needed whenever cells
    #: of one design use *different* DGP configurations that must still be
    #: compared at the same tau: the quantile rule reads the observed-time
    #: distribution, which censoring shifts, so a censoring-regime contrast would
    #: otherwise silently evaluate its two arms at different target times.
    target_times: Optional[Sequence[float]] = None
    #: Replicates to additionally run through ``concrete``'s second stage, on the
    #: same seeds and the same injected nuisances. 0 disables it. It is a
    #: replicate *count* rather than a flag because concrete is single-process
    #: and an order of magnitude slower per fit, so it runs on a subset; the
    #: comparison stays paired because the seeds are shared.
    concrete_reps: int = 0
    params_override: Dict = field(default_factory=dict)

    def dgp_params(self) -> DGPParams:
        p = get_config(self.config)
        return p.with_(**self.params_override) if self.params_override else p


def target_times_for(
    p: DGPParams,
    quantiles: Sequence[float] = (0.3, 0.5, 0.7),
    n_mc: int = 200_000,
    seed: int = 20250301,
) -> List[float]:
    """Population quantiles of the observed event-time distribution.

    Computed once and frozen, so the estimand is the same in every replicate.
    """
    sm = sample(n_mc, p, np.random.default_rng(seed))
    return [float(round(q, 6)) for q in np.quantile(sm.event_times, list(quantiles))]


def _one_rep(args) -> pd.DataFrame:
    (cell, taus, seed_state, rep) = args
    # ``seed_state`` is an already-spawned SeedSequence; default_rng consumes it
    # directly, whereas re-wrapping it would raise.
    rng = np.random.default_rng(seed_state)
    p = cell.dgp_params()
    try:
        sm = sample(cell.n, p, rng)
        if sm.event_times.max() < max(taus):
            raise ValueError(
                f"max observed event time {sm.event_times.max():.4g} < max target "
                f"time {max(taus):.4g}"
            )
        ie, diag = build(sm, cell.spec, return_diagnostics=True)
        res = run_all(
            sm.df,
            ie,
            taus,
            target_events=list(range(1, p.n_causes + 1)),
            which=cell.estimators,
            min_nuisance=cell.min_nuisance,
            max_updates=cell.max_updates,
        )
        res = res.copy()
        res["cell"] = cell.name
        res["rep"] = rep
        res["n"] = cell.n
        res["ps_mae"] = diag.ps_mae
        res["ps_min"] = diag.ps_min
        res["ps_max"] = diag.ps_max
        res["cumhaz_mae"] = diag.cumhaz_mae
        res["n_failed_fits"] = diag.n_failed_fits
        res["tmle_converged"] = res.attrs.get("tmle_converged", np.nan)
        res["tmle_steps"] = res.attrs.get("tmle_steps", np.nan)
        res["censored_frac"] = float((sm.event_indicator == 0).mean())
        # grid size and target count: second-stage cost is O(n * n_times) per
        # update step, so runtimes are only interpretable alongside these
        res["n_times"] = int(len(np.unique(sm.event_times)))
        res["n_target_times"] = len(taus)
        return res
    except Exception as exc:  # a replicate must never kill the run
        return pd.DataFrame(
            [
                {
                    "cell": cell.name, "rep": rep, "n": cell.n,
                    "estimator": None, "estimand": None, "event": np.nan,
                    "time": np.nan, "group": np.nan, "est": np.nan, "se": np.nan,
                    "ci_lo": np.nan, "ci_hi": np.nan, "converged": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=3),
                    "seconds": np.nan,
                }
            ]
        )


def run_cell(
    cell: Cell,
    output_dir: Path,
    master_seed: int = 20250301,
    n_jobs: Optional[int] = None,
    chunk: int = 50,
    overwrite: bool = False,
) -> Path:
    """Run one cell, writing parquet shards and skipping any already complete."""
    output_dir = Path(output_dir)
    shard_dir = output_dir / cell.name
    shard_dir.mkdir(parents=True, exist_ok=True)
    if n_jobs is None:
        n_jobs = default_n_jobs(cell.n)

    p = cell.dgp_params()
    taus = (list(cell.target_times) if cell.target_times is not None
            else target_times_for(p, cell.tau_quantiles))
    (shard_dir / "meta.json").write_text(
        json.dumps({"cell": cell.name, "config": cell.config, "n": cell.n,
                    "reps": cell.reps, "spec": cell.spec.__dict__,
                    "target_times": taus, "min_nuisance": cell.min_nuisance},
                   indent=2, default=str)
    )

    ss = np.random.SeedSequence(master_seed)
    # a deterministic, cell-specific stream so cells are independent but reproducible
    cell_seed = np.random.SeedSequence(
        [master_seed, int.from_bytes(cell.name.encode()[:8].ljust(8, b"\0"), "little") % (2**31)]
    )
    children = cell_seed.spawn(cell.reps)

    chunks = [range(i, min(i + chunk, cell.reps)) for i in range(0, cell.reps, chunk)]
    for ci, rng_chunk in enumerate(chunks):
        shard = shard_dir / f"shard_{ci:04d}.parquet"
        if shard.exists() and not overwrite:
            continue
        tasks = [(cell, taus, children[r], r) for r in rng_chunk]
        frames: List[pd.DataFrame] = []
        if n_jobs == 1:
            frames = [_one_rep(t) for t in tasks]
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                futs = [ex.submit(_one_rep, t) for t in tasks]
                for f in as_completed(futs):
                    frames.append(f.result())
        out = pd.concat(frames, ignore_index=True)
        # Error rows are built with plain Python types and can reintroduce the
        # mixed-dtype problem that estimators.run_all guards against, so the
        # same normalisation is applied once more at the write boundary. A
        # failure here would otherwise discard a whole chunk of completed work.
        if "converged" in out:
            out["converged"] = pd.array(
                [None if pd.isna(v) else bool(v) for v in out["converged"]],
                dtype="boolean",
            )
        out.to_parquet(shard, index=False)

    _run_concrete(cell, shard_dir, taus, master_seed, overwrite)
    return shard_dir


def _run_concrete(cell: Cell, shard_dir: Path, taus: Sequence[float],
                  master_seed: int, overwrite: bool) -> None:
    """Second implementation, same cell, same seeds -- written beside the shards.

    Kept inside ``run_cell`` rather than offered as a separate script so that one
    ``sim.run`` reproduces the whole study. It runs after the replicate shards so
    an interrupted run still leaves the Python results complete and resumable,
    and it is skipped when already present, like the shards themselves.

    A concrete failure is reported and swallowed: it is a comparator, and losing
    it must not cost the cell it was comparing against.
    """
    out_path = shard_dir / "concrete.parquet"
    if cell.concrete_reps <= 0 or (out_path.exists() and not overwrite):
        return
    from .concrete_bridge import run_cell_concrete

    # The seed stream is spawned to `cell.reps`, so asking for more concrete
    # replicates than the cell has runs off the end of it. That happens whenever
    # `--reps` shrinks a cell for a pilot while the config's concrete_reps stays
    # put, which is exactly when a hard failure is least wanted.
    reps = min(cell.concrete_reps, cell.reps)
    try:
        got = run_cell_concrete(cell, taus=taus, reps=reps,
                                master_seed=master_seed,
                                min_nuisance=cell.min_nuisance or 0.01)
    except Exception as exc:
        print(f"  [concrete] {cell.name}: FAILED {type(exc).__name__}: {exc}",
              flush=True)
        return

    # The grid size is a property of the replicate, not of the implementation,
    # and concrete does not report it -- without it a runtime is not
    # interpretable, since second-stage cost is O(n * n_times) per step.
    try:
        grid = (load_cell(shard_dir).drop_duplicates("rep")
                .set_index("rep")["n_times"])
        got["n_times"] = got["rep"].map(grid)
    except Exception:
        pass
    if "converged" in got:
        got["converged"] = pd.array(
            [None if pd.isna(v) else bool(v) for v in got["converged"]],
            dtype="boolean",
        )
    got.to_parquet(out_path, index=False)
    print(f"  [concrete] {cell.name}: {got['rep'].nunique()} replicates",
          flush=True)


def load_cell(shard_dir: Path) -> pd.DataFrame:
    """Every estimate for one cell, PyTMLE's and concrete's alike.

    ``concrete.parquet`` sits alongside the replicate shards and is loaded with
    them, so downstream code sees one frame per cell and needs no special case
    for the second implementation. Its rows carry ``estimator = "tmle
    (concrete)"`` and their own replicate count; ``metrics.summarise`` groups by
    estimator, so the two are never pooled into a single Monte Carlo SE.
    """
    shard_dir = Path(shard_dir)
    shards = sorted(shard_dir.glob("shard_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no shards under {shard_dir}")
    frames = [pd.read_parquet(s) for s in shards]
    concrete = shard_dir / "concrete.parquet"
    if concrete.exists():
        frames.append(pd.read_parquet(concrete))
    return pd.concat(frames, ignore_index=True)


def run_cells(
    cells: Sequence[Cell],
    output_dir: Path,
    master_seed: int = 20250301,
    n_jobs: Optional[int] = None,
    chunk: int = 50,
    overwrite: bool = False,
) -> Dict[str, Path]:
    out = {}
    for cell in cells:
        out[cell.name] = run_cell(
            cell, output_dir, master_seed=master_seed, n_jobs=n_jobs,
            chunk=chunk, overwrite=overwrite,
        )
    return out
