"""A matched second-stage benchmark: PyTMLE against concrete, single-threaded.

    python -m sim.bench_stage2 --study-dir results/study_c

Study C's own `stage2_seconds` is **not** a fair comparison and this module
exists because of that. Two things spoil it, and they compound:

*Contention.* Every Study C stage runs `n_jobs`-way parallel -- the concrete and
comparator stages as concurrent `Rscript` shards, the PyTMLE stage as a worker
pool. PyTMLE's recorded single-process median at n = 500 is 1.59 s against 4.10 s
inside the pool, a ~2.6x penalty.

*Threading, which is worse.* Neither side is single-threaded by default and they
thread to **different degrees**: numpy links OpenBLAS with 20 threads, while R
links the *pthreads* OpenBLAS build and `data.table` claims half the cores (10 of
20 here). concrete's Armadillo/C++ links `SHLIB_OPENMP_CXXFLAGS` and BLAS/LAPACK
on top. So the previously reported number compares an 8-way-contended 20-thread Python
process against an 8-way-contended 10-thread R process.

**The measurement contradicted the reason this module was written.** The
expectation recorded here was that the reported ~4x gap would prove "mostly
artefact". It is not. Pinning both sides to one thread and serialising them
roughly halves *both* times -- contention was costing them about equally -- and
the ratio moves only from ~3.8x to 3.26x at n = 500. PyTMLE is genuinely slower
than concrete per fit, by 3.3x at n = 500 rising to 4.4x at n = 2000, and the
fair comparison establishes that rather than explaining it away. See FINDINGS 15.

This module fixes both, and **verifies rather than assumes** that it did:

    single-threaded   thread limits are placed in the child's environment
                      *before* it starts, because OpenBLAS reads its count at
                      load time; `data.table::setDTthreads(1)` covers the part
                      no environment variable reaches; `threadpoolctl` confirms
                      the Python side; and the orchestrator samples the child's
                      live thread count with `psutil` -- the only check that
                      works for R, where nothing can introspect from inside.
    serialised        one fit at a time, never concurrent, alternating which
                      implementation goes first so a warm cache cannot favour
                      one of them systematically.
    idle              refuses to start on a loaded machine, and records CPU time
                      beside wall time so `wall/cpu` shows afterwards whether the
                      run really was uncontended.

Deliberately **not** a stage in `study_c._MARKERS`. A marker file records that
work happened, not that the machine was quiet while it happened; a stale
`bench.parquet` from a contended run would be silently accepted as done. It is
also the only part of the study that must not be parallelised, which sits badly
inside a driver whose every other stage is.

Scope is PyTMLE `tmle` against concrete `tmle`, second stage only, on
byte-identical injected nuisances -- the same algorithm implemented twice. Not
`aipw`/`gcomp`/`ipw`: those are different algorithms with different comparators,
and `gcomp` has no separable cost at all since it rides the TMLE fit.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["THREAD_VARS", "single_thread_env", "check_idle", "run_bench",
           "summarise_bench"]

#: Every thread-count variable the two stacks read. OpenBLAS and MKL consult
#: these when the library loads, so they must be in the environment before the
#: interpreter starts -- setting them from inside the process is too late.
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "RAYON_NUM_THREADS")

#: A live thread *count* is a weak signal and is recorded as context, not used
#: as the gate. Both runtimes carry idle pool threads that never compute -- R
#: showed 7 while its CPU time equalled its wall time to within 1 %, i.e. one
#: thread of actual work. The gate is `wall_over_cpu` below: a single-threaded
#: fit that was left alone spends its wall time computing, so the ratio sits at
#: ~1; parallel work drives CPU above wall, and contention drives wall above CPU.
THREAD_TOLERANCE = 3

#: How far `wall/cpu` may stray from 1 before a row is not a fair measurement.
#: Wide enough for process startup and I/O, tight enough that a second busy core
#: (ratio ~0.5) or a competing job (ratio >> 1) fails.
WALL_CPU_BAND = (0.75, 1.35)


def single_thread_env(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """A copy of the environment with every thread pool pinned to one."""
    env = dict(os.environ if base is None else base)
    for var in THREAD_VARS:
        env[var] = "1"
    return env


def verify_python_threads() -> List[Dict]:
    """What each native pool in *this* process actually reports.

    Recorded, not just asserted: "we set the variable" and "the library obeyed"
    are different claims, and only the second one makes the benchmark fair.
    """
    try:
        import threadpoolctl
    except ImportError:  # pragma: no cover - threadpoolctl is in the env
        return []
    return [{"user_api": p.get("user_api"), "prefix": p.get("prefix"),
             "num_threads": p.get("num_threads")}
            for p in threadpoolctl.threadpool_info()]


def check_idle(max_load: float = 2.0, allow_busy: bool = False) -> float:
    """Refuse to benchmark a busy machine.

    Advisory only -- the load average is a lagging, coarse signal. The check that
    actually settles it is `wall_over_cpu` in the output: a single-threaded fit
    that was left alone spends its wall time computing, so wall/cpu ~ 1.
    """
    load1 = os.getloadavg()[0]
    if load1 > max_load and not allow_busy:
        raise RuntimeError(
            f"1-minute load average is {load1:.2f} (> {max_load}); a runtime "
            f"measured against other work is not a fair one. Wait for the "
            f"machine to go quiet, or pass --allow-busy for a smoke test.")
    return float(load1)


def _max_threads_while(proc, poll: float = 0.02) -> int:
    """Poll a child's live thread count until it exits.

    This is the verification that covers R. `setDTthreads` and the environment
    variables are instructions; this is the observation. Without it the whole
    single-threaded claim rests on the libraries having done as they were told.
    """
    try:
        import psutil
        p = psutil.Process(proc.pid)
    except Exception:  # pragma: no cover - psutil is in the env
        proc.wait()
        return -1
    peak = 0
    while proc.poll() is None:
        try:
            peak = max(peak, p.num_threads())
        except Exception:
            break
        time.sleep(poll)
    proc.wait()
    return peak


def _bench_pytmle(stub: Path, taus: Sequence[float], min_nuisance: float,
                  repeats: int) -> Dict:
    """Time PyTMLE's targeted update, in-process and single-threaded."""
    from pytmle import PyTMLE

    from .validate import initial_estimates_from_r
    from .study_c import _load_replicate

    rep = _load_replicate(stub)
    best_wall, best_cpu, steps, n_times = np.inf, np.inf, np.nan, np.nan
    # `repeats + 1` passes, first discarded: the first fit in a fresh process
    # pays import and first-call costs that have nothing to do with the
    # algorithm. Measured at n = 80 it was 33 s against 0.9 s for the next --
    # enough to decide the comparison on its own if it landed on one side.
    for _pass in range(repeats + 1):
        # Rebuilt every repeat, outside the clock. `fit()` mutates the estimates
        # in place, so a second repeat over the same object would start from
        # *targeted* values and time a fit that has nothing left to do.
        ie = initial_estimates_from_r(rep)
        model = PyTMLE(rep["df"], target_times=list(taus), initial_estimates=ie,
                       g_comp=False, evalues_benchmark=False, verbose=0)
        w0, c0 = time.perf_counter(), time.process_time()
        model.fit(min_nuisance=min_nuisance, max_updates=200)
        wall, cpu = time.perf_counter() - w0, time.process_time() - c0
        if _pass == 0:
            continue                      # warm-up, never scored
        if wall < best_wall:
            best_wall, best_cpu = wall, cpu
            steps = int(model.step_num)
            # The grid the update ran on, not the injected one. PyTMLE truncates
            # at `max(target_times)`; concrete reports its own working grid, so
            # recording `len(ie[1].times)` here divided the two implementations
            # by different denominators and understated PyTMLE's per-cell cost.
            n_times = int(len(model._updated_estimates[1].times))
    return {"stage2_seconds": best_wall, "stage2_cpu_seconds": best_cpu,
            "tmle_steps": steps, "n_times": n_times, "max_threads": np.nan}


def _bench_concrete(out_dir: Path, rep_id: int, min_nuisance: float,
                    repeats: int) -> Dict:
    """Time concrete's targeted update, one replicate in its own process."""
    from .concrete_bridge import RSCRIPT

    args = [RSCRIPT, "R/run_concrete_injected.R", "--dir", str(out_dir),
            "--from", str(rep_id), "--to", str(rep_id + 1),
            "--shard", f"bench{rep_id:04d}", "--reps", str(rep_id + 1),
            "--min-nuisance", str(min_nuisance), "--repeats", str(repeats)]
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True,
                            env=single_thread_env())
    peak = _max_threads_while(proc)
    err = proc.stderr.read() if proc.stderr else ""
    shard = out_dir / f"concrete_estimates_bench{rep_id:04d}.parquet"
    eic = out_dir / f"concrete_eic_bench{rep_id:04d}.parquet"
    if proc.returncode != 0 or not shard.exists():
        for f in (shard, eic):
            f.unlink(missing_ok=True)
        raise RuntimeError(f"concrete bench failed on rep {rep_id}: "
                           f"{err.strip()[-400:]}")
    d = pd.read_parquet(shard)
    for f in (shard, eic):
        f.unlink(missing_ok=True)
    row = d.drop_duplicates("rep").iloc[0]
    return {"stage2_seconds": float(row["stage2_seconds"]),
            "stage2_cpu_seconds": float(row.get("stage2_cpu_seconds", np.nan)),
            "tmle_steps": float(row.get("steps", np.nan)),
            "n_times": float(row.get("n_times", np.nan)),
            "max_threads": peak}


def run_bench(study_dir: Path | str, sizes: Sequence[Dict],
              min_nuisance: float = 0.01, repeats: int = 3,
              max_load: float = 2.0, allow_busy: bool = False) -> pd.DataFrame:
    """Both implementations, one at a time, on the same replicates."""
    study_dir = Path(study_dir)
    load_before = check_idle(max_load, allow_busy)
    pools = verify_python_threads()
    bad = [p for p in pools if (p.get("num_threads") or 1) != 1]
    if bad and not allow_busy:
        raise RuntimeError(
            f"native thread pools are not pinned: {bad}. Set the thread "
            f"variables before starting Python -- OpenBLAS reads its count when "
            f"the library loads, so os.environ afterwards has no effect.")

    rows: List[Dict] = []
    for spec in sizes:
        n, reps = int(spec["n"]), int(spec["reps"])
        out_dir = study_dir / f"n{n}"
        taus = [float(t) for t in pd.read_parquet(out_dir / "taus.parquet")["time"]]
        stubs = sorted(out_dir.glob("rep*_grid.parquet"))[:reps]
        if not stubs:
            raise FileNotFoundError(
                f"{out_dir} has no exported replicates. The benchmark reuses "
                f"Study C's nuisances, so the main run must come first and its "
                f"rep*.parquet files must not have been cleaned up yet.")
        print(f"[bench] n={n}: {len(stubs)} replicates x {repeats} repeats",
              flush=True)
        for k, stub in enumerate(stubs):
            s = str(stub)[: -len("_grid.parquet")]
            rep_id = int(Path(s).name[len("rep"):])
            # Alternate which implementation runs first: whichever goes second
            # benefits from a warm page cache, and fixing the order would hand
            # that advantage to the same package at every single replicate.
            order = ("pytmle", "concrete") if k % 2 == 0 else ("concrete", "pytmle")
            for impl in order:
                if impl == "pytmle":
                    r = _bench_pytmle(Path(s), taus, min_nuisance, repeats)
                else:
                    r = _bench_concrete(out_dir, rep_id, min_nuisance, repeats)
                rows.append({"n": n, "rep": rep_id, "implementation": impl,
                             "ran_first": order[0] == impl, "repeats": repeats,
                             "contended": False, **r})
    out = pd.DataFrame(rows)
    out["load_before"] = load_before
    out["load_after"] = os.getloadavg()[0]
    out["wall_over_cpu"] = out["stage2_seconds"] / out["stage2_cpu_seconds"]
    lo, hi = WALL_CPU_BAND
    out["fair"] = out["wall_over_cpu"].between(lo, hi)
    out.attrs["thread_pools"] = json.dumps(pools)
    return out


def summarise_bench(bench: pd.DataFrame) -> pd.DataFrame:
    """Per (n, implementation), on the three units that are comparable.

    Seconds per fit is not interpretable on its own: second-stage cost is
    O(n * n_times) per update step, and the two implementations take different
    numbers of steps on *different grids* -- concrete builds its own, 209 points
    where PyTMLE has 300 at n = 300. So the per-step and per-cell figures are
    reported beside the raw one, each implementation normalised by **its own**
    grid. Normalising both by PyTMLE's would hand concrete its coarser grid as
    free speed.

    Both packages count *accepted* steps only; each hides its rejected-iteration
    count. Per-step cost therefore understates work symmetrically.

    **The step counts are not on the same convention.** Measured at n = 500 over
    500 replicates with byte-identical injected nuisances, concrete reports
    exactly one step more than PyTMLE on 487 of them (mean difference 1.014;
    medians 12 against 11), while both converge on 100 % of replicates and both
    drive `|PnEIC|` below their own `seEIC/(sqrt(n) log n)` on 100 % of targets.
    So the offset is a counting convention -- one side counts the initial
    evaluation as a step -- not extra work, and at a median of ~11 steps it
    flatters concrete's per-step figure by roughly 8 %.

    `median_s_per_step` and `median_ns_per_step_cell` are therefore **not**
    like-for-like across implementations; they are the right unit for comparing
    one implementation across `n`. The cross-package headline is
    `median_stage2_seconds` per fit, which is convention-free.
    """
    from .report import _runtime_agg

    if bench.empty:
        return pd.DataFrame()
    if "fair" in bench:
        unfair = int((~bench["fair"]).sum())
        if unfair:
            print(f"[bench] dropping {unfair} row(s) whose wall/cpu left "
                  f"{WALL_CPU_BAND} -- not single-threaded or not alone",
                  flush=True)
        bench = bench[bench["fair"]]
    return _runtime_agg(bench, ["n", "implementation"])


def main(argv=None) -> int:
    import argparse

    import yaml

    ap = argparse.ArgumentParser(prog="sim.bench_stage2", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study-dir", type=Path, default=Path("results/study_c"))
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--reps-per-n", type=int, default=30,
                    help="cap on replicates timed per sample size (default 30). "
                         "Applied to --config too: a study config's `reps` are "
                         "its *study* replicate counts, which are one to two "
                         "orders of magnitude more than a benchmark needs")
    ap.add_argument("--max-load", type=float, default=2.0)
    ap.add_argument("--allow-busy", action="store_true",
                    help="skip the idle and thread-pin gates; smoke tests only, "
                         "the resulting timings are not comparable")
    a = ap.parse_args(argv)

    if a.config and a.config.exists():
        cfg = yaml.safe_load(a.config.read_text())
        sizes = cfg["sizes"]
        repeats = int(cfg.get("repeats", a.repeats))
    else:
        sizes = [{"n": 500, "reps": 30}, {"n": 1000, "reps": 30},
                 {"n": 2000, "reps": 15}]
        repeats = a.repeats

    # A study config's `reps` are its *study* replicate counts -- 500/500/150 for
    # study_c.yaml -- and timing all of them, each `repeats + 1` times on both
    # implementations, is ~19 hours rather than the ~1 the benchmark needs. The
    # cap makes `--config` mean "which sample sizes" and not "how many".
    if a.reps_per_n:
        sizes = [{**s, "reps": min(int(s["reps"]), a.reps_per_n)} for s in sizes]

    bench = run_bench(a.study_dir, sizes, repeats=repeats,
                      max_load=a.max_load, allow_busy=a.allow_busy)
    out_dir = a.study_dir / "_bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    bench.to_parquet(out_dir / "bench_stage2.parquet", index=False)
    (out_dir / "thread_pools.json").write_text(bench.attrs["thread_pools"])

    summ = summarise_bench(bench)
    summ.to_csv(out_dir / "bench_stage2_summary.csv", index=False)
    print(summ.to_string(index=False))

    peak = bench.loc[bench["implementation"] == "concrete", "max_threads"]
    if len(peak):
        print(f"\nconcrete peak live threads: {peak.max():.0f} "
              f"(context only; idle pool threads are counted too)")
    print(f"wall/cpu: median {bench['wall_over_cpu'].median():.3f}, "
          f"range [{bench['wall_over_cpu'].min():.3f}, "
          f"{bench['wall_over_cpu'].max():.3f}]  -- 1.0 means one thread, alone")
    print(f"rows passing the fairness band {WALL_CPU_BAND}: "
          f"{int(bench['fair'].sum())}/{len(bench)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
