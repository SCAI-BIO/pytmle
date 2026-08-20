"""Command-line entry point.

    python -m sim.run --config sim/configs/reference.yaml --reps 500 \
        --seed 20250301 --output-dir results/debug-study-a --cells C1

Run from the ``simulation-study`` directory. BLAS threading should be pinned to
one thread (the committed ``.vscode/launch.json`` does this), since parallelism
here is across replicates.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def _pin_blas() -> None:
    """Avoid oversubscription: replicates are the unit of parallelism."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sim.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="YAML design file")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--cells", nargs="*", default=None,
                    help="subset of cell names; default is all cells in the config")
    ap.add_argument("--reps", type=int, default=None,
                    help="override the replicate count (useful for pilots)")
    ap.add_argument("--seed", type=int, default=20250301)
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="workers; default is chosen per cell from n. The second "
                         "stage is memory-bandwidth bound, so throughput ceilings "
                         "near 8 workers and 20 costs ~2.4x the CPU for less "
                         "throughput (see runner.default_n_jobs)")
    ap.add_argument("--chunk", type=int, default=50,
                    help="replicates per parquet shard; also the resume granularity")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute shards that already exist")
    ap.add_argument("--summarise", action="store_true",
                    help="after running, print the performance and diagnostics tables")
    ap.add_argument("--report", action="store_true",
                    help="after running, write the figures and tables. Together with "
                         "the config this makes one command reproduce the whole study, "
                         "concrete included -- there is no separate merge step")
    ap.add_argument("--fig-dir", type=Path, default=None,
                    help="figures directory for --report (default <output-dir>/figures)")
    ap.add_argument("--table-dir", type=Path, default=None,
                    help="tables directory for --report (default <output-dir>/tables)")
    ap.add_argument("--truth-mc", type=int, default=10_000_000,
                    help="Monte Carlo draws for the closed-form truth")
    return ap


def main(argv=None) -> int:
    _pin_blas()
    args = build_parser().parse_args(argv)

    from .config import load_cells
    from .runner import run_cells

    cells = load_cells(args.config, only=args.cells, reps=args.reps)
    total = sum(c.reps for c in cells)
    print(f"[sim.run] {len(cells)} cell(s), {total} replicates -> {args.output_dir}",
          flush=True)
    from .runner import default_n_jobs
    for c in cells:
        j = args.n_jobs if args.n_jobs else default_n_jobs(c.n)
        print(f"  {c.name:<10s} config={c.config:<12s} n={c.n:<6d} reps={c.reps:<5d} "
              f"spec={c.spec.label} n_jobs={j:<3d} est={','.join(c.estimators)}", flush=True)

    run_cells(cells, args.output_dir, master_seed=args.seed, n_jobs=args.n_jobs,
              chunk=args.chunk, overwrite=args.overwrite)
    print("[sim.run] done", flush=True)

    if args.report:
        from .report_study_a import build

        fig = args.fig_dir or Path(args.output_dir) / "figures"
        tab = args.table_dir or Path(args.output_dir) / "tables"
        build(args.output_dir, fig, tab,
              summary_path=Path(args.output_dir) / "summary.parquet")
        print(f"[sim.run] report -> {fig}, {tab}", flush=True)

    if args.summarise:
        from .report import diagnostics, runtimes, summarise_dir

        pd.set_option("display.width", 200, "display.max_columns", 50)
        print("\n=== diagnostics ===")
        print(diagnostics(args.output_dir, cells=[c.name for c in cells]).to_string(index=False))
        print("\n=== second-stage runtime (targeted update only) ===")
        rt = runtimes(args.output_dir, cells=[c.name for c in cells])
        if rt.empty:
            print("(no timed implementations in this run)")
        else:
            print(rt.round(4).to_string(index=False))
        print("\n=== performance (risk difference) ===")
        s = summarise_dir(args.output_dir, cells=[c.name for c in cells],
                          n_mc=args.truth_mc)
        cols = ["cell", "estimator", "event", "time", "truth", "bias", "bias_mc_se",
                "emp_sd", "mean_se", "se_ratio", "coverage", "coverage_mc_se", "ci_width"]
        sub = s[s["estimand"] == "rd"][[c for c in cols if c in s.columns]]
        print(sub.sort_values(["cell", "event", "time", "estimator"]).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
