"""Regenerate every Study A deliverable from the stored shards.

Both implementations come out of one place. ``sim.run`` writes PyTMLE's
per-replicate rows as parquet shards and concrete's beside them as
``concrete.parquet``, and ``report.collect`` loads them together -- so nothing
here needs to know that a second implementation exists. It did once, and the
special-casing was the thing that made the study awkward to reproduce.

    python -m sim.report_study_a --output-dir results/study_a
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .plots import make_all
from .report import diagnostics, runtimes, summarise_dir
from .tables import write_tables

__all__ = ["build"]

#: Both implementations of the targeted update. Everything else -- the plug-in,
#: the one-step, IPW -- rides along on work already done and has no separately
#: meaningful second-stage cost.
IMPLEMENTATIONS = ("tmle", "tmle (concrete)")


def build(
    output_dir: Path | str = "results/study_a",
    fig_dir: Path | str = "results/figures",
    table_dir: Path | str = "results/tables",
    summary_path: Path | str = "results/study_a_summary.parquet",
    estimand: str = "rd",
    event: int = 1,
    n_mc: int = 4_000_000,
) -> pd.DataFrame:
    output_dir = Path(output_dir)

    summary = summarise_dir(output_dir, n_mc=n_mc)
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(summary_path, index=False)

    diag = diagnostics(output_dir)
    rt = runtimes(output_dir, implementations=IMPLEMENTATIONS)

    Path(table_dir).mkdir(parents=True, exist_ok=True)
    write_tables(summary, Path(table_dir), estimand=estimand, event=event,
                 diagnostics=diag, runtimes=rt)
    make_all(summary, Path(fig_dir), estimand=estimand, event=event, runtimes=rt)
    return summary


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="sim.report_study_a", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="results/study_a")
    ap.add_argument("--fig-dir", default="results/figures")
    ap.add_argument("--table-dir", default="results/tables")
    ap.add_argument("--estimand", default="rd")
    ap.add_argument("--event", type=int, default=1)
    a = ap.parse_args(argv)
    s = build(a.output_dir, a.fig_dir, a.table_dir, estimand=a.estimand,
              event=a.event)
    print(f"[sim.report_study_a] {len(s)} summary rows; "
          f"figures -> {a.fig_dir}, tables -> {a.table_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
