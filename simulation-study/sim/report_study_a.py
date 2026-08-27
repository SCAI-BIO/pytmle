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
    fig_dir: Path | str | None = None,
    table_dir: Path | str | None = None,
    summary_path: Path | str | None = None,
    estimand: str = "rd",
    event: int = 1,
    n_mc: int = 4_000_000,
) -> pd.DataFrame:
    """Every Study A deliverable, written **under `output_dir`** by default.

    The three paths used to default to flat `results/...` locations that ignored
    `output_dir`, so reporting two output directories overwrote one summary with
    the other's and `--output-dir` silently failed to move the figures.
    """
    output_dir = Path(output_dir)
    fig_dir = Path(fig_dir) if fig_dir is not None else output_dir / "figures"
    table_dir = Path(table_dir) if table_dir is not None else output_dir / "tables"
    summary_path = (Path(summary_path) if summary_path is not None
                    else output_dir / "summary.parquet")

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
    # Derived from --output-dir, matching `sim.run --report` (run.py:86-87).
    # These used to default to a flat `results/figures` / `results/tables`, so
    # reporting the same study through the two entry points wrote to two
    # different places and `--output-dir` silently did not move the figures.
    ap.add_argument("--fig-dir", default=None,
                    help="default <output-dir>/figures")
    ap.add_argument("--table-dir", default=None,
                    help="default <output-dir>/tables")
    ap.add_argument("--estimand", default="rd")
    ap.add_argument("--event", type=int, default=1)
    a = ap.parse_args(argv)
    fig = Path(a.fig_dir) if a.fig_dir else Path(a.output_dir) / "figures"
    tab = Path(a.table_dir) if a.table_dir else Path(a.output_dir) / "tables"
    s = build(a.output_dir, fig, tab, estimand=a.estimand, event=a.event)
    print(f"[sim.report_study_a] {len(s)} summary rows; "
          f"figures -> {fig}, tables -> {tab}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
