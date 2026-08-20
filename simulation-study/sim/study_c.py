"""Study C: estimator comparison and cross-package validation.

Every estimator here sees a **correctly specified** model. Specification is
Study A's subject; this study asks what the estimator itself contributes, and
what a practitioner gets from the conventional analysis instead.

Nuisances flow **R -> parquet -> everyone**. They are fitted once per replicate
in R (`CSC` for the cause-specific hazards, `coxph` for censoring, `glm` for the
propensity) and exported as full counterfactual cumulative hazards on the
replicate's own event-time grid. Fitting them separately in Python would confound
the estimator comparison with ties handling, centring and cross-fitting
differences between the two implementations -- which is the very thing the study
is trying to measure around.

Three tiers of sharing, and the report has to say which applies to which row:

    tier 1  byte-identical injected nuisances   PyTMLE (tmle/gcomp/aipw/ipw), concrete
    tier 2  identical fitted model objects      riskRegression::ate, conventional CSC
    tier 3  same model class, refit internally  AdjCuminc::adjDR

Agreement tolerances are tiered for the same reason. PyTMLE and concrete share
the discrete-hazard convention, so they should agree to ~1e-3 and that is the
primary gate; `riskRegression` uses the exact Aalen-Johansen product limit and
differs from a discrete plug-in *by construction*, so ~1e-2 there is a
consistency check, not a defect.

One command runs the whole study:

    python -m sim.study_c --config sim/configs/study_c.yaml --output-dir results/study_c

Each sample size passes through four stages -- export the replicates, fit the
nuisances once in R and run the R comparators, run concrete's second stage on
those same arrays, run PyTMLE's estimators on them -- and then the figures and
tables are written. Every stage is skipped when its output is already present,
so an interrupted run resumes rather than restarting. The individual stages
remain available as subcommands for debugging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .dgp import DGPParams, get_config, sample
from .runner import target_times_for
from .validate import initial_estimates_from_r, load_r_replicate

__all__ = ["export_replicates", "run_python_estimators", "collect_study_c"]

#: One DGP for the whole study, with no misspecification device switched on:
#: plain logistic assignment, no threshold term, no control resampling. The
#: comparison is between estimators, so nothing in the data should favour one.
CONFIG = "base"


def export_replicates(
    out_dir: Path | str,
    n: int,
    reps: int,
    config: str = CONFIG,
    master_seed: int = 20250901,
) -> List[Path]:
    """Write the replicate datasets and the frozen target times.

    Seeds are spawned from one ``SeedSequence`` so that replicate *r* is the same
    dataset for every consumer, and so that a re-export reproduces it exactly.
    Target times come from ``target_times_for``, computed once from a large draw
    rather than per replicate -- data-dependent targets would move the estimand
    between replicates and make bias meaningless.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = get_config(config)

    taus = target_times_for(p)
    pd.DataFrame({"time": list(taus)}).to_parquet(out_dir / "taus.parquet", index=False)
    pd.DataFrame({"n": [n], "reps": [reps], "config": [config],
                  "master_seed": [master_seed]}).to_parquet(
        out_dir / "manifest.parquet", index=False)

    children = np.random.SeedSequence(
        [master_seed, n, int.from_bytes(config.encode()[:8].ljust(8, b"\0"), "little")
         % (2 ** 31)]
    ).spawn(reps)

    written = []
    for i in range(reps):
        sm = sample(n, p, np.random.default_rng(children[i]))
        path = out_dir / f"rep{i:03d}_data.parquet"
        sm.df.to_parquet(path, index=False)
        written.append(path)
    return written


def _load_taus(out_dir: Path) -> List[float]:
    return [float(t) for t in pd.read_parquet(out_dir / "taus.parquet")["time"]]


def _load_replicate(stub: Path) -> Dict:
    """Read one replicate in Study C's column convention.

    ``validate.load_r_replicate`` reads the reference study's columns
    (``obs_time``/``status``/``treatment``); here the data comes from our own DGP
    and is already tidy, so only the nuisance matrices need loading.
    """
    df = pd.read_parquet(f"{stub}_data.parquet")[
        ["event_time", "event_indicator", "group"]]
    grid = pd.read_parquet(f"{stub}_grid.parquet")["time"].to_numpy(dtype=float)
    ps1 = pd.read_parquet(f"{stub}_ps.parquet")["ps1"].to_numpy(dtype=float)
    mats = {
        nm: pd.read_parquet(f"{stub}_{nm}.parquet").to_numpy(dtype=float)
        for nm in ("H1_1", "H1_0", "H2_1", "H2_0", "HC_1", "HC_0")
    }
    return {"df": df, "grid": grid, "ps1": ps1, **mats}


def _one_python_replicate(args) -> pd.DataFrame:
    stub, taus, which, min_nuisance = args
    from .estimators import run_all

    stub = Path(stub)
    rep_id = int(stub.name[len("rep"):])
    try:
        rep = _load_replicate(stub)
        ie = initial_estimates_from_r(rep)
        res = run_all(rep["df"], ie, list(taus), target_events=[1, 2],
                      which=which, min_nuisance=min_nuisance)
    except Exception as exc:  # one bad replicate must not cost the rest
        res = pd.DataFrame([{"estimator": None, "estimand": None, "event": np.nan,
                             "time": np.nan, "group": np.nan, "est": np.nan,
                             "error": f"{type(exc).__name__}: {exc}"}])
    res = res.copy()
    res["rep"] = rep_id
    res["tier"] = 1
    res["source"] = "pytmle"
    return res


def run_python_estimators(
    out_dir: Path | str,
    min_nuisance: float = 0.01,
    which: Sequence[str] = ("tmle", "gcomp", "aipw", "ipw"),
    reps: Optional[int] = None,
    n_jobs: int = 8,
) -> pd.DataFrame:
    """Tier 1: PyTMLE's four estimators on the nuisances R fitted.

    Parallel over replicates, like everything else here. The second stage is
    memory-bandwidth bound, so throughput ceilings well below the core count --
    see ``runner.default_n_jobs`` for the measurements behind that.
    """
    from concurrent.futures import ProcessPoolExecutor

    out_dir = Path(out_dir)
    taus = _load_taus(out_dir)
    stubs = sorted(out_dir.glob("rep*_grid.parquet"))
    if reps is not None:
        stubs = stubs[:reps]
    tasks = [(str(g)[: -len("_grid.parquet")], taus, tuple(which), min_nuisance)
             for g in stubs]

    if n_jobs <= 1:
        rows = [_one_python_replicate(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            rows = list(ex.map(_one_python_replicate, tasks))
    return pd.concat(rows, ignore_index=True)


def collect_study_c(out_dir: Path | str) -> pd.DataFrame:
    """Stack every implementation's estimates for one (n) directory.

    PyTMLE's rows come from ``python_estimates.parquet``, concrete's from
    ``concrete_estimates.parquet`` (written by ``R/run_concrete_injected.R``) and
    the R comparators' from ``r_estimates.parquet``. Each keeps its tier so the
    agreement gates can be applied at the right tolerance.
    """
    out_dir = Path(out_dir)
    n = int(pd.read_parquet(out_dir / "manifest.parquet")["n"].iloc[0])
    parts: List[pd.DataFrame] = []

    py = out_dir / "python_estimates.parquet"
    if py.exists():
        parts.append(pd.read_parquet(py))

    con = out_dir / "concrete_estimates.parquet"
    if con.exists():
        c = pd.read_parquet(con)
        # concrete emits both its targeted and its plug-in estimate in one frame
        # (`Estimator` in {tmle, gcomp}), so the label has to come from that
        # column -- tagging every row "tmle (concrete)" would silently score the
        # plug-in as the targeted estimator.
        c = c.rename(columns={"Event": "event", "Time": "time",
                              "Pt Est": "est", "CI Low": "ci_lo", "CI Hi": "ci_hi",
                              "steps": "tmle_steps"})
        c = c[c["Estimand"].astype(str).str.strip() == "Risk Diff"].copy()
        c["estimand"] = "rd"
        c["estimator"] = c["Estimator"].astype(str) + " (concrete)"
        c["tier"] = 1
        c["source"] = "concrete"
        parts.append(c.drop(columns=["Estimand", "Estimator", "Intervention"]))

    r = out_dir / "r_estimates.parquet"
    if r.exists():
        parts.append(pd.read_parquet(r))

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["n"] = n
    return out


#: Stage outputs. A stage is skipped when its marker exists, which is what makes
#: the run resumable at the granularity work is actually lost at -- the R
#: nuisance fit for one sample size is tens of minutes.
_MARKERS = {
    "export": "manifest.parquet",
    "nuisances": "r_estimates.parquet",
    "concrete": "concrete_estimates.parquet",
    "estimators": "python_estimates.parquet",
}


def _shards(reps: int, n_jobs: int) -> List[tuple]:
    """Contiguous replicate ranges, one per worker."""
    n_jobs = max(1, min(n_jobs, reps))
    edges = np.linspace(0, reps, n_jobs + 1).astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _rscript_parallel(script: str, out_dir: Path, stem: str, reps: int,
                      extra: List[str], n_jobs: int, label: str) -> None:
    """Run one R script over sharded replicate ranges, then combine.

    R is single-threaded and both scripts are a plain loop over replicates, so
    the ranges are independent and the only shared state is the output
    directory -- which is why each shard writes its own file. Combining happens
    here rather than in R so that a shard that dies leaves the others' work
    intact and visible.
    """
    import subprocess

    from .concrete_bridge import RSCRIPT

    ranges = _shards(reps, n_jobs)
    procs = []
    for k, (lo, hi) in enumerate(ranges):
        args = [RSCRIPT, script, "--dir", str(out_dir), "--from", str(lo),
                "--to", str(hi), "--shard", f"{k:02d}", *extra]
        procs.append((k, lo, hi, subprocess.Popen(
            args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)))

    failed = []
    for k, lo, hi, proc in procs:
        _, err = proc.communicate()
        if proc.returncode != 0:
            failed.append(f"  shard {k} (reps {lo}-{hi}): {err.strip()[-400:]}")

    parts = sorted(out_dir.glob(f"{stem}_*.parquet"))
    if not parts:
        raise RuntimeError(f"{label} produced no output:\n" + "\n".join(failed))
    if failed:
        print(f"  [{out_dir.name}] {label}: {len(failed)} of {len(ranges)} shards "
              f"failed; continuing with the rest", flush=True)
        for f in failed:
            print(f, flush=True)
    pd.concat([pd.read_parquet(f) for f in parts],
              ignore_index=True).to_parquet(out_dir / f"{stem}.parquet", index=False)
    for f in parts:
        f.unlink()


def run_size(out_dir: Path | str, n: int, reps: int, config: str = CONFIG,
             min_nuisance: float = 0.01, master_seed: int = 20250901,
             overwrite: bool = False, n_jobs: int = 8) -> Path:
    """Every stage for one sample size, skipping whatever is already done."""
    out_dir = Path(out_dir)

    def _todo(stage: str) -> bool:
        return overwrite or not (out_dir / _MARKERS[stage]).exists()

    if _todo("export"):
        export_replicates(out_dir, n, reps, config=config, master_seed=master_seed)
        print(f"  [{out_dir.name}] exported {reps} replicates", flush=True)

    if _todo("nuisances"):
        print(f"  [{out_dir.name}] fitting nuisances in R and running the R "
              f"comparators, {n_jobs} shards ...", flush=True)
        _rscript_parallel("R/study_c_nuisances.R", out_dir, "r_estimates", reps,
                          [], n_jobs, "study_c_nuisances.R")

    if _todo("concrete"):
        print(f"  [{out_dir.name}] concrete second stage, {n_jobs} shards ...",
              flush=True)
        _rscript_parallel("R/run_concrete_injected.R", out_dir,
                          "concrete_estimates", reps,
                          ["--reps", str(reps), "--min-nuisance", str(min_nuisance)],
                          n_jobs, "run_concrete_injected.R")

    if _todo("estimators"):
        res = run_python_estimators(out_dir, min_nuisance=min_nuisance,
                                    n_jobs=n_jobs)
        res.to_parquet(out_dir / _MARKERS["estimators"], index=False)
        bad = int(res["estimator"].isna().sum())
        print(f"  [{out_dir.name}] PyTMLE estimators: {len(res)} rows"
              + (f", {bad} replicate failures" if bad else ""), flush=True)
    return out_dir


def run_study(config_path: Path | str, output_dir: Path | str,
              only_n: Optional[Sequence[int]] = None,
              reps: Optional[int] = None, overwrite: bool = False,
              n_jobs: int = 8) -> List[Path]:
    """Drive every sample size in a config, then hand back the directories."""
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text())
    d = cfg.get("defaults", {})
    output_dir = Path(output_dir)
    dirs: List[Path] = []
    for size in cfg["sizes"]:
        n = int(size["n"])
        if only_n and n not in only_n:
            continue
        r = int(reps or size.get("reps", d.get("reps", 500)))
        print(f"[sim.study_c] n = {n}, {r} replicates", flush=True)
        dirs.append(run_size(output_dir / f"n{n}", n, r,
                             config=size.get("config", d.get("config", CONFIG)),
                             min_nuisance=float(d.get("min_nuisance", 0.01)),
                             overwrite=overwrite, n_jobs=n_jobs))
    return dirs


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="sim.study_c", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None,
                    help="YAML design file; runs every stage for every sample size")
    ap.add_argument("--output-dir", type=Path, default=Path("results/study_c"))
    ap.add_argument("--only-n", type=int, nargs="*", default=None,
                    help="restrict to these sample sizes")
    ap.add_argument("--reps", type=int, default=None, help="override replicate count")
    ap.add_argument("--n-jobs", type=int, default=8,
                    help="parallel shards for the two R stages and the Python "
                         "estimator loop. All three are loops over independent "
                         "replicates; R is single-threaded, so this is the main "
                         "lever on wall time")
    ap.add_argument("--overwrite", action="store_true",
                    help="redo stages whose output already exists")
    ap.add_argument("--no-report", action="store_true",
                    help="skip the figures and tables at the end")
    sub = ap.add_subparsers(dest="cmd")

    e = sub.add_parser("export", help="write replicate datasets and target times")
    e.add_argument("--dir", required=True)
    e.add_argument("--n", type=int, required=True)
    e.add_argument("--reps", type=int, default=500)
    e.add_argument("--config", default=CONFIG)
    e.add_argument("--seed", type=int, default=20250901)

    r = sub.add_parser("run", help="run PyTMLE's estimators on R's nuisances")
    r.add_argument("--dir", required=True)
    r.add_argument("--reps", type=int, default=None)
    r.add_argument("--min-nuisance", type=float, default=0.01)

    a = ap.parse_args(argv)

    if a.cmd is None:
        if a.config is None:
            ap.error("give --config to run the study, or a subcommand for one stage")
        dirs = run_study(a.config, a.output_dir, only_n=a.only_n, reps=a.reps,
                         overwrite=a.overwrite, n_jobs=a.n_jobs)
        print("[sim.study_c] all stages done", flush=True)
        if not a.no_report:
            from .study_c_report import build

            build(dirs, a.output_dir)
            print(f"[sim.study_c] report -> {a.output_dir}", flush=True)
        return 0

    if a.cmd == "export":
        paths = export_replicates(a.dir, a.n, a.reps, config=a.config,
                                  master_seed=a.seed)
        print(f"[sim.study_c] exported {len(paths)} replicates to {a.dir}")
    else:
        res = run_python_estimators(a.dir, min_nuisance=a.min_nuisance, reps=a.reps)
        out = Path(a.dir) / "python_estimates.parquet"
        res.to_parquet(out, index=False)
        bad = int(res["estimator"].isna().sum())
        print(f"[sim.study_c] {len(res)} rows -> {out}"
              + (f"  ({bad} replicate failures)" if bad else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
