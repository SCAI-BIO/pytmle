"""Study B: Wald against bootstrap intervals, on identical replicates.

    python -m sim.study_b --config sim/configs/study_b.yaml --output-dir results/study_b

Every interval procedure is computed for the *same* replicate from the *same*
resamples, so differences between them isolate the interval construction rather
than the data or the resampling. That pairing matters: the differences of
interest (93 % against 95 %) are the same size as the marginal Monte Carlo SE at
these replicate counts, and paired contrasts have far smaller SEs than marginal
ones.

Procedures, per (estimand, event, tau, group):

    wald          Pt Est +- z * SE, the package default
    logwald       risk ratio, on the log scale
    logitwald     risk, on the logit scale
    atanhwald     risk difference, on the Fisher-z scale
                  -- the three transformed scales cost nothing (they are
                  functions of the stored estimate and SE) and each is confined
                  to its estimand's support, where a symmetric interval is not:
                  at F = 0.005 the Wald lower bound is -0.0048, an impossible
                  cumulative incidence. If one of these restores coverage, the
                  finding is "the interval is on the wrong scale", not "the
                  asymptotics fail" -- a one-line fix rather than a bootstrap.
    pct_*         percentile
    basic_*       reverse-percentile
    bca_*         bias-corrected and accelerated

Each of the three constructions is emitted under each of four filtering rules, so
the procedure label is `{construction}_{filter}` -- `pct_all`, `bca_all`,
`basic_convfilter`, and so on. They come from one call to
`intervals_from_draws`, so the full cross costs nothing at run time, and it is
the only way to read construction and filter apart: `basic` and `bca` were once
emitted under the convergence filter alone, which made their weak showing a
measurement of the filter rather than of the construction.

The four filtering rules exist to attribute coverage loss to the bootstrap's
failure modes. Draws are tagged rather than filtered, so the interval can be
recomputed with and without the affected draws:

    pct_all         no filtering -- PyTMLE's behaviour once the per-target
                    `Converged` filter was removed from `pytmle/bootstrap.py`
    pct_convfilter  mode-2 draws dropped -- PyTMLE's behaviour *before* that
                    removal, retained here as the historical comparator
    pct_dropmode1   mode-1 draws dropped
    pct_strict      both dropped

`pct_convfilter - pct_all` is mode 2's effect on coverage; `pct_dropmode1 -
pct_all` is mode 1's. See BOOTSTRAP_FAILURES.md.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .dgp import get_config, sample
from .nuisance import Spec, build
from .runner import target_times_for

__all__ = ["BCell", "run_cell_b", "run_study_b", "progress_b"]

ALPHA = 0.05

#: The filtering rules, as (label, drop mode 1, drop mode 2).
FILTERS = [("pct_all", False, False), ("pct_convfilter", False, True),
           ("pct_dropmode1", True, False), ("pct_strict", True, True)]

#: Keys a cell may set. Unknown keys raise rather than being ignored -- the
#: previous loader silently dropped anything it did not recognise, so a per-cell
#: `min_nuisance:` looked like it worked and did nothing.
_CELL_KEYS = {
    "name", "n", "arm", "reps", "config", "n_bootstrap", "min_nuisance",
    "max_updates", "tau_quantiles", "target_times", "params_override",
    "q_arm", "pi_arm", "g_arm", "seed_key", "axis", "level",
}

#: DGPParams fields that must be numpy arrays; YAML gives lists.
_ARRAY_FIELDS = {"gamma", "alpha", "beta", "theta", "delta", "eta", "beta_c"}


def _coerce_override(ov: Dict) -> Dict:
    """YAML lists -> numpy arrays for the array-valued DGP fields."""
    return {k: (np.asarray(v, dtype=float) if k in _ARRAY_FIELDS else v)
            for k, v in (ov or {}).items()}


def _check_seed_keys(cells: Sequence["BCell"]) -> None:
    """Distinct seed keys must not collide in their first 8 bytes.

    `run_cell_b` hashes only that prefix, so two cells sharing it draw identical
    datasets. That is wanted between an oracle and a correct arm of the same
    condition -- they deliberately share a key -- and silently wrong between two
    different conditions. Checked here so it fails loudly at load time.
    """
    seen: Dict[bytes, str] = {}
    for c in cells:
        key = c.seed_key or c.name
        pref = key.encode()[:8].ljust(8, b"\0")
        if pref in seen and seen[pref] != key:
            raise ValueError(
                f"seed keys {seen[pref]!r} and {key!r} share their first 8 bytes, "
                f"so those cells would silently draw identical datasets; "
                f"rename one or give them the same seed_key deliberately")
        seen[pref] = key


@dataclass
class BCell:
    name: str
    n: int
    arm: str                       # "oracle" or "correct"; per-component override below
    reps: int
    config: str = "base"
    n_bootstrap: int = 0           # 0 = Wald-only cell
    min_nuisance: float = 0.01
    max_updates: int = 200
    tau_quantiles: Sequence[float] = (0.10, 0.50, 0.85)

    #: DGP fields overridden for this cell, applied via ``DGPParams.with_``. This
    #: is how every stress level is expressed -- scaled ``gamma`` for overlap,
    #: lowered ``alpha[0]`` for rarity, scaled censoring coefficients -- without
    #: a named entry in ``dgp.CONFIGS`` per level. Mirrors ``Cell.params_override``.
    params_override: Dict = field(default_factory=dict)

    #: Explicit target times, overriding ``tau_quantiles``. Required whenever
    #: cells of one design use *different* DGPs that must be compared at the same
    #: tau: ``target_times_for`` reads quantiles of the observed event-time
    #: distribution, which rarity and censoring both shift, so leaving it to the
    #: quantile rule would silently evaluate each condition at a different clock
    #: time and confound the axis with tau. Mirrors ``Cell.target_times``.
    target_times: Optional[Sequence[float]] = None

    #: Per-component nuisance specification. ``None`` inherits ``arm``; setting
    #: one lets a cell hold, say, Q and pi correct while varying G.
    q_arm: Optional[str] = None
    pi_arm: Optional[str] = None
    g_arm: Optional[str] = None

    #: What the replicate seed stream is derived from; defaults to ``name``.
    #:
    #: Seeds hash only the *first 8 bytes* of the key (see ``run_cell_b``), which
    #: made pairing accidental: ``BS12_n250_oracle`` and ``BS12_n250_correct``
    #: happened to share a dataset while ``W_n250_oracle`` and ``W_n250_correct``
    #: happened not to. Setting this to the DGP identity -- config, n and every
    #: overridden parameter, but *not* the arm, ``min_nuisance`` or the number of
    #: resamples -- makes each of those contrasts exactly paired by construction,
    #: which matters because the differences of interest are the size of the
    #: marginal MC-SE. Existing cells pin it to their own name so their completed
    #: shards stay valid.
    #:
    #: Only the first 8 bytes are hashed, and that formula is kept rather than
    #: fixed so the completed cells still reproduce -- so two keys sharing a
    #: prefix silently share a dataset. A design-gate test asserts the keys in a
    #: config have distinct 8-byte prefixes, which turns the hazard into a loud
    #: failure instead of a quiet one.
    seed_key: Optional[str] = None

    #: Which hypothesis this cell belongs to and where on its ladder it sits.
    #: Carried into meta.json purely so the report can group by axis rather than
    #: parsing cell names.
    axis: str = "base"
    level: str = "base"

    def spec(self) -> Spec:
        return Spec(Q=self.q_arm or self.arm,
                    pi=self.pi_arm or self.arm,
                    G=self.g_arm or self.arm)

    def dgp_params(self):
        """The DGP for this cell: named config plus any per-cell overrides."""
        p = get_config(self.config)
        return p.with_(**self.params_override) if self.params_override else p

    def resolved_target_times(self) -> List[float]:
        if self.target_times is not None:
            return [float(t) for t in self.target_times]
        return target_times_for(self.dgp_params(), self.tau_quantiles)


def _main_fit(sm, ie, taus, events, min_nuisance, max_updates) -> pd.DataFrame:
    """Point estimates, EIC standard errors, and the IC itself (for BCa)."""
    from pytmle import PyTMLE

    model = PyTMLE(sm.df, target_times=list(taus), initial_estimates=ie,
                   g_comp=False, evalues_benchmark=False, verbose=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(min_nuisance=min_nuisance, max_updates=max_updates)
    rows = []
    for estimand in ("rd", "rr", "risk"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = model.predict(estimand if estimand != "risk" else "risks")
        except Exception:
            continue
        p = p.copy()
        p["type"] = estimand
        if "Group" not in p:
            p["Group"] = -1
        rows.append(p)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out, model


def _ic_for_bca(model, key_1: int = 1, key_0: int = 0) -> Dict:
    """Per-subject influence values, keyed by (type, Event, Time, Group).

    BCa's acceleration is a jackknife over observations in the textbook, which
    would cost `n` extra second-stage fits per replicate. For a smooth functional
    the jackknife influence values are asymptotically the influence function,
    which the fit has already produced, so it is taken from there instead.
    """
    out = {}
    try:
        ue = model._updated_estimates
        ic1 = ue[key_1].ic.set_index(["ID", "Event", "Time"])["IC"]
        ic0 = ue[key_0].ic.set_index(["ID", "Event", "Time"])["IC"]
        d = (ic1 - ic0).reset_index()
        for (ev, t), g in d.groupby(["Event", "Time"]):
            out[("rd", int(ev), float(t), -1)] = g["IC"].to_numpy()
    except Exception:
        pass
    return out


def _condition_diagnostics(sm, ie, nd, p, taus, min_nuisance) -> Dict:
    """What this replicate's stress level actually looked like.

    A stress axis has to be shown to vary what it claims, per replicate and not
    only in the design tables -- FINDINGS 4 records what happens when a study
    reports coverage without checking which replicates produced it. Everything
    here already exists in the runner path; none of it was ever wired into
    Study B.

    The two truncation rates are reported separately because the `concrete`
    paper asks for both: the fraction of nuisance *weights* at the floor, and the
    fraction of *subjects* with any weight at the floor. They differ sharply --
    a few subjects truncated at every time point is a very different situation
    from many subjects truncated at the last one.
    """
    out: Dict = {
        "ps_mae": nd.ps_mae, "ps_min": nd.ps_min, "ps_max": nd.ps_max,
        "cumhaz_mae": nd.cumhaz_mae, "n_failed_fits": int(nd.n_failed_fits),
        "censored_frac": float((sm.event_indicator == 0).mean()),
        "n_times": int(len(np.unique(sm.event_times))),
        "tmle_converged": False, "tmle_steps": -1,
    }
    for j in range(1, p.n_causes + 1):
        out[f"n_events_{j}"] = int((sm.event_indicator == j).sum())

    # realised overlap, as percentiles rather than min/max: FINDINGS 6 notes that
    # an extreme-value statistic describes a tail that n never samples
    ps = np.asarray(ie[1].propensity_scores, dtype=float)
    two_sided = np.minimum(ps, 1.0 - ps)
    out["ps_p01"] = float(np.quantile(two_sided, 0.01))
    out["frac_ps_below_05"] = float(np.mean(two_sided < 0.05))

    # pi * G(tau-) at the last target time, the quantity min_nuisance floors
    try:
        grid = np.asarray(ie[1].times, dtype=float)
        k = int(np.searchsorted(grid, max(taus), side="right")) - 1
        k = max(k, 0)
        piG = np.concatenate([
            np.asarray(ie[key].propensity_scores, dtype=float)
            * np.asarray(ie[key].censoring_survival_function, dtype=float)[:, k]
            for key in (1, 0)])
        floor = min_nuisance if min_nuisance is not None else 0.0
        out["piG_p01"] = float(np.quantile(piG, 0.01))
        out["frac_weights_truncated"] = float(np.mean(piG < floor))
        n = len(sm.event_indicator)
        subj = (piG < floor).reshape(2, n).any(axis=0)
        out["frac_subjects_truncated"] = float(subj.mean())
    except Exception:
        out["piG_p01"] = np.nan
        out["frac_weights_truncated"] = np.nan
        out["frac_subjects_truncated"] = np.nan
    return out


def _one_rep_b(args) -> tuple:
    """One replicate: `(interval rows, raw bootstrap draws)`.

    The draws are returned rather than discarded so that a future interval
    construction -- studentized, a different alpha, a different filter -- can be
    computed from stored output instead of costing another full run. Only the
    first element feeds the report; the second is archived beside the shard.
    """
    cell, taus, seed_state, rep = args
    from .bootstrap_ci import (atanh_wald_interval, bootstrap_draws,
                               intervals_from_draws, log_wald_interval,
                               logit_wald_interval, wald_interval)

    rng = np.random.default_rng(seed_state)
    p = cell.dgp_params()
    events = list(range(1, p.n_causes + 1))
    t0 = time.time()
    cond: Dict = {}
    draws_out = pd.DataFrame()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm = sample(cell.n, p, rng)
            # runner._one_rep applies this guard and study_b never did. It matters
            # once tau is pinned across DGPs: a rare-event or heavily-censored
            # replicate can end before the last target time, and extrapolating
            # past the observed support would be a silent fabrication.
            if float(sm.event_times.max()) < max(taus):
                raise RuntimeError(
                    f"follow-up ends at {sm.event_times.max():.4g} "
                    f"before the last target time {max(taus):.4g}")
            ie, nd = build(sm, cell.spec(), return_diagnostics=True)
            cond = _condition_diagnostics(sm, ie, nd, p, taus, cell.min_nuisance)
        est, model = _main_fit(sm, ie, taus, events, cell.min_nuisance,
                               cell.max_updates)
        if est.empty:
            raise RuntimeError("main fit produced no estimates")
        cond["tmle_converged"] = bool(getattr(model, "has_converged", False))
        cond["tmle_steps"] = int(getattr(model, "step_num", -1))
    except Exception as exc:
        return pd.DataFrame([{"cell": cell.name, "rep": rep, "n": cell.n,
                              "arm": cell.arm, "procedure": None,
                              "error": f"{type(exc).__name__}: {exc}",
                              "seconds": time.time() - t0, **cond}]), draws_out

    rows: List[Dict] = []

    def _emit(proc, typ, ev, tt, grp, point, lo, hi, **extra):
        rows.append(dict(cell=cell.name, rep=rep, n=cell.n, arm=cell.arm,
                         procedure=proc, type=typ, event=int(ev), time=float(tt),
                         group=int(grp), est=float(point) if point is not None else np.nan,
                         ci_lo=lo, ci_hi=hi, error=None, **extra))

    # --- Wald, from the main fit -------------------------------------------
    for _, r in est.iterrows():
        typ, ev, tt = r["type"], r["Event"], r["Time"]
        grp = r.get("Group", -1)
        point, se = r.get("Pt Est", np.nan), r.get("SE", np.nan)
        lo, hi = wald_interval(point, se, ALPHA) if np.isfinite(se) else (np.nan, np.nan)
        _emit("wald", typ, ev, tt, grp, point, lo, hi, se=se)
        # Transformed scales, one per estimand's natural support. Free: they are
        # functions of the stored point estimate and SE, so they add no fitting.
        # Where the estimand sits near its boundary a symmetric interval puts a
        # bound outside the parameter space; these cannot.
        if typ == "rr":
            _emit("logwald", typ, ev, tt, grp, point,
                  *log_wald_interval(point, se, ALPHA), se=se)
        elif typ == "risk":
            _emit("logitwald", typ, ev, tt, grp, point,
                  *logit_wald_interval(point, se, ALPHA), se=se)
        elif typ == "rd":
            _emit("atanhwald", typ, ev, tt, grp, point,
                  *atanh_wald_interval(point, se, ALPHA), se=se)

    diag = {"mode1": 0, "mode3": 0, "n_usable": np.nan, "first_error": None,
            "boot_seconds": np.nan, "median_steps": np.nan, **cond}

    if cell.n_bootstrap > 0:
        tb = time.time()
        bd = bootstrap_draws(ie, sm.event_times, sm.event_indicator, taus, events,
                             n_bootstrap=cell.n_bootstrap,
                             rng=np.random.default_rng(seed_state.spawn(1)[0]),
                             n_jobs=1, min_nuisance=cell.min_nuisance,
                             max_updates=cell.max_updates)
        diag.update(mode1=bd.mode1_nonconverged, mode3=bd.mode3_errors,
                    n_usable=bd.n_usable, first_error=bd.first_error,
                    boot_seconds=time.time() - tb,
                    median_steps=float(np.median(bd.steps)) if bd.steps else np.nan)
        ics = _ic_for_bca(model)
        d = bd.draws
        if len(d):
            draws_out = d.copy()
            draws_out["cell"] = cell.name
            draws_out["rep"] = rep
            draws_out["n"] = cell.n
            draws_out["arm"] = cell.arm
            pt = {(r["type"], int(r["Event"]), float(r["Time"]),
                   int(r.get("Group", -1))): float(r["Pt Est"])
                  for _, r in est.iterrows()}
            for (typ, ev, tt, grp), g in d.groupby(
                    ["type", "Event", "Time", "Group"]):
                key = (typ, int(ev), float(tt), int(grp))
                point = pt.get(key, np.nan)
                for label, drop1, drop2 in FILTERS:
                    sub = g
                    if drop1:
                        sub = sub[sub["loop_converged"]]
                    if drop2:
                        sub = sub[sub["Converged"]]
                    iv = intervals_from_draws(sub["Pt Est"].to_numpy(), point,
                                              ALPHA, ics.get(key))
                    eff = int(sub["boot"].nunique())
                    # All three constructions under *every* filter.
                    # `intervals_from_draws` computes them together, so this is
                    # free at run time -- and previously `basic` and `bca` were
                    # emitted only under the convergence filter, which confounded
                    # the interval construction with the filter. Their weak
                    # showing at OV4 (0.493 and 0.708 against the percentile's
                    # 0.960) measured the filter, not the construction.
                    suffix = label[len("pct_"):]
                    for kind in ("percentile", "basic", "bca"):
                        name = ("pct" if kind == "percentile" else kind)
                        _emit(f"{name}_{suffix}", typ, ev, tt, grp, point,
                              *iv[kind], eff_b=eff)

    out = pd.DataFrame(rows)
    for k, v in diag.items():
        out[k] = v
    out["seconds"] = time.time() - t0
    return out, draws_out


def _shard_is_intact(path: Path) -> bool:
    """Can this shard actually be read back?

    A shard is skipped on resume because the file exists, so a file that exists
    but is unreadable would be skipped forever and then crash the report. That
    is exactly what a kill during `to_parquet` used to leave behind. Writes are
    now atomic (below), so this is the belt to that braces -- it also catches a
    shard truncated by a full disk or an unclean unmount.

    Only the footer is read, not the data, so checking a completed study costs
    milliseconds.
    """
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows > 0
    except Exception:
        return False


#: What `chunk` was before it became cell-dependent. Any shard written by the
#: earlier code used this, and must keep using it.
_LEGACY_CHUNK = 25


def _chunk_for(cell: BCell, chunk: int, shard_dir: Optional[Path] = None,
               n_jobs: int = 8) -> int:
    """Replicates per shard, bounded by what a restart may throw away.

    Resume granularity is the shard, so an interrupted shard is redone from
    scratch. At 25 replicates that is seconds for a Wald cell but nearly two
    hours for a bootstrap cell at a stressed level -- one replicate there is
    100 resamples of a fit that already takes 20 s. Bootstrap cells therefore
    get much smaller shards: more files, but a machine restart costs minutes
    instead of most of an evening.

    **The chunk of a cell that already has shards is never changed.** Shard `i`
    means replicates `[i*chunk, (i+1)*chunk)`, so re-chunking silently redefines
    what every existing file contains: switching 25 -> 5 on a finished 250-rep
    cell would read its 10 shards as replicates 0-49, then recompute 50-249 that
    it already holds, and the duplicates would inflate every downstream count.
    So the chunk is persisted in meta.json, and an older cell whose meta predates
    the field falls back to the legacy value.
    """
    if shard_dir is not None and any(shard_dir.glob("shard_*.parquet")):
        meta = shard_dir / "meta.json"
        if meta.exists():
            try:
                prev = json.loads(meta.read_text()).get("chunk")
                if prev:
                    return int(prev)
            except Exception:
                pass
        return _LEGACY_CHUNK
    # A shard's *wall* time is one replicate whenever `chunk <= n_jobs`, since
    # the replicates in it run concurrently. So shrinking the chunk below the
    # worker count costs throughput and buys no reduction in restart loss
    # whatsoever -- it just leaves workers idle. Measured the hard way: B_OV4 at
    # chunk 5 with --n-jobs 10 ran on 5 of 10 workers for 16 h.
    #
    # The right size is therefore exactly `n_jobs`: every worker busy, and a
    # restart costs one replicate's work rather than a chunk of them.
    if cell.n_bootstrap > 0:
        return max(1, min(n_jobs, cell.reps))
    return chunk


def run_cell_b(cell: BCell, output_dir: Path, master_seed: int = 20250301,
               n_jobs: int = 8, chunk: int = 25, overwrite: bool = False) -> Path:
    output_dir = Path(output_dir)
    shard_dir = output_dir / cell.name
    shard_dir.mkdir(parents=True, exist_ok=True)

    taus = cell.resolved_target_times()
    step = _chunk_for(cell, chunk, shard_dir, n_jobs=n_jobs)
    (shard_dir / "meta.json").write_text(json.dumps(
        {"cell": cell.name, "n": cell.n, "arm": cell.arm, "reps": cell.reps,
         "chunk": step,
         "n_bootstrap": cell.n_bootstrap, "config": cell.config,
         "target_times": taus, "min_nuisance": cell.min_nuisance,
         "max_updates": cell.max_updates,
         "spec": cell.spec().__dict__,
         "seed_key": cell.seed_key or cell.name,
         "params_override": {k: (v.tolist() if hasattr(v, "tolist") else v)
                             for k, v in cell.params_override.items()},
         "axis": cell.axis, "level": cell.level}, indent=2))

    children = np.random.SeedSequence(
        [master_seed,
         int.from_bytes((cell.seed_key or cell.name).encode()[:8].ljust(8, b"\0"),
                        "little") % (2 ** 31)]
    ).spawn(cell.reps)

    chunks = [range(i, min(i + step, cell.reps)) for i in range(0, cell.reps, step)]
    done = 0
    for ci, rng_chunk in enumerate(chunks):
        shard = shard_dir / f"shard_{ci:04d}.parquet"
        if shard.exists() and not overwrite:
            if _shard_is_intact(shard):
                done += 1
                continue
            # exists but unreadable: a kill mid-write, or a truncated file.
            # Redo it rather than skipping it forever.
            print(f"  [{cell.name}] shard {ci + 1} unreadable, redoing", flush=True)
            shard.unlink()
        tasks = [(cell, taus, children[r], r) for r in rng_chunk]
        if n_jobs == 1:
            results = [_one_rep_b(t) for t in tasks]
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                results = list(ex.map(_one_rep_b, tasks))
        frames = [r[0] for r in results]
        draws = [r[1] for r in results if len(r[1])]
        out = pd.concat(frames, ignore_index=True)
        for c in ("ci_lo", "ci_hi", "est", "se"):
            if c in out:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        # Atomic: write beside the target, then rename. os.replace is atomic on
        # POSIX, so a kill at any instant leaves either the old state or the new
        # one, never a half-written parquet that resume would skip as "done".
        #
        # The draws archive goes first, and the interval shard -- the file that
        # `_shard_is_intact` and resume key off -- goes last. A kill between the
        # two then leaves an orphan draws file that the next attempt overwrites,
        # rather than a shard marked done with no draws beside it.
        if draws:
            dpath = shard_dir / f"draws_{ci:04d}.parquet"
            dtmp = dpath.with_suffix(".parquet.tmp")
            pd.concat(draws, ignore_index=True).to_parquet(dtmp, index=False)
            os.replace(dtmp, dpath)
        tmp = shard.with_suffix(".parquet.tmp")
        out.to_parquet(tmp, index=False)
        os.replace(tmp, shard)
        done += 1
        print(f"  [{cell.name}] shard {ci + 1}/{len(chunks)}", flush=True)
    return shard_dir


def progress_b(config_path: Path | str, output_dir: Path | str,
               only: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Per-cell completion, for checking where an interrupted run got to.

    `only` takes the same cell names as `run_study_b`, so `--progress --only X`
    reports on exactly the cells `--only X` would run.
    """
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path(output_dir)
    want = set(only) if only else None
    rows = []
    for key in ("wald_cells", "boot_cells"):
        for c in cfg.get(key, []):
            if want is not None and c["name"] not in want:
                continue
            d = out / c["name"]
            step = None
            meta = d / "meta.json"
            if meta.exists():
                try:
                    step = json.loads(meta.read_text()).get("chunk")
                except Exception:
                    step = None
            if not step:
                # same rule as _chunk_for: an existing cell keeps the legacy
                # chunk, or its shards would be re-indexed and recomputed
                if d.exists() and any(d.glob("shard_*.parquet")):
                    step = _LEGACY_CHUNK
                else:
                    B = c.get("n_bootstrap", 0)
                    step = 2 if B >= 500 else (5 if B else 25)
            total = -(-int(c["reps"]) // int(step))
            got = sum(1 for s in d.glob("shard_*.parquet") if _shard_is_intact(s)) \
                if d.exists() else 0
            rows.append({"cell": c["name"], "axis": c.get("axis", "base"),
                         "n": c["n"], "reps": c["reps"], "B": c.get("n_bootstrap", 0),
                         "shards_done": got, "shards_total": total,
                         "pct": round(100.0 * got / total, 1) if total else 0.0})
    return pd.DataFrame(rows)


def run_study_b(config_path: Path | str, output_dir: Path | str,
                only: Optional[Sequence[str]] = None, reps: Optional[int] = None,
                n_jobs: int = 8, overwrite: bool = False) -> List[Path]:
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text())
    d = cfg.get("defaults", {})
    cells: List[BCell] = []
    for key in ("wald_cells", "boot_cells"):
        for c in cfg.get(key, []):
            unknown = set(c) - _CELL_KEYS
            if unknown:
                raise ValueError(
                    f"cell {c.get('name', '?')!r}: unknown key(s) "
                    f"{sorted(unknown)}; allowed are {sorted(_CELL_KEYS)}")
            # Per-cell first, falling back to `defaults`. Previously min_nuisance,
            # max_updates and tau_quantiles were read from `defaults` *only*, so a
            # per-cell value was silently ignored -- which is why the unknown-key
            # check above exists too.
            def pick(k, default):
                return c.get(k, d.get(k, default))

            cells.append(BCell(
                name=c["name"], n=int(c["n"]), arm=c["arm"],
                reps=int(reps or c["reps"]),
                config=pick("config", "base"),
                n_bootstrap=int(c.get("n_bootstrap", 0)),
                min_nuisance=float(pick("min_nuisance", 0.01)),
                max_updates=int(pick("max_updates", 200)),
                tau_quantiles=tuple(pick("tau_quantiles", (0.10, 0.50, 0.85))),
                target_times=pick("target_times", None),
                params_override=_coerce_override(c.get("params_override", {})),
                q_arm=c.get("q_arm"), pi_arm=c.get("pi_arm"), g_arm=c.get("g_arm"),
                seed_key=c.get("seed_key"),
                axis=c.get("axis", "base"), level=c.get("level", "base")))
    if only:
        cells = [c for c in cells if c.name in set(only)]

    _check_seed_keys(cells)

    out = []
    for c in cells:
        kind = "wald-only" if c.n_bootstrap == 0 else f"B={c.n_bootstrap}"
        print(f"[sim.study_b] {c.name}: n={c.n} arm={c.arm} reps={c.reps} {kind}",
              flush=True)
        out.append(run_cell_b(c, Path(output_dir), n_jobs=n_jobs,
                              overwrite=overwrite))
    return out


def main(argv=None) -> int:
    import argparse
    import os

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    ap = argparse.ArgumentParser(prog="sim.study_b", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("results/study_b"))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--progress", action="store_true",
                    help="report per-cell completion and exit; use after an "
                         "interrupted run to see where it got to")
    a = ap.parse_args(argv)
    if a.progress:
        df = progress_b(a.config, a.output_dir, only=a.only)
        done = int((df["pct"] >= 100).sum())
        print(df.to_string(index=False))
        print(f"\n{done}/{len(df)} cells complete; "
              f"{df['shards_done'].sum()}/{df['shards_total'].sum()} shards")
        return 0
    run_study_b(a.config, a.output_dir, only=a.only, reps=a.reps,
                n_jobs=a.n_jobs, overwrite=a.overwrite)
    print("[sim.study_b] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
