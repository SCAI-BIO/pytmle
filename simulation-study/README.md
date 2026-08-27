# PyTMLE simulation study

An isolated validation project. It does not modify the `pytmle` package — it only
imports its public API, plus two documented internals (`get_influence_curve.get_eic`
for the one-step estimator, and `tmle_update` for the bootstrap instrumentation).

## Environment

The shared Mamba environment `pytmle-sim` contains the Python stack, the R runtime,
`adjustedCurves`, `riskRegression`, and `concrete` in one place. To recreate it:

```bash
mamba env create --file simulation-study/environment.yml
mamba activate pytmle-sim
python -m pip install -e '.[dev]'
Rscript -e 'remotes::install_github("imbroglio-dc/concrete", upgrade = "never")'
```

Verify the R side with `Rscript simulation-study/R/install_packages.R`.

## Reproducing the studies

Everything below is run **from the `simulation-study/` directory** with the
`pytmle-sim` environment active:

```bash
mamba activate pytmle-sim
cd simulation-study
python -m pytest tests -q          # all must pass before anything else
```

The three studies are independent of one another and can be run in any order. They
share no state, so a failure in one does not invalidate the others.

**Budget before you start.** These are the measured costs of the runs that produced
the committed results, on a 20-core machine:

| study | cells | replicates | CPU-hours | disk |
|---|---|---|---|---|
| validation gate | 4 scenarios + 8 rung-4 cells | 720 R + 3 200 Python | not separately measured | 401 MB |
| **A** — double robustness | 24 | 12 000 + 3 600 concrete | **63** | 44 MB |
| **B** — interval coverage | 53 | 36 600 | **789** | 69 MB |
| **C** — cross-package | 3 sizes | 1 150 | ~10 | **~27 GB** |

CPU-hours are measured, summed from the per-replicate `seconds` recorded in the
shards. Divide by your worker count for wall time, but not below ~8 workers'
worth — throughput ceilings there (see the `--n-jobs` note below), so 8 workers
puts Study A at roughly 8 hours and Study B at **two to four days**.

Study B is the long one, and two thirds of it is bootstrap: the seven cells with
`n_bootstrap > 0` account for **518 of the 789** CPU-hours. Dropping them from the
config leaves a ~270 CPU-hour run that still answers the coverage question on
every stress axis — only the Wald-versus-bootstrap comparison is lost.

Study C's disk figure is the exported per-replicate nuisance arrays, which are
`(n × K)` doubles and dominate everything else. They scale as O(n²) — 5 MB per
replicate at n = 500, 21 MB at n = 1000 — so the three sizes cost roughly 2.5 GB,
11 GB and 13 GB. They are deletable once the benchmark has run (see below), and
nothing downstream reads them after that.

### The validation gate

Nothing else is trusted until the harness reproduces Hage et al. (2025) — see
[VALIDATION.md](VALIDATION.md) for the rungs and what each one proves.

```bash
Rscript R/install_packages.R
Rscript R/truth_adjcuminc.R --out results/validation/truth_r.parquet \
                            --times 0.25,0.5,1.0
Rscript R/run_adjcuminc.R   --scenarios s1,s2,s3,s4 --n 1500 --reps 150 \
                            --times 0.25,0.5,1.0 \
                            --out results/validation/adjcuminc_est.parquet
Rscript R/rung3_export.R    --scenarios s1,s2,s3,s4 --n 400 --reps 30 \
                            --times 0.25,0.5,1.0 --out-dir results/validation/rung3
python -m sim.run --config sim/configs/validation_rung4.yaml \
                  --output-dir results/validation/rung4
python -m sim.validate --dir results/validation
```

`sim.validate` reports whichever rungs have completed and **exits non-zero** if a
completed rung fails its gate, so it works as a precondition in a script and not
only as a report. It recomputes the closed-form truth by Monte Carlo on each call
and takes a couple of minutes as a result; `--truth-mc` trades accuracy for speed
when you only want to see the shape of the tables.

Rung 4 prints a 2×2×2 — censoring informative or not, outcome model right or
wrong, censoring model right or wrong. Read it down the `q_model` column:
`bias_change` should be ~0 in every `correct` row, which is double robustness
holding even against a misspecified censoring model, and can only be non-zero in
the `wrong` rows, where that protection is switched off and `G` carries the
estimate alone.

### Study A — double robustness

One command. `--report` writes the figures and tables at the end, and `concrete`
is not a separate step: a cell with `concrete_reps > 0` runs concrete's second
stage on the same seeds and the same injected nuisances straight after its own
replicates.

```bash
python -m sim.run --config sim/configs/study_a_dr.yaml \
    --output-dir results/study_a --report
```

Outputs land in `results/study_a/figures/` and `results/study_a/tables/`. To
redraw them from stored shards without recomputing anything:

```bash
python -m sim.report_study_a --output-dir results/study_a
```

Smoke test first if you like — two cells, 12 replicates, tables printed to stdout:

```bash
python -m sim.run --config sim/configs/study_a_dr.yaml \
    --output-dir results/pilot --cells C1_n250 C5_n250 --reps 12 --summarise
```

### Study B — where confidence intervals fail

Three commands, because the run is long enough that you will want to inspect the
tables and redraw the figures without touching the replicates.

```bash
# 1. run  (the long one -- see the budget table above)
python -m sim.study_b --config sim/configs/study_b.yaml \
    --output-dir results/study_b --n-jobs 8

# 2. tables
python -m sim.study_b_report --config sim/configs/study_b.yaml \
    --output-dir results/study_b

# 3. figures
python -m sim.plots_study_b --output-dir results/study_b
```

Steps 2 and 3 are cheap and read only stored shards, so re-run them freely.
**Always re-run step 2 before step 3** — the figures read
`study_b_performance.csv`, so plotting against a stale table silently omits
whatever finished since it was written.

Check progress at any time, including from another shell while the run is going:

```bash
python -m sim.study_b --config sim/configs/study_b.yaml \
    --output-dir results/study_b --progress
```

To run a subset — useful for reproducing one axis rather than all of them:

```bash
python -m sim.study_b --config sim/configs/study_b.yaml \
    --output-dir results/study_b --only OV3_n250_correct B_OV3_n250_correct
```

### Study C — agreement with `concrete`, and a fair runtime comparison

Study C is a separate entry point because its nuisances are fitted **in R** and
shared outward rather than built in Python. Each sample size passes through four
stages — export the replicates, fit the nuisances once in R and run the R
comparators, run concrete's second stage on those same arrays, run PyTMLE's
estimators on them — and the tables and figures are written at the end unless
`--no-report` is passed.

```bash
# 1. the four stages, for every sample size in the config
python -m sim.study_c --config sim/configs/study_c.yaml \
    --output-dir results/study_c --n-jobs 8
```

**The runtime benchmark is deliberately not one of those stages.** It has to run
alone on an idle machine, so it is a separate command and the ordering is not
optional:

```bash
# 2. wait until the machine is idle, then benchmark -- nothing else running
uptime                                   # 1-minute load should be near zero
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 \
python -m sim.bench_stage2 --study-dir results/study_c \
    --config sim/configs/study_c.yaml --repeats 3

# 3. re-run the report so the runtime table and figure are picked up
python -m sim.study_c_report results/study_c/n500 results/study_c/n1000 \
    results/study_c/n2000 --out-dir results/study_c
```

**The thread variables must be on the command line, not inside the script.**
OpenBLAS reads its thread count when the library loads, which happens before any
line of `sim.bench_stage2` runs, so setting them in Python is too late. The module
checks with `threadpoolctl` and **refuses to start** if the parent process is not
pinned — it will tell you exactly this — so a forgotten prefix costs a message,
not a bad measurement.

`sim.bench_stage2` also refuses to start when the 1-minute load average exceeds
`--max-load` (default 2.0), pins the R child's environment the same way, adds
`data.table::setDTthreads(1)` where no environment variable reaches, runs one fit
at a time alternating which implementation goes first, and samples the child's
live thread count with `psutil`. Afterwards it reports `wall/cpu` per row and
drops anything outside [0.75, 1.35]: a single-threaded fit left alone spends its
wall time computing, so the ratio sits at ~1.0. `--allow-busy` skips the gates and
exists for smoke tests only; timings taken with it are not comparable.

`--reps-per-n` (default 30) caps how many replicates are timed per sample size.
It applies to `--config` too, so a study config's `reps` choose *which* sample
sizes to benchmark, not how many replicates — timing all 1 150 of Study C's would
take ~19 hours rather than ~2.

Step 2 reads the per-replicate nuisance stubs written by step 1, so **do not
delete `results/study_c/n*/rep*_*.parquet` until the benchmark has run.** Once it
has, they are ~12 GB of reclaimable scratch:

```bash
rm results/study_c/n*/rep*_*.parquet     # after step 3
```

### Checking you got the right thing

Each study writes a fixed set of deliverables. If any are missing, the run did not
finish — check with `--progress` (Study B) or by re-running the command, which
resumes rather than restarting.

| study | tables | figures |
|---|---|---|
| A | `results/study_a/tables/study_a_{bias_rd, coverage_rd, detail_rd, diagnostics, runtimes}.{csv,md}` | 5, in `results/study_a/figures/` |
| B | `results/study_b/study_b_{performance, conditions, breakpoints, type_i_error, attribution, failures}.csv` | 8, in `results/study_b/figures/` |
| C | `results/study_c/study_c_{agreement, agreement_summary, performance, score, runtime}.{csv,md}` plus `study_c_{estimates,eic}.parquet` | one per agreement quantity, plus performance, score and runtime, in `results/study_c/figures/` |

**Start with `study_c_agreement_summary.md`.** It is the same content as
`study_c_agreement.csv` with the sample sizes turned into columns, one row per
comparison and quantity, and it is the form that answers the question the study
asks. Read across a row: a *numerical* difference shrinks with `n`, a
*structural* one does not, and the `shrink_ratio` column (largest `n`'s
discrepancy over smallest `n`'s) states which it is. Eight of the nine rows sit
between 0.10 and 0.46; the one at 0.96 is the finding.

Study C's `study_c_runtime.*` and its figure appear only after the benchmark has
run and the report has been re-run (steps 2 and 3 above); everything else is
written by step 1.

**Seeding.** Every study is seeded and every replicate's seed is derived from the
master seed together with the cell identity, so re-running one cell reproduces
exactly the replicates it recomputes and a resumed run is indistinguishable from
an uninterrupted one. The defaults are `20250301` for Studies A and B (`--seed`)
and `20250901` for Study C. This fixes the *data*; it does not promise
bit-identical floating point across different BLAS builds or CPU architectures,
so expect agreement to many digits rather than all of them.

The headline results these reproduce are written up in [STUDY_B.md](STUDY_B.md)
and [FINDINGS.md](FINDINGS.md).

### Resuming an interrupted run

All three studies resume from disk, at different granularities:

| study | unit of resume | how to force recomputation |
|---|---|---|
| A | one parquet shard (`--chunk` replicates) | `--overwrite` |
| B | one parquet shard; the chunk size is **pinned in `meta.json`** | `--overwrite` |
| C | one stage of one sample size | `--overwrite` |

Nothing needs to be cleaned up first and no partial state is carried in memory —
kill the process, restart the same command, and it continues from the last
completed unit. Study B's chunk size is pinned per cell on purpose: re-chunking a
cell that already has shards would re-index them and silently recompute
replicates it already holds.

Study A and B write per-cell directories, `results/<study>/<cell>/`, holding
`shard_*.parquet`, `concrete.parquet` where concrete ran, and a `meta.json`
recording the config, spec, target times and `min_nuisance` actually used.

A Study B cell with `n_bootstrap > 0` also writes `draws_*.parquet` — the **raw
bootstrap draws**, one row per (resample, estimand, event, tau, arm), with the
per-draw convergence flags. Nothing in the report reads them; they are archived
so that a new interval construction can be computed from stored output instead of
re-running the fits, which for the seven bootstrap cells is ~519 CPU-hours. They
are safe to delete if disk matters more than that option.

## Layout

```
sim/
  dgp.py         continuous-time competing-risks DGP; named configurations
  truth.py       closed-form counterfactual CIF, cached per configuration
  nuisance.py    the three-way correct/wrong/oracle toggles, and injection
                 into PyTMLE's InitialEstimates
  estimators.py  tmle / gcomp / aipw / ipw, all from identical nuisances
  metrics.py     bias, coverage, SE ratio, ... each with a Monte Carlo SE
  runner.py      replicate loop, parquet shards, resume; drives concrete too
  concrete_bridge.py  export nuisances, run concrete's second stage on them
  report.py      aggregation: performance, diagnostics, runtimes
  report_study_a.py   figures + tables from a finished output directory
  plots.py       Study A figures;  tables.py  Markdown + CSV tables
  calibrate.py   the estimand-level guard;  validate.py  the AdjCuminc ladder
  config.py      YAML -> design cells
  run.py         CLI
  study_b.py     Study B: stress axes, tagged bootstrap draws, resumable shards
  bootstrap_ci.py     resampling that keeps the draws and the diagnostics
  study_b_report.py   Study B tables;  plots_study_b.py  Study B figures
  study_c.py     Study C: R-fitted nuisances shared outward, four stages
  bench_stage2.py     single-threaded, serialised cross-package runtime bench
  study_c_report.py   Study C tables;  plots_study_c.py  Study C figures
R/               comparator bridges (concrete, riskRegression, AdjCuminc)
tests/           harness unit tests
```

### Documents

| file | what it holds |
|---|---|
| [DGP.md](DGP.md) | the data-generating process and every lever, for Studies A and B |
| [STUDY_A.md](STUDY_A.md) | Study A's results: double robustness, and what it does *not* buy |
| [STUDY_B.md](STUDY_B.md) | Study B's results: where intervals fail, and what fixes them |
| [STUDY_C.md](STUDY_C.md) | Study C's results: agreement with `concrete`, and the runtime cost |
| [BOOTSTRAP_FAILURES.md](BOOTSTRAP_FAILURES.md) | the three bootstrap failure modes and how they are attributed |
| [FINDINGS.md](FINDINGS.md) | defects in `pytmle` and traps the study could fall into again |
| [VALIDATION.md](VALIDATION.md) | the validation ladder and its rungs |
| [AIPW.md](AIPW.md) | the one-step estimator's construction |

## Design notes worth knowing before editing

**Target times are frozen per configuration**, computed once from a large draw
(`runner.target_times_for`) rather than from each replicate's own quantiles.
Data-dependent target times would move the estimand between replicates and make
"bias" meaningless.

**The nuisance grid must match exactly.** `PyTMLE._check_inputs` requires
`np.unique(event_times) == np.unique(initial_estimates[k].times)`, so every
builder keys its columns off the observed unique times of that replicate.

**Memory is O(n²).** With continuous event times the grid is the ~n distinct
observed times, so every nuisance is an `(n, K, ·)` array with `K ≈ n`. n = 2000
peaks around 1.25 GB per replicate; n = 20 000 does not fit. Large-*n* checks must
use the propensity fitters alone, not the full build.

**`min_nuisance` is honoured — but raising it is not a remedy.** It used to be
discarded after the first update step (FINDINGS 1); that is **fixed upstream in
`c43539e`**, and Study B's sweep confirms the passed value now governs every step.
What the sweep also shows is that turning the dial up makes positivity stress
*worse*: at OV3, going from 0.01 to 0.10 costs 0.107 coverage at τ = 0.48 while
SE/SD falls 1.21 → 0.68 and standardised bias rises — variance traded for bias,
losing on both. See [STUDY_B.md](STUDY_B.md) §8.

**Do not raise `--n-jobs` to the core count.** The second stage is elementwise
arithmetic over `(n, K, ·)` arrays and is memory-bandwidth bound, so throughput
ceilings near 8 workers. Measured at n = 500: 8 workers give 98 reps/min for 84 s
of CPU per 24 reps, while 20 workers give *fewer* reps/min (85) for 169 s. The
default is chosen per cell from `n` (`runner.default_n_jobs`); overriding it
upward costs CPU and buys nothing.

**A cell's results live in one directory.** `results/<study>/<cell>/` holds
`shard_*.parquet` (PyTMLE, one per chunk of replicates), `concrete.parquet` when
concrete ran, and `meta.json` recording the config, spec, target times and
`min_nuisance` the cell actually used. Both estimate files share a schema and
`metrics.summarise` groups by estimator, so concrete's smaller replicate count
never pools into PyTMLE's Monte Carlo SE.

**Runtime logging is second-stage only.** `stage2_seconds` times the targeted
update alone — initial estimates are injected, so it excludes all nuisance
fitting. `report.runtimes()` aggregates it over replicates alongside `n`, the grid
size and the iteration count, without which a wall-clock number is not
interpretable.

**But `stage2_seconds` from a study run is not a fair cross-package comparison.**
The *definition* is like-for-like; the *conditions* are not. Study runs execute
8-way parallel and neither side is single-threaded — numpy links a 20-thread
OpenBLAS, while R carries its own pthreads OpenBLAS plus `data.table` defaulting
to 10 cores — so the two are contended differently and threaded differently. Use
`python -m sim.bench_stage2` for the comparison: it pins both sides to one thread
before the interpreter starts, runs them serialised rather than concurrently,
refuses to start on a busy machine, and verifies the thread count at runtime
rather than assuming it.

## Calibration decisions

Two parameters were set by measurement rather than assumption, because a
double-robustness design fails silently when a "misspecified" model turns out to
be nearly right:

- **`gamma = (1.0, -0.8, 0.6)`** in the `threshold` config. The plain-logistic
  propensity fit then carries an asymptotic mean absolute error of ~0.018 against
  ~0.003 for the correct family, while the true propensity stays within
  [0.09, 0.67] even at n = 50 000 — so the misspecification is real and positivity
  is nowhere near violated. Weaker values make it inert; stronger ones drive
  `e(W)` to zero, which belongs in the boundary study instead.
- **`delta_j = 4`**, matching the `e⁴` hazard contrast of Hage et al. across the
  `w_cont > 1` threshold.

The correct propensity family plugs in `q = P(A = 1) = mean(A)` rather than
fitting the offset freely: the two-parameter version is weakly identified and
measurably worse (0.039 vs 0.029 mean absolute error at n = 500).
