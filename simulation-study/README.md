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

## Running

From this directory:

```bash
# the whole of Study A -- every cell, both implementations, figures and tables
python -m sim.run --config sim/configs/study_a_dr.yaml \
    --output-dir results/study_a --report

# pilot: two cells, 12 replicates each, with summary tables printed
python -m sim.run --config sim/configs/study_a_dr.yaml \
    --output-dir results/pilot --cells C1_n250 C5_n250 --reps 12 --summarise
```

Study C is a second entry point, because its nuisances are fitted in R and shared
outward rather than built in Python:

```bash
python -m sim.study_c --config sim/configs/study_c.yaml --output-dir results/study_c
```

That runs all four stages for every sample size — export the replicates, fit the
nuisances once in R and run the R comparators, run `concrete`'s second stage on
those same arrays, run PyTMLE's estimators on them — then writes the tables and
three figures. Stages are skipped when their output exists, so an interrupted run
resumes; the stages remain available as `export` / `run` subcommands for
debugging.

**One command reproduces a study.** `concrete` is not a separate step: a cell
with `concrete_reps > 0` runs concrete's second stage on the same seeds and the
same injected nuisances straight after its own replicates, writing
`concrete.parquet` beside the shards. `report.collect` loads the two together,
so every downstream table and figure carries both implementations without any
merge step. `--report` then writes the figures and tables into the output
directory. Earlier versions needed three commands and a bespoke merge; that was
the main obstacle to reproducing the study.

Results are written as parquet shards under `<output-dir>/<cell>/`. Shards that
already exist are skipped, so an interrupted run resumes rather than restarting;
pass `--overwrite` to force recomputation. `--chunk` sets both the shard size and
the resume granularity.

Tests: `python -m pytest tests -q`.

## Validation gate

Before any study is trusted, the harness is validated against Hage et al. (2025)
— see [VALIDATION.md](VALIDATION.md) for the four rungs and their results.

```bash
Rscript R/install_packages.R
Rscript R/truth_adjcuminc.R --out results/validation/truth_r.parquet --times 0.25,0.5,1.0
Rscript R/run_adjcuminc.R   --scenarios s1,s2,s3,s4 --n 1500 --reps 150 \
                            --times 0.25,0.5,1.0 --out results/validation/adjcuminc_est.parquet
Rscript R/rung3_export.R    --scenarios s1,s2,s3,s4 --n 400 --reps 30 \
                            --times 0.25,0.5,1.0 --out-dir results/validation/rung3
python -m sim.run --config sim/configs/validation_rung4.yaml --output-dir results/validation/rung4
python -m sim.validate --dir results/validation      # reports and gates every completed rung
```

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
R/               comparator bridges (concrete, riskRegression, AdjCuminc)
tests/           harness unit tests
```

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

**`min_nuisance` cannot be pinned — see [FINDINGS.md](FINDINGS.md).** The configs
pass 0.01, but PyTMLE discards it after the first update step and reverts to its
n-dependent default `5/(√n·log n)`. Any design that relies on controlling the
truncation bound (Study B's n-ladder, Study D's `min_nuisance` sweep) has to
account for that. It happens not to matter in the calibration runs done so far,
where the nuisance denominators sit an order of magnitude above either value and
truncation never binds.

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
fitting. That is the like-for-like number to compare PyTMLE against `concrete`;
`report.runtimes()` aggregates it over replicates alongside `n`, the grid size and
the iteration count, without which a wall-clock number is not interpretable.

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
