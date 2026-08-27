# Study C — does the implementation agree with the reference, and what does it cost?

Studies A and B ask what the *estimator* does. This study asks what the
*implementation* does: given byte-identical nuisances, does PyTMLE produce the
same point estimate, the same standard error and the same **score** as an
independent implementation of the same algorithm — and how long does it take?

Every model here is correctly specified. Specification is Study A's subject; this
study holds it fixed so that any disagreement is an implementation difference and
nothing else.

**Scope.** Three sample sizes — 500, 1000 and 2000 replicates 500/500/150 — on the
`base` DGP, at τ ∈ {1.61, 3.12, 5.42}. Nuisances are fitted **once per replicate
in R** (`CSC` for the cause-specific hazards, `coxph` for censoring, `glm` for the
propensity) and exported as full counterfactual cumulative hazards on that
replicate's own event-time grid, so PyTMLE, `concrete` and `riskRegression::ate`
all consume the same arrays. Fitting them separately in Python would confound the
comparison with ties handling, centring and cross-fitting differences between the
two languages — which is the very thing the study exists to measure around.

Two tiers of sharing, and each is entitled to a different tolerance:

| tier | what is shared | comparators |
|---|---|---|
| 1 | byte-identical injected nuisances | `concrete` |
| 2 | identical fitted model objects | `riskRegression::ate` |

---

## 1. Start here: the agreement summary

`study_c_agreement_summary.md` is the table to read first. It is the long
agreement table with the sample sizes turned into columns, one row per comparison
and quantity, because the question is *"does this discrepancy shrink with n"* and
that is a comparison along a row.

| comparison | tier | quantity | unit | n=500 | n=1000 | n=2000 | shrink |
|---|--:|---|---|--:|--:|--:|--:|
| gcomp (concrete) vs gcomp | 1 | point estimate | abs. diff | 7.9e-04 | 3.9e-04 | 1.9e-04 | 0.24 |
| tmle (concrete) vs tmle | 1 | point estimate | abs. diff | 1.1e-02 | 7.2e-03 | 4.9e-03 | 0.46 |
| ate:AIPTW vs aipw | 2 | point estimate | abs. diff | 1.0e-04 | 4.9e-05 | 2.3e-05 | 0.23 |
| ate:GFORMULA vs gcomp | 2 | point estimate | abs. diff | 1.0e-04 | 4.9e-05 | 2.3e-05 | 0.23 |
| ate:IPTW vs ipw | 2 | point estimate | abs. diff | 1.6e-05 | 6.8e-06 | 3.2e-06 | 0.20 |
| tmle (concrete) vs tmle | 1 | **standard error** | abs. log ratio | 6.6e-04 | 3.3e-04 | 1.6e-04 | 0.24 |
| ate:AIPTW vs aipw | 2 | **standard error** | abs. log ratio | 6.0e-03 | 3.8e-03 | 2.4e-03 | 0.40 |
| ate:IPTW vs ipw | 2 | **standard error** | abs. log ratio | **0.134** | **0.130** | **0.128** | **0.96** |
| tmle (concrete) vs tmle | 1 | **score (PnEIC)** | abs. diff | 2.2e-05 | 2.6e-06 | 2.4e-06 | 0.11 |

`shrink` is the largest sample size's discrepancy over the smallest's, and it is
the column that does the work: **a numerical difference decays with `n`, a
structural one does not.** Eight rows sit between 0.10 and 0.46. One sits at 0.96,
and that is the finding.

---

## 2. The score: the two packages solve the same estimating equation

A point estimate cannot distinguish "same algorithm" from "same answer". Two
implementations can agree on `Psi` while driving different estimating equations to
zero. The quantity that separates them is the score `Pn D*(Q*)`, which both
packages compute — with the same formula — and both throw away before their output
tables.

Extracting the post-update `SummEIC` from each and merging on
`(rep, time, event, arm)` matches **9000 of 9000** rows at n = 500 and n = 1000,
and 2700 of 2700 at n = 2000. Two facts make the result conclusive rather than
merely close:

1. **Both drive `|PnEIC|` below their own `seEIC/(sqrt(n) log n)` on 100 % of
   targets**, at every sample size and both causes.
2. **No row has a cross-package `PnEIC` difference exceeding that threshold.**

So the residual gap is not a discrepancy — it is the width of the tolerance band
both are aiming inside, and it shrinks with `n` exactly as the band does. Median
`|PnEIC|` / criterion runs 0.13–0.38 on both sides, and the median score reduction
over the update is ~7.7× at n = 500 rising to ~9.0× at n = 2000, again matched
between the two.

This is the strongest validation result in the harness, and it is what licenses
reading the point-estimate divergence below as a *targeting* difference rather
than a bridge or an input difference.

---

## 3. Standard errors: one agreement and one disagreement

**concrete: agrees three orders of magnitude better than the point estimate.**
Mean absolute log ratio 6.6e-04 → 1.6e-04. That is the expected shape and it is
worth stating why: the FINDINGS 9 defect moves `Psi` through the targeted update
but never touches the influence curve, so the two packages' standard errors are
built from the same object even where their estimates diverge. `TIERS` pins
`tol_se = 2e-3` with `expect_se = "agree"` for this pair, separately from the
point estimate's `expect = "diverge"`.

**`ipw`: a stable ~13 % gap that does not shrink, and PyTMLE is on the wrong side
of it.** The point estimates agree to 1.6e-05; the standard errors differ by
0.128–0.134 on the log scale, flat in `n`. The empirical sampling distribution
settles which is right, and the answer is not the intuitive one:

| | SE / empirical SD | coverage |
|---|--:|--:|
| PyTMLE `ipw` | **1.08 – 1.12** | 0.969 – 0.975 |
| `ate:IPTW` | 0.93 – 1.01 | 0.934 – 0.954 |

`riskRegression`'s SE is the **smaller** of the two (0.877×) and the calibrated
one. PyTMLE's IPW interval is too wide and over-covers by up to 0.025.

The cause is in `sim/estimators.py`: `run_ipw` builds the influence function as
`w - est` with the weights `1{A=a} / (pi_a(W) G(t- | a, W))` treated as **known**,
while `riskRegression::ate` propagates the estimation of `pi` and `G` into its
influence function. An IPW estimator using *estimated* propensities has a smaller
asymptotic variance than the same estimator with the true ones, so treating them
as known overstates it. The τ dependence confirms the mechanism: the overstatement
grows with the evaluation time, because `G` is also treated as known and matters
more the further out the evaluation goes. See FINDINGS 14.

`aipw` against `ate:AIPTW`, on the same fitted models, shrinks as O(1/n) and stays
inside 1e-2 throughout — so this is specific to the IPW influence function, not to
the bridge.

---

## 4. Point estimates: the targeting increment, and what concrete's update costs

`gcomp` agrees across packages to 7.9e-04 and tightens to 1.9e-04. `tmle` does
not: 1.1e-02 falling only to 4.9e-03. Since the plug-in agrees and the score
agrees, the difference is in the update itself, and it is the expected result
rather than a defect in the harness — concrete 1.0.8 still carries the
`g.star.obs` defect that PyTMLE has fixed (FINDINGS 9). The comparison is made on
the **targeting increment** `tmle - gcomp` rather than the CIF level, because the
two build different time grids and that shifts the plug-in by ~1e-3 in both
estimators alike (FINDINGS 8).

Under *correct* specification the defect costs concrete almost nothing in bias
— −0.0009 against PyTMLE's −0.0011 at n = 500 — but it shows up clearly in the
dispersion:

| estimator | n | bias | empirical SD | SE/SD | coverage |
|---|--:|--:|--:|--:|--:|
| `gcomp` | 500 | −0.0021 | 0.0277 | — | — |
| `tmle` | 500 | −0.0011 | 0.0379 | **0.993** | **0.950** |
| `tmle (concrete)` | 500 | −0.0009 | **0.0301** | **1.271** | **0.981** |
| `tmle` | 2000 | 0.0002 | 0.0194 | 0.971 | 0.940 |
| `tmle (concrete)` | 2000 | 0.0006 | **0.0158** | **1.211** | **0.987** |

concrete's estimates are **under-dispersed** — its empirical SD sits much closer to
the plug-in's (0.0301 against `gcomp`'s 0.0277) than to a fully updated
estimator's (0.0379). Its standard error, computed from the same influence curve
as PyTMLE's and agreeing with it to 1.6e-04, therefore describes a spread its
estimates do not have, and the interval over-covers at 0.98. So an under-updated
TMLE looks *better* on coverage while being closer to a plug-in — which is exactly
the reading Study B's SE/SD panel exists to prevent.

`riskRegression::ate` agrees with PyTMLE's corresponding estimators to 1.0e-04 or
better on every point estimate, tightening as O(1/n) — a tier-2 comparison against
an exact Aalen–Johansen product limit rather than a discrete plug-in, so agreement
at this level is a stronger result than the tolerance demanded.

---

## 5. Runtime, measured under conditions where it means something

Study C's own `stage2_seconds` is **not** a fair cross-package number, and
`sim/bench_stage2.py` exists because of that. Every study stage runs 8-way
parallel, and neither side is single-threaded by default: numpy links a 20-thread
OpenBLAS while R links its own pthreads OpenBLAS with `data.table` claiming 10 of
20 cores. The reported figure compared an 8-way-contended 20-thread Python process
against an 8-way-contended 10-thread R process.

The benchmark removes both and **verifies rather than assumes** that it did:
`threadpoolctl` confirms the Python side at `num_threads = 1`;
`data.table::setDTthreads(1)` covers what no environment variable reaches; the
orchestrator samples the R child's live thread count with `psutil`; fits run one at
a time, alternating which implementation goes first so a warm page cache cannot
favour one systematically. **All 180 rows passed the fairness band**, with
`wall/cpu` median 1.002 and maximum 1.027, and `ran_first` exactly 0.50 per cell.

| n | concrete | PyTMLE | ratio | grid (both) | per step × cell |
|---|--:|--:|--:|--:|---|
| 500 | 0.624 s | 2.038 s | **3.26×** | 352 | 301 → 1026 ns = **3.41×** |
| 1000 | 2.278 s | 7.936 s | **3.48×** | 706 | 265 → 1011 ns = **3.82×** |
| 2000 | 8.639 s | 37.778 s | **4.37×** | 1412 | 239 → 1113 ns = **4.65×** |

**The hypothesis the benchmark was built on was wrong.** The expectation was that
the reported ~4× gap would prove mostly artefact. Pinning halves *both* times —
contention was costing them about equally — and the ratio moves only from ~3.8× to
3.26× at n = 500. The gap is real, and it widens with `n`.

**The two run on the same grid.** PyTMLE truncates the injected grid at
`max(target_times)` (`estimates.py:237-247`); concrete builds
`{0} ∪ {observed times ≤ max(τ)} ∪ {target times}`, which is the same set. Their
working grids are **identical on 100 % of replicates** at all three sample sizes.
So none of the gap is grid size — **all of it is per-cell cost**.

Two things the normalisation shows that raw seconds do not:

1. **The two scale in opposite directions.** concrete's cost per (step × grid
   cell) *falls* with `n`, 301 → 265 → 239 ns, while PyTMLE's is flat to rising,
   1026 → 1011 → 1113 ns. That is why the headline ratio widens.
2. **Per-step figures flatter concrete by ~8 %.** The step counts are on
   different conventions: concrete reports exactly one more at every size
   (12.5 / 12.5 / 13 against 11.5 / 11.5 / 12), at 487 of 500 replicates at
   n = 500 — while both converge on 100 % and both solve their own stopping
   criterion. One side counts the initial evaluation as a step.
   `median_s_per_step` is therefore **not** like-for-like across implementations;
   per-fit seconds is.

**Where the per-cell cost goes.** Four avoidable Python-level costs inside the
`O(n × K)` array work account for **1.30× at n = 500** and 1.15× at n = 1000, with
step counts identical and estimates and standard errors bit-identical: a
per-subject loop in `get_haz_ls`, an `O(n)` scatter with an `O(K)` scan per
subject in `get_ic`, ~200k dict constructions building the IC frame, and a
`cumsum` recomputed once per target event when it depends only on `j`. These are
upstream changes to `pytmle`, specified but not applied here.

> **Correction (2026-08-27).** An earlier version of this section attributed part
> of the gap to grid size — PyTMLE on all `n` observed times against concrete's
> ~70 % — and decomposed it as "1.42× grid × 2.37× per-cell". That was an
> instrumentation artefact: `sim/estimators.py` and `sim/bench_stage2.py` recorded
> the *input* grid while concrete reported its *working* grid. Both now record the
> working grid, `n_times_input` carries the injected size separately, and the
> stored outputs were backfilled from `python_eic.parquet`, which had it right all
> along.

See FINDINGS 15.

---

## 6. What to tell a user

1. **The port is faithful.** Same estimating equation, solved to the same
   tolerance, with standard errors agreeing to ~1e-04. Where the point estimates
   differ, the difference is concrete's unfixed defect, not PyTMLE's.
2. **PyTMLE's second stage is 3–4× slower than concrete's**, and the gap grows
   with `n`. Roughly a third of that is evaluating on a finer grid, which is a
   choice with a cost rather than an inefficiency.
3. **Do not use `ipw`'s interval as a yardstick.** It is ~13 % too wide and
   over-covers; its coverage should be read as "over-covers because the SE is
   wrong", not as good calibration.
4. **A high coverage number is not automatically the better one.** concrete's
   0.98 here comes from under-dispersed estimates against a correctly computed
   standard error, which is a worse position than PyTMLE's 0.95.

---

## 7. Limitations

- **Second stage only.** `stage2_seconds` and the benchmark both exclude nuisance
  fitting, which is where a user of either package would usually spend most of
  their time. The comparison isolates the targeted update and says nothing about
  end-to-end cost.
- **Single-threaded by construction.** The benchmark says nothing about how either
  implementation scales with cores — only that at one thread each, under identical
  conditions, PyTMLE is slower.
- **Correct specification throughout.** Everything here is conditional on the
  models being right; Study A is where misspecification lives, and the two
  compose rather than overlap.
- **Rejected iterations are hidden on both sides.** Each package reports accepted
  steps only (`iter_num` / `IterNum` are not exposed), so per-step cost understates
  work symmetrically.
- **`gcomp` has no standard error or score** in either package, by design, so it
  appears only in the point-estimate rows and its `se`/`pn_eic` comparisons are
  emitted as explicit skips rather than silently dropped.
- **n = 2000 runs 150 replicates** against 500 at the smaller sizes, so its Monte
  Carlo SE is ~1.8× larger. The agreement comparisons are paired per replicate and
  far sharper than that; the marginal coverage figures are not.
