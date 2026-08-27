# Study B — when do the confidence intervals fail, and what fixes them?

Study A asks whether the point estimate is right. This study asks whether the
**interval around it** is right, and it asks the question along three axes chosen
because the literature predicts failure on each: near-violation of treatment
positivity, rare events, and dependent censoring. A fourth condition sets the true
effect to zero so type-I error can be read directly.

The design, the levers and their calibration live in [DGP.md](DGP.md). This
document is the result.

**Scope.** 53 cells, 1 590 shards, 36 600 replicates. Sample sizes 250 and 500 on
every stress axis (1 000 and 2 000 additionally at the base condition), three
evaluation times τ ∈ {0.48, 3.12, 8.61}, both a `correct` and an `oracle` nuisance
arm on the diagnostic cells. Unless stated otherwise every number below is the
**cause-1 risk difference** under `correct` nuisances, and Monte Carlo SE on a
coverage of 0.95 is 0.007 at 1 000 replicates and 0.010 at 500.

---

## 1. Reading the three panels

Every axis figure has the same three rows, and they answer different questions.
The middle one is the reason the study is interpretable at all.

| panel | what it says |
|---|---|
| **coverage** | *that* the interval is wrong |
| **mean SE / empirical SD** | *why* — is the interval the right width? |
| **mean width** | what the coverage cost, in precision |

For a symmetric Wald interval with negligible bias and a roughly normal sampling
distribution, coverage follows from the ratio alone:

> coverage ≈ 2Φ(1.96 · SE/SD) − 1

which is 0.95 at ratio 1, 0.92 at 0.90, 0.90 at 0.85 and 0.77 at 0.61. Coverage is
a *consequence* of three things — right width, right centre, right shape — and
only the ratio isolates the first. The third panel exists because coverage can
always be bought by widening; reporting it beside width is the discipline
borrowed from Fan et al. (2024), whose debiased LASSO holds nominal coverage under
heavy tails only by inflating mean length from 0.483 to 1.418.

Two further diagnostics do the discriminating work: **standardised bias**
(bias / empirical SD) separates a mis-centred interval from a mis-sized one, and
the **left/right miss split** separates a symmetric failure from a skewed one.

---

## 2. Baseline: the asymptotics are fine when nothing is stressed

| n | τ = 0.48 | τ = 3.12 | τ = 8.61 |
|---|---|---|---|
| 250 | 0.948 | 0.942 | 0.939 |
| 500 | 0.963 | 0.956 | 0.949 |
| 1 000 | 0.952 | 0.943 | 0.944 |
| 2 000 | 0.960 | 0.950 | 0.954 |

SE/SD sits in [0.967, 1.080] throughout and standardised bias never exceeds 0.14.
This matters as a control: every failure below is caused by the axis, not by the
estimator, the grid, or the evaluation time.

---

## 3. Axis 1 — treatment positivity: a variance failure

Coverage degrades monotonically with `gamma`, and SE/SD tracks it almost exactly.
At n = 250, τ = 3.12:

| level | P(min(e, 1−e) < 0.05) | ESS frac. | coverage | SE/SD | std. bias | miss L / R |
|---|---|---|---|---|---|---|
| base | 0.000 | 0.94 | 0.942 | 0.967 | −0.007 | 0.030 / 0.028 |
| OV1 | 0.000 | 0.74 | 0.930 | 0.983 | −0.032 | 0.027 / 0.043 |
| OV2 | 0.027 | 0.42 | 0.918 | 0.959 | 0.021 | 0.044 / 0.038 |
| OV3 | 0.120 | 0.17 | 0.895 | 0.917 | 0.088 | 0.061 / 0.044 |
| OV4 | 0.332 | 0.013 | **0.703** | **0.610** | 0.084 | 0.164 / 0.133 |

The stress is real well before `P(e < 0.05)` moves: OV1 already costs a quarter of
the effective sample size while producing no draws below 0.05 at all.

Standardised bias stays below 0.09 and the misses stay symmetric. **This is not a
bias problem — the point estimate is fine and the interval around it is too
short.** The predicted coverage from SE/SD alone is 0.768 at OV4 against 0.703
observed; the residual is heavy tails, not centring.

**More data does not help.** At n = 500, OV4 gives 0.736 with SE/SD 0.631 — a
sample-size doubling buys 0.033 coverage. The binding constraint is the condition,
not n.

Where the interval first breaks (`study_b_breakpoints.csv`): **OV3** at τ ≥ 3.12,
**OV4** at τ = 0.48.

---

## 4. Axis 2 — rare events: a shape failure, confined to early τ

The rare-event axis lowers cause-1 incidence at the last τ from 0.29 to 0.02 while
leaving cause 2 common. At n = 250:

| level | τ = 0.48 | τ = 3.12 | τ = 8.61 |
|---|---|---|---|
| base | 0.948 | 0.942 | 0.939 |
| RA1 | 0.890 | 0.944 | 0.942 |
| RA2 | 0.746 | 0.946 | 0.941 |
| RA3 | **0.470** | 0.903 | 0.929 |

RA3 at τ = 0.48 has SE/SD 0.701 — but the misses are **0.530 left and 0.000
right**. A purely under-dispersed interval misses symmetrically; this one misses
on one side only. The RD is being estimated near the boundary of its support with
a mean width of 0.017, and the sampling distribution is skewed, not merely narrow.
This is why the transformed-scale intervals are in the design: at F = 0.005 the
Wald lower bound is −0.0048, an impossible cumulative incidence.

The failure **does not persist**: by τ = 3.12 coverage is back to 0.903 and by
τ = 8.61 to 0.929. n = 500 halves the shortfall (RA3 at τ = 0.48 → 0.726). So the honest
statement is not "rare events break the intervals" but "estimating a risk
difference at a time when almost nothing has happened breaks the intervals".

---

## 5. Axis 3 — censoring positivity: no failure from the amount, a sign flip with τ

Holding the censored fraction fixed at 30 % and raising only the *dependence* of
censoring on covariates and treatment, the failure is not monotone in the way the
other two are. At n = 250, CN4:

| τ | coverage | SE/SD |
|---|---|---|
| 0.48 | 0.988 | 1.452 |
| 3.12 | 0.954 | 1.025 |
| 8.61 | 0.895 | 0.852 |

Early τ **over-covers** — nominally excellent, but the interval is 45 % wider than
it needs to be. Late τ under-covers. Both are visible only in the ratio; coverage
alone reads the first as success. At n = 500 the late-τ end reaches 0.854 with
SE/SD 0.776, and the breakpoint moves down a level to CN3.

---

## 6. The null condition: type-I error is close to nominal

Setting `theta = [0, 0]` (both causes — zeroing only cause 1 leaves RD = −0.023,
because cause 2's hazard enters cause 1's CIF through Λ):

| n | τ = 0.48 | τ = 3.12 | τ = 8.61 |
|---|---|---|---|
| 250 | 0.055 | 0.065 | **0.077** |
| 500 | 0.040 | 0.058 | 0.046 |

Against a nominal 0.05 with MC-SE 0.008. The 0.077 at n = 250 and the latest τ is
a genuine ~3σ excess and is the one place the null condition shows anything; it
disappears by n = 500. No systematic inflation.

---

## 7. Which failures belong to the estimator and which to the nuisances

The `oracle` arm substitutes the true nuisances. Comparing it to `correct` at
n = 250 separates "the variance estimator is wrong under this condition" from "the
nuisance fit is noisy under this condition", and the answer differs by axis:

| cell | τ | correct | oracle | reading |
|---|---|---|---|---|
| base | 8.61 | 0.939 | 0.939 | control |
| **OV4** | 3.12 | 0.703 | 0.709 | **unchanged** |
| **OV4** | 8.61 | 0.718 | 0.710 | **unchanged** |
| **RA3** | 0.48 | 0.470 | 0.470 | **unchanged** |
| CN3 | 8.61 | 0.902 | **0.950** | repaired |
| CN4 | 8.61 | 0.895 | 0.921 | mostly repaired |

**The overlap and rare-event failures survive the true nuisances.** They are
properties of the influence-curve variance estimator under those conditions, not
artefacts of estimating the propensity or the hazards — no better nuisance model
would fix them. The late-τ censoring failure is the opposite: give it the true
censoring mechanism and it is gone. That is a nuisance-estimation problem, and it
*is* fixable by a better model.

This split is the most actionable result in the study, and it also says the two
remedies below are being asked to fix the right things.

---

## 8. Remedy 1 — propensity truncation makes overlap *worse*

`min_nuisance` is the obvious lever for a positivity problem. It does not work.
At OV3, n = 250:

| `min_nuisance` | τ = 0.48 | τ = 3.12 | τ = 8.61 | SE/SD (τ=0.48) | std. bias (τ=3.12) |
|---|---|---|---|---|---|
| 0.01 | 0.955 | 0.895 | 0.887 | 1.206 | 0.088 |
| 0.025 | 0.886 | 0.830 | 0.886 | 0.783 | 0.117 |
| 0.05 | 0.856 | 0.826 | 0.892 | 0.667 | 0.113 |
| 0.10 | 0.848 | 0.848 | 0.876 | 0.679 | 0.120 |

Coverage falls at every τ, SE/SD collapses from 1.21 to 0.68, and standardised
bias rises. Truncation trades variance for bias in roughly equal measure and loses
on both. Under CN3 the same sweep is flat (0.983/0.961/0.902 → 0.962/0.944/0.910),
so this is specific to the positivity axis — precisely the axis it is normally
recommended for.

**Recommendation: do not raise `min_nuisance` in response to poor overlap.**

## 9. Remedy 2 — a larger update budget does nothing

RA3's median resample uses the full 200 update steps, which invites the theory
that the loop is simply running out. Raising `max_updates` to 1 000:

| `max_updates` | τ = 0.48 | τ = 3.12 | τ = 8.61 |
|---|---|---|---|
| 200 | 0.470 | 0.903 | 0.929 |
| 1 000 | 0.470 | 0.895 | 0.929 |

Identical. The rare-event failure is not a convergence failure.

---

## 10. Remedy 3 — the bootstrap, and the filter that was destroying it

### 10.1 The filter

`pytmle/bootstrap.py` dropped, per `(Event, Time)` target, every resample whose
`Converged` flag was false. Study B tags draws instead of filtering them, so the
same resamples can be scored under four rules (see
[BOOTSTRAP_FAILURES.md](BOOTSTRAP_FAILURES.md)) — `pct_all` (no filtering, which
is what PyTMLE does now), `pct_convfilter` (the old behaviour), `pct_dropmode1`,
`pct_strict`.

The filter's cost, paired at the replicate level (`study_b_attribution.csv`):

| cell | τ = 0.48 | τ = 3.12 | τ = 8.61 |
|---|---|---|---|
| B_OV2 | −0.055 | −0.020 | −0.007 |
| B_OV3 | **−0.188** | −0.101 | −0.067 |
| B_OV4 | **−0.257** | −0.141 | −0.076 |
| B_RA2 | −0.067 | −0.004 | 0.000 |
| B_RA3 | −0.121 | +0.052 | +0.013 |
| base | +0.007 | 0.000 | 0.000 |

Why it is so destructive is visible in `study_b_failures.csv`: the fraction of
resamples flagged non-converged rises from 0.167 at the base condition to 0.745 /
0.920 / **0.973** at OV2 / OV3 / OV4. At OV4 the surviving effective B is **6.1 of
100** and every replicate is below 40 — the quantile is being read off six numbers.
Worse, the filter is **selection on the outcome**: what survives is the resamples
that solved the score equation, which are systematically the narrow ones.

**Non-convergence of a resample is not a reason to discard it.** `pct_all` beats
`pct_dropmode1` and `pct_strict` on every stress cell, and those filters were never
shipped — they exist here only to attribute the loss.

### 10.2 Where the bootstrap helps and where it hurts

Unfiltered percentile intervals against Wald, both on the same replicates:

| cell | τ | Wald | percentile (all draws) |
|---|---|---|---|
| B_OV2 | 3.12 | 0.920 | 0.960 |
| B_OV3 | 3.12 | 0.900 | **0.947** |
| B_OV3 | 8.61 | 0.887 | **0.940** |
| B_OV4 | 3.12 | 0.733 | **0.960** |
| B_OV4 | 8.61 | 0.787 | **0.960** |
| B_RA2 | 0.48 | 0.746 | **0.540** |
| B_RA3 | 0.48 | 0.450 | **0.248** |

**The two axes want opposite remedies.** Under positivity stress the bootstrap
repairs Wald completely — it widens on exactly the replicates where the analytic
SE is too small, which is what the analytic SE cannot do. Under rare events at
early τ it makes things substantially worse, and the mechanism is plain: the
resamples contain even fewer events than the original sample, so the bootstrap
distribution degenerates in the direction the estimator was already failing.

### 10.3 Percentile against reverse-percentile: same width, different place

The reverse-percentile (`basic`) interval is the percentile interval reflected
about the point estimate, `(2θ̂ − q_hi, 2θ̂ − q_lo)`. The two therefore have
**identical width** by construction, and any difference in coverage is purely a
difference in *location* — which makes the pair a clean read on whether the
bootstrap distribution sits off-centre from the point estimate.

It is a third axis-dependent reversal, and the sharpest of them:

| cell | τ | `pct_all` | `basic_all` | width (both) |
|---|---|---|---|---|
| B_OV3 | 3.12 | **0.947** | 0.807 | 0.250 |
| B_OV4 | 3.12 | **0.960** | 0.613 | 0.344 |
| B_OV4 | 8.61 | **0.960** | 0.640 | 0.420 |
| B_RA2 | 0.48 | 0.540 | **0.727** | 0.029 |
| B_RA3 | 0.48 | 0.248 | **0.450** | 0.015 |

Under positivity stress the reflection points the wrong way and costs up to 0.347
coverage at no saving in width. Under rare events at early τ it corrects the
one-sided miss and nearly doubles coverage, again at the same width. So the
bootstrap distribution is offset in opposite directions on the two axes, and no
single construction is right for both.

These numbers are **derived, not measured**: the reflection is exact given the
stored percentile bounds and point estimate, verified to 0.0 across 3 391 rows on
cells that carry both. `_derive_basic` in `study_b_report.py` adds them at load
time, so every filter gets its `basic_*` counterpart without a re-run.

### 10.4 B was never the constraint

`B_BASEb500` runs the base condition at B = 500 against `BS12`'s B = 100:

| B | τ = 0.48 | τ = 3.12 | τ = 8.61 |
|---|---|---|---|
| 100 | 0.912 | 0.928 | 0.928 |
| 500 | 0.910 | 0.910 | 0.970 |

Indistinguishable at 100–250 replicates. Raising B would not have rescued the
filtered bootstrap; removing the filter did.

---

## 11. What to tell a user

1. **Report SE/SD, or at least be aware of it.** Coverage alone cannot distinguish
   an honest interval from a conservative one from a broken one.
2. **Under poor overlap, use the bootstrap.** The analytic interval under-covers
   by up to 0.22 at n = 500 and does not improve with n. Do not raise
   `min_nuisance` — it makes it worse.
3. **Under rare events, do not use the percentile bootstrap.** At early τ it is
   worse than Wald (0.540 against 0.746 at RA2). If a bootstrap interval is
   wanted there, the reverse-percentile is the one to use — same width, and it
   recovers most of the gap — but treat any τ at which the estimand sits near its
   support boundary as unreliable regardless of procedure, and wait for a later τ
   if the question allows it.
4. **Late-τ dependent censoring is a nuisance-modelling problem**, not an interval
   problem: the oracle arm recovers nominal coverage.
5. **`max_updates` is not a lever** for any failure observed here.

---

## 12. Limitations

- **`bca` is confounded with the filter, and is not reported.** It was computed
  on the convergence-filtered draws only, so its weak showing measures the filter
  rather than the construction. Unlike `basic` it cannot be recovered from stored
  output — its bias-correction needs the draw distribution and its acceleration
  the influence curve, neither of which was saved — and re-running the bootstrap
  cells to obtain it was judged not worth ~90 CPU-hours once `basic` had answered
  the same question for free. Future runs emit all three constructions under all
  four filters and archive the raw draws, so the question is cheap from here on.
- **The bootstrap resamples the second stage only**, never refitting nuisances —
  as PyTMLE does, and for the reason it cites. Under `oracle` that is the correct
  bootstrap; under `correct` it omits a genuine source of variability, which is
  measured here rather than worked around.
- **Nine base cells lack condition diagnostics.** `W_*` and `BS12` were run before
  those columns existed. Every stress cell has them, so no result above is
  affected.
- **The correlated-noise sub-axis was dropped** after measuring no effect. The
  cause is structural: unpenalised MLEs depend only on the column span of the
  design, so AR(1) correlation among noise covariates is inert. `noise_rho`
  remains in the DGP with the inertness documented and pinned by a test.
- **Bootstrap cells run at 150 replicates**, so their MC-SE is 0.018 — the
  Wald-vs-bootstrap contrasts are paired and therefore much sharper than that, but
  the marginal coverages are not.

---

## 13. Reproducing the bootstrap results without re-running them

The bootstrap cells cost ~519 of Study B's 789 CPU-hours, so anything that avoids
re-running them is worth knowing about.

**`basic_*` is derived, not measured.** `_derive_basic` in `study_b_report.py`
reflects each stored `pct_*` interval about its point estimate at load time, so
every filter gains its reverse-percentile counterpart with no compute. Exact by
construction, verified to 0.0 against cells that carry both.

**Raw draws are archived.** A cell with `n_bootstrap > 0` now writes
`draws_*.parquet` beside each shard: one row per (resample, estimand, event, τ,
arm), carrying the per-draw convergence flags. Nothing in the report reads them.
They exist so that a *new* interval construction — studentized, a different α, a
filter not currently defined — is a table operation rather than another 519
CPU-hours. The cells run before this change have no archive, which is why `bca`
on unfiltered draws is absent from this document rather than merely unreported.
