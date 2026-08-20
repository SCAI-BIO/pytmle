# Findings

Things discovered while building the harness that are worth keeping, either
because they are defects in `pytmle` or because they are traps that a future
version of this study could fall into again.

---

## 1. `min_nuisance` is silently discarded after the first update step

**Status:** confirmed, reported, and **fixed upstream in `c43539e`** ("fix: keep
min_nuisance in updated estimates"). Verified after the fix: passing 0.01 / 0.2 /
0.5 now yields exactly those values. Kept here because the diagnosis is what the
sweep in Study D depends on.

`PyTMLE.fit(min_nuisance=...)` is honoured when the initial `UpdatedEstimates`
are built (`tmle_update.py:344`), but the targeting loop reconstructs those
objects on **every iteration** and does not forward the argument:

```python
# pytmle/tmle_update.py:210 -- inside tmle_loop
new_ests[trt] = UpdatedEstimates(
    times=eval_times,
    hazards=new_hazards,
    ...
    g_star_obs=est_a.g_star_obs,
    g_comp_est=est_a.g_comp_est,
)   # <-- min_nuisance is not passed
```

`UpdatedEstimates.__post_init__` (`estimates.py:122`) then sees `min_nuisance is
None` and recomputes the default `5 / sqrt(n) / log(n)`. So the user's value
governs step 0 only; every subsequent step, and therefore the final estimates and
their influence curve, uses the default.

**Reproduction** (n = 600, where the default is 0.031910):

| passed | final `min_nuisance` | honoured? | RD |
|---|---|---|---|
| 0.01 | 0.031910 | no | −0.27750 |
| 0.2 | 0.031910 | no | −0.27753 |
| 0.5 | 0.031910 | no | −0.27766 |
| `None` | 0.031910 | n/a | −0.27750 |

A 50× change in the requested bound moves the estimate by 1.6e-4, and passing
0.01 is indistinguishable from passing nothing.

**The fix** was to add `min_nuisance=est_a.min_nuisance` to that constructor call.

**Consequences for this study.** Study B's n-ladder was designed around pinning
the bound so an n-dependent truncation could not confound it, and Study D
includes a `min_nuisance` sweep — neither works as specified. It was *not* affecting any measured result before the fix: the nuisance
denominators span roughly [0.12, 0.79], an order of magnitude above both 0.01 and
0.0319, so truncation never bound. Post-fix confirmation: the risk difference is
identical at 0.01 and at the default, and moves only at 0.5, where truncation
finally binds. It would have bitten in a positivity-stressed regime, which is
exactly where Study D looks.

---

## 2. A nuisance-level discrepancy is not evidence of an estimand-level one

**Status:** design trap, fixed.

The first version of the double-robustness design used a threshold term with the
same coefficient in both arms (`delta_j = 4`, no interaction). Omitting it from
the outcome model produced a **twelvefold** error in the cumulative hazard
(`cumhaz_mae` 0.54 → 6.34) and moved the risk difference by **0.002** against a
truth of −0.149. The misspecification was real at the nuisance level and
invisible at the estimand level, which would have made four of Study A's eight
cells inert.

Cause: a term entering both arms identically shifts `F^1` and `F^0` together, so
the *contrast* survives. The fix is the treatment × threshold interaction `eta`,
which makes the omitted term arm-specific.

The guard that caught it is `sim/calibrate.py`, which measures asymptotic bias in
the **estimand** using `gcomp` as the probe. The guard that missed it was the
nuisance-level `cumhaz_mae` check. Both are kept; only the first is decisive.

---

## 3. `delta = 4` is not portable from a discrete-time DGP to a continuous-time one

**Status:** calibration error, fixed.

`delta_j = 4` was inherited from Hage et al., where it is a log-hazard contrast
across a threshold. In this continuous-time parameterisation it multiplies the
hazard **rate** by `e^4 ≈ 55`, giving a 150× spread across subjects:

| config | u frac | hazard rate median / p99 / max | p01 event time | TMLE failures |
|---|---|---|---|---|
| threshold at 1, `delta = 4` | 0.18 | 0.17 / 25.9 / 38.0 | 0.0028 | **50–75 %** |
| threshold at 0, `delta = 1.5` | 0.55 | 0.80 / 1.62 / 2.4 | 0.011 | 0 |

Subjects above the threshold failed almost immediately, event-free survival
collapsed to zero within a few grid points, and PyTMLE hit its survival floor.
Lowering the threshold to 0 and `delta` to 1.5 fixed the conditioning and, as a
bonus, put the misspecification on 55 % of the sample instead of 18 %, which
strengthens the censoring-model lever too.

---

## 4. A guard that drops failed replicates is worse than no guard

**Status:** fixed in `sim/calibrate.py`.

`asymptotic_bias` originally returned `None` for a failed replicate and filtered
those out. When finding 3 was causing 50–75 % failures, the guard reported
confident-looking bias estimates computed from the surviving quarter — a subset
selected on the fit having succeeded, and therefore not comparable across cells.
It also produced an apparently impossible result (`gcomp` differing between cells
that share the same outcome model) which was really just different surviving
subsets.

It now tracks attempted-vs-usable replicates per cell and raises a
`RuntimeWarning` naming any cell below 95 % completion.

---

## 5. Hage et al.'s `trueCSH()` targets a different population in Scenarios 2 and 4

See [VALIDATION.md](VALIDATION.md#rung-2b--which-population-is-the-target--finding).
Their estimators track the realised (mixture) covariate distribution to within
0.00085, while `trueCSH()` with default arguments returns the unconditional-population
value, differing by up to 0.072 on the per-arm risk scale.


---

## 6. Misspecifying a bounded nuisance by distorting its shape fights its own bounds

**Status:** design lesson, resolved after three failed attempts.

The propensity lever went through four designs before one worked. All three
failures share a cause worth stating once, because it will recur:

| device | why it failed |
|---|---|
| control-resampling (Hage et al. Scenario 2) | true `e(W) = sigma/(sigma + 1 - q)`; a plain logistic tracks it to ~4 % relative error. Real at the nuisance level (`ps_mae` 0.003 vs 0.018), invisible at the estimand: `ipw` moved by 0.0001 between the correct and wrong cells. |
| threshold `delta_pi * u` | `u = 1{w_cont > 0}` is nearly a monotone transform of `w_cont`, so a logistic in `w_cont` absorbs it by steepening its coefficient. \|z\| <= 1.8 at every `delta_pi`; also pushed `e(W)` to 0.99. |
| centred threshold `delta_pi * (u - 0.5)` | fixed the asymmetry, made the lever *weaker* still (z 4.0 -> 0.5). Range still [0.02, 0.99]. |
| quadratic `delta_pi * (w_cont^2 - 1)` | not absorbable by a linear term, but large in *both* tails, so `e(W) -> 1` above the 95th percentile and `ipw` was biased (+0.0145) even with `pi` correct. |

The pattern: a propensity is bounded in (0, 1), so any additive term in the logit
strong enough to misspecify meaningfully also drives `e(W)` to the boundary.
Strength and positivity are in direct conflict *as long as the misspecification
lives inside the true propensity*.

**What works:** an **omitted confounder**. The truth stays plain logistic in `W`,
so positivity is governed solely by `gamma`, and the wrong fit simply drops
`w_cont` -- a covariate that genuinely drives both treatment and outcome.
Measured: `ps_mae` 0.0069 correct vs 0.1009 wrong (14.6x), `e(W)` centred at 0.51
with P(e < 0.05) = P(e > 0.95) ~ 0.

This does not contradict the threshold working for `Q` and `G`: there the term
enters a *hazard*, which is unbounded above, and a Cox model cannot reshape its
baseline the way a logistic can rescale a linear predictor.

**Diagnostic note.** For much of this I was reporting `min`/`max` of `e(W)` over
20 000 draws as the positivity check. That is an extreme-value statistic about a
tail never sampled at n = 800. Percentiles and `P(e < 0.05)` are the right
summaries, and switching to them changed which configurations looked acceptable.

---

## 7. TMLE's substitution estimator retains ~35 % of plug-in bias where the one-step retains ~10 %

**Status:** **traced to a defect in the targeted update, shared by PyTMLE and
concrete.** An earlier version of this section called it "a property of the
algorithm, not of PyTMLE ... a *result*, not a defect". That was wrong, and
section 9 below has the diagnosis and the one-line experiment that settles it.
The measurements in this section stand; only their interpretation changes.

In cell C5 (outcome model wrong, propensity and censoring both correct) double
robustness says TMLE should be consistent -- and the one-step built from the same
influence function, on the same injected nuisances, is clean. TMLE is not.

Ruled out first, in order: the `min_nuisance` defect (finding 1 -- fixed, and the
bound never binds here anyway); a nuisance mismatch between the two estimator
paths (propensity, censoring survival and `nuisance_weight` compared elementwise,
max\|diff\| = 0); and non-convergence (12/12 replicates converged, median 26 steps,
\|\|PnEIC\|\| driven down 97.7 %, 108/108 targets meeting the stopping rule, zero
weights at the truncation bound).

The decisive test was to run **`concrete`'s second stage on byte-identical
injected nuisances**. PyTMLE's targeted update is a port of concrete's, so
agreement separates "the algorithm does this" from "the port does this".
`R/run_concrete_injected.R` overwrites every component of concrete's `Estimates`
object with the Python values and calls `getEIC` -> `doTmleUpdate` -> `getOutput`;
`getInitialEstimate` runs once only, to obtain the scaffolding.

Bias vs truth, 30 replicates, n = 800, identical nuisances:

| event | tau | `aipw` | `gcomp` | `tmle` (PyTMLE) | `tmle` (concrete) |
|---|---|---|---|---|---|
| 1 | 0.51 | −0.0016 | +0.0303 | **+0.0183** | **+0.0188** |
| 1 | 1.15 | −0.0001 | +0.0505 | **+0.0315** | **+0.0323** |
| 1 | 2.67 | +0.0059 | +0.0578 | **+0.0375** | **+0.0392** |
| 2 | 0.51 | −0.0014 | −0.0318 | **−0.0196** | **−0.0202** |
| 2 | 1.15 | −0.0028 | −0.0471 | **−0.0298** | **−0.0306** |
| 2 | 2.67 | −0.0160 | −0.0461 | **−0.0344** | **−0.0360** |

Mean \|bias\|: `gcomp` 0.0439, **`tmle` 0.0285, `concrete` 0.0295**, `aipw` 0.0046.

The two implementations agree to within 0.002 at every target (~4 % relative,
inside Monte Carlo noise), with concrete marginally *more* biased. Both converged
everywhere (concrete 180/180, median 39 steps).

So under a badly misspecified `Q` with correct `g` at n = 800, the targeted
substitution estimator removes ~35 % of the plug-in bias while the one-step built
from the same influence function removes ~90 %. Both are first-order equivalent;
the gap is the second-order remainder, which the substitution form carries and
the additive correction does not.

### Does it shrink? -- settled by Study A: no

The calibration n-ladder (40 replicates at n = 400 / 800 / 1600) gave 0.0298 /
0.0299 / 0.0239 and could not separate "slowly shrinking" from "persistent".
Study A's full budget can. Bias in the cause-1 risk difference, C5, 500
replicates per cell:

| n | `gcomp` | `aipw` | `tmle` | `tmle (concrete)` |
|---|---|---|---|---|
| 250 | +0.0438 | +0.0024 | **+0.0283** | +0.0356 |
| 500 | +0.0433 | −0.0001 | **+0.0274** | +0.0318 |
| 1000 | +0.0439 | +0.0007 | **+0.0283** | +0.0272 |

**The TMLE residual is flat in `n`.** Quadrupling the sample size moves it by
under 4 %, where `1/sqrt(n)` predicts a halving; `aipw` over the same range falls
to the Monte Carlo floor, and `gcomp` is flat as an asymptotically biased plug-in
should be. concrete tracks PyTMLE at every `n` (the 250 gap is the largest at
0.007, on 150 replicates against 500). So the ~35 % retained bias is an
asymptotic property of the substitution form under this misspecification, not a
second-order remainder that estimation noise was hiding.

The same picture holds in C6-C8, and the `1/sqrt(n)` guide line in
`results/figures/study_a_bias_vs_n_rd.png` makes it visible directly: `tmle` and
`tmle (concrete)` run parallel to the horizontal in the whole bottom row while
`aipw` descends.

The consequence for inference is in the coverage table: because the interval
width shrinks like `1/sqrt(n)` while the bias does not, Wald coverage in C5
*degrades* with sample size -- 0.952 / 0.937 / 0.887 at n = 250 / 500 / 1000, and
0.907 / 0.879 / 0.754 in C8. `aipw` holds 0.927 / 0.937 / 0.944 in C5. This is
the natural setup for Study B.

This is also the contrast the plan called the sharpest in the study: `tmle` and
`aipw` share an identical influence function by construction, so the difference
between them is purely the substitution step.

## 8. PyTMLE and concrete agree on the *targeting*; the residual gap is grid discretisation

**Status:** measured. It changes how the tier-1 agreement gate should be read,
and it makes an assertion in `R/run_concrete_injected.R` false — now corrected
there.

The bridge injects our nuisances into concrete by evaluating our cumulative
hazards at concrete's own time points. The header comment justified the constant
interpolation with "our grid is the unique observed times, concrete's `Times` is
a subset of it, so constant interpolation is exact rather than approximate."
**`Times` is not a subset of our grid.** At n = 300 concrete builds a 209-point
grid where we have 300 unique observed times, `all(Times %in% grid)` is `FALSE`,
and — the part that matters — the target times land *on* concrete's grid and not
on ours:

```
taus     [1.60539 3.11961 5.4209 ]
our grid nearest <= tau  [1.59469 3.11675 5.40743]
concrete Times contains every tau exactly
```

So both implementations compute the same discrete plug-in `sum(S(t-) h(t))`, from
the same hazards, on **different discretisations of the time axis**. That is a
real difference in the estimand being computed, not in the algorithm computing
it.

Two things identify it as discretisation rather than targeting:

* **it moves `gcomp` too.** The plug-in involves no targeting at all, and it
  carries the same gap as `tmle` (mean 0.0016 vs 0.0017 over the Study C pilot).
  Snapping the target times down onto our grid changes PyTMLE's `gcomp` by
  exactly 0 — it already evaluates at the last grid point at or before `tau`.
* **the targeting increment agrees ~6x more tightly.** Comparing `tmle - gcomp`,
  which is what the update step actually contributes:

| quantity | mean | max |
|---|---|---|
| level, \|`tmle` − `tmle (concrete)`\| | 0.00116 | 0.00328 |
| increment, \|(`tmle`−`gcomp`) − (`tmle`−`gcomp`)_concrete\| | **0.00018** | **0.00057** |

**Consequence for Study C.** The primary tier-1 gate is the *targeting
increment*, not the CIF level: at the level, a ~1e-3 disagreement is expected
from the grid and would be there even if the two implementations were the same
code. Quoting the level alone would either mask a real algorithmic difference of
that size or manufacture one. Study A's PyTMLE-vs-concrete columns carry the same
component, which is part of why the two differ by 0.002–0.007 there.

**Not a defect in either package.** Neither grid is wrong; they are different
conventions for where a step function is evaluated, and both converge to the same
limit as the grid refines.

## 9. The targeted update moves only the subjects observed in the arm it is updating

**Status:** located, reproduced, and confirmed by a controlled experiment.
Present in **both** PyTMLE and `concrete` 1.0.8, PyTMLE having inherited it.
This is the cause of finding 7.

### The contradiction

Both source papers say the substitution estimator is consistent in C5 (`Q` wrong,
`pi` and `G` correct). Chen et al. Section 7.1 report exactly that. The algebra
says why: for this parameter the second-order remainder is an **exact product**,

```
R2 = int E_L[ S0(t-) c_{j,l,t}(Q) (lam_l - lam_0l) (1 - (pi0/pi)(S0^c/S^c)) ] dt
```

so it vanishes for *any* `lambda`, however wrong, once `g` is correct. Study A
measures +0.028, flat from n = 250 to n = 1000.

### What it is not

Each of these was measured, not argued:

| candidate | test | result |
|---|---|---|
| targeting stops before solving the score | evaluate `Pn D*` at the final estimates | **solved**: mean \|Pn D*\| 0.0006-0.0018 against the loop's own threshold of 0.005-0.008; `frac_over_threshold` = 0.000. Adding the residual back changes the estimate by 0.0003. |
| nuisance estimation error in `g` | rerun with **oracle** `pi` and `G`, making `R2` exactly zero | bias unchanged: 0.0181 / 0.0264 / 0.0319 against 0.0183 / 0.0277 / 0.0323 |
| the path is traversed too coarsely | `one_step_eps` = 0.1 / 0.01 / 0.002, `max_updates` 2000 | 0.0264 / 0.0263 / 0.0263 -- no dependence at all |
| mixing `S = exp(-Lambda)` with `F = sum S(t-) dLambda` | measure `sum_j F_j(tau) + S(tau) - 1` | real but an order of magnitude too small: mean 0.002 against a bias of 0.028 |

### Where the bias is created

With oracle `g` the identity `Psi(Q_eps) - psi_0 + Pn D*(Q_eps) = (Pn - P0) D*(Q_eps)`
holds at every `eps`, so tracking that quantity along the path localises the
damage exactly. Averaged over 24 replicates at n = 500, cause 1, tau = 1.15:

| step | bias | `Pn D*` | `(Pn - P0) D*` |
|---|---|---|---|
| 0 | +0.0460 | -0.0408 | **+0.0053** (~0: this is why `aipw` is unbiased) |
| 1 | +0.0292 | +0.0087 | **+0.0379** |
| 40 | +0.0310 | -0.0000 | +0.0310 |

**The whole bias appears in the first update step and never recovers.** One step
moves the score by +0.0495 -- past zero, 121 % of what was needed -- while moving
`Psi` by only -0.0168, 34 % as far. That ratio *is* finding 7's headline number:
the update reaches "score solved" having carried the parameter about a third of
the distance, and the other two thirds of the plug-in bias stays.

### The cause

`tmle_update.update_hazards` (and `concrete:::updateHazardsCpp`) fluctuates the
hazard by `exp(eps * h)` with

```
h = g_star_obs * nuisance_weight * (1{l = j} - h_fs),     g_star_obs = 1{A_i = a*}
```

`g_star_obs` is the **observed** treatment indicator. Subjects with `A != a*`
therefore get `h = 0` and their hazard is multiplied by `exp(0) = 1` -- never
updated. That is correct in the *influence curve*, whose martingale term only
runs over what was observed. It is not correct here: `est_a["Hazards"]` is the
**counterfactual** hazard for arm `a*` (that is why there is one `Estimates`
entry per intervention), and the plug-in averages `F_j(tau | a*, L_i)` over all
`n` subjects -- of whom only the fraction in arm `a*` were moved.

The indicator belongs to the fluctuation evaluated at `a = a*`, where it is
`1{a* = a*} = 1` for everyone. **`concrete` stores precisely that quantity** and
does not use it here:

```
A=1   g.star.intervention: values={1}   mean=1.000  |  g.star.obs: values={0,1}  mean=0.537
A=0   g.star.intervention: values={1}   mean=1.000  |  g.star.obs: values={0,1}  mean=0.463
```

while `doTmleUpdate` passes `GStar = attr(est.a[["PropScore"]], "g.star.obs")`.

### The controlled experiment

Identical path, identical influence curve, 24 paired replicates, n = 500, oracle
`g`, C5. The **only** change is the multiplier inside the update:

| tau | `g.star.obs` (as shipped) | intervention indicator | MC-SE |
|---|---|---|---|
| 0.514 | +0.0190 | **-0.0003** | 0.005 |
| 1.150 | +0.0310 | **+0.0029** | 0.007 |
| 2.670 | +0.0329 | **+0.0007** | 0.009 |

The bias vanishes -- every target within one Monte Carlo standard error of zero.
The residual score is the same in both arms of the experiment (0.0008 / 0.0007 /
0.0002), so both solve the efficient influence curve equation equally well. The
difference is entirely *which subjects' counterfactual hazards get moved*.

### Caveat for anyone acting on this

The substitution is not free. Applying `1/pi(a*|L)` to every subject rather than
only to those observed in the arm exposes the update to small propensities for
subjects who were never in that arm: the experiment raised overflow warnings in
`get_influence_curve.py:120` and lost 1 of 24 replicates. A real fix has to pair
the intervention indicator with the truncation/stability handling, and should be
validated against `concrete` -- which needs the same change.

**Nothing in `pytmle/` was modified.** The experiment overrides the argument at
the call site from `sim/`, leaving the package untouched.

### Outcome of the fix (applied upstream; Study A n = 250, C1/C5/C7/C8, 500 reps)

Both changes were made in `pytmle/tmle_update.py`: the intervention indicator in
the update, and `h_fs` clipped to `[0, 1]`. The `g_star` parameter was dropped
from `update_hazards` entirely.

**The isolation is exact.** `gcomp`, `ipw` and `aipw` are bit-identical before
and after across every paired replicate (max \|difference\| = 0.000e+00) -- none
of them routes through `update_hazards`. Only `tmle` moves.

| cell | bias before | bias after | median after | 1 %-trimmed after |
|---|---|---|---|---|
| C1 all correct | -0.0000 | +0.0020 | +0.0045 | +0.0026 |
| **C5 `Q` wrong** | **+0.0282** | **-0.0149** | **+0.0005** | **-0.0010** |
| C7 `Q`,`pi` wrong | +0.0266 | +0.0074 | +0.0060 | +0.0074 |
| C8 all wrong | +0.0311 | +0.0160 | +0.0145 | +0.0160 |

**The systematic bias is gone.** C5's median and trimmed mean land on zero, which
is what double robustness predicts and what the pre-fix code never produced at
any `n`. C7 and C8 improve by half; they retain legitimate bias, since `pi` and
`G` are also wrong there and the theory offers no protection. Run health improved
too: replicate failures fell from 1.0-6.2 % to 0.0-0.2 %, non-convergence from
1.4-6.5 % to 0.8-4.4 %.

**But two costs remain, and neither should be glossed.**

*Rare divergence.* C5's untrimmed mean is `-0.0149` and its RMSE `0.279` --
driven entirely by **2 replicates out of 500**, one reaching `-6.25`. Trimming
1 % restores RMSE to `0.0489`, against `0.0474` pre-fix. The divergent replicates
are not the ones with extreme propensities (their `ps_min` is 0.066 against
0.0125 for the rest); they are the ones that **stop after 7 update steps instead
of the usual 26**, converged flag set. So a single large early step lands
somewhere that satisfies the stopping rule while being far from the truth. The
`h_fs` clip bounds the clever covariate but not the cumulative movement, and the
`||PnEIC||`-decrease guard does not catch it because the norm does decrease.

*Variance and coverage.* Even trimmed, C1's RMSE rises from 0.0390 to 0.0465, and
the influence-curve standard error stops tracking the spread: the ratio of mean
SE to empirical SD falls from 1.19/1.27/1.14/1.14 (conservative) to
0.86/0.24/0.91/0.92. Wald coverage falls accordingly -- C1 0.968 -> 0.921, C5
0.952 -> 0.913, C7 0.929 -> 0.912, C8 0.907 -> 0.909. Some of this is the honest
price of updating every subject rather than half of them, but a coverage of 0.92
in the all-correct cell is a regression that needs resolving before the change
can be called finished.

Figures: `results/figures/fix_comparison.png` (bias, spread, coverage) and
`results/figures/fix_replicates.png` (every replicate, with the off-scale ones
flagged rather than dropped).

### Final state of the fix (three changes, Study A n = 250, C1/C5/C7/C8, 500 reps)

`pytmle/tmle_update.py` now carries three changes: the intervention indicator in
`update_hazards`, `h_fs` clipped to `[0, 1]`, and `cif_within_bounds` used as a
step-acceptance test that halves `working_eps` and retries -- reusing the
backtracking the loop already had for the `||PnEIC||` test. The intermediate
`+-5` clip on the exponent was tried and **removed**: it fixed C5 but created a
`-38.7` estimate in C8 by altering the trajectory, which is what a fix that
constrains the step rather than the estimate will do.

| cell | bias before | bias after | RMSE before | RMSE after | divergent reps |
|---|---|---|---|---|---|
| C1 all correct | -0.0000 | +0.0022 | 0.0445 | 0.0549 | 0 |
| **C5 `Q` wrong** | **+0.0282** | **+0.0026** | 0.0511 | 0.0549 | 0 |
| C7 `Q`,`pi` wrong | +0.0266 | +0.0075 | 0.0501 | 0.0528 | 0 |
| C8 all wrong | +0.0311 | +0.0161 | 0.0521 | 0.0532 | 0 |

C5's bias falls by a factor of 11 (`z = -13.9`) and no cell has a single
divergent replicate -- against 2 in C5 with the indicator fix alone and 1 in C8
with the exponent clip. Replicate failures are 0.0 % everywhere, from 1.0-6.2 %
originally. `gcomp`, `ipw` and `aipw` remain bit-identical to the original run.

**The `min_nuisance` tolerance is not slack.** `tol = 1e-6`, the first value
tried, rejected the *first step of every fit*: `S` is built as `exp(-Lambda)`
while `F_j` accumulates as `sum S(t-) dLambda_j`, so the initial estimates
already have a median `max_j sum F_j` of 1.02, reaching 1.075, before any
targeting. Only 11.7 % of untargeted estimates passed. The symptom is silent --
`steps = 0` and a "targeted" estimate exactly equal to the plug-in -- which is
worth a warning in the fit path.

**Non-convergence rose but is benign.** C1 5.5 % -> 17.2 %, C5 6.5 % -> 21.4 %:
the guard rejects steps, so more fits exhaust `max_updates`. Splitting the bias
by the flag shows the non-converged fits are *not* the bad ones --

| cell | converged: reps / bias | non-converged: reps / bias |
|---|---|---|
| C1 | 414 / +0.0029 | 86 / **-0.0010** |
| C5 | 393 / +0.0039 | 107 / **-0.0023** |
| C7 | 489 / +0.0078 | 11 / -0.0035 |
| C8 | 466 / +0.0159 | 34 / +0.0198 |

-- with comparable spread. They are stopped mid-descent, not failing. Raising
`max_updates` is the remedy if the flag matters; the estimates do not need it.

**The open item is coverage.** C1 0.968 -> 0.927, C5 0.952 -> 0.924, C7 0.929 ->
0.913, C8 0.907 -> 0.909, with the SE/SD ratio moving from conservative
(1.14-1.27) to roughly calibrated (0.91-0.97). So the intervals are no longer
conservative and coverage sits 2-3 points below nominal even where the estimator
is unbiased and correctly specified. That is a separate question from the bias
this section is about, and it is unresolved.

## 10. Post-fix Wald coverage: the shortfall is SE calibration, and the *pre*-fix coverage was two errors cancelling

**Status:** decomposed at n = 250; the n-scaling that decides whether it needs a
code change is pending.

After the finding-9 fix, Wald coverage sits 2-3 points below nominal (C1 0.927,
C5 0.924) where it had been at or above it (0.968, 0.952). Coverage can fall
short from bias, from an under-stated standard error, or from non-normal tails,
and these call for different responses, so they were separated: for each cell,
the coverage a normal estimator with *that* bias and *that* SE ratio would attain
was computed and compared with the observed value.

| cell | bias | SE/SD | coverage | coverage if the SE were exact |
|---|---|---|---|---|
| C1 all correct | +0.0022 | 0.967 | 0.927 | **0.9498** |
| C5 `Q` wrong | +0.0026 | 0.973 | 0.924 | **0.9497** |
| C7 | +0.0075 | 0.910 | 0.913 | 0.9477 |
| C8 | +0.0161 | 0.925 | 0.909 | 0.9386 |

**In C1 and C5 the bias contributes nothing** -- with a calibrated SE, coverage
would be 0.950 exactly. The whole shortfall is the influence-curve SE
understating the estimator's spread by 3-9 %. Excess kurtosis is between -0.03
and 0.16, so tails are not the explanation either.

The mechanism is visible directly: the fix leaves `mean_se` *unchanged* (C1 at
tau = 2.67: 0.0633 before and after) while the empirical SD rises from 0.0550 to
0.0670. The influence curve is a function of the nuisances and the data; it does
not see the extra variability introduced by a targeting step that now moves three
times as far.

**The pre-fix coverage was not what it looked like.** Pre-fix C5 coverage of
0.952 came with an SE ratio of **1.27** -- intervals 27 % too wide -- against a
bias of +0.028. With a calibrated SE, pre-fix C5 coverage would have been
**0.897**. The apparently-nominal coverage was an over-wide interval compensating
for a mis-centred one. So the before/after comparison is not "coverage got
worse"; it is *biased with intervals that hid it* against *unbiased with
intervals ~5 % too narrow*.

**What decides the response.** If the excess variance is the finite-sample price
of data-adaptive targeting, the SE ratio should approach 1 with `n`. The n = 500
and n = 1000 tiers settle it: a ratio moving 0.967 -> ~0.99 makes this a
documented small-sample caveat, while a ratio flat near 0.96 means the influence
curve is missing a variance component and the package needs a change.

### Resolved by the full run: the SE shortfall is confined to the pi-wrong cells

The n = 250 analysis above pooled all eight cells and concluded the influence-
curve SE understates the spread everywhere. With all three sample sizes and the
complete factorial, the shortfall splits cleanly by whether the **propensity**
model is correct:

| n | SE/SD, `pi` correct | SE/SD, `pi` wrong | coverage, `pi` correct | coverage, `pi` wrong |
|---|---|---|---|---|
| 250 | 0.989 | 0.906 | 0.929 | 0.915 |
| 500 | 0.961 | 0.918 | 0.929 | 0.921 |
| 1000 | **1.004** | 0.923 | **0.946** | 0.910 |

At n = 1000 the `pi`-correct cells are C1 1.052, C2 0.998, C5 0.985, C6 0.983 --
calibrated -- against C3 0.934, C4 0.877, C7 0.936, C8 0.944.

**This is what the theory predicts, not a defect.** The efficient influence
curve is the estimator's actual influence function only when `g` is correctly
specified and estimated fast enough. Under `g`-misspecification the estimator
remains asymptotically linear but with a *different* influence function, one
carrying a contribution from estimating the misspecified `g`'s limit, so the
EIF-based variance is simply the wrong one. The plan said as much from the
start -- "double robustness buys consistency, not inference" -- and this is that
sentence measured.

So: **where inference is guaranteed, it is valid.** SE/SD 1.004 and coverage
0.946 at n = 1000 across the `pi`-correct cells, with the earlier apparent
regression explained by pooling those cells with the ones where no guarantee
applies. The `pi`-wrong cells sit at 0.92 and are climbing slowly with `n`
(0.906 -> 0.918 -> 0.923), consistent with a second-order term rather than a
fixed defect.

Nothing in `pytmle` needs changing for this. It belongs in the write-up as a
caveat on interval interpretation.
