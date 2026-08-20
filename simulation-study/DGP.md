# The data-generating process

A short description of the simulation design used in Study A, intended for
inclusion in a larger report. Implementation: `sim/dgp.py`; the closed-form truth
is in `sim/truth.py`; the calibration evidence behind each choice is in
[FINDINGS.md](FINDINGS.md).

## What the design has to support

The study asks whether a targeted estimator is consistent when some of its
nuisance models are wrong. That imposes four requirements that together decide
almost every parameter value:

1. **Continuous event times.** PyTMLE targets continuously distributed
   event times, so a discrete-grid design would exercise a different code path
   than the one under test.
2. **Competing risks**, since the cause-specific structure is what distinguishes
   this estimator from a survival TMLE.
3. **A truth known in closed form**, so that bias is measured rather than
   approximated by a reference estimator.
4. **Misspecification that moves the estimand.** A model can be badly wrong at
   the nuisance level and leave the target parameter untouched; a design that
   does not check this produces cells that look informative and are inert.

## Structure

Two baseline covariates: a three-level categorical `w_cat` in equal thirds,
entered as dummies `d2`, `d3`, and a standard normal `w_cont`. The design matrix
is `X = [d2, d3, w_cont]`. A threshold indicator `u = 1{w_cont > c}` enters the
treatment-free part of the model and is the device by which models are
misspecified.

```
A      ~ Bernoulli(expit(gamma'X))
lam_j  = exp(alpha_j + X'beta_j + theta_j A + delta_j u + eta_j A u)     j = 1, 2
lam_C  = exp(alpha_C + X'beta_C + theta_C A + delta_C u + eta_C A u)
```

Event times are drawn as latent exponentials per cause, with `T = min(T_1, T_2)`
and `J = argmin`; censoring is drawn from its own exponential hazard, and the
observed data are `(min(T, C), Delta, A, W)`. Constant (exponential)
cause-specific hazards are what make the truth available in closed form; the
covariate and treatment dependence still make the *marginal* estimand a
non-trivial average.

## Coefficient values

Every weight in the Study A configuration (`threshold` in `sim/dgp.py`).
Covariate coefficients are ordered `(d2, d3, w_cont)`, matching
`X = [d2, d3, w_cont]`.

**Treatment assignment.** `A ~ Bernoulli(expit(gamma'X))`

| symbol | role | value |
|---|---|---|
| `gamma` | covariate effects on assignment | `(1.0, -0.8, 0.6)` |

There is no intercept: `E[expit(gamma'X)] ≈ 0.5` by construction of the covariate
distribution, so the arms are close to balanced in size without one.

**Cause-specific hazards.** `lam_j = exp(alpha_j + X'beta_j + theta_j A + delta_j u + eta_j A u)`

| symbol | role | cause 1 | cause 2 |
|---|---|---|---|
| `alpha_j` | log baseline rate | `-2.3` | `-2.6` |
| `beta_j` | covariate effects | `(0.4, -0.2, 0.1)` | `(-0.3, 0.3, 0.2)` |
| `theta_j` | treatment effect | `-0.6` | `+0.2` |
| `delta_j` | threshold main effect | `1.5` | `1.5` |
| `eta_j` | treatment x threshold | `-1.0` | `+0.7` |

The two causes are given opposite-signed treatment effects (`theta_1 < 0 <
theta_2`) so that treatment shifts the *cause mix* as well as overall event
timing — which is the competing-risks feature the estimator has to handle, and
the reason cause-specific biases in the study run opposite-signed and must never
be pooled.

**Censoring hazard.** `lam_C = exp(alpha_C + X'beta_C + theta_C A + delta_C u + eta_C A u)`

| symbol | role | value |
|---|---|---|
| `alpha_C` | log baseline censoring rate | `-3.0` |
| `beta_C` | covariate effects | `(0.1, 0.1, -0.1)` |
| `theta_C` | treatment effect | `0.0` |
| `delta_C` | threshold main effect | `2.0` |
| `eta_C` | treatment x threshold | `0.0` |

**Threshold.** `u = 1{w_cont > 0}`, so `P(u = 1) = 0.5`.

### Variants

The validation ladder and the no-misspecification configurations reuse the same
family and change only what is listed.

| configuration | purpose | differences from the above |
|---|---|---|
| `base` | Studies B–D; no misspecification device armed | `gamma = (0.5, -0.4, 0.3)`, threshold at `w_cont > 1`, `delta = eta = (0, 0)`, `delta_C = 0` |
| `rung4_cens_none` | rung 4, uninformative censoring | `alpha_C = -2.482`, `beta_C = (0, 0, 0)`, `theta_C = 0`, `delta_C = 0` |
| `rung4_cens_info` | rung 4, informative censoring | `alpha_C = -2.947`, `beta_C = (0.3, -0.3, 0.25)`, `theta_C = 0.15`, `delta_C = 0`, `eta_C = 1.6` |

The two rung-4 variants differ **only** in the censoring mechanism and are held
at a matched ~22 % censoring rate by re-solving `alpha_C` for each, so the
contrast isolates dependence on covariates and treatment rather than amount of
censoring. `delta_C = 0` there is deliberate: carrying the arm gap through the
interaction alone rather than splitting it with a main effect keeps the
`pi * G` denominator away from its truncation bound in the treated,
above-threshold cell, which is what sets the positivity floor.

## Estimand and truth

The target is the counterfactual cumulative incidence of cause `j` at time
`tau` under a static intervention on treatment, and the contrast between arms:

```
psi_j^a(tau) = E_W[ F_j(tau | a, W) ],     F_j(tau | a, W) = lam_j / Lam * (1 - exp(-Lam tau))
```

with `Lam = lam_1 + lam_2`. Because the conditional CIF is exact, the only
approximation in the truth is the Monte Carlo average over the covariate
distribution, which is computed once per configuration at 10^7 draws and cached;
its Monte Carlo standard error is reported alongside the value, so the precision
of the reference is auditable rather than assumed.

Target times are frozen per configuration at the 30th, 50th and 70th percentiles
of the observed event-time distribution, computed once from a large draw rather
than from each replicate. Data-dependent target times would move the estimand
between replicates and make "bias" meaningless.

## How models are misspecified

One device per nuisance, so the eight cells of the factorial differ only in what
the fitted models can represent — the data are identical across cells at a given
sample size, and every comparison is paired.

| nuisance | correct model | wrong model |
|---|---|---|
| outcome `Q` | Cox per cause on `[X, u, A, A*u]` | drops `u` and `A*u` |
| censoring `G` | Cox on `[X, u, A]` | drops `u` |
| propensity `pi` | logistic on `X` | drops `w_cont` |

The propensity lever is an **omitted confounder** rather than a distortion of the
true propensity, and that asymmetry is deliberate. A propensity is bounded in
(0, 1), so any additive term in the logit strong enough to misspecify it
meaningfully also drives `e(W)` toward 0 or 1; strength and positivity are in
direct conflict as long as the misspecification lives inside the truth. Keeping
the truth plain-logistic in `W` and dropping a genuine confounder from the fit
gives a large, clean error with positivity untouched (FINDINGS 6). The hazard
levers do not have this problem, because a hazard is unbounded above.

## Calibration

Three parameters were set by measurement, not assumption.

**`gamma = (1.0, -0.8, 0.6)`.** The misspecified propensity then carries a mean
absolute error of ~0.018 against ~0.003 for the correct family — a 7.6-fold gap —
while the true `e(W)` stays within [0.09, 0.67] even at n = 50 000. Weaker values
make the lever inert; stronger ones drive `e(W)` to the boundary, which is a
positivity question and belongs in a separate study.

**Threshold at `w_cont > 0`, `delta_j = 1.5`.** Both were inherited from a
discrete-time reference design as `w_cont > 1` and `delta_j = 4`, and neither
ports. In continuous time `delta` multiplies the hazard *rate*, so `e^4 ≈ 55`
gives a 150-fold spread across subjects, a first-percentile event time of 0.003,
and 50–75 % of fits failing. Lowering the threshold also widens the support of
the misspecified term from 18 % to 55 % of the sample (FINDINGS 3).

**A treatment × threshold interaction, `eta = (-1.0, 0.7)`.** This is the
requirement that is easy to miss. A term entering both arms identically shifts
`F^1` and `F^0` together, so the risk *difference* survives: measured, omitting
such a term produced a twelvefold error in the cumulative hazard and moved the
risk difference by 0.002 against a truth of −0.149. The interaction is what makes
the omitted term arm-specific and therefore visible in the contrast
(FINDINGS 2). The same argument applies to the censoring hazard, where `eta_C`
plays the identical role.

Calibration is verified by `sim/calibrate.py`, which measures the **asymptotic
bias of the plug-in estimator** at large `n` rather than any nuisance-level
discrepancy — the latter is not evidence of the former, and conflating the two is
how a double-robustness simulation fails silently.

## Realised characteristics

For the Study A configuration: roughly 28 % censoring, both causes well
represented, `P(A = 1) ≈ 0.5` with `e(W)` centred at 0.51 and
`P(e < 0.05) = P(e > 0.95) ≈ 0`. Target times are approximately 0.51, 1.15 and
2.67, at which the true cause-1 risk differences are −0.099, −0.175 and −0.257,
and the cause-2 differences +0.114, +0.184 and +0.239. No estimand sits near
zero, so relative comparisons remain meaningful.

Confounding is real rather than nominal: the crude arm difference departs from
the truth by 0.020–0.102 for cause 1.

## Limitations worth stating

- **Cause 2 carries less confounding** than cause 1 (crude-versus-truth gap
  0.004–0.030 against 0.020–0.102), so cause-2 cells are a weaker test. Reported
  results focus on cause 1. Fixable by giving `beta_2` more overlap with `gamma`;
  not done.
- **The censoring lever is the weakest of the three.** `u` is monotone in
  `w_cont`, so a censoring model that keeps the linear term absorbs much of the
  omitted main effect, and the cells that differ only in `G` separate less than
  those that differ in `Q` or `pi`.
- **Constant baseline hazards** buy the closed-form truth at the cost of
  proportional-hazards-friendly data. The Cox nuisance models are therefore
  correctly specified in functional form when they include `u`, which is exactly
  what "correct" is meant to mean here, but it does mean the design does not
  probe baseline-hazard misspecification.
