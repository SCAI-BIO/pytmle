# The data-generating process

One simulation design serves both studies in this project, and this document
describes it once. Implementation: `sim/dgp.py`; the closed-form truth is in
`sim/truth.py`; the calibration evidence behind each choice is in
[FINDINGS.md](FINDINGS.md).

| | question | what it varies | configuration |
|---|---|---|---|
| **Study A** | Is the estimator consistent when some nuisance models are wrong? | which of `Q`, `pi`, `G` the fitted models can represent | `threshold` |
| **Study B** | Are its confidence intervals valid, and where do they fail? | how hard the *estimation problem* is, with all models correct | `base` + per-cell overrides |

The two are complementary rather than parallel: Study A holds the problem easy
and breaks the models; Study B holds the models correct and breaks the problem.
Because they share a DGP family, an estimand and a truth, results transfer
directly between them — and, as it turns out, **Study A's propensity sits exactly
on the first rung of Study B's positivity ladder** (see [Cross-study
calibration](#cross-study-calibration)), which is what lets Study A's
"positivity is comfortable here" be a measurement rather than an assertion.

## What the design has to support

Five requirements together decide almost every parameter value:

1. **Continuous event times.** PyTMLE targets continuously distributed event
   times, so a discrete-grid design would exercise a different code path than the
   one under test.
2. **Competing risks**, since the cause-specific structure is what distinguishes
   this estimator from a survival TMLE.
3. **A truth known in closed form**, so that bias is measured rather than
   approximated by a reference estimator.
4. **Misspecification that moves the estimand** (Study A). A model can be badly
   wrong at the nuisance level and leave the target parameter untouched; a design
   that does not check this produces cells that look informative and are inert.
5. **Stress that is a dial, not a switch** (Study B). "Does the interval fail
   here" is not answerable; "how far can the problem be pushed before it fails"
   is. Every stress axis is therefore a monotone ladder with a calibrated target
   at each rung, and each rung is asserted in `tests/test_study_b_design.py`
   before any compute is spent.

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

Two further fields exist and are inert at their defaults: `n_noise` adds that
many independent N(0,1) covariates that enter no hazard, and `noise_rho` makes
`(w_cont, z_1, ..., z_k)` jointly AR(1) with `corr = rho^|i-j|` and unit
marginals. They are groundwork for a deferred high-dimensional study and are set
to `0` throughout Studies A and B — see [Levers that exist but are not
used](#levers-that-exist-but-are-not-used).

## The two configurations

Covariate coefficients are ordered `(d2, d3, w_cont)`, matching
`X = [d2, d3, w_cont]`. `base` is `DGPParams()`; `threshold` is the Study A
family. Cells differing from `base` are marked.

**Treatment assignment.** `A ~ Bernoulli(expit(gamma'X))` — no intercept, since
`E[expit(gamma'X)] ~ 0.5` by construction of the covariate law, so the arms are
close to balanced in size without one.

| symbol | `base` (Study B) | `threshold` (Study A) |
|---|---|---|
| `gamma` | `(0.5, -0.4, 0.3)` | **`(1.0, -0.8, 0.6)`** |

**Cause-specific hazards.**
`lam_j = exp(alpha_j + X'beta_j + theta_j A + delta_j u + eta_j A u)`

| symbol | role | `base` cause 1 / 2 | `threshold` cause 1 / 2 |
|---|---|---|---|
| `alpha_j` | log baseline rate | `-2.3` / `-2.6` | `-2.3` / `-2.6` |
| `beta_j` | covariate effects | `(0.4, -0.2, 0.1)` / `(-0.3, 0.3, 0.2)` | same |
| `theta_j` | treatment effect | `-0.6` / `+0.2` | same |
| `delta_j` | threshold main effect | `0` / `0` | **`1.5` / `1.5`** |
| `eta_j` | treatment x threshold | `0` / `0` | **`-1.0` / `+0.7`** |

The two causes are given opposite-signed treatment effects (`theta_1 < 0 <
theta_2`) so that treatment shifts the *cause mix* as well as overall event
timing — the competing-risks feature the estimator has to handle, and the reason
cause-specific biases run opposite-signed and must never be pooled.

**Censoring hazard.**
`lam_C = exp(alpha_C + X'beta_C + theta_C A + delta_C u + eta_C A u)`

| symbol | role | `base` | `threshold` |
|---|---|---|---|
| `alpha_C` | log baseline censoring rate | `-3.0` | `-3.0` |
| `beta_C` | covariate effects | `(0.1, 0.1, -0.1)` | same |
| `theta_C` | treatment effect | `0.0` | same |
| `delta_C` | threshold main effect | `0.0` | **`2.0`** |
| `eta_C` | treatment x threshold | `0.0` | same |

**Threshold.** `u = 1{w_cont > c}` with `c = 1` in `base` (`P(u = 1) = 0.16`) and
`c = 0` in `threshold` (`P(u = 1) = 0.50`).

---

# Study A's levers: misspecification

One device per nuisance, so the eight cells of the 2³ factorial differ only in
what the fitted models can represent. The data are identical across cells at a
given sample size, and every comparison is paired.

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

### Calibration

**`gamma = (1.0, -0.8, 0.6)`.** The misspecified propensity then carries a mean
absolute error of ~0.018 against ~0.003 for the correct family — a 7.6-fold gap —
while the true `e(W)` stays within [0.09, 0.67] even at n = 50 000. Weaker values
make the lever inert; stronger ones drive `e(W)` to the boundary, which is a
positivity question and is Study B's subject.

**Threshold at `w_cont > 0`, `delta_j = 1.5`.** Both were inherited from a
discrete-time reference design as `w_cont > 1` and `delta_j = 4`, and neither
ports. In continuous time `delta` multiplies the hazard *rate*, so `e^4 ~ 55`
gives a 150-fold spread across subjects, a first-percentile event time of 0.003,
and 50–75 % of fits failing. Lowering the threshold also widens the support of
the misspecified term from 18 % to 55 % of the sample (FINDINGS 3).

**A treatment x threshold interaction, `eta = (-1.0, 0.7)`.** The requirement
that is easy to miss. A term entering both arms identically shifts `F^1` and
`F^0` together, so the risk *difference* survives: measured, omitting such a term
produced a twelvefold error in the cumulative hazard and moved the risk
difference by 0.002 against a truth of -0.149. The interaction is what makes the
omitted term arm-specific and therefore visible in the contrast (FINDINGS 2). The
same argument applies to the censoring hazard, where `eta_C` plays the identical
role.

Calibration is verified by `sim/calibrate.py`, which measures the **asymptotic
bias of the plug-in estimator** at large `n` rather than any nuisance-level
discrepancy — the latter is not evidence of the former, and conflating the two is
how a double-robustness simulation fails silently.

---

# Study B's levers: stress

All three nuisances are correctly specified throughout. What varies is how hard
the estimation problem is. Every level below is expressed as a per-cell override
of `base` in `sim/configs/study_b.yaml`, and every calibration target was solved
numerically and is asserted as a test.

What these levers produced is in **[STUDY_B.md](STUDY_B.md)**; this section is the
design only.

### Axis 1 — treatment positivity

Scale `gamma` by `kappa`. Measured on 2 x 10^6 draws:

| level | `gamma` | `P(min(e, 1-e) < 0.05)` | `e(W)` 0.1–99.9 pct | ESS fraction |
|---|---|---|---|---|
| `base` | `(0.5, -0.4, 0.3)` | 0.000 | [0.228, 0.790] | 0.94 |
| `OV1` | `(1.0, -0.8, 0.6)` | 0.000 | [0.080, 0.934] | 0.74 |
| `OV2` | `(1.5, -1.2, 0.9)` | 0.027 | [0.025, 0.981] | 0.42 |
| `OV3` | `(2.0, -1.6, 1.2)` | 0.120 | [0.007, 0.995] | 0.17 |
| `OV4` | `(3.0, -2.4, 1.8)` | 0.332 | [0.001, 1.000] | 0.013 |

`P(e < 0.05)` and percentiles are the reported summaries, not `min`/`max`: an
extreme-value statistic describes a tail that `n` never samples, and switching to
percentiles changed which configurations looked acceptable (FINDINGS 6).

### Axis 2 — rare events

Lower `alpha_1` only, leaving cause 2 common. This is the realistic
competing-risks rare-event problem, and it keeps the observed-time support wide
so the last target time stays inside it — the runner rejects a replicate whose
follow-up ends before `max(tau)`, so making *both* causes rare would measure
replicate loss rather than rarity.

| level | `alpha` | cause-1 risk at `tau = 8.61`, `a = 1 / a = 0` | true RD | ~cause-1 events at n = 250 |
|---|---|---|---|---|
| `base` | `(-2.3, -2.6)` | 0.290 / 0.468 | `-0.178` | ~90 |
| `RA1` | `(-3.5, -2.6)` | 0.103 / 0.187 | `-0.084` | ~36 |
| `RA2` | `(-4.2, -2.6)` | 0.053 / 0.099 | `-0.046` | ~19 |
| `RA3` | `(-5.0, -2.6)` | 0.024 / 0.046 | `-0.022` | ~9 |

**This axis moves the truth**, unlike the others, so bias along it must be read
standardised (`bias / SD`) rather than raw.

### Axis 3 — censoring positivity

Scale `(beta_C, theta_C, eta_C)` by `k`, re-solving `alpha_C` at each level so the
**censored fraction is held fixed**. The axis therefore varies dependence on
covariates and treatment, not the amount of censoring — the same discipline as
the rung-4 validation pair. Coefficients are
`beta_C = k*(0.4, -0.4, 0.35)`, `theta_C = 0.2k`, `eta_C = 1.0k`, `delta_C = 0`:

| level | `k` | `alpha_C` | censored | `G(tau_last)` 1st pct | `P(pi*G < 0.01)` |
|---|---|---|---|---|---|
| `CN0` | 0 | `-2.6341` | 0.30 | 0.539 | 0.000 |
| `CN1` | 0.5 | `-2.7466` | 0.30 | 0.187 | 0.000 |
| `CN2` | 1.0 | `-2.8868` | 0.30 | 0.012 | 0.012 |
| `CN3` | 1.5 | `-3.0425` | 0.30 | 0.000 | 0.075 |
| `CN4` | 1.0 | `-1.9612` | **0.50** | 0.000 | 0.100 |

`CN4` repeats `CN2`'s dependence at a higher censoring rate, so the pair
separates *amount* from *dependence*. `delta_C = 0` throughout is deliberate:
carrying the arm gap through the interaction alone rather than splitting it with
a main effect keeps `pi * G` away from its truncation bound in the treated,
above-threshold cell, which is what sets the positivity floor.

### Axis 4 — the null condition

`theta = (0, 0)`, so the true risk difference is **exactly zero at every `tau`,
for both causes**, and the risk ratio is exactly 1. Coverage here is a direct
type-I error rate.

Zeroing only `theta_1` is *not* enough, and the reason is a competing-risks trap
worth stating: cause 2's hazard still depends on treatment and enters cause 1's
cumulative incidence through the total hazard `Lam`, leaving a cause-1 RD of
`-0.023` at the last `tau`. A null on the cause-specific hazard is not a null on
the cumulative incidence.

### Remedy dials

Two candidate fixes are swept rather than assumed:

| dial | values | applied at |
|---|---|---|
| `min_nuisance` (truncation floor on `pi * G`) | `0.01`, `0.025`, `0.05`, `0.10` | `OV3`, `CN3` |
| `max_updates` (targeting-loop budget) | `200`, `1000` | `RA3` |

A third — building the Wald interval on a transformed scale (logit for risks, log
for the risk ratio, Fisher-z for the risk difference) — costs nothing at run time,
since it is a function of the stored point estimate and standard error, and is
computed for every cell.

---

## Which levers move the estimand

A property worth stating explicitly, because it is what makes the stress axes
interpretable as difficulty dials rather than as different questions:

| lever | moves the truth? | why |
|---|---|---|
| `gamma` (overlap) | **no** | with `control_resample = False` the covariate law is marginal, and the estimand averages `F_j(tau | a, W)` over it; treatment assignment does not enter |
| censoring parameters | **no** | censoring enters the observed-data likelihood, not `psi` |
| `n_noise`, `noise_rho` | **no** | the noise covariates enter no hazard, and an AR(1) with unit variances preserves `w_cont`'s marginal law |
| `alpha_1` (rarity) | **yes**, by design | the cause-1 hazard is the estimand's own input |
| `theta` (null) | **yes**, by design | it is the treatment effect |
| misspecification (Study A) | **no** | it lives in the *fitted* models, not the truth |

The three "no" rows are asserted as tests: `closed_form` must return
bit-identical values across every level of those axes.

## Estimand and truth

The target is the counterfactual cumulative incidence of cause `j` at time `tau`
under a static intervention on treatment, and the contrast between arms:

```
psi_j^a(tau) = E_W[ F_j(tau | a, W) ],     F_j(tau | a, W) = lam_j / Lam * (1 - exp(-Lam tau))
```

with `Lam = lam_1 + lam_2`. Because the conditional CIF is exact, the only
approximation in the truth is the Monte Carlo average over the covariate
distribution, computed once per configuration at 10^7 draws and cached; its Monte
Carlo standard error is reported alongside the value, so the precision of the
reference is auditable rather than assumed. Each Study B cell's truth is computed
from **its own** parameters, since the rarity and null axes move it.

**Target times.**

| study | rule | values |
|---|---|---|
| A | 30th / 50th / 70th percentile of observed event times under `threshold` | 0.51, 1.15, 2.67 |
| B | 10th / 50th / 85th percentile under `base`, then **pinned** in every cell | 0.477676, 3.11961, 8.608445 |

Study B pins the absolute times rather than re-deriving quantiles per cell. The
quantile rule reads the *observed* event-time distribution, which both rarity and
censoring shift, so leaving it in place would evaluate each condition at a
different clock time and confound every axis with `tau`. Pinning means "rare"
denotes a smaller cumulative incidence at the same time, which is the intended
comparison. Data-dependent target times would additionally move the estimand
between replicates and make "bias" meaningless.

## Cross-study calibration

Study A's `gamma = (1.0, -0.8, 0.6)` is **identical to Study B's `OV1`**. That is
not a coincidence in the design so much as a useful accident of it, and it lets
Study A's qualitative claim be checked: at `OV1` the propensity has
`P(min(e, 1-e) < 0.05) = 0.0003` and an effective sample size fraction of 0.74,
and Study B measures Wald coverage there at 0.923–0.966 across `tau` and `n`. So
Study A operates at the first rung of the positivity ladder, one rung above
`base`, and any residual under-coverage in Study A's `pi`-correct cells is
attributable to misspecification elsewhere rather than to a near-violation.

The converse also holds: Study B never varies specification, so the two studies'
findings compose rather than confound.

## Realised characteristics

| | `base` (Study B) | `threshold` (Study A) |
|---|---|---|
| censored fraction | 0.244 | 0.275 |
| `P(u = 1)` | 0.159 | 0.501 |
| `P(A = 1)` | ~0.50 | ~0.50 |
| `e(W)` | centred 0.50, `P(e < 0.05) ~ 0` | centred 0.51, `P(e < 0.05) = P(e > 0.95) ~ 0` |
| true cause-1 RD at the three `tau` | `-0.023`, `-0.110`, `-0.178` | `-0.098`, `-0.175`, `-0.257` |
| true cause-2 RD | `+0.008`, `+0.050`, `+0.113` | `+0.113`, `+0.184`, `+0.239` |

Both causes are well represented and, in Study A, no estimand sits near zero, so
relative comparisons remain meaningful. Confounding is real rather than nominal:
the crude arm difference departs from the truth by 0.020–0.102 for cause 1.

In Study B the cause-2 risk difference at the earliest `tau` *is* small
(`+0.008`), which is deliberate — that corner is where a symmetric interval is
expected to strain, and the rarity axis makes it a controlled ladder rather than
an incidental feature.

## Levers that exist but are not used

`n_noise` and `noise_rho` implement a covariate-dimension axis that neither study
runs. `n_noise` adds inert N(0,1) columns, which reach all three nuisance fits;
`noise_rho` makes `(w_cont, z_1, ..., z_k)` jointly AR(1), following the
correlated design matrix used in the high-dimensional literature.

The axis was implemented, measured and then dropped, for a reason worth recording
rather than repeating: under this project's **injected, unpenalised** nuisances,
"high-dimensional" can only mean "an unpenalised GLM overfits", which is a
statement about the analyst's model and not about the estimator. And the
correlation structure is *algebraically inert* here — an unpenalised MLE depends
on the design only through its column span, and adding `z = rho*w_cont + noise`
to a design that already contains `w_cont` spans exactly what independent noise
spans. Measured: max change in fitted propensity 6.5e-4 (optimiser tolerance),
max change in fitted hazard 6e-7. Correlation bites only where the span genuinely
changes — under a penalised or selection-based fit, or under Study A's `pi`-wrong
arm, where `w_cont` is omitted so correlated noise partially recovers it
(measured 9.6e-2, a real effect).

A study that shows PyTMLE *handling* high dimensions is a different design: it
drops the injection and exercises the package's own state learner and propensity
super learner. Both fields are retained as groundwork for it, with the inertness
result pinned by a test so that study inherits a verified starting point.

## Limitations worth stating

- **Cause 2 carries less confounding** than cause 1 in Study A (crude-versus-truth
  gap 0.004–0.030 against 0.020–0.102), so cause-2 cells are a weaker test.
  Reported results focus on cause 1. Fixable by giving `beta_2` more overlap with
  `gamma`; not done.
- **The censoring misspecification lever is the weakest of the three** in Study A.
  `u` is monotone in `w_cont`, so a censoring model keeping the linear term
  absorbs much of the omitted main effect, and cells differing only in `G`
  separate less than those differing in `Q` or `pi`.
- **Constant baseline hazards** buy the closed-form truth at the cost of
  proportional-hazards-friendly data. The Cox nuisance models are therefore
  correctly specified in functional form when they include `u`, which is exactly
  what "correct" means here — but the design does not probe baseline-hazard
  misspecification.
- **Study B's sample-size ladder is uneven.** The base condition spans
  n in {250, 500, 1000, 2000}; the stress ladders span {250, 500} only, because
  second-stage cost scales as n^2.05 and stressed cells run 6–11x base (they
  exhaust the update budget without converging). Trends in `n` along a stress axis
  rest on two points.
- **No heavy-tailed or contaminated outcome mechanism.** The survival analogue of
  a contaminated error distribution would be a mixture hazard; the DGP has no
  device for it, so robustness to outlying event times is untested.
