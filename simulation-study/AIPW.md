# The one-step (AIPTW) estimator, built from PyTMLE's own influence function

How `sim/estimators.py:run_aipw` is constructed, and why it is built that way.

## Why it reuses PyTMLE's EIF rather than reimplementing it

The sharpest contrast in the study is `tmle` against `aipw`. Both are
asymptotically linear with the *same* efficient influence function, so any
difference between them is attributable to one thing only: TMLE fluctuates the
nuisance estimates until the empirical EIF equation is solved and then plugs in,
while the one-step leaves the estimates alone and adds the empirical mean of the
EIF as a correction.

That attribution only holds if the two really do share an influence function. An
independent reimplementation would put a second, uncontrolled difference into the
comparison — a sign convention, a lagging choice, a truncation applied in one
place and not the other — and any gap between the estimators would then be
uninterpretable. So `run_aipw` calls
`pytmle.get_influence_curve.get_eic`, the same function the targeting loop calls,
and never writes an influence function of its own.

## The estimator

For cause `j` at time `tau` under intervention `a`, with `Q` the cause-specific
hazards and `g = (pi, G)` the treatment and censoring mechanisms:

```
psi_aipw = Psi(Q_hat) + Pn D*(Q_hat, g_hat)
```

`Psi(Q_hat)` is the g-computation plug-in — the covariate average of the
counterfactual cumulative incidence implied by the initial hazards — and
`Pn D*` is the sample mean of the efficient influence function evaluated **at
those same initial estimates**. The correction is applied once; nothing is
iterated.

Contrast with the TMLE in the same file: it fluctuates `Q_hat` to `Q*` until
`Pn D*(Q*, g_hat) ~ 0` and reports `Psi(Q*)` with no additive correction.

## Implementation

Three steps, all in `run_aipw`.

**1. Wrap the injected nuisances in the object the EIF code expects.**

```python
ue = {
    k: UpdatedEstimates.from_initial_estimates(
        initial_estimates[k],
        target_events=list(target_events),
        target_times=list(target_times),
        min_nuisance=min_nuisance,
    )
    for k in (key_1, key_0)
}
```

`UpdatedEstimates.__post_init__` derives the nuisance weight
`1 / max(pi * G(t-), min_nuisance)` from the propensity and censoring survival.
Passing `min_nuisance` explicitly matters: left as `None` it is silently replaced
by the n-dependent default `5 / (sqrt(n) log n)`, and the one-step and the TMLE
would then be truncating at different bounds (FINDINGS 1).

**2. Evaluate the influence function and the plug-in together.**

```python
ue = get_eic(ue, event_times=..., event_indicator=..., g_comp=True)
```

`g_comp=True` makes the same call populate `g_comp_est`, so the plug-in and the
correction are computed from one pass over one set of arrays. There is no
opportunity for them to be evaluated at different estimates.

**3. Add the correction, per arm.**

```python
risk    = ue[k].g_comp_est.set_index(["Event", "Time"])["Risk"]
mean_ic = ue[k].ic.groupby(["Event", "Time"])["IC"].mean()
per_arm[k] = risk + mean_ic
```

`ue[k].ic` is long-format, one row per (subject, cause, target time). The
per-arm estimate is the plug-in plus the mean influence-function value at that
cause and time.

## Standard errors

The influence function *is* the variance estimator, so no separate machinery is
needed. Per arm, over the `n` subjects at a given `(cause, time)`:

```python
se = sqrt(mean(IC**2) / n)
```

`mean(IC**2)` rather than `var(IC)`: the influence function has mean zero at the
truth, and the second-moment form is what PyTMLE uses internally, so the two
estimators' standard errors are computed identically as well.

**Risk difference.** The contrast's influence function is the difference of the
arms' influence functions *for the same subject*, so the subtraction is done at
the subject level before taking the second moment:

```python
ic1 = ic_by_arm[key_1].set_index(["ID", "Event", "Time"])["IC"]
ic0 = ic_by_arm[key_0].set_index(["ID", "Event", "Time"])["IC"]
se_rd = (ic1 - ic0).groupby(["Event", "Time"]).apply(
    lambda x: np.sqrt(np.mean(x**2) / len(x)))
```

This keeps the correlation between the arms — both are functions of the same
observation — which a variance-sum would discard and which would inflate the
interval.

Note the indexing includes `ID`. PyTMLE's own `predict_ate.ate_diff` subtracts
with only `["Event", "Time"]` in the index, which aligns the two arms
*positionally* within each group. That gives the same answer here, because both
frames are built by the same loop over `range(n)` and so are in the same subject
order, but aligning on `ID` states the intent and does not depend on that
invariant holding.

**Risk ratio.** Delta method on `R1 / R0`, matching
`predict_ate.ate_ratio`:

```python
se_rr = sqrt(mean((IC1 / R0 - IC0 * R1 / R0**2)**2) / n)
```

## What this estimator does not do

- **It is not a substitution estimator.** `Psi(Q_hat) + Pn D*` need not respect
  the bounds of the parameter space; a risk can fall outside `[0, 1]` and a
  cumulative incidence need not be monotone in `t`. TMLE's plug-in form is what
  buys those guarantees, and it is the reason the two estimators differ in finite
  samples even when they agree asymptotically.
- **It does not iterate.** The correction is evaluated once, at the initial
  estimates. Whether that is enough is a second-order question, and the
  divergence between the two estimators under a badly misspecified `Q` was what
  first exposed the targeted-update defect in FINDINGS 9.
- **It does not re-estimate anything.** `pi`, `G` and the hazards are whatever
  was injected, so `aipw` and `tmle` see byte-identical nuisances by
  construction.

## Verification

`aipw` is checked against `riskRegression::ate(estimator = "AIPTW")`, which is an
independent implementation of the same estimator fitted from the same model
objects — tier 2 in Study C's scheme, because the two share fitted objects rather
than nuisance arrays and `riskRegression` uses the exact Aalen-Johansen product
limit where this is a discrete plug-in.

Measured mean absolute paired difference: **~1.2e-4**, against a tolerance of
1e-2 that had been set on the expectation that the discretisation difference
would dominate. It does not; the two agree far more closely than the design
anticipated.

The stronger internal check is behavioural rather than numerical. Across Study A
the one-step is unbiased in exactly the cells double robustness predicts — clean
wherever `Q` is correct *or* both `g` components are, biased only in C6–C8 — and
its agreement with `tmle` after the FINDINGS 9 fix, in cells where both should be
consistent, is what confirms that the shared-EIF construction is doing what it
claims.
