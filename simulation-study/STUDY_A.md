# Study A — is the point estimate right, and when?

Study B asks whether the interval around the estimate is right. This study asks
the prior question: **does the estimator recover the truth, and which nuisance
models does it need to get right in order to?** It is a 2×2×2 over correct and
misspecified outcome (`Q`), propensity (`π`) and censoring (`G`) models, which is
the design double robustness makes a prediction about.

The data-generating process, the misspecification levers and their calibration
live in [DGP.md](DGP.md). This document is the result.

**Scope.** 24 cells — eight specification combinations × three sample sizes —
500 replicates each, plus 150 `concrete` replicates per cell on the same seeds and
the same injected nuisances. 12 000 PyTMLE replicates and 3 600 concrete ones,
~63 CPU-hours. Three evaluation times, τ ∈ {0.51, 1.15, 2.67}, at which the true
cause-1 risk difference is −0.099, −0.175 and −0.257. Tables pool the three τ
unless stated, so Monte Carlo SE on a coverage of 0.95 is 0.006 for PyTMLE and
0.010 for concrete.

**Zero replicate failures** in any cell, and no nuisance fit failed anywhere.

---

## 1. The eight cells

| cell | `Q` | `π` | `G` | what theory says about `aipw`/`tmle` |
|---|---|---|---|---|
| C1 | ✓ | ✓ | ✓ | consistent |
| C2 | ✓ | ✓ | ✗ | consistent |
| C3 | ✓ | ✗ | ✓ | consistent — `Q` carries it |
| C4 | ✓ | ✗ | ✗ | consistent |
| C5 | ✗ | ✓ | ✓ | consistent — `π` carries it |
| C6 | ✗ | ✓ | ✗ | consistent |
| C7 | ✗ | ✗ | ✓ | **no guarantee** |
| C8 | ✗ | ✗ | ✗ | **no guarantee** |

The misspecifications are real and were calibrated to be so (FINDINGS 2, 3, 6):
the wrong propensity carries a mean absolute error of 0.104 against 0.023 for the
correct family, and the wrong outcome model 0.431 against 0.123 on the cumulative
hazard. Neither is near a positivity boundary — the propensity spans [0.24, 0.78]
in the `π`-wrong cells at n = 1000, which Study B later identified as its `OV1`
rung, one step above comfortable.

---

## 2. Double robustness holds, cell by cell

Mean absolute bias in the cause-1 RD at n = 1000, against truths of −0.099 to
−0.257:

| cell | spec | `gcomp` | `ipw` | `aipw` | `tmle` |
|---|---|--:|--:|--:|--:|
| C1 | Q✓ π✓ G✓ | 0.000 | 0.000 | 0.000 | 0.000 |
| C2 | Q✓ π✓ G✗ | 0.000 | 0.006 | 0.000 | 0.000 |
| C3 | Q✓ π✗ G✓ | −0.001 | **0.031** | −0.001 | −0.001 |
| C4 | Q✓ π✗ G✗ | −0.002 | **0.035** | −0.002 | −0.002 |
| C5 | Q✗ π✓ G✓ | **0.044** | 0.001 | 0.001 | 0.000 |
| C6 | Q✗ π✓ G✗ | **0.044** | 0.005 | 0.006 | 0.006 |
| C7 | Q✗ π✗ G✓ | **0.043** | **0.031** | 0.010 | 0.008 |
| C8 | Q✗ π✗ G✗ | **0.044** | **0.038** | 0.018 | 0.015 |

Read it as three claims, all of which hold:

1. **Each singly-robust estimator fails on exactly the nuisance it depends on.**
   `gcomp` is clean in C1–C4 and carries ~0.044 in C5–C8; `ipw` is clean in
   C1, C2, C5, C6 and carries 0.031–0.038 wherever `π` is wrong. Neither is ever
   biased for a reason other than its own model.
2. **`aipw` and `tmle` are consistent whenever *either* `Q` or `π` is correct** —
   C1 through C6, at or below 0.006 on an effect of 0.099–0.257.
3. **Where nothing is guaranteed, the damage is bounded rather than absent.** In
   C8 both singly-robust estimators carry ~0.04; `tmle` retains 0.015, about 35 %
   of the plug-in bias, and `aipw` 0.018.

`G` misspecification alone never bites: C1 against C2, and C3 against C4, differ
by less than 0.002 in every estimator that targets the RD. That is the same
result Study B's rung 4 reaches from the other direction — a wrong censoring model
is harmless while the outcome model can still carry the estimate.

---

## 3. The targeted-update defect, before and after the fix

The single most consequential number in this study is C5: outcome model wrong,
propensity and censoring correct. Double robustness says the targeted update must
remove *all* of `gcomp`'s 0.044 bias.

| | C5 bias, n = 1000 | fraction of plug-in bias retained |
|---|--:|--:|
| `gcomp` (plug-in) | 0.0439 | 100 % |
| `tmle`, **pre-fix** (FINDINGS 7, n = 800) | 0.0285 | ~65 % |
| `tmle`, **post-fix** | **0.0003** | **0.6 %** |
| `tmle (concrete)` 1.0.8 | 0.0272 | 62 % |

FINDINGS 7 recorded the pre-fix state and initially misread it as a property of
the substitution estimator. FINDINGS 9 traced it instead to a defect in the
targeted update — it moved only the subjects observed in the arm being updated —
shared by both implementations because PyTMLE's update is a port of concrete's.
PyTMLE has been fixed; **concrete 1.0.8 has not**, and this table is what that
costs an end user.

Across all four `Q`-wrong cells at n = 1000, concrete retains **62–76 %** of the
plug-in bias where PyTMLE retains 0.6–35 %:

| cell | `gcomp` | `tmle` | `tmle (concrete)` |
|---|--:|--:|--:|
| C5 | 0.0439 | 0.0003 | 0.0272 |
| C6 | 0.0438 | 0.0057 | 0.0326 |
| C7 | 0.0432 | 0.0080 | 0.0283 |
| C8 | 0.0438 | 0.0155 | 0.0334 |

This is the finding worth reporting upstream, and it is the one Study C's
cross-package agreement tables quantify at three sample sizes.

---

## 4. Double robustness buys consistency, not inference

Coverage does **not** follow bias. Splitting the `tmle` cells at n = 1000 by
whether `π` is correct:

| | SE / empirical SD | coverage |
|---|--:|--:|
| `π` correct (C1, C2, C5, C6) | **1.004** | **0.946** |
| `π` wrong (C3, C4, C7, C8) | 0.923 | 0.910 |

`aipw` behaves identically (1.009 / 0.946 against 0.936 / 0.909). So where
inference is guaranteed it is valid — SE/SD within 0.5 % of 1 and coverage at
nominal — and where `π` is misspecified the interval is ~8 % too short and covers
at ~0.91.

**This is what the theory predicts, not a defect.** The efficient influence curve
is the estimator's actual influence function only when `g` is correctly specified.
Under `g`-misspecification the estimator stays asymptotically linear but with a
*different* influence function, one carrying a contribution from estimating the
misspecified `g`'s limit — so the EIF-based variance is simply the wrong one. The
shortfall does not close with `n` (C3: 0.909 → 0.923 → 0.924), which is the
signature of a wrong variance rather than a slow one.

`ipw` shows the same thing far more violently, because it has no `Q` to fall back
on: C3 coverage runs **0.917 → 0.869 → 0.797** across n = 250/500/1000. Coverage
that *degrades* with sample size is the clearest possible sign that the interval
is centred on the wrong thing — the bias is fixed at 0.031 while the interval
shrinks around it.

FINDINGS 10 has the longer analysis, including why the pre-fix coverage looked
better: two errors were cancelling.

---

## 5. concrete's intervals

concrete's coverage in this study is not comparable to PyTMLE's, and the reason is
the defect rather than the interval construction. At n = 1000 it runs 0.993 and
0.995 in C1 and C2 — over-covering, because its under-updated estimates are less
dispersed than the influence curve its standard error is built from — and 0.884,
0.825, 0.807, **0.729** across C5–C8, where the retained bias moves the estimate
away from the truth faster than the interval widens.

Study C measures the same phenomenon under correct specification and at three
sample sizes, where it shows up cleanly as SE/SD ≈ 1.26 and coverage ≈ 0.98.

---

## 6. Convergence and cost

`tmle` non-convergence within 200 updates is a small-sample phenomenon that
disappears with `n`, and it is worst where the outcome model is correct and the
propensity comfortable:

| cell | n = 250 | n = 500 | n = 1000 |
|---|--:|--:|--:|
| C2 (Q✓ π✓ G✗) | 0.320 | 0.116 | 0.028 |
| C6 (Q✗ π✓ G✗) | 0.314 | 0.088 | 0.014 |
| C7 (Q✗ π✗ G✓) | 0.022 | 0.000 | 0.000 |

Non-convergence here means the loop budget was exhausted, not that the estimate
is unusable: the median cell takes 17–31 accepted steps, and Study B's attribution
found that discarding non-converged bootstrap resamples is actively harmful
(FINDINGS 11). No result in this document conditions on convergence.

Second-stage cost for `tmle`, median seconds per replicate across cells, is
1.5–3.3 s at n = 250, 6.1–9.9 s at n = 500 and 25–40 s at n = 1000 — measured
single-process but *not* single-thread, and 8-way contended. The cross-package
runtime comparison belongs to Study C, which measures it under conditions where
it means something.

---

## 7. What to tell a user

1. **Get one of `Q` or `π` right.** Either alone is enough for consistency, and
   the study measures the cost of neither: ~35 % of the plug-in bias survives.
2. **Do not read coverage as validation when `π` may be wrong.** The interval is
   ~8 % too short there and will not improve with more data. If `π` is doubtful,
   the point estimate is still trustworthy and the interval is not.
3. **`G` is the forgiving one.** Misspecifying the censoring model alone changed
   nothing measurable here, at 27 % censoring.
4. **Prefer `tmle` or `aipw` to `ipw` outright.** `ipw` is the only estimator whose
   coverage gets *worse* with sample size.
5. **On concrete 1.0.8, treat a `Q`-misspecified analysis as a plug-in.** The
   targeted update removes about a third of what it should.

---

## 8. Limitations

- **One DGP, one misspecification of each nuisance.** The levers were calibrated
  to be real and non-degenerate (FINDINGS 2, 3, 6), but "wrong" here is a
  particular wrongness — an omitted confounder for `π`, an omitted threshold
  interaction for `Q` and `G`. A different misspecification could order C7/C8
  differently.
- **Positivity is comfortable throughout.** Study B's cross-calibration shows
  Study A operates at its `OV1` rung, so nothing here speaks to the
  near-violation regime; that is Study B's subject and the two compose rather
  than overlap.
- **`gcomp` has no standard error** in either implementation, by design, so it
  appears in the bias tables only.
- **concrete runs 150 replicates against PyTMLE's 500**, so its Monte Carlo SE is
  1.7× larger; the bias differences reported above are 5–20× that, but the
  coverage figures should be read with the wider band in mind.
