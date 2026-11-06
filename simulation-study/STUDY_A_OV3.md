# Study A-OV3 — double robustness under overlap stress

**Status:** complete — 10 cells, 5 000 replicates, **39 CPU-hours**, 20 MB. Zero
replicate failures and zero nuisance-fit failures in every cell.

Study A asks whether the point estimate is right when some nuisance models are
wrong, at comfortable overlap. Study B shows that at its `OV3` rung the Wald
interval becomes a variance failure. This sub-study puts the two together: it
repeats a focused subset of Study A's specification cells under Study B's OV3
condition while holding Study A's estimand and target times fixed.

## 1. Design

The DGP is Study A's `threshold` family with only `gamma` replaced by OV3's
`(2.0, -1.6, 1.2)`. `gamma` enters treatment assignment only; the estimand
averages the conditional CIF over the marginal covariate law, so it does not
depend on `gamma`. The truth is therefore bit-identical to Study A's, and the
target times are pinned to Study A's exact values.

| cell | Q | π | G | role |
|---|---|---|---|---|
| C1 | ✓ | ✓ | ✓ | all-correct reference under OV3 |
| C2 | ✓ | ✓ | ✗ | censoring lever alone |
| C3 | ✓ | ✗ | ✓ | **mandatory**: π-wrong, Q must carry the estimate |
| C5 | ✗ | ✓ | ✓ | **mandatory**: Q-wrong, π must carry the estimate |
| C7 | ✗ | ✗ | ✓ | neither Q nor π correct; no guarantee |

- n = 250 and 500, 500 replicates per cell; 5 000 replicates total.
- Estimators: `tmle`, `gcomp`, `aipw`, `ipw`; Wald intervals only.
- No `concrete`, no bootstrap, no oracle arm.
- `min_nuisance = 0.01`, `max_updates = 200`, as in Study A.
- Target times: `0.513946, 1.149562, 2.669820`, inherited from
  `results/study_a/*/meta.json`.
- Seeds are the existing per-cell-name streams, so the specification cells are
  independent samples rather than a paired design.

## 2. Calibration

OV3 is defined by `gamma` and the covariate law, both unchanged from Study B.
Measured on 2 × 10⁶ draws: `P(min(e, 1−e) < 0.05) = 0.120` and an effective
sample-size fraction of 0.17, against Study A's OV1 values of 0.000 and 0.74.
The truth at the pinned times is also unchanged: cause-1 risk differences are
−0.0989, −0.1747 and −0.2574.

The late Study B target time (8.608) is deliberately not used. Study A's
`threshold` DGP carries `delta_c = 2.0`; at 8.608 that collapses the censoring
survival and empties the above-threshold risk set, so the late target would
measure a different failure from the one intended.

## 3. Pre-registered expectations

- `gcomp` is biased wherever Q is wrong (C5, C7); `ipw` wherever π or G is
  wrong (C2, C3, C7).
- `aipw` and `tmle` remain consistent in C1, C2, C3 and C5 — C5 is the sharp
  test that Q-wrong is carried by a correct π even under OV3.
- C7 has no theoretical guarantee; its residual bias is the bound.
- Wald coverage is expected to sit below nominal, most visibly in the
  π-wrong and Q-wrong cells, following Study B's OV3 result. Non-convergence
  is reported, not used to filter estimates.

## 4. Results

All estimates are the cause-1 risk difference, averaged over the three pinned
taus; coverage is on the same scale. At 500 replicates the coverage MC-SE is
0.006–0.012 and the bias MC-SE is 0.001–0.003.

### 4.1 Point estimates: double robustness survives OV3

Mean absolute bias, averaged over taus:

| n | cell | spec | `gcomp` | `ipw` | `aipw` | `tmle` |
|---|---|---|---|---|---|---|
| 250 | C1 | Q✓ π✓ G✓ | −0.0009 | 0.0078 | 0.0015 | 0.0049 |
| 250 | C2 | Q✓ π✓ G✗ | −0.0013 | 0.0016 | −0.0068 | 0.0011 |
| 250 | C3 | Q✓ π✗ G✓ | −0.0032 | **0.0557** | −0.0013 | −0.0011 |
| 250 | C5 | Q✗ π✓ G✓ | **0.0467** | 0.0043 | 0.0014 | 0.0100 |
| 250 | C7 | Q✗ π✗ G✓ | **0.0461** | **0.0578** | **0.0242** | **0.0214** |
| 500 | C1 | Q✓ π✓ G✓ | −0.0006 | 0.0020 | 0.0003 | 0.0053 |
| 500 | C2 | Q✓ π✓ G✗ | −0.0016 | 0.0062 | −0.0022 | −0.0013 |
| 500 | C3 | Q✓ π✗ G✓ | −0.0009 | **0.0583** | 0.0006 | 0.0012 |
| 500 | C5 | Q✗ π✓ G✓ | **0.0475** | 0.0023 | −0.0013 | −0.0041 |
| 500 | C7 | Q✗ π✗ G✓ | **0.0471** | **0.0557** | **0.0225** | **0.0184** |

The double-robustness prediction holds in both mandatory cells. In C3 (π
wrong, Q correct) `aipw` and `tmle` stay within ~0.001 of truth while `ipw`
carries 0.056–0.058; in C5 (Q wrong, π correct) `aipw` and `tmle` stay within
~0.01 while `gcomp` carries 0.047. `gcomp` is clean wherever Q is correct, and
`ipw` is clean in C1/C2/C5. C7, where neither Q nor π is correct, is biased as
expected: `aipw`/`tmle` retain 0.018–0.024, roughly 40–50 % of the singly-robust
bias rather than nothing.

Two small residuals are worth stating rather than smoothing over. C1's `tmle`
has a +0.005 bias at both sizes (z ≈ 2.5–3.4), while `aipw` is essentially zero;
C5's `tmle` bias is +0.010 at n = 250 and −0.004 at n = 500, so it does not
reproduce a consistent sign. These are the size of the OV3-induced
non-convergence effect (§4.3), not the fixed plug-in bias that C5's `gcomp`
retains. `aipw` is the cleaner DR point estimator under this condition.

### 4.2 Intervals: the variance failure composes with misspecification

SE / empirical SD (ratio), with Wald coverage in brackets:

| n | cell | `ipw` | `aipw` | `tmle` |
|---|---|---|---|---|
| 250 | C1 | 0.90 [0.850] | 0.84 [0.897] | **0.82 [0.827]** |
| 250 | C2 | 0.90 [0.873] | 0.89 [0.927] | 0.92 [0.872] |
| 250 | C3 | 1.04 [**0.768**] | 0.80 [0.875] | 0.72 [0.827] |
| 250 | C5 | 0.93 [0.860] | 0.88 [0.897] | 0.99 [0.872] |
| 250 | C7 | 1.05 [**0.768**] | 0.85 [0.853] | 0.79 [0.835] |
| 500 | C1 | 0.98 [0.899] | 0.92 [0.910] | **0.82 [0.839]** |
| 500 | C2 | 0.94 [0.886] | 0.90 [0.913] | 0.80 [0.829] |
| 500 | C3 | 1.03 [**0.653**] | 0.79 [0.875] | 0.72 [0.831] |
| 500 | C5 | 1.04 [0.915] | 0.98 [0.935] | 0.92 [0.889] |
| 500 | C7 | 1.06 [**0.676**] | 0.86 [0.853] | 0.80 [0.845] |

`gcomp` has no standard error or interval by design.

**The interval is the part that does not survive.** Even C1, with every model
correct, under-covers: `tmle` 0.827 → 0.839 and `aipw` 0.897 → 0.910. The
`tmle` SE/SD ratio is 0.82 at both sizes, so the shortfall is a variance
underestimate, not a centring problem, and it does not close with n. This is
Study B's OV3 result reproducing inside Study A's design.

Adding misspecification does not repair it. C3's `tmle` sits at 0.72–0.83, and
`ipw` is the clearest failure: its bias is fixed at 0.056–0.058 while the
interval shrinks, so coverage falls from 0.768 to 0.653 as n doubles. C5 is the
least damaged cell — `aipw` reaches 0.935 at n = 500 — but still below nominal,
and its point estimate is the one double robustness says should be clean.

### 4.3 Run health

| cell | TMLE non-converged | median steps | mean `ps_mae` | min / max fitted π |
|---|---:|---:|---:|---|
| C1_n250 | 0.838 | 28 | 0.037 | 0.0007 / 0.9999 |
| C2_n250 | 0.884 | 28 | 0.039 | 0.0005 / 1.0000 |
| C3_n250 | 0.120 | 22 | 0.154 | 0.110 / 0.955 |
| C5_n250 | 0.898 | 28 | 0.038 | 0.0005 / 0.9997 |
| C7_n250 | 0.060 | 23 | 0.155 | 0.092 / 0.933 |
| C1_n500 | 0.508 | 33 | 0.027 | 0.0010 / 0.9997 |
| C2_n500 | 0.534 | 32 | 0.027 | 0.0004 / 0.9999 |
| C3_n500 | 0.008 | 19.5 | 0.154 | 0.120 / 0.907 |
| C5_n500 | 0.604 | 33 | 0.027 | 0.0004 / 0.9999 |
| C7_n500 | 0.002 | 20.5 | 0.154 | 0.121 / 0.935 |

The π-correct cells exhaust the 200-update budget far more often under OV3
(84–90 % at n = 250, 51–60 % at n = 500); the π-wrong cells converge because the
fitted propensity is less extreme. No estimates were filtered on convergence.
The fitted propensity range confirms the condition is real — values reach
0.0004 and 0.9999 — and the π-wrong cells' larger `ps_mae` (0.154 against 0.027)
shows the omitted-confounder lever is stronger, not weaker, under OV3.

### 4.4 What survives, and what does not

1. **The DR point estimate survives.** C3 and C5 remain consistent in the
   two-component sense double robustness promises; the singly-robust estimators
   fail on exactly their own nuisance.
2. **The Wald interval does not.** Under OV3 it under-covers even in C1
   (`tmle` 0.83, `aipw` 0.90–0.91), and misspecification makes it worse. This
   is consistent with Study B's finding that the OV3 failure is a property of
   the influence-curve variance, not of the nuisance fit.
3. **`aipw` is the better-inferred DR estimator here.** Its point estimate is
   cleaner than `tmle`'s in C1/C5 and its coverage is closer to nominal, though
   still short of 0.95.
4. **`ipw` should not be used as a yardstick under π-misspecification.** Its
   coverage degrades with n (0.768 → 0.653 in C3), exactly the signature Study A
   reported at OV1 and amplified here.
5. **Do not read a high coverage figure as validation under poor overlap.**
   Study B's percentile bootstrap, not measured in this sub-study, is the
   interval remedy the evidence supports.

## 5. Limitations

- Five of eight specification cells: the G-wrong crosses (C4, C6) and the
  all-wrong bound (C8) are not run.
- One overlap level (OV3); no dose-response in overlap within this design.
- Wald intervals only; the percentile bootstrap that repairs OV3 in Study B is
  not measured here.
- n = 250 and 500 only; the n-trend is two points.
- Specification cells are unpaired; cross-cell differences include independent
  Monte Carlo noise. Estimators within a cell are paired.
- The same misspecification levers as Study A: an omitted confounder for π and
  an omitted threshold interaction for Q and G.

## 6. Reproduce

From `simulation-study/`, with `pytmle` importable:

```bash
python -m sim.run --config sim/configs/study_a_ov3.yaml \
    --output-dir results/study_a_ov3 --n-jobs 8 --report
```

Pilot in a separate directory; resume skips existing shards, so a pilot shard
in the full-run directory would silently drop replicates. Artefact filenames
still carry the `study_a_*` prefix and the figures label cells `C1`, `C3`, …
without an OV3 tag; this directory is the disambiguator.
