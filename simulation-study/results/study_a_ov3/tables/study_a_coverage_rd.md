# Study A — 95% Wald coverage by cell (rd, cause 1)

Nominal is 0.95. Double robustness buys consistency, not inference: under-coverage where a nuisance is wrong is expected, not a defect.

| n | cell | spec | gcomp | ipw | aipw | tmle |
|---|---|---|---|---|---|---|
| 250 | C1 | Q✓ π✓ G✓ |  | 0.8500 | 0.8967 | 0.8267 |
| 250 | C2 | Q✓ π✓ G✗ |  | 0.8733 | 0.9267 | 0.8720 |
| 250 | C3 | Q✓ π✗ G✓ |  | 0.7680 | 0.8747 | 0.8267 |
| 250 | C5 | Q✗ π✓ G✓ |  | 0.8600 | 0.8967 | 0.8720 |
| 250 | C7 | Q✗ π✗ G✓ |  | 0.7680 | 0.8533 | 0.8353 |
| 500 | C1 | Q✓ π✓ G✓ |  | 0.8987 | 0.9100 | 0.8393 |
| 500 | C2 | Q✓ π✓ G✗ |  | 0.8860 | 0.9133 | 0.8287 |
| 500 | C3 | Q✓ π✗ G✓ |  | 0.6527 | 0.8747 | 0.8313 |
| 500 | C5 | Q✗ π✓ G✓ |  | 0.9153 | 0.9353 | 0.8893 |
| 500 | C7 | Q✗ π✗ G✓ |  | 0.6760 | 0.8533 | 0.8453 |
