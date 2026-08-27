| comparison                |   tier | quantity_label   | unit            |   tolerance | expect   |       n=500 |      n=1000 |      n=2000 | as_expected   |   shrink_ratio |
|:--------------------------|-------:|:-----------------|:----------------|------------:|:---------|------------:|------------:|------------:|:--------------|---------------:|
| gcomp (concrete) vs gcomp |      1 | point estimate   | abs. difference |       0.005 | agree    | 0.000793446 | 0.000388087 | 0.000190104 | True          |       0.239593 |
| tmle (concrete) vs tmle   |      1 | point estimate   | abs. difference |       0.002 | diverge  | 0.0105991   | 0.00719452  | 0.00487859  | True          |       0.460283 |
| ate:AIPTW vs aipw         |      2 | point estimate   | abs. difference |       0.01  | agree    | 0.000101727 | 4.88497e-05 | 2.29315e-05 | True          |       0.225421 |
| ate:GFORMULA vs gcomp     |      2 | point estimate   | abs. difference |       0.01  | agree    | 0.000102094 | 4.87241e-05 | 2.32473e-05 | True          |       0.227704 |
| ate:IPTW vs ipw           |      2 | point estimate   | abs. difference |       0.01  | agree    | 1.61318e-05 | 6.79147e-06 | 3.21557e-06 | True          |       0.199331 |
| tmle (concrete) vs tmle   |      1 | standard error   | abs. log ratio  |       0.002 | agree    | 0.000658569 | 0.000330874 | 0.000159683 | True          |       0.242469 |
| ate:AIPTW vs aipw         |      2 | standard error   | abs. log ratio  |       0.01  | agree    | 0.00599626  | 0.00381782  | 0.00240165  | True          |       0.400525 |
| ate:IPTW vs ipw           |      2 | standard error   | abs. log ratio  |       0.01  | diverge  | 0.133508    | 0.130494    | 0.128389    | True          |       0.961655 |
| tmle (concrete) vs tmle   |      1 | score (PnEIC)    | abs. difference |       0.001 | agree    | 2.23236e-05 | 2.56463e-06 | 2.36438e-06 | True          |       0.105914 |