| implementation | vs | tier | quantity | statistic | tolerance | expect | note | n | compared | pairs | mean_diff | mean_abs_diff | max_abs_diff | as_expected | skipped | skip_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tmle (concrete) | tmle | 1 | est | difference | 0.0020 | agree | agrees on the targeting increment; the residual is numerical | 500.0000 | targeting increment | 1500 | 0.0000 | 0.0002 | 0.0038 | True | False |  |
| tmle (concrete) | tmle | 1 | est | difference | 0.0020 | agree | agrees on the targeting increment; the residual is numerical | 1000.0000 | targeting increment | 1500 | -0.0000 | 0.0001 | 0.0011 | True | False |  |
| tmle (concrete) | tmle | 1 | est | difference | 0.0020 | agree | agrees on the targeting increment; the residual is numerical | 2000.0000 | targeting increment | 450 | -0.0000 | 0.0000 | 0.0004 | True | False |  |
| gcomp (concrete) | gcomp | 1 | est | difference | 0.0050 | agree | plug-in on concrete's coarser grid | 500.0000 | level | 1500 | -0.0008 | 0.0008 | 0.0027 | True | False |  |
| gcomp (concrete) | gcomp | 1 | est | difference | 0.0050 | agree | plug-in on concrete's coarser grid | 1000.0000 | level | 1500 | -0.0004 | 0.0004 | 0.0010 | True | False |  |
| gcomp (concrete) | gcomp | 1 | est | difference | 0.0050 | agree | plug-in on concrete's coarser grid | 2000.0000 | level | 450 | -0.0002 | 0.0002 | 0.0004 | True | False |  |
| ate:GFORMULA | gcomp | 2 | est | difference | 0.0100 | agree | exact Aalen-Johansen vs discrete plug-in | 500.0000 | level | 1500 | -0.0001 | 0.0001 | 0.0006 | True | False |  |
| ate:GFORMULA | gcomp | 2 | est | difference | 0.0100 | agree | exact Aalen-Johansen vs discrete plug-in | 1000.0000 | level | 1500 | -0.0000 | 0.0000 | 0.0002 | True | False |  |
| ate:GFORMULA | gcomp | 2 | est | difference | 0.0100 | agree | exact Aalen-Johansen vs discrete plug-in | 2000.0000 | level | 450 | -0.0000 | 0.0000 | 0.0001 | True | False |  |
| ate:IPTW | ipw | 2 | est | difference | 0.0100 | agree |  | 500.0000 | level | 1500 | 0.0000 | 0.0000 | 0.0002 | True | False |  |
| ate:IPTW | ipw | 2 | est | difference | 0.0100 | agree |  | 1000.0000 | level | 1500 | 0.0000 | 0.0000 | 0.0000 | True | False |  |
| ate:IPTW | ipw | 2 | est | difference | 0.0100 | agree |  | 2000.0000 | level | 450 | 0.0000 | 0.0000 | 0.0000 | True | False |  |
| ate:AIPTW | aipw | 2 | est | difference | 0.0100 | agree |  | 500.0000 | level | 1500 | -0.0001 | 0.0001 | 0.0006 | True | False |  |
| ate:AIPTW | aipw | 2 | est | difference | 0.0100 | agree |  | 1000.0000 | level | 1500 | -0.0000 | 0.0000 | 0.0002 | True | False |  |
| ate:AIPTW | aipw | 2 | est | difference | 0.0100 | agree |  | 2000.0000 | level | 450 | -0.0000 | 0.0000 | 0.0001 | True | False |  |
| tmle (concrete) | tmle | 1 | se | log ratio | 0.0020 | agree | same IC formula on both sides, so the SE must agree | 500.0000 | log ratio | 1500 | -0.0007 | 0.0007 | 0.0044 | True | False |  |
| tmle (concrete) | tmle | 1 | se | log ratio | 0.0020 | agree | same IC formula on both sides, so the SE must agree | 1000.0000 | log ratio | 1500 | -0.0003 | 0.0003 | 0.0010 | True | False |  |
| tmle (concrete) | tmle | 1 | se | log ratio | 0.0020 | agree | same IC formula on both sides, so the SE must agree | 2000.0000 | log ratio | 450 | -0.0002 | 0.0002 | 0.0004 | True | False |  |
| gcomp (concrete) | gcomp | 1 | se | log ratio |  | agree | plug-in on concrete's coarser grid |  | log ratio | 0 |  |  |  |  | True | se not available for gcomp (concrete) or gcomp |
| ate:GFORMULA | gcomp | 2 | se | log ratio |  | agree | exact Aalen-Johansen vs discrete plug-in |  | log ratio | 0 |  |  |  |  | True | se not available for ate:GFORMULA or gcomp |
| ate:IPTW | ipw | 2 | se | log ratio | 0.0100 | diverge | PyTMLE's IPW SE is conservative -- it treats the weights as known; riskRegression propagates nuisance estimation and is the calibrated one | 500.0000 | log ratio | 1500 | 0.1335 | 0.1335 | 0.2683 | True | False |  |
| ate:IPTW | ipw | 2 | se | log ratio | 0.0100 | diverge | PyTMLE's IPW SE is conservative -- it treats the weights as known; riskRegression propagates nuisance estimation and is the calibrated one | 1000.0000 | log ratio | 1500 | 0.1305 | 0.1305 | 0.2312 | True | False |  |
| ate:IPTW | ipw | 2 | se | log ratio | 0.0100 | diverge | PyTMLE's IPW SE is conservative -- it treats the weights as known; riskRegression propagates nuisance estimation and is the calibrated one | 2000.0000 | log ratio | 450 | 0.1284 | 0.1284 | 0.2173 | True | False |  |
| ate:AIPTW | aipw | 2 | se | log ratio | 0.0100 | agree |  | 500.0000 | log ratio | 1500 | 0.0036 | 0.0060 | 0.0464 | True | False |  |
| ate:AIPTW | aipw | 2 | se | log ratio | 0.0100 | agree |  | 1000.0000 | log ratio | 1500 | 0.0015 | 0.0038 | 0.0210 | True | False |  |
| ate:AIPTW | aipw | 2 | se | log ratio | 0.0100 | agree |  | 2000.0000 | log ratio | 450 | 0.0009 | 0.0024 | 0.0107 | True | False |  |
| tmle (concrete) | tmle | 1 | pn_eic | difference | 0.0010 | agree | both solve their own stopping criterion; the residual gap is the width of that band, not a discrepancy | 500.0000 | level | 3000 | 0.0000 | 0.0000 | 0.0030 | True | False |  |
| tmle (concrete) | tmle | 1 | pn_eic | difference | 0.0010 | agree | both solve their own stopping criterion; the residual gap is the width of that band, not a discrepancy | 1000.0000 | level | 3000 | 0.0000 | 0.0000 | 0.0011 | True | False |  |
| tmle (concrete) | tmle | 1 | pn_eic | difference | 0.0010 | agree | both solve their own stopping criterion; the residual gap is the width of that band, not a discrepancy | 2000.0000 | level | 900 | -0.0000 | 0.0000 | 0.0005 | True | False |  |
| gcomp (concrete) | gcomp | 1 | pn_eic | difference |  | agree | plug-in on concrete's coarser grid |  | difference | 0 |  |  |  |  | True | pn_eic not available for gcomp (concrete) or gcomp |
| ate:GFORMULA | gcomp | 2 | pn_eic | difference |  | agree | exact Aalen-Johansen vs discrete plug-in |  | difference | 0 |  |  |  |  | True | pn_eic not available for ate:GFORMULA or gcomp |
| ate:IPTW | ipw | 2 | pn_eic | difference |  | agree |  |  | difference | 0 |  |  |  |  | True | pn_eic not available for ate:IPTW or ipw |
| ate:AIPTW | aipw | 2 | pn_eic | difference |  | agree |  |  | difference | 0 |  |  |  |  | True | pn_eic not available for ate:AIPTW or aipw |