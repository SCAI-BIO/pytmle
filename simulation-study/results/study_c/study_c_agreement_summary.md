| comparison | tier | quantity_label | unit | tolerance | expect | n=500 | n=1000 | n=2000 | as_expected | shrink_ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| gcomp (concrete) vs gcomp | 1 | point estimate | abs. difference | 0.0050 | agree | 0.0008 | 0.0004 | 0.0002 | True | 0.2396 |
| tmle (concrete) vs tmle | 1 | point estimate | abs. difference | 0.0020 | agree | 0.0002 | 0.0001 | 0.0000 | True | 0.1053 |
| ate:AIPTW vs aipw | 2 | point estimate | abs. difference | 0.0100 | agree | 0.0001 | 0.0000 | 0.0000 | True | 0.2254 |
| ate:GFORMULA vs gcomp | 2 | point estimate | abs. difference | 0.0100 | agree | 0.0001 | 0.0000 | 0.0000 | True | 0.2277 |
| ate:IPTW vs ipw | 2 | point estimate | abs. difference | 0.0100 | agree | 0.0000 | 0.0000 | 0.0000 | True | 0.1993 |
| tmle (concrete) vs tmle | 1 | standard error | abs. log ratio | 0.0020 | agree | 0.0007 | 0.0003 | 0.0002 | True | 0.2349 |
| ate:AIPTW vs aipw | 2 | standard error | abs. log ratio | 0.0100 | agree | 0.0060 | 0.0038 | 0.0024 | True | 0.4005 |
| ate:IPTW vs ipw | 2 | standard error | abs. log ratio | 0.0100 | diverge | 0.1335 | 0.1305 | 0.1284 | True | 0.9617 |
| tmle (concrete) vs tmle | 1 | score (PnEIC) | abs. difference | 0.0010 | agree | 0.0000 | 0.0000 | 0.0000 | True | 0.1038 |