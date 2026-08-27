"""Performance measures, every one carrying a Monte Carlo standard error.

Follows Morris, White & Crowther (2019): a simulation result without its MC-SE
cannot be read, because the interesting differences (93 % vs 95 % coverage) are
often the same size as the Monte Carlo noise.

Coverage is always reported next to mean interval width. A procedure can buy
coverage by widening, and reporting coverage alone hides that -- the lesson
taken from Fan et al. (2024).
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

__all__ = ["summarise", "GROUP_KEYS"]

GROUP_KEYS = ["cell", "estimator", "estimand", "event", "time", "group"]


def _mc_se_mean(x: np.ndarray) -> float:
    m = len(x)
    return float(np.std(x, ddof=1) / np.sqrt(m)) if m > 1 else np.nan


def _mc_se_prop(p: float, m: int) -> float:
    return float(np.sqrt(p * (1 - p) / m)) if m > 0 else np.nan


#: Every metric `_summarise_group` produces, in order. The schema must not depend
#: on the contents of a group: if some groups returned a shorter Series, pandas
#: would silently stack the whole result into long format instead of raising,
#: which turns a handful of failed replicates into a mangled results table.
_FIELDS = [
    "n_rep", "n_used", "truth", "bias", "bias_mc_se", "rel_bias", "emp_sd",
    "mean_se", "se_ratio", "rmse", "coverage", "coverage_mc_se", "miss_low",
    "miss_high", "ci_width", "ci_width_mc_se", "n_ci", "n_error",
    "frac_nonconverged", "n_converged_known",
]


def _summarise_group(g: pd.DataFrame) -> pd.Series:
    est = g["est"].to_numpy(dtype=float)
    truth = g["truth"].to_numpy(dtype=float)
    se = g["se"].to_numpy(dtype=float)
    lo = g["ci_lo"].to_numpy(dtype=float)
    hi = g["ci_hi"].to_numpy(dtype=float)
    conv = g["converged"]

    ok = np.isfinite(est) & np.isfinite(truth)
    n_rep = int(len(g))
    n_ok = int(ok.sum())
    if n_ok == 0:
        out = {f: np.nan for f in _FIELDS}
        out.update(n_rep=n_rep, n_used=0, n_error=int(g["error"].notna().sum()))
        return pd.Series(out)[_FIELDS]

    e, t = est[ok], truth[ok]
    err = e - t
    bias = float(err.mean())
    emp_sd = float(np.std(e, ddof=1)) if n_ok > 1 else np.nan
    rmse = float(np.sqrt(np.mean(err**2)))

    ci_ok = ok & np.isfinite(lo) & np.isfinite(hi)
    n_ci = int(ci_ok.sum())
    if n_ci:
        cov = (lo[ci_ok] <= truth[ci_ok]) & (truth[ci_ok] <= hi[ci_ok])
        coverage = float(cov.mean())
        miss_lo = float((truth[ci_ok] < lo[ci_ok]).mean())  # truth below interval
        miss_hi = float((truth[ci_ok] > hi[ci_ok]).mean())
        width = float(np.mean(hi[ci_ok] - lo[ci_ok]))
        width_se = _mc_se_mean(hi[ci_ok] - lo[ci_ok])
    else:
        coverage = miss_lo = miss_hi = width = width_se = np.nan

    se_ok = se[ok][np.isfinite(se[ok])]
    mean_se = float(se_ok.mean()) if len(se_ok) else np.nan

    return pd.Series(
        {
            "n_rep": n_rep,
            "n_used": n_ok,
            "truth": float(t[0]),
            "bias": bias,
            "bias_mc_se": _mc_se_mean(err),
            "rel_bias": bias / t[0] if t[0] not in (0.0,) else np.nan,
            "emp_sd": emp_sd,
            "mean_se": mean_se,
            "se_ratio": mean_se / emp_sd if emp_sd and np.isfinite(emp_sd) else np.nan,
            "rmse": rmse,
            "coverage": coverage,
            "coverage_mc_se": _mc_se_prop(coverage, n_ci) if n_ci else np.nan,
            "miss_low": miss_lo,
            "miss_high": miss_hi,
            "ci_width": width,
            "ci_width_mc_se": width_se,
            "n_ci": n_ci,
            "n_error": int(g["error"].notna().sum()),
            # `converged` is nullable: PyTMLE reports no flag for g-computation
            # rows. Unknown is not the same as "did not converge", so the rate is
            # taken over the rows that actually carry a flag, and how many did is
            # reported alongside it.
            "frac_nonconverged": (
                float((~conv.dropna().astype(bool)).mean()) if conv.notna().any() else np.nan
            ),
            "n_converged_known": int(conv.notna().sum()),
        }
    )[_FIELDS]


def summarise(
    results: pd.DataFrame,
    truth: pd.DataFrame,
    group_keys: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Join per-replicate estimates to the truth and reduce to one row per cell.

    ``truth`` must carry columns ``estimand, event, time, group, truth``.
    """
    keys = list(group_keys) if group_keys is not None else GROUP_KEYS
    keys = [k for k in keys if k in results.columns]

    merged = results.merge(
        truth, on=["estimand", "event", "time", "group"], how="left", validate="many_to_one"
    )
    missing = merged["truth"].isna() & merged["est"].notna()
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} estimate rows have no matching truth; "
            "check that target times are frozen across replicates."
        )
    out = (
        merged.groupby(keys, dropna=False)
        .apply(_summarise_group, include_groups=False)
        .reset_index()
    )
    # Guard the failure mode the fixed schema exists to prevent: if apply ever
    # stacks into long format again, fail loudly rather than return a table that
    # looks plausible but is not.
    missing = set(_FIELDS) - set(out.columns)
    if missing:
        raise RuntimeError(
            f"summarise() produced a malformed table (missing {sorted(missing)}); "
            "this means _summarise_group returned inconsistent shapes."
        )
    return out


def truth_long(truth_df: pd.DataFrame) -> pd.DataFrame:
    """Reshape ``truth.closed_form`` output into the long form ``summarise`` wants."""
    rows = []
    for _, r in truth_df.iterrows():
        rows.append(
            dict(estimand="risk", event=int(r["event"]), time=float(r["time"]),
                 group=float(r["arm"]), truth=float(r["risk"]))
        )
    contrasts = truth_df.drop_duplicates(["event", "time"])
    for _, r in contrasts.iterrows():
        rows.append(dict(estimand="rd", event=int(r["event"]), time=float(r["time"]),
                         group=np.nan, truth=float(r["rd"])))
        rows.append(dict(estimand="rr", event=int(r["event"]), time=float(r["time"]),
                         group=np.nan, truth=float(r["rr"])))
    return pd.DataFrame(rows)
