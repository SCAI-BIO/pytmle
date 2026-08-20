"""Closed-form ground truth for the competing-risks DGP.

With constant (exponential) cause-specific hazards the counterfactual cumulative
incidence is available in closed form conditional on covariates:

    F_j^a(tau | W) = lam_j / Lam * (1 - exp(-Lam * tau)),   Lam = sum_j lam_j

The marginal estimand averages that over the covariate distribution *of the
population actually simulated*. Without the control-resampling device that is
the marginal covariate law. With it, the treated arm's covariates are tilted by
the logistic selection while the control arm's are marginal, so the realised law
has density ``p(w) * [sigma(gamma'w) + 1 - q]`` relative to the marginal and the
average must be weighted accordingly:

    psi_j^a(tau) = E_{W ~ p}[ F_j^a(tau | W) * (sigma(gamma'W) + 1 - q) ]

The weight integrates to one by construction. Ignoring it targets a different
population; see ``target="unweighted"`` for the comparison the validation ladder
reports.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

from .dgp import DGPParams, _design, _draw_covariates, cause_rates, expit, marginal_treated_fraction

__all__ = ["closed_form", "config_hash", "cached_truth"]

Target = Literal["realised", "unweighted"]


def config_hash(p: DGPParams, taus: Sequence[float], target: Target) -> str:
    payload = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(p).items()
    }
    payload["_taus"] = list(map(float, taus))
    payload["_target"] = target
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def closed_form(
    p: DGPParams,
    taus: Sequence[float],
    n_mc: int = 10_000_000,
    seed: int = 20250301,
    chunk: int = 1_000_000,
    target: Target = "realised",
) -> pd.DataFrame:
    """Monte-Carlo average of the exact conditional CIF.

    Returns one row per (arm, cause, tau) with the point value and its Monte
    Carlo standard error, so the precision of the truth is auditable rather than
    assumed.
    """
    taus = np.asarray(list(taus), dtype=float)
    rng = np.random.default_rng(seed)

    q = marginal_treated_fraction(p) if p.control_resample else None
    weighted = p.control_resample and target == "realised"

    n_arms, n_causes, n_taus = 2, p.n_causes, len(taus)
    tot = np.zeros((n_arms, n_causes, n_taus))
    tot_sq = np.zeros((n_arms, n_causes, n_taus))
    n_seen = 0

    remaining = n_mc
    while remaining > 0:
        m = min(chunk, remaining)
        remaining -= m
        w_cat, w_cont = _draw_covariates(m, rng)
        X = _design(w_cat, w_cont)
        u = (w_cont > p.threshold).astype(float)

        wt = expit(X @ p.gamma) + (1.0 - q) if weighted else np.ones(m)

        for ai, a in enumerate((1.0, 0.0)):
            rates = cause_rates(X, u, a, p)
            lam_tot = np.sum(rates, axis=0)
            # (m, n_taus): 1 - exp(-Lam * tau)
            frac = 1.0 - np.exp(-np.outer(lam_tot, taus))
            for j in range(n_causes):
                cif = (rates[j] / lam_tot)[:, None] * frac
                contrib = cif * wt[:, None]
                tot[ai, j] += contrib.sum(axis=0)
                tot_sq[ai, j] += (contrib**2).sum(axis=0)
        n_seen += m

    mean = tot / n_seen
    var = np.maximum(tot_sq / n_seen - mean**2, 0.0)
    mc_se = np.sqrt(var / n_seen)

    rows = []
    for ai, a in enumerate((1, 0)):
        for j in range(n_causes):
            for ti, tau in enumerate(taus):
                rows.append(
                    {
                        "arm": a,
                        "event": j + 1,
                        "time": float(tau),
                        "risk": float(mean[ai, j, ti]),
                        "mc_se": float(mc_se[ai, j, ti]),
                    }
                )
    out = pd.DataFrame(rows)

    # contrasts
    wide = out.pivot_table(index=["event", "time"], columns="arm", values="risk")
    se_w = out.pivot_table(index=["event", "time"], columns="arm", values="mc_se")
    contrasts = pd.DataFrame(
        {
            "event": wide.index.get_level_values("event"),
            "time": wide.index.get_level_values("time"),
            "rd": (wide[1] - wide[0]).to_numpy(),
            "rr": (wide[1] / wide[0]).to_numpy(),
            "rd_mc_se": np.sqrt(se_w[1] ** 2 + se_w[0] ** 2).to_numpy(),
        }
    )
    return out.merge(contrasts, on=["event", "time"], how="left")


def cached_truth(
    p: DGPParams,
    taus: Sequence[float],
    cache_dir: Path,
    target: Target = "realised",
    n_mc: int = 10_000_000,
    seed: int = 20250301,
) -> pd.DataFrame:
    """Compute the truth once per configuration and reuse it thereafter."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"truth_{config_hash(p, taus, target)}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    out = closed_form(p, taus, n_mc=n_mc, seed=seed, target=target)
    out.to_parquet(path, index=False)
    return out
