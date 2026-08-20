"""Nuisance construction and injection into PyTMLE.

Three components, each toggled independently between a correctly specified and
an incorrectly specified fit. That independence is what generates Study A's
2^3 factorial from a single DGP:

    Q   cause-specific hazards (and hence event-free survival)
    pi  propensity score
    G   censoring survival

``correct`` and ``wrong`` differ *only in the fitted model* -- same data, same
truth -- so the eight cells stay paired and nothing incidental can be mistaken
for misspecification.

The truth is linear in ``[X, u]`` with ``u = 1{w_cont > 1}``, so a Cox model on
those columns is exactly correct and one omitting ``u`` is genuinely
misspecified: ``u`` is a step and ``w_cont`` already enters linearly, so no
linear term can absorb it.

For the propensity, the control-resampling device makes the truth
``logit e = log sigma(gamma'X) - log(1 - q)``; the correct fit is an MLE over
that one-extra-parameter family and the wrong fit is a plain logistic
regression, which cannot represent the soft hinge of ``log sigma``.

A critical API constraint: ``PyTMLE._check_inputs`` requires
``np.unique(event_times)`` to equal ``np.unique(initial_estimates[k].times)``
exactly, so every builder keys its columns off the observed unique times of that
specific replicate.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit as _expit, log_expit
from sklearn.linear_model import LogisticRegression
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from pytmle import InitialEstimates

from .dgp import DGPParams, Sample, cause_rates, censoring_rate, marginal_treated_fraction

__all__ = ["Spec", "build", "NuisanceDiagnostics"]

Mode = Literal["oracle", "correct", "wrong"]


@dataclass(frozen=True)
class Spec:
    """Which specification each nuisance component gets."""

    Q: Mode = "correct"
    pi: Mode = "correct"
    G: Mode = "correct"

    @property
    def label(self) -> str:
        short = {"oracle": "O", "correct": "C", "wrong": "W"}
        return f"Q{short[self.Q]}-pi{short[self.pi]}-G{short[self.G]}"


@dataclass
class NuisanceDiagnostics:
    """Evidence that the misspecification is real and that positivity holds."""

    ps_mae: float  # mean |pi_hat - pi_true|
    ps_min: float
    ps_max: float
    cumhaz_mae: float  # time-averaged mean |Lambda_hat - Lambda_true|
    n_failed_fits: int


# ---------------------------------------------------------------------------
# Cox helpers
# ---------------------------------------------------------------------------


def _cox_cumhaz(
    Xfit: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    Xpred_1: np.ndarray,
    Xpred_0: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Fit a Cox model and return counterfactual cumulative hazards on ``grid``.

    Uses the proportional-hazards factorisation ``H_i(t) = H0(t) * exp(lp_i)``
    rather than materialising one step function per subject, which keeps this
    O(K + n) instead of O(K * n).
    """
    n = Xpred_1.shape[0]
    if event.sum() < 2:
        # Not enough events to identify anything; a zero hazard is the honest
        # degenerate answer and is recorded as a failure by the caller.
        return np.zeros((n, len(grid))), np.zeros((n, len(grid))), True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = CoxPHSurvivalAnalysis(alpha=1e-6)
            model.fit(Xfit, Surv.from_arrays(event.astype(bool), time))
    except Exception:
        return np.zeros((n, len(grid))), np.zeros((n, len(grid))), True

    base = model.cum_baseline_hazard_
    h0 = np.asarray(base(np.clip(grid, base.x[0], base.x[-1])), dtype=float)
    h0 = np.where(grid < base.x[0], 0.0, h0)

    out = []
    for Xp in (Xpred_1, Xpred_0):
        lp = np.asarray(model.predict(Xp), dtype=float)
        out.append(np.exp(lp)[:, None] * h0[None, :])
    return out[0], out[1], False


def _with_treatment(
    cov: np.ndarray, a: np.ndarray | float, u: Optional[np.ndarray] = None
) -> np.ndarray:
    """Prepend treatment, and add its interaction with the threshold if supplied.

    The interaction column is what lets a correctly specified outcome model
    represent an arm-specific threshold effect. Omitting it (the ``wrong`` arm)
    is what makes the misspecification bias the *contrast* rather than shifting
    both arms together.
    """
    a_col = np.broadcast_to(np.asarray(a, dtype=float), (cov.shape[0],))
    parts = [a_col, *cov.T]
    if u is not None:
        parts.append(a_col * u)
    return np.column_stack(parts)


# ---------------------------------------------------------------------------
# Propensity
# ---------------------------------------------------------------------------


def _fit_propensity_correct(sm: Sample, cov: np.ndarray, q: Optional[float] = None) -> np.ndarray:
    """MLE over the exactly-correct family.

    Without the resampling device the truth is plain logistic. With it the truth
    is ``logit e = log sigma(gamma'X) - log(1 - q)`` with ``q = E[sigma(gamma'X)]``.

    The offset is *not* fitted as a free parameter. Under the device ``q`` is
    simply ``P(A = 1)``, which the realised treated fraction estimates directly,
    so it is plugged in and only ``gamma`` is fitted. Fitting both is weakly
    identified -- ``log sigma`` flattens towards a constant as ``gamma -> 0`` and
    towards linear in the other direction, so the two parameters trade off --
    and measurably worse: mean absolute error 0.039 vs 0.029 at n = 500 and
    0.0050 vs 0.0034 at n = 20 000. Using the known structure keeps the
    "correct" arm genuinely correct at the sample sizes actually run.
    """
    A = sm.group
    if not sm.params.control_resample:
        # truth is logistic in (W, u); cov already carries u when correct
        clf = LogisticRegression(penalty=None, max_iter=1000)
        clf.fit(cov, A)
        return clf.predict_proba(cov)[:, 1]

    b0 = -np.log(1.0 - (q if q is not None else float(A.mean())))

    def nll(g):
        eta = log_expit(cov @ g) + b0
        return -np.mean(A * eta - np.logaddexp(0.0, eta))

    start = LogisticRegression(penalty=None, max_iter=1000).fit(cov, A).coef_.ravel()
    res = minimize(nll, start, method="L-BFGS-B")
    return np.asarray(_expit(log_expit(cov @ res.x) + b0), dtype=float)


def _fit_propensity_wrong(sm: Sample, cov: np.ndarray) -> np.ndarray:
    """Logistic regression with the continuous confounder omitted.

    An *omitted confounder* rather than a wrong functional form. Every attempt to
    misspecify the shape -- threshold, quadratic -- either got absorbed by the
    linear term or pushed e(W) to the boundary (see DGPParams.delta_pi). Dropping
    w_cont leaves the true propensity untouched, so positivity stays exactly as
    gamma calibrates it, while guaranteeing the fit cannot recover e(W): w_cont
    genuinely drives both treatment and outcome.
    """
    clf = LogisticRegression(penalty=None, max_iter=1000)
    clf.fit(cov, sm.group)
    return clf.predict_proba(cov)[:, 1]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build(
    sm: Sample,
    spec: Spec = Spec(),
    key_1: int = 1,
    key_0: int = 0,
    q: Optional[float] = None,
    return_diagnostics: bool = False,
):
    """Build ``{key_1: InitialEstimates, key_0: InitialEstimates}`` for one replicate."""
    p = sm.params
    grid = np.unique(sm.event_times)
    K, n = len(grid), sm.n
    A = sm.group
    n_failed = 0

    # -- Q: cause-specific cumulative hazards, counterfactual in A ----------
    if spec.Q == "oracle":
        cum1 = [np.outer(r, grid) for r in cause_rates(sm.X, sm.u, 1.0, p)]
        cum0 = [np.outer(r, grid) for r in cause_rates(sm.X, sm.u, 0.0, p)]
    else:
        q_ok = spec.Q == "correct"
        cov = sm.design(include_threshold=q_ok)
        # A correct outcome model also needs the treatment x threshold
        # interaction, otherwise it cannot represent an arm-specific threshold
        # effect and "correct" would not be correct.
        u_int = sm.u if q_ok else None
        Xfit = _with_treatment(cov, A, u_int)
        cum1, cum0 = [], []
        for j in range(1, p.n_causes + 1):
            c1, c0, failed = _cox_cumhaz(
                Xfit,
                sm.event_times,
                (sm.event_indicator == j).astype(int),
                _with_treatment(cov, 1.0, u_int),
                _with_treatment(cov, 0.0, u_int),
                grid,
            )
            n_failed += int(failed)
            cum1.append(c1)
            cum0.append(c0)

    # -- G: censoring survival ---------------------------------------------
    if spec.G == "oracle" and p.censoring != "hazard":
        raise ValueError(
            "G='oracle' assumes an exponential censoring hazard, but this "
            f"configuration uses censoring={p.censoring!r}. Use 'correct' "
            "instead, which estimates G from the data either way."
        )
    if spec.G == "oracle":
        cumc1 = np.outer(censoring_rate(sm.X, sm.u, 1.0, p), grid)
        cumc0 = np.outer(censoring_rate(sm.X, sm.u, 0.0, p), grid)
    else:
        g_ok = spec.G == "correct"
        cov = sm.design(include_threshold=g_ok)
        # Mirrors the outcome model above: when the censoring hazard carries a
        # treatment x threshold interaction (``eta_c``), a correct censoring
        # model has to carry it too, or "correct" would not be correct -- and
        # the wrong model would differ from it only by a main effect that the
        # linear term in w_cont largely absorbs.
        # ... but only when the truth actually has one. With eta_c = 0 a model
        # without the interaction already *is* correct, and adding a column whose
        # coefficient is known to be zero only adds estimation noise.
        u_int = sm.u if (g_ok and p.eta_c != 0.0) else None
        Xfit = _with_treatment(cov, A, u_int)
        cumc1, cumc0, failed = _cox_cumhaz(
            Xfit,
            sm.event_times,
            (sm.event_indicator == 0).astype(int),
            _with_treatment(cov, 1.0, u_int),
            _with_treatment(cov, 0.0, u_int),
            grid,
        )
        n_failed += int(failed)

    # -- pi -----------------------------------------------------------------
    if spec.pi == "oracle":
        ps1 = sm.ps_true.copy()
    else:
        # correct: the full design. wrong: the continuous confounder dropped.
        cov = sm.X if spec.pi == "correct" else sm.X[:, :2]
        if spec.pi == "correct":
            ps1 = _fit_propensity_correct(sm, cov, q)
        else:
            ps1 = _fit_propensity_wrong(sm, cov)
    ps1 = np.clip(ps1, 1e-9, 1 - 1e-9)

    # -- assemble ------------------------------------------------------------
    def _arm(cum_list, cumc, ps, g_star) -> InitialEstimates:
        haz = np.stack([np.diff(c, prepend=0.0, axis=1) for c in cum_list], axis=-1)
        haz = np.maximum(haz, 0.0)
        surv = np.exp(-np.sum(cum_list, axis=0))
        return InitialEstimates(
            times=grid,
            g_star_obs=g_star,
            propensity_scores=ps,
            hazards=haz,
            event_free_survival_function=surv,
            censoring_survival_function=np.exp(-cumc),
        )

    est = {
        key_1: _arm(cum1, cumc1, ps1, A.astype(float)),
        key_0: _arm(cum0, cumc0, 1.0 - ps1, 1.0 - A.astype(float)),
    }

    if not return_diagnostics:
        return est

    true1 = [np.outer(r, grid) for r in cause_rates(sm.X, sm.u, 1.0, p)]
    cumhaz_mae = float(
        np.mean([np.abs(a - b).mean() for a, b in zip(cum1, true1)])
    )
    diag = NuisanceDiagnostics(
        ps_mae=float(np.abs(ps1 - sm.ps_true).mean()),
        ps_min=float(ps1.min()),
        ps_max=float(ps1.max()),
        cumhaz_mae=cumhaz_mae,
        n_failed_fits=n_failed,
    )
    return est, diag


#: The eight cells of Study A's factorial, in the order used throughout.
STUDY_A_CELLS: Dict[str, Spec] = {
    "C1": Spec("correct", "correct", "correct"),
    "C2": Spec("correct", "correct", "wrong"),
    "C3": Spec("correct", "wrong", "correct"),
    "C4": Spec("correct", "wrong", "wrong"),
    "C5": Spec("wrong", "correct", "correct"),
    "C6": Spec("wrong", "correct", "wrong"),
    "C7": Spec("wrong", "wrong", "correct"),
    "C8": Spec("wrong", "wrong", "wrong"),
}
