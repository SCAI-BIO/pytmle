"""Continuous-time competing-risks data generating process.

The design follows Hage et al. (2025, Stat. Med. 44(18-19), doi:10.1002/sim.70066),
whose code lives at https://github.com/survival-lumc/AdjCuminc, and generalises it
with an informative censoring mechanism.

Covariates
----------
``w_cat``   three-level categorical, equal thirds, entered as two dummies
``w_cont``  continuous, standard normal

The design matrix is ``X = [d2, d3, w_cont]``, matching the three-element
coefficient vectors of the reference study.

Structure
---------
    u(W)  = 1{w_cont > 1}                       threshold term
    A     ~ Bern(expit(gamma'X))                optionally followed by the
                                                control-arm resampling device
    lam_j = exp(alpha_j + X'beta_j + theta_j A + delta_j u)      j = 1, 2
    lam_C = exp(alpha_c + X'beta_c + theta_c A + delta_c u)

Latent exponential times per cause, ``T = min(T_1, T_2)``, ``J = argmin``;
censoring either from its own exponential hazard or, for the ``adjcuminc``
configuration, from ``U(P20(T), P95(T))``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Optional

import numpy as np
import pandas as pd

__all__ = ["DGPParams", "Sample", "expit", "sample", "CONFIGS", "get_config"]


def expit(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class DGPParams:
    """Parameters of the data generating process.

    ``delta`` / ``delta_c`` switch the threshold term on; ``control_resample``
    switches the non-logistic treatment-assignment device on. Both stay on for
    every cell of a given configuration -- what varies across cells is only what
    the fitted nuisance models can represent (see ``nuisance.py``).
    """

    # --- treatment -------------------------------------------------------
    gamma: np.ndarray = field(default_factory=lambda: np.array([0.5, -0.4, 0.3]))
    #: Deprecated: every attempt to put the propensity misspecification *inside*
    #: the true propensity failed the same way. A threshold term is nearly a
    #: monotone transform of w_cont, so a logistic in w_cont absorbs it (|z| <= 1.8);
    #: a quadratic is not absorbable but is large in both tails, driving e(W) to
    #: 0.96-1.00 above the 95th percentile and destabilising IPW even when pi is
    #: correct (+0.0145 bias). Strengthening the lever and preserving positivity
    #: are in direct conflict while the misspecification lives in the truth.
    #:
    #: The pi lever is therefore an *omitted confounder*: the truth stays plain
    #: logistic in W, so positivity is governed solely by gamma, and the wrong fit
    #: simply drops w_cont. See nuisance._fit_propensity_wrong.
    delta_pi: float = 0.0
    control_resample: bool = False

    #: Location of the threshold, ``u(W) = 1{w_cont > threshold}``. At 1.0 only
    #: ~16 % of subjects are above it, which leaves the treatment x threshold
    #: interaction non-zero for ~8 % of the sample -- a strongly predictive column
    #: supported by very few observations, which conditions the Cox fit badly.
    #: Lowering it widens that support and gives the misspecification more of the
    #: sample to act on.
    threshold: float = 1.0

    # --- cause-specific hazards (two causes) ------------------------------
    alpha: np.ndarray = field(default_factory=lambda: np.array([-2.3, -2.6]))
    beta: np.ndarray = field(
        default_factory=lambda: np.array([[0.4, -0.2, 0.1], [-0.3, 0.3, 0.2]])
    )
    theta: np.ndarray = field(default_factory=lambda: np.array([-0.6, 0.2]))
    delta: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    #: treatment x threshold interaction. Without this the threshold term shifts
    #: both arms identically, so omitting it from the outcome model moves F^1 and
    #: F^0 together and the risk *difference* survives almost intact -- a large
    #: nuisance-level discrepancy that produces no estimand-level bias. The
    #: interaction is what makes omitting the term bias the contrast.
    eta: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))

    # --- censoring --------------------------------------------------------
    censoring: Literal["hazard", "uniform-quantile"] = "hazard"
    alpha_c: float = -3.0
    beta_c: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.1, -0.1]))
    theta_c: float = 0.0
    delta_c: float = 0.0
    #: treatment x threshold interaction in the *censoring* hazard -- the exact
    #: counterpart of ``eta``, and needed for the same reason. Without it the
    #: threshold term shifts G identically in both arms, so omitting it from the
    #: fitted censoring model leaves the risk *difference* essentially untouched
    #: and the G lever is inert: measured at delta_c = 1.0 with eta_c = 0, the
    #: G-correct and G-wrong cells of rung 4 differ by <= 0.0014 under
    #: *informative* censoring. Worse, u = 1{w_cont > threshold} is monotone in
    #: w_cont, so a censoring model that keeps the linear term absorbs most of
    #: the omitted main effect anyway (FINDINGS 6). The interaction is what it
    #: cannot absorb.
    eta_c: float = 0.0

    # --- extra noise covariates (Study D, dimension axis) -----------------
    n_noise: int = 0

    # --- administrative ---------------------------------------------------
    n_causes: int = 2

    def with_(self, **kwargs) -> "DGPParams":
        """Return a copy with fields replaced (keeps the dataclass frozen)."""
        return replace(self, **kwargs)


@dataclass
class Sample:
    """One simulated dataset plus everything the nuisance builders need."""

    df: pd.DataFrame  # event_time, event_indicator, group, covariates
    X: np.ndarray  # (n, 3) design matrix [d2, d3, w_cont]
    u: np.ndarray  # (n,) threshold indicator 1{w_cont > 1}
    noise: np.ndarray  # (n, n_noise) inert covariates
    ps_true: np.ndarray  # (n,) true P(A = 1 | W)
    params: DGPParams

    @property
    def n(self) -> int:
        return len(self.df)

    @property
    def event_times(self) -> np.ndarray:
        return self.df["event_time"].to_numpy()

    @property
    def event_indicator(self) -> np.ndarray:
        return self.df["event_indicator"].to_numpy()

    @property
    def group(self) -> np.ndarray:
        return self.df["group"].to_numpy()

    def design(self, include_threshold: bool, include_noise: bool = True) -> np.ndarray:
        """Covariate matrix handed to a nuisance regression.

        ``include_threshold`` is the switch that makes a fit correctly specified
        or not: the truth is linear in ``[X, u]``, so omitting ``u`` cannot be
        absorbed by the linear term in ``w_cont``.
        """
        parts = [self.X]
        if include_threshold:
            parts.append(self.u[:, None])
        if include_noise and self.noise.shape[1] > 0:
            parts.append(self.noise)
        return np.column_stack(parts)


def _design(w_cat: np.ndarray, w_cont: np.ndarray) -> np.ndarray:
    """Two dummies for the three-level categorical, plus the continuous covariate."""
    d2 = (w_cat == 1).astype(float)
    d3 = (w_cat == 2).astype(float)
    return np.column_stack([d2, d3, w_cont])


def _draw_covariates(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    w_cat = rng.integers(0, 3, size=n)
    w_cont = rng.normal(size=n)
    return w_cat, w_cont


def cause_rates(X: np.ndarray, u: np.ndarray, a: np.ndarray | float, p: DGPParams):
    """Cause-specific hazard rates (constant in t) for treatment value ``a``."""
    a = np.broadcast_to(np.asarray(a, dtype=float), (X.shape[0],))
    return [
        np.exp(
            p.alpha[j] + X @ p.beta[j] + p.theta[j] * a
            + p.delta[j] * u + p.eta[j] * a * u
        )
        for j in range(p.n_causes)
    ]


def censoring_rate(X: np.ndarray, u: np.ndarray, a: np.ndarray | float, p: DGPParams):
    a = np.broadcast_to(np.asarray(a, dtype=float), (X.shape[0],))
    return np.exp(p.alpha_c + X @ p.beta_c + p.theta_c * a + p.delta_c * u
                  + p.eta_c * a * u)


def marginal_treated_fraction(p: DGPParams, n_mc: int = 2_000_000, seed: int = 20250301) -> float:
    """q = E[expit(gamma'X)], needed by the control-resampling propensity."""
    rng = np.random.default_rng(seed)
    w_cat, w_cont = _draw_covariates(n_mc, rng)
    return float(expit(_design(w_cat, w_cont) @ p.gamma).mean())


def true_propensity(X: np.ndarray, p: DGPParams, q: Optional[float] = None,
                    u: Optional[np.ndarray] = None) -> np.ndarray:
    """True P(A = 1 | W).

    Without the resampling device this is just ``expit(gamma'X)``. With it, the
    treated arm's covariates are tilted by the logistic selection while the
    control arm's are marginal, which gives

        e(x) = sigma(gamma'x) / (sigma(gamma'x) + 1 - q),  q = E[sigma(gamma'X)]

    i.e. ``logit e(x) = log sigma(gamma'x) - log(1 - q)``.
    """
    sig = expit(X @ p.gamma)
    if not p.control_resample:
        return sig
    if q is None:
        q = marginal_treated_fraction(p)
    return sig / (sig + (1.0 - q))


def sample(n: int, p: DGPParams, rng: np.random.Generator, q: Optional[float] = None) -> Sample:
    """Draw one replicate."""
    w_cat, w_cont = _draw_covariates(n, rng)
    X = _design(w_cat, w_cont)

    u_pre = (w_cont > p.threshold).astype(float)
    a_prob = expit(X @ p.gamma)
    A = rng.binomial(1, a_prob).astype(int)

    if p.control_resample:
        # Hage et al. Scenario 2: the control arm's covariates are redrawn from
        # the marginal distribution *before* event times are simulated, so W
        # remains the genuine driver of the outcome while P(A=1|W) stops being
        # logistic.
        ctl = A == 0
        n_ctl = int(ctl.sum())
        if n_ctl:
            w_cat_new, w_cont_new = _draw_covariates(n_ctl, rng)
            w_cat = w_cat.copy()
            w_cont = w_cont.copy()
            w_cat[ctl] = w_cat_new
            w_cont[ctl] = w_cont_new
            X = _design(w_cat, w_cont)

    u = (w_cont > p.threshold).astype(float)

    rates = cause_rates(X, u, A, p)
    latent = np.column_stack([rng.exponential(1.0 / r) for r in rates])
    t_event = latent.min(axis=1)
    cause = latent.argmin(axis=1) + 1

    if p.censoring == "hazard":
        c_time = rng.exponential(1.0 / censoring_rate(X, u, A, p))
    elif p.censoring == "uniform-quantile":
        lo, hi = np.quantile(t_event, [0.20, 0.95])
        c_time = rng.uniform(lo, hi, size=n)
    else:  # pragma: no cover - guarded by the dataclass Literal
        raise ValueError(f"unknown censoring mechanism {p.censoring!r}")

    obs = np.minimum(t_event, c_time)
    delta = np.where(t_event <= c_time, cause, 0).astype(int)

    noise = rng.normal(size=(n, p.n_noise)) if p.n_noise else np.zeros((n, 0))

    data = {
        "event_time": obs,
        "event_indicator": delta,
        "group": A,
        "d2": X[:, 0],
        "d3": X[:, 1],
        "w_cont": X[:, 2],
    }
    for k in range(p.n_noise):
        data[f"z{k + 1}"] = noise[:, k]

    return Sample(
        df=pd.DataFrame(data),
        X=X,
        u=u,
        noise=noise,
        ps_true=true_propensity(X, p, q=q, u=u),
        params=p,
    )


# --------------------------------------------------------------------------
# Named configurations
# --------------------------------------------------------------------------

#: Hage et al. parameters. ``lambda_{k,0} = 0`` for their Scenarios 1-2 and
#: ``= 2`` with the ``-4 * 1{w_cont < 1}`` term for Scenarios 3-4; up to a
#: baseline constant the latter is ``+4 * 1{w_cont > 1}``, which is how it is
#: written here.
_ADJCUMINC = DGPParams(
    gamma=np.array([1.0, -1.0, 1.0]),
    alpha=np.array([0.0, 0.0]),
    beta=np.array([[1.0, -1.0, 0.5], [-1.0, 1.0, -0.5]]),
    theta=np.array([-1.0, -0.5]),
    delta=np.array([0.0, 0.0]),
    censoring="uniform-quantile",
)

#: Study A's calibrated family. Named rather than inlined because rung 4 of the
#: validation ladder varies *only* the censoring regime off it -- keeping the
#: outcome side literally the same object is what makes that contrast clean.
_THRESHOLD = DGPParams(
    gamma=np.array([1.0, -0.8, 0.6]),
    threshold=0.0,
    delta=np.array([1.5, 1.5]),
    eta=np.array([-1.0, 0.7]),
    delta_c=2.0,
    control_resample=False,
)

CONFIGS: dict[str, DGPParams] = {
    # Studies B and C: no misspecification intended.
    "base": DGPParams(),
    # Study A: both devices on, one DGP serving all eight specification cells.
    # gamma is calibrated (guard step 1): at (1.0, -0.8, 0.6) the plain-logistic
    # propensity fit carries an asymptotic mean absolute error of ~0.018 against
    # ~0.002 for the correct family -- a 7.6x gap, so the misspecification is
    # real -- while the true propensity stays within [0.09, 0.67] even at
    # n = 50 000, so positivity is nowhere near violated. Weaker gamma makes the
    # misspecification inert; stronger drives e(W) to zero, which belongs in
    # Study D rather than here.
    # Calibrated by `sim.calibrate --study-a`, not chosen a priori. See
    # FINDINGS.md 2 and 3 for why the obvious settings fail:
    #   * threshold at 0 rather than 1 -- at 1 only 18 % of subjects are above it
    #     and the treatment x threshold interaction is supported by 11 %, which
    #     conditions the Cox fit badly enough to fail 50-75 % of TMLE fits.
    #   * delta = 1.5 rather than 4 -- in continuous time delta multiplies the
    #     hazard *rate*, so 4 means e^4 = 55x and a 150x spread across subjects.
    #   * eta non-zero and same-signed as theta -- without the interaction the
    #     omitted term shifts both arms together and the risk difference is
    #     untouched (0.002 against a truth of -0.149).
    # Verified: gcomp |z| ~ 0.8 with Q correct and ~13 with Q wrong; every
    # replicate usable; estimands -0.10 to -0.27 and +0.11 to +0.25, none near
    # zero.
    # One device, three components: the threshold u(W) enters the propensity, the
    # cause-specific hazards and the censoring hazard, and each nuisance is
    # misspecified by omitting u from its design matrix. Uniform, easy to
    # calibrate, and it removes the mixture-weighted target that the
    # control-resampling device forces on the truth. That device is retained only
    # in the adjcuminc_* configs, where faithful reproduction requires it.
    "threshold": _THRESHOLD,
    # Validation ladder, reproducing the reference study's four scenarios.
    # Scenarios 3-4 carry their non-linearity. Their code writes it as
    # ``exp(ifelse(x3 > 1, 2, -2))`` and the paper as ``lambda_{k,0} = 2`` with a
    # ``-4 * 1{x3 < 1}`` term; both give log-hazard +2 above the threshold and -2
    # below. Reproducing that needs ``alpha = -2`` *and* ``delta = 4``: using
    # ``alpha = 0`` would scale both cause hazards by e^2, which leaves the
    # cause-mix untouched but shifts the CIF in time, so the curves would not
    # line up with theirs.
    "adjcuminc_s1": _ADJCUMINC,
    "adjcuminc_s2": _ADJCUMINC.with_(control_resample=True),
    "adjcuminc_s3": _ADJCUMINC.with_(
        alpha=np.array([-2.0, -2.0]), delta=np.array([4.0, 4.0])
    ),
    "adjcuminc_s4": _ADJCUMINC.with_(
        alpha=np.array([-2.0, -2.0]), delta=np.array([4.0, 4.0]),
        control_resample=True,
    ),
    # Rung 4: their DGP with censoring made informative. alpha_c = -1.4 is
    # calibrated to reproduce their ~24 % censoring, so the only thing that
    # changes is *whether censoring depends on covariates and treatment* --
    # which is what their design, having no censoring model at all, cannot
    # detect. delta_c gives the censoring hazard a threshold term, so omitting
    # it from the fitted model is a genuine G misspecification.
    # Rung 4, first attempt: their DGP with informative censoring. Retained for
    # the record but superseded -- their sharp assignment (omega = (1,-1,1)) puts
    # min pi*G at 0.003, so min_nuisance truncation masks any censoring effect
    # before it can bias anything, and the very fast event rates drove 7-9 %
    # replicate failures. The rung4_* configs below rebuild it on the calibrated
    # threshold family instead.
    "adjcuminc_cens_none": _ADJCUMINC,
    "adjcuminc_cens_info": _ADJCUMINC.with_(
        censoring="hazard",
        alpha_c=-1.4,
        beta_c=np.array([0.5, -0.5, 0.4]),
        theta_c=0.3,
        delta_c=2.0,
    ),
    # Rung 4: the censoring regime is the only thing that varies. Both variants
    # censor ~22 %, so the contrast is about *dependence on covariates and
    # treatment*, not about how much censoring there is. control_resample is off
    # so the propensity is plain logistic and "pi correct" is unambiguous.
    # They are built by `with_` off the calibrated `threshold` family rather than
    # spelled out, so the outcome side is *identical* to Study A's -- same
    # threshold location, same delta = 1.5, same treatment x threshold eta. That
    # matters twice over: delta = 4 at threshold = 1.0 multiplies the hazard rate
    # by e^4 in continuous time and drove most fits to fail (FINDINGS 3), and
    # without eta the Q-wrong row carries no estimand-level bias at all
    # (FINDINGS 2) -- which is precisely the row this rung reads its answer from.
    "rung4_cens_none": _THRESHOLD.with_(
        censoring="hazard",
        alpha_c=-2.482,
        beta_c=np.zeros(3),
        theta_c=0.0,
        delta_c=0.0,
    ),
    # The dependence coefficients are half the size they look like they "should"
    # be, and that is deliberate. At full strength -- beta_c = (0.6, -0.6, 0.5),
    # theta_c = 0.3, delta_c = 2.0 -- the 1st percentile of the true pi * G at the
    # last target time is 0.0075, *below* the min_nuisance bound of 0.01. The
    # clever covariate's denominator then sits on the truncation floor for the
    # subjects who matter most, and the rung measures a positivity failure in G
    # rather than the effect of censoring dependence: measured 2.5-55 % replicate
    # failures and 36-67 % TMLE non-convergence, in the all-correct cell too.
    # Halving the coefficients puts that percentile at 0.043 -- four times the
    # bound -- while G(tau_max) still spans an interquartile range of 0.55-0.86
    # across subjects, so censoring remains firmly covariate- and
    # treatment-dependent. Same lesson as FINDINGS 6, on a different nuisance.
    # eta_c is the piece that makes omitting `u` from the censoring model cost
    # anything at all -- see the field's own note. delta_c is *zero* here, and
    # that is the calibrated choice rather than an oversight: the two terms
    # compound in the treated-and-above-threshold cell, and it is that cell which
    # sets the positivity floor. Carrying the same arm gap through the
    # interaction alone is strictly better on both counts:
    #
    #   delta_c  eta_c   min(pi*G)   0.5th pct   arm gap in G(tau_max) | u = 1
    #      1.0     1.2      0.0082      0.0561              -0.440
    #      0.0     1.6      0.0142      0.0760              -0.466
    #
    # At (1.0, 1.2) the treated arm's IC overflowed in 1.7-13 % of replicates;
    # alpha_c is re-solved each time to hold censoring at 22 %, so the comparison
    # is at matched censoring.
    "rung4_cens_info": _THRESHOLD.with_(
        censoring="hazard",
        alpha_c=-2.947,
        beta_c=np.array([0.3, -0.3, 0.25]),
        theta_c=0.15,
        delta_c=0.0,
        eta_c=1.6,
    ),
}


def get_config(name: str) -> DGPParams:
    if name not in CONFIGS:
        raise KeyError(f"unknown DGP config {name!r}; have {sorted(CONFIGS)}")
    return CONFIGS[name]
