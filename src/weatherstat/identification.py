"""Causal identification for behavior-driven couplings.

Effector gains are identified by quasi-experiment: the controller flips
devices on its own schedule, thousands of times. Environment couplings
(shades, windows — user-operated) have no such luxury: the "treatment" is
human habit, correlated with time of day, season, and weather. A batch
coefficient on observational habit data can be confounded in either
direction — inflated by correlation (a NE shade "cooling" a SW room) or
attenuated by it (a shade's true effect on its own room diluted 3x).

The test: refit with sample weights concentrated in Gaussian windows around
the device's state *transitions*. Within a few hours of a transition, slow
confounders are approximately constant while the state flips, so the
coefficient is identified by within-window contrast. A coupling that holds
(same sign, still significant) as the window tightens is CAUSAL; one that
collapses was CORRELATIONAL; too little signal near transitions is
UNDERPOWERED (no verdict).

The tags gate *use*, not existence: correlational couplings still improve
conditional prediction (habit genuinely predicts, in distribution) — they
are only excluded from counterfactual evaluation (the advisory sweep's
what-if toggles), where correlation lies. See simulator.py enforcement and
docs/plans/PLAN-36-identification-tagging.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

# Test windows, widest to tightest. σ=1h is deliberately excluded from the
# verdict: too little solar variance inside one hour for stable estimates.
SIGMAS_HOURS: tuple[float, ...] = (8.0, 4.0, 2.0)

# Minimum weighted signal energy Σw·F² for a σ to contribute a verdict.
INFO_MIN = 30.0

# A causal coupling must retain |t| >= this (with the batch sign) at every
# test σ. Batch keep-threshold is 2.0; transition windows have less data, so
# the persistence bar is slightly lower than the discovery bar.
T_CAUSAL = 1.5

# ... and must retain at least this fraction of the batch magnitude. High-SNR
# couplings can shrink several-fold under identification yet stay
# "significant"; a coupling that loses most of its magnitude near transitions
# is mostly habit, and crediting the batch value counterfactually would
# over-credit it. (Causal-but-batch-attenuated couplings strengthen instead
# and pass trivially.)
MAGNITUDE_RETENTION_MIN = 0.5


class IdentificationTag(StrEnum):
    CAUSAL = "causal"
    CORRELATIONAL = "correlational"
    UNDERPOWERED = "underpowered"


@dataclass(frozen=True)
class IdentificationResult:
    tag: str
    beta_batch: float
    t_batch: float
    by_sigma: dict[str, dict[str, float]] = field(default_factory=dict)  # "4h" -> {beta, t, info}


def transition_weights(state: np.ndarray, ts_hours: np.ndarray, sigma_hours: float) -> np.ndarray:
    """Gaussian bump around each state transition, two-sided, max over transitions."""
    change = np.zeros(len(state), dtype=bool)
    change[1:] = state[1:] != state[:-1]
    t_transitions = ts_hours[change]
    if len(t_transitions) == 0:
        return np.zeros(len(state))
    idx = np.searchsorted(t_transitions, ts_hours)
    d_prev = np.where(idx > 0, ts_hours - t_transitions[np.clip(idx - 1, 0, None)], np.inf)
    d_next = np.where(
        idx < len(t_transitions), t_transitions[np.clip(idx, None, len(t_transitions) - 1)] - ts_hours, np.inf
    )
    d = np.minimum(np.abs(d_prev), np.abs(d_next))
    return np.exp(-(d**2) / (2 * sigma_hours**2))


def weighted_ridge(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Selectively standardized ridge with row weights.

    Replicates the batch solve in sysid._fit_sensor_model (same selective
    standardization, λ scaled by effective sample size) so weighted and
    batch coefficients are directly comparable.
    """
    scale = _selective_scale(X, feature_names)
    return _weighted_ridge_scaled(X / scale, scale, y, w)


def _selective_scale(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Column scales for the selective standardization (solar/advisory by std, the rest 1)."""
    scale = np.ones(X.shape[1])
    for j, name in enumerate(feature_names):
        if name.startswith("_solar_") or name.startswith("_adv_"):
            s = np.std(X[:, j])
            if s > 0:
                scale[j] = s
    return scale


def _weighted_ridge_scaled(
    X_s: np.ndarray, scale: np.ndarray, y: np.ndarray, w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """weighted_ridge on an already-scaled design (X_s = X / scale).

    Solved from the weighted Gram XᵀWX rather than a √w-reweighted copy of X;
    the scaling is shared across the σ sweep of one coupling.
    """
    from weatherstat.gram import weighted_grams

    G, b, _ = weighted_grams(X_s, y, w[None, :])
    n_eff = float(w.sum())
    lam = 0.01 * n_eff
    A = G[0] + lam * np.eye(X_s.shape[1])
    beta_s = np.linalg.solve(A, b[0])
    beta = beta_s / scale
    r = y - X_s @ beta_s
    dof = max(n_eff - X_s.shape[1], 1.0)
    s2 = float(w @ (r * r)) / dof
    try:
        cov = s2 * np.linalg.inv(A)
        se = np.sqrt(np.abs(np.diag(cov))) / scale
    except np.linalg.LinAlgError:
        se = np.full(len(beta), np.nan)
    t = np.where(se > 0, beta / se, 0.0)
    return beta, t


def classify_coupling(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    feature: str,
    state: np.ndarray,
    ts_hours: np.ndarray,
    *,
    beta_batch: float,
    t_batch: float,
) -> IdentificationResult:
    """Classify one coupling as causal / correlational / underpowered.

    X, y are the (already NaN-filtered) batch design and target; state and
    ts_hours are the device state and timestamps filtered identically.
    """
    j = feature_names.index(feature)
    scale = _selective_scale(X, feature_names)
    X_s = X / scale
    sign_batch = np.sign(beta_batch)
    by_sigma: dict[str, dict[str, float]] = {}
    holds = True
    for sigma in SIGMAS_HOURS:
        w = transition_weights(state, ts_hours, sigma)
        info = float((w * X[:, j] ** 2).sum())
        key = f"{sigma:.0f}h"
        if info < INFO_MIN:
            by_sigma[key] = {"info": round(info, 1)}
            return IdentificationResult(
                IdentificationTag.UNDERPOWERED, round(beta_batch, 4), round(t_batch, 2), by_sigma,
            )
        beta, t = _weighted_ridge_scaled(X_s, scale, y, w)
        by_sigma[key] = {"beta": round(float(beta[j]), 4), "t": round(float(t[j]), 2), "info": round(info, 1)}
        if (
            np.sign(beta[j]) != sign_batch
            or abs(t[j]) < T_CAUSAL
            or abs(beta[j]) < MAGNITUDE_RETENTION_MIN * abs(beta_batch)
        ):
            holds = False
    tag = IdentificationTag.CAUSAL if holds else IdentificationTag.CORRELATIONAL
    return IdentificationResult(tag, round(beta_batch, 4), round(t_batch, 2), by_sigma)
