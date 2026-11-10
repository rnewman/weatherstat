"""Gram-based ridge solves must match the direct computations they replace.

Bootstrap, VIF and the transition-weighted identification refits were
rewritten to solve from XᵀWX instead of resampled / reweighted copies of X
(sysid refit performance). Each test pins the optimized path to a
straightforward reference — the previous implementation — on designs shaped
like sysid's: 0/1 lag columns, continuous weather/solar columns, a
near-collinear pair.
"""

from __future__ import annotations

import numpy as np
import pytest

from weatherstat.gram import bootstrap_counts, weighted_grams
from weatherstat.identification import weighted_ridge
from weatherstat.validate import compute_bootstrap_stability, compute_vif


def _design(n: int = 3000, p: int = 9, seed: int = 3) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, :4] = (X[:, :4] > 0.7).astype(float)  # effector lag indicators
    X[:, 5] = X[:, 4] * 0.97 + rng.normal(scale=0.05, size=n)  # near-collinear with col 4
    X[:, 6] = np.clip(np.sin(np.arange(n) / 40.0), 0, None)  # solar-like, zero half the time
    y = X @ rng.normal(size=p) + rng.normal(size=n)
    names = [f"_lag_e{i}" for i in range(4)] + ["_weather_a", "_weather_b", "_solar_elev", "_adv_tau_w", "_tod_sin1"]
    return X, y, names


def _scale(X: np.ndarray, names: list[str]) -> np.ndarray:
    scale = np.ones(X.shape[1])
    for j, name in enumerate(names):
        if name.startswith(("_solar_", "_adv_")):
            scale[j] = np.std(X[:, j]) or 1.0
    return scale


class TestWeightedGrams:
    @pytest.mark.parametrize("K", [1, 3, 20])  # per-vector path and batched path
    def test_matches_direct_products(self, K: int) -> None:
        X, y, _ = _design(n=1777)  # not a multiple of the block size
        W = np.random.default_rng(1).random((K, len(y)))
        G, b, yy = weighted_grams(X, y, W)
        for k in range(K):
            Xw = X * np.sqrt(W[k])[:, None]
            np.testing.assert_allclose(G[k], Xw.T @ Xw, rtol=1e-12, atol=1e-9)
            np.testing.assert_allclose(b[k], X.T @ (W[k] * y), rtol=1e-12, atol=1e-9)
            assert yy[k] == pytest.approx(float(W[k] @ (y * y)), rel=1e-12)

    def test_batched_grams_are_symmetric(self) -> None:
        X, y, _ = _design()
        G, _, _ = weighted_grams(X, y, np.random.default_rng(2).random((12, len(y))))
        np.testing.assert_array_equal(G, np.transpose(G, (0, 2, 1)))

    def test_bootstrap_counts_replay_the_index_draws(self) -> None:
        n = 500
        counts = bootstrap_counts(np.random.RandomState(42), n, 4)
        rng = np.random.RandomState(42)
        for i in range(4):
            np.testing.assert_array_equal(counts[i], np.bincount(rng.randint(0, n, size=n), minlength=n))
        assert counts.sum(axis=1).tolist() == [n] * 4


class TestMatchesPreviousImplementations:
    def test_bootstrap_cvs(self) -> None:
        X, y, names = _design()
        scale = _scale(X, names)
        lam = 0.01 * len(y)
        # Reference: gather each resample and solve it directly (the previous code).
        rng = np.random.RandomState(42)
        X_s = X / scale
        betas = []
        for _ in range(50):
            idx = rng.randint(0, len(y), size=len(y))
            Xb, yb = X_s[idx], y[idx]
            betas.append(np.linalg.solve(Xb.T @ Xb + lam * np.eye(X.shape[1]), Xb.T @ yb) / scale)
        betas = np.array(betas)
        means, stds = betas.mean(axis=0), betas.std(axis=0)
        ref = np.where(np.abs(means) > 1e-10, stds / np.abs(means), 0.0)
        np.testing.assert_allclose(compute_bootstrap_stability(X, y, scale, lam), ref, rtol=1e-8, atol=1e-12)

    def test_vif(self) -> None:
        X, _, names = _design()
        X_s = X / _scale(X, names)
        got = compute_vif(X_s, names)
        for j, name in enumerate(names):
            # Reference: drop the column, regress it on the rest (the previous code).
            y_j = X_s[:, j]
            X_o = np.delete(X_s, j, axis=1)
            beta = np.linalg.solve(X_o.T @ X_o + 0.001 * len(y_j) * np.eye(X_o.shape[1]), X_o.T @ y_j)
            ss_res = np.sum((y_j - X_o @ beta) ** 2)
            ss_tot = np.sum((y_j - y_j.mean()) ** 2)
            ref = 1.0 / (1.0 - min(1.0 - ss_res / ss_tot, 0.9999))
            assert got[name] == pytest.approx(ref, rel=1e-9), name
        assert got["_weather_a"] > 10  # the collinear pair is still flagged

    def test_weighted_ridge(self) -> None:
        X, y, names = _design()
        w = np.exp(-(((np.arange(len(y)) % 300) - 150.0) ** 2) / (2 * 40.0**2))  # transition bumps
        scale = _scale(X, names)
        # Reference: √w-reweighted copy of X (the previous code).
        sw = np.sqrt(w)
        Xw, yw = (X / scale) * sw[:, None], y * sw
        A = Xw.T @ Xw + 0.01 * w.sum() * np.eye(X.shape[1])
        beta_s = np.linalg.solve(A, Xw.T @ yw)
        resid = yw - Xw @ beta_s
        s2 = float(resid @ resid) / max(w.sum() - X.shape[1], 1.0)
        se = np.sqrt(np.abs(np.diag(s2 * np.linalg.inv(A)))) / scale
        beta_ref = beta_s / scale
        beta, t = weighted_ridge(X, y, w, names)
        np.testing.assert_allclose(beta, beta_ref, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(t, beta_ref / se, rtol=1e-9, atol=1e-9)
