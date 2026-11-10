"""Weighted Gram matrices for many ridge solves on one design.

Bootstrap resamples, transition-weighted refits and VIF all solve ridge
problems on the same X with different row weights (or column subsets).
Each solve needs only XᵀWX, XᵀWy and yᵀWy — p×p, not N×p — so they can be
built without materializing a resampled or reweighted copy of X per solve.

A bootstrap resample is the special case of integer weights: drawing row i
c_i times contributes c_i·xᵢxᵢᵀ, so ``Σ_resampled xxᵀ = Xᵀ diag(c) X``
exactly (up to float summation order).
"""

from __future__ import annotations

import numpy as np

# Above this many weight vectors, build all Grams with one GEMM over the
# row-wise outer products instead of one weighted product per vector.
_BATCH_MIN_K = 8
# Rows per block of outer products. Measured best at ~500 rows for both
# p=33 (2 MB scratch) and p=60 (8 MB): small enough to stay in cache, large
# enough for an efficient GEMM. 32 MB blocks were 2.6× slower.
_BLOCK_ROWS = 512


def weighted_grams(X: np.ndarray, y: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each row w of ``W`` (K × N): (XᵀWX, XᵀWy, yᵀWy) with W = diag(w).

    Returns G (K, p, p), b (K, p), yy (K,).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    W = np.atleast_2d(np.asarray(W, dtype=float))
    n, p = X.shape
    K = W.shape[0]
    if W.shape[1] != n or y.shape != (n,):
        raise ValueError(f"shape mismatch: X {X.shape}, y {y.shape}, W {W.shape}")

    if K < _BATCH_MIN_K:
        G = np.empty((K, p, p))
        b = np.empty((K, p))
        yy = np.empty(K)
        for k in range(K):
            Xw = X * W[k][:, None]
            G[k] = Xw.T @ X
            b[k] = Xw.T @ y
            yy[k] = W[k] @ (y * y)
        return G, b, yy

    # Batched: G_k[i, j] = Σ_r W[k, r] · X[r, i] · X[r, j]. Stack the upper-
    # triangle outer products of each row as Z (rows × p(p+1)/2); then every
    # Gram is one row of W @ Z — a single large GEMM, half the flops of K
    # separate full products and no per-k copy of X. Blocked over rows to
    # bound the scratch.
    iu, ju = np.triu_indices(p)
    m = len(iu)
    rows = _BLOCK_ROWS
    G_upper = np.zeros((K, m))
    b = np.zeros((K, p))
    yy = np.zeros(K)
    for s in range(0, n, rows):
        Xb = X[s : s + rows]
        yb = y[s : s + rows]
        Wb = W[:, s : s + rows]
        G_upper += Wb @ (Xb[:, iu] * Xb[:, ju])
        b += Wb @ (Xb * yb[:, None])
        yy += Wb @ (yb * yb)
    G = np.empty((K, p, p))
    G[:, iu, ju] = G_upper
    G[:, ju, iu] = G_upper
    return G, b, yy


def bootstrap_counts(rng: np.random.RandomState, n: int, n_boot: int) -> np.ndarray:
    """(n_boot × n) resample multiplicities, from the same draws as ``rng.randint(0, n, size=n)`` per resample."""
    counts = np.empty((n_boot, n))
    for i in range(n_boot):
        counts[i] = np.bincount(rng.randint(0, n, size=n), minlength=n)
    return counts
