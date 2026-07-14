"""Multi-timescale residual trackers.

Every sysid coefficient is conceptually a state with a timescale; the batch
fit is the timescale-∞ special case. A *tracked block* is a group of
design-matrix features (selected by name prefix) whose coefficients are
re-estimated on a trailing window against the batch fit's context terms,
regularized toward their previous values (random-walk prior):

    minimize ‖r − F·γ‖² + λ·‖γ − γ_prev‖²

where r is the window's Newton residual minus the batch model's non-block
contributions, F is the block's columns of the same design matrix the batch
fit used (built by sysid.build_design_matrix — no skew), and λ scales with
the window's information so γ moves quickly under strong signal and holds
under weak signal.

A block is only safe to track fast if its carrier is sharp and exogenous.
Solar is the first (and currently only) block: its carrier is zero at night,
plane-specific in phase, astronomically scheduled, and independent of
controller actions. The tracked γ absorbs slow directional drift — deciduous
canopy, pruning, new structures — without modeling any of them explicitly.
Tau (broadband carrier) and effector gains (closed-loop-confounded carrier)
must NOT be tracked without new identification evidence; see
docs/plans/PLAN-35-directional-solar.md.

State persists in ~/.weatherstat/tracker_state.json; every update appends to
tracker_history.jsonl, so the γ time series is itself a measurement (e.g.,
per-room canopy phenology).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from weatherstat.config import DATA_DIR, delta_temp

TRACKER_STATE_FILE = DATA_DIR / "tracker_state.json"
TRACKER_HISTORY_FILE = DATA_DIR / "tracker_history.jsonl"


@dataclass(frozen=True)
class TrackedBlock:
    """A group of design-matrix features whose coefficients drift.

    prefixes: feature-name prefixes owned by the block. Everything else in
        the design matrix is context, predicted with batch coefficients.
    window_days: trailing window length.
    prior_ratio: λ = prior_ratio × (block signal energy per column). 1.0
        weighs the previous estimate equal to a full-signal window, giving
        ~half-life-one-update movement under strong signal.
    min_information: minimum per-column signal energy (tr(FᵀF)/k) to update
        at all — below this (e.g., a dark Seattle fortnight for solar), hold.
    lower/upper: coefficient bounds (physical sign + magnitude cap).
    """

    name: str
    prefixes: tuple[str, ...]
    window_days: float = 14.0
    prior_ratio: float = 1.0
    min_information: float = 25.0
    lower: float = 0.0
    upper: float = delta_temp(5.0)


# Solar transmission: the first tracked block. Owns _solar_elev, _solar_east,
# _solar_west (NOT _adv_solar_* shade interactions — those are batch-fit).
SOLAR_BLOCK = TrackedBlock(name="solar", prefixes=("_solar_",))

TRACKED_BLOCKS: tuple[TrackedBlock, ...] = (SOLAR_BLOCK,)


@dataclass(frozen=True)
class BlockUpdate:
    """Result of one tracked-block update for one sensor."""

    coefficients: dict[str, float]
    information: float
    n_rows: int
    updated: bool  # False = held previous estimate (information gate)


def solve_tracked_block(
    F: np.ndarray,
    r: np.ndarray,
    gamma_prev: np.ndarray,
    block: TrackedBlock,
) -> tuple[np.ndarray, float, bool]:
    """Ridge with random-walk prior; returns (gamma, information, updated).

    Pure math — unit-testable without data plumbing. information is the mean
    per-column signal energy tr(FᵀF)/k. When it is below the block's gate,
    gamma_prev is returned unchanged.
    """
    k = F.shape[1]
    FtF = F.T @ F
    information = float(np.trace(FtF)) / max(k, 1)
    if information < block.min_information:
        return gamma_prev, information, False

    lam = block.prior_ratio * information
    A = FtF + lam * np.eye(k)
    b = F.T @ r + lam * gamma_prev
    gamma = np.linalg.solve(A, b)
    gamma = np.clip(gamma, block.lower, block.upper)
    return gamma, information, True


def _load_state() -> dict:
    if TRACKER_STATE_FILE.exists():
        try:
            return json.loads(TRACKER_STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_state(state: dict) -> None:
    TRACKER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRACKER_STATE_FILE.write_text(json.dumps(state, indent=2))


def _append_history(record: dict) -> None:
    with open(TRACKER_HISTORY_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def update_block_for_sensor(
    block: TrackedBlock,
    df_window,  # preprocessed DataFrame (sysid._preprocess output)
    sensor,  # SensorSpec
    effectors,  # list[EffectorSpec]
    model: dict,  # serialized batch fit: {tau_base, feature_names, coefficients}
    prev_coefficients: dict[str, float] | None,
) -> BlockUpdate | None:
    """One tracked-block update for one sensor on a preprocessed window.

    Returns None if the design matrix can't be built or the block has no
    features in it. Context features are predicted with batch coefficients
    matched by name; features absent from the batch fit contribute zero.
    """
    from weatherstat.sysid import build_design_matrix, newton_residual

    dm = build_design_matrix(df_window, sensor, effectors)
    if dm is None:
        return None
    y = newton_residual(df_window, sensor, model["tau_base"])
    if y is None:
        return None

    block_idx = [j for j, n in enumerate(dm.feature_names) if n.startswith(block.prefixes)]
    if not block_idx:
        return None
    block_names = [dm.feature_names[j] for j in block_idx]
    ctx_idx = [j for j in range(len(dm.feature_names)) if j not in set(block_idx)]

    batch_coef = dict(zip(model["feature_names"], model["coefficients"], strict=True))
    ctx_beta = np.array([batch_coef.get(dm.feature_names[j], 0.0) for j in ctx_idx])

    valid = ~np.isnan(y)
    for j in range(dm.X.shape[1]):
        valid &= ~np.isnan(dm.X[:, j])
    if int(valid.sum()) < 10 * len(block_idx):
        return BlockUpdate(prev_coefficients or {}, 0.0, int(valid.sum()), False)

    X = dm.X[valid]
    r = y[valid] - X[:, ctx_idx] @ ctx_beta
    F = X[:, block_idx]

    # Prior: previous tracked estimate; cold start from the batch fit itself
    gamma_prev = np.array([
        (prev_coefficients or {}).get(n, batch_coef.get(n, 0.0)) for n in block_names
    ])

    gamma, information, updated = solve_tracked_block(F, r, gamma_prev, block)
    coefficients = {n: round(float(g), 4) for n, g in zip(block_names, gamma, strict=True)}
    return BlockUpdate(coefficients, round(information, 2), int(valid.sum()), updated)


def update_trackers(log=print, params_path: Path | None = None, now: datetime | None = None) -> str:
    """Update all tracked blocks from recent collector data.

    Called after each quality-gated sysid refit (hourly). Loads the trailing
    window (longest window across blocks), preprocesses once, and updates
    every block × sensor. Returns a one-line summary.
    """
    import pandas as pd

    from weatherstat.extract import load_collector_snapshots
    from weatherstat.sysid import _enumerate_effectors, _enumerate_sensors, _preprocess

    p = params_path or DATA_DIR / "thermal_params.json"
    if not p.exists():
        return "tracker: no thermal_params.json — skipped"
    params = json.loads(p.read_text())
    models = params.get("regression_models", {})
    if not models:
        return "tracker: batch fit has no regression_models (pre-tracker params) — skipped"

    max_window = max(b.window_days for b in TRACKED_BLOCKS)
    cutoff = pd.Timestamp(now or datetime.now(UTC)) - pd.Timedelta(days=max_window)

    df = load_collector_snapshots()
    ts = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df = df[ts >= cutoff].reset_index(drop=True)
    if len(df) < 500:
        return f"tracker: only {len(df)} window rows — skipped"

    effectors = _enumerate_effectors()
    sensors = _enumerate_sensors()
    df = _preprocess(df, effectors, sensors)

    state = _load_state()
    blocks_state = state.setdefault("blocks", {})
    now_iso = (now or datetime.now(UTC)).isoformat()
    n_updated = 0
    n_held = 0

    for block in TRACKED_BLOCKS:
        block_state = blocks_state.setdefault(block.name, {}).setdefault("sensors", {})
        for sensor in sensors:
            model = models.get(sensor.name)
            if model is None:
                continue
            prev = block_state.get(sensor.name, {}).get("coefficients")
            upd = update_block_for_sensor(block, df, sensor, effectors, model, prev)
            if upd is None:
                continue
            if upd.updated:
                n_updated += 1
            else:
                n_held += 1
                if not upd.coefficients:
                    continue  # nothing previous to hold, nothing new — skip
            block_state[sensor.name] = {
                "coefficients": upd.coefficients,
                "updated_at": now_iso if upd.updated else block_state.get(sensor.name, {}).get("updated_at", now_iso),
                "information": upd.information,
                "n_rows": upd.n_rows,
            }
            _append_history({
                "ts": now_iso,
                "block": block.name,
                "sensor": sensor.name,
                "coefficients": upd.coefficients,
                "information": upd.information,
                "updated": upd.updated,
            })

    _save_state(state)
    msg = f"tracker: {n_updated} sensor-block(s) updated, {n_held} held (low information)"
    log(f"[tracker] {msg}")
    return msg


def load_tracked_solar_gains(max_age_hours: float = 48.0) -> dict[str, dict[str, float]]:
    """Effective solar coefficients from tracker state, for the simulator.

    Returns {sensor: {feature_name: γ}} for entries updated within
    max_age_hours. Empty dict = no usable tracker state (simulator falls
    back to batch gains).
    """
    state = _load_state()
    sensors = state.get("blocks", {}).get("solar", {}).get("sensors", {})
    out: dict[str, dict[str, float]] = {}
    now = datetime.now(UTC)
    for sensor, rec in sensors.items():
        try:
            updated_at = datetime.fromisoformat(rec["updated_at"])
        except (KeyError, ValueError):
            continue
        if (now - updated_at).total_seconds() > max_age_hours * 3600:
            continue
        coefs = rec.get("coefficients", {})
        if coefs:
            out[sensor] = {str(k): float(v) for k, v in coefs.items()}
    return out


def main() -> None:
    print(update_trackers())


if __name__ == "__main__":
    main()
