"""Grey-box forward simulator using sysid parameters.

Physics-based temperature prediction: Euler integration of Newton cooling
with effector gains, delays, and solar profiles from system identification.

Usage:
  from weatherstat.simulator import HouseState, load_sim_params, predict
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from weatherstat.config import DATA_DIR, PREDICTION_ROOMS
from weatherstat.types import BlowerDecision, MiniSplitDecision, TrajectoryScenario

# ── SimParams: loaded once from thermal_params.json ──────────────────────


@dataclass(frozen=True)
class TauModel:
    """Thermal envelope model with learned window effects.

    tau_base: sealed envelope time constant (hours, all windows closed).
    window_betas: per-window additional cooling rate coefficient.
        When window is open, effective 1/tau += beta.
    interaction_betas: cross-breeze coefficients for window pairs.
        When both windows in a pair are open, effective 1/tau += beta.
    """

    tau_base: float
    window_betas: dict[str, float] = field(default_factory=dict)
    interaction_betas: dict[str, float] = field(default_factory=dict)

    def effective_tau(self, window_states: dict[str, bool]) -> float:
        """Compute effective tau given current window states."""
        inv_tau = 1.0 / self.tau_base
        for win, beta in self.window_betas.items():
            if window_states.get(win, False):
                inv_tau += beta
        for key, beta in self.interaction_betas.items():
            w1, w2 = key.split("+")
            if window_states.get(w1, False) and window_states.get(w2, False):
                inv_tau += beta
        return 1.0 / max(inv_tau, 0.01)  # safety floor


@dataclass(frozen=True)
class SimParams:
    """Lookup structures for fast simulation from sysid output."""

    taus: dict[str, TauModel]  # sensor -> TauModel
    gains: dict[tuple[str, str], tuple[float, float]]  # (effector, sensor) -> (gain_f/hr, lag_min)
    solar: dict[tuple[str, int], float]  # (sensor, hour) -> gain_f/hr
    sensors: list[str]  # sensor names with params
    effectors: list[dict]  # raw effector dicts (name, encoding, device_type)


@dataclass(frozen=True)
class HouseState:
    """Current state of the house for prediction.

    Bundles everything about "where the house is now" into a single object.
    Scenarios describe "what HVAC actions to evaluate"; HouseState describes
    the starting point and environment.
    """

    current_temps: dict[str, float]  # sensor/room name -> temperature (F)
    outdoor_temp: float  # current outdoor temperature (F)
    forecast_temps: list[float]  # hourly outdoor temps [h+1, h+2, ..., h+N]
    window_states: dict[str, bool]  # window_name -> is_open
    hour_of_day: float  # fractional hour of day (0-23.99)
    recent_history: dict[str, list[float]] = field(default_factory=dict)
    # effector_name -> recent activity values (oldest first)


def load_sim_params(path: Path | None = None) -> SimParams:
    """Load sysid parameters from thermal_params.json."""
    p = path or DATA_DIR / "thermal_params.json"
    with open(p) as f:
        data = json.load(f)

    # Tau lookup: build TauModel from fitted_taus
    taus: dict[str, TauModel] = {}
    for ft in data["fitted_taus"]:
        sensor = ft["sensor"]
        if "tau_base" in ft:
            # New format: tau_base + window_betas + interaction_betas
            taus[sensor] = TauModel(
                tau_base=ft["tau_base"],
                window_betas=ft.get("window_betas", {}),
                interaction_betas=ft.get("interaction_betas", {}),
            )
        else:
            # Legacy format: tau_sealed / tau_ventilated
            tau_s = ft["tau_sealed"]
            taus[sensor] = TauModel(tau_base=tau_s)

    # Gain lookup (only non-negligible gains)
    gains: dict[tuple[str, str], tuple[float, float]] = {}
    for g in data["effector_sensor_gains"]:
        if g["negligible"]:
            continue
        gains[(g["effector"], g["sensor"])] = (g["gain_f_per_hour"], g["best_lag_minutes"])

    # Solar lookup
    solar: dict[tuple[str, int], float] = {}
    for sg in data["solar_gains"]:
        solar[(sg["sensor"], sg["hour_of_day"])] = sg["gain_f_per_hour"]

    sensors = [s["name"] for s in data["sensors"]]

    return SimParams(
        taus=taus,
        gains=gains,
        solar=solar,
        sensors=sensors,
        effectors=data["effectors"],
    )


# ── Helpers ──────────────────────────────────────────────────────────────


def _find_split(splits: tuple[MiniSplitDecision, ...], name: str) -> MiniSplitDecision | None:
    for sd in splits:
        if sd.name == name:
            return sd
    return None


def _find_blower(blowers: tuple[BlowerDecision, ...], name: str) -> BlowerDecision | None:
    for bd in blowers:
        if bd.name == name:
            return bd
    return None


# ── Activity timeline: single-scenario utility ───────────────────────────

_HISTORY_STEPS = 18  # 90 minutes of 5-min history


def build_activity_timeline(
    scenario_activity: float,
    recent_history: list[float],
    n_future_steps: int,
    switch_on_step: int = 0,
    switch_off_step: int | None = None,
) -> list[float]:
    """Build complete activity timeline: [...history, future...].

    Index 0 = oldest history step. Index len(history) = t=0 (start of scenario).
    Future steps use scenario_activity when active, 0.0 otherwise.

    Args:
        scenario_activity: Activity level when active.
        recent_history: Recent activity values (oldest first).
        n_future_steps: Number of future steps to generate.
        switch_on_step: Future step at which activity begins (0 = immediate).
        switch_off_step: Future step at which activity ends (None = never).
    """
    padded = [0.0] * max(0, _HISTORY_STEPS - len(recent_history)) + recent_history[-_HISTORY_STEPS:]
    if switch_on_step == 0 and switch_off_step is None:
        # Fast path: constant activity over entire future
        return padded + [scenario_activity] * n_future_steps
    # Segment-based construction: [OFF × delay] + [ON × duration] + [OFF × remainder]
    start = min(switch_on_step, n_future_steps)
    end = min(switch_off_step if switch_off_step is not None else n_future_steps, n_future_steps)
    on_len = max(0, end - start)
    off_after = max(0, n_future_steps - end)
    return padded + [0.0] * start + [scenario_activity] * on_len + [0.0] * off_after


# ── Core simulation: single sensor, single scenario ─────────────────────

_DT_HOURS = 5.0 / 60.0  # 5-minute timestep in hours


def simulate_sensor(
    sensor: str,
    current_temp: float,
    outdoor_temp: float,
    forecast_temps: list[float],
    tau: float,
    effector_timelines: dict[str, list[float]],
    gains: dict[str, tuple[float, float]],
    solar_profile: dict[int, float],
    start_hour: float,
    n_steps: int,
) -> list[float]:
    """Euler-integrate temperature for one sensor under one scenario.

    Args:
        sensor: Sensor name (for logging only).
        current_temp: Temperature at t=0 (F).
        outdoor_temp: Current outdoor temp (F). Used if no forecast.
        forecast_temps: Hourly outdoor temps [h+1, h+2, ..., h+N].
        tau: Effective envelope time constant (hours), pre-computed from
            TauModel.effective_tau(window_states).
        effector_timelines: effector_name -> full timeline (history + future).
            Timeline index `history_len` corresponds to t=0.
        gains: effector_name -> (gain_f_per_hour, lag_minutes) for this sensor.
        solar_profile: hour_of_day -> gain_f_per_hour for this sensor.
        start_hour: Fractional hour of day at t=0.
        n_steps: Number of 5-min steps to simulate.

    Returns:
        Temperature at each step [t+1, t+2, ..., t+n_steps].
    """
    if tau <= 0:
        tau = 40.0  # safety fallback

    history_len = _HISTORY_STEPS
    temps: list[float] = []
    t = current_temp

    for step in range(1, n_steps + 1):
        # Hours from t=0 for this step
        hours_from_start = step * _DT_HOURS

        # Outdoor temp: piecewise hourly from forecast
        t_out = _outdoor_at(outdoor_temp, forecast_temps, hours_from_start)

        # Envelope loss
        dTdt = (t_out - t) / tau

        # Effector contributions
        for eff_name, (gain, lag_min) in gains.items():
            timeline = effector_timelines.get(eff_name)
            if timeline is None:
                continue
            # Look back by lag: timeline index for the activity that affects this step
            lag_steps = int(round(lag_min / 5.0))
            # Current step in timeline coordinates = history_len + step
            tl_idx = history_len + step - lag_steps
            if 0 <= tl_idx < len(timeline):
                activity = timeline[tl_idx]
            elif tl_idx < 0:
                activity = 0.0  # before any known history
            else:
                activity = timeline[-1]  # clamp to last known
            dTdt += gain * activity

        # Solar gain
        current_hour = int((start_hour + hours_from_start) % 24)
        solar_gain = solar_profile.get(current_hour, 0.0)
        dTdt += solar_gain

        # Euler step
        t = t + _DT_HOURS * dTdt
        temps.append(t)

    return temps


def _outdoor_at(current: float, forecast: list[float], hours_ahead: float) -> float:
    """Get outdoor temp at hours_ahead using piecewise hourly forecast.

    forecast[0] = temp at h+1, forecast[1] = temp at h+2, etc.
    For hours_ahead < 1, interpolate between current and forecast[0].
    """
    if not forecast or hours_ahead <= 0:
        return current
    if hours_ahead < 1.0:
        # Linear interpolation: current -> forecast[0]
        return current + hours_ahead * (forecast[0] - current)
    # Which hour segment are we in?
    idx = int(hours_ahead) - 1  # forecast[0] is h+1
    if idx >= len(forecast):
        return forecast[-1]
    return forecast[idx]


# ── Vectorized batch prediction ──────────────────────────────────────────


def _build_activity_matrices(
    scenarios: list[TrajectoryScenario],
    params: SimParams,
    recent_history: dict[str, list[float]],
    n_future: int,
) -> dict[str, np.ndarray]:
    """Build per-effector activity matrices for all scenarios.

    Uses numpy broadcasting to construct (n_scenarios, n_total_steps) matrices
    for each effector, replacing the per-scenario Python loop.

    Returns:
        effector_name -> np.ndarray of shape (n_scenarios, _HISTORY_STEPS + n_future).
    """
    from weatherstat.config import BLOWERS as _BLOWERS

    n = len(scenarios)
    n_total = _HISTORY_STEPS + n_future
    steps = np.arange(n_future)  # [0, 1, ..., n_future-1]

    # Pre-extract thermostat parameters as numpy arrays
    up_heating = np.array([s.upstairs.heating for s in scenarios])
    up_delay = np.array([s.upstairs.delay_steps for s in scenarios])
    up_dur = np.array([
        s.upstairs.duration_steps if s.upstairs.duration_steps is not None
        else (n_future - s.upstairs.delay_steps)
        for s in scenarios
    ])
    dn_heating = np.array([s.downstairs.heating for s in scenarios])
    dn_delay = np.array([s.downstairs.delay_steps for s in scenarios])
    dn_dur = np.array([
        s.downstairs.duration_steps if s.downstairs.duration_steps is not None
        else (n_future - s.downstairs.delay_steps)
        for s in scenarios
    ])

    # Thermostat active masks: (n, n_future) boolean
    # Broadcasting: (n, 1) op (1, n_future) -> (n, n_future)
    up_active = (
        up_heating[:, None]
        & (steps[None, :] >= up_delay[:, None])
        & (steps[None, :] < (up_delay + up_dur)[:, None])
    )
    dn_active = (
        dn_heating[:, None]
        & (steps[None, :] >= dn_delay[:, None])
        & (steps[None, :] < (dn_delay + dn_dur)[:, None])
    )

    blower_zones = {b.name: b.zone for b in _BLOWERS}
    matrices: dict[str, np.ndarray] = {}

    for eff in params.effectors:
        name = eff["name"]
        # For future scenarios, use command encoding (what we intend to set).
        # For history, encoding (measured state) was already applied above.
        encoding = eff.get("command_encoding") or eff["encoding"]
        dtype = eff["device_type"]

        # History: shared across all scenarios
        hist = recent_history.get(name, [])
        padded = [0.0] * max(0, _HISTORY_STEPS - len(hist)) + hist[-_HISTORY_STEPS:]
        hist_arr = np.array(padded)  # (_HISTORY_STEPS,)

        # Future: varies by device type
        if dtype == "thermostat":
            zone = name.removeprefix("thermostat_")
            active = up_active if zone == "upstairs" else dn_active
            future = active.astype(np.float64) * encoding.get("heating", 1.0)

        elif dtype == "boiler":
            boiler_on = encoding.get("Space Heating", 1.0)
            future = (up_active | dn_active).astype(np.float64) * boiler_on

        elif dtype == "blower":
            blower_name = name.removeprefix("blower_")
            enc_vals = np.array([
                encoding.get(bd.mode, 0.0) if (bd := _find_blower(s.blowers, blower_name)) else 0.0
                for s in scenarios
            ])
            zone = blower_zones.get(blower_name, "downstairs")
            zone_active = up_active if zone == "upstairs" else dn_active
            future = zone_active.astype(np.float64) * enc_vals[:, None]

        elif dtype == "mini_split":
            split_name = name.removeprefix("mini_split_")
            enc_vals = np.array([
                encoding.get(sd.mode, 0.0) if (sd := _find_split(s.mini_splits, split_name)) else 0.0
                for s in scenarios
            ])
            future = np.broadcast_to(enc_vals[:, None], (n, n_future))

        else:
            future = np.zeros((n, n_future))

        # Combine history + future into (n, n_total)
        matrix = np.empty((n, n_total))
        matrix[:, :_HISTORY_STEPS] = hist_arr  # broadcasts across rows
        matrix[:, _HISTORY_STEPS:] = future
        matrices[name] = matrix

    return matrices


def predict(
    state: HouseState,
    scenarios: list[TrajectoryScenario],
    params: SimParams,
    horizons: list[int],
) -> tuple[list[str], np.ndarray]:
    """Predict room temperatures for multiple HVAC scenarios.

    Vectorized Euler integration: loops over timesteps (72) but processes
    all scenarios simultaneously via numpy. Activity matrices are built
    once using numpy broadcasting, and effector forcing is pre-computed
    per sensor as a single slice operation per effector.

    Args:
        state: Current house state (temps, weather, windows, effector history).
        scenarios: HVAC trajectory scenarios to evaluate.
        params: Thermal model parameters from sysid.
        horizons: Prediction horizons in 5-min steps (e.g., [12, 24, 48, 72]).

    Returns:
        (target_names, predictions) where predictions shape is (n_scenarios, n_targets).
        target_names like "bedroom_temp_t+12".
    """
    max_horizon = max(horizons)
    n_scenarios = len(scenarios)
    n_future = max_horizon + 1
    n_total = _HISTORY_STEPS + n_future
    horizon_set = set(horizons)

    # Build target names: room_temp_t+horizon for each room in PREDICTION_ROOMS
    room_to_sensor = _room_to_sensor_map(params.sensors)
    target_names: list[str] = []
    target_info: list[tuple[str, int]] = []
    for room in PREDICTION_ROOMS:
        sensor_col = room_to_sensor.get(room)
        if sensor_col is None:
            continue
        for h in horizons:
            target_names.append(f"{room}_temp_t+{h}")
            target_info.append((sensor_col, h))

    n_targets = len(target_names)
    result = np.empty((n_scenarios, n_targets))

    # Build activity matrices for all effectors: (n_scenarios, n_total) each
    activity_matrices = _build_activity_matrices(scenarios, params, state.recent_history, n_future)

    # Group targets by sensor for efficient per-sensor simulation
    sensor_targets: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for j, (sc, h) in enumerate(target_info):
        sensor_targets[sc].append((j, h))

    # Pre-compute outdoor temperature and solar gain vectors (shared across sensors)
    outdoor_vec = np.array([
        _outdoor_at(state.outdoor_temp, state.forecast_temps, step * _DT_HOURS)
        for step in range(1, max_horizon + 1)
    ])

    # Simulate each sensor (vectorized across all scenarios)
    for sensor_col, col_targets in sensor_targets.items():
        tau_model = params.taus.get(sensor_col, TauModel(tau_base=40.0))
        tau = tau_model.effective_tau(state.window_states)
        if tau <= 0:
            tau = 40.0

        # Gains for this sensor
        sensor_gains: dict[str, tuple[float, float]] = {}
        for (eff, sens), (gain, lag) in params.gains.items():
            if sens == sensor_col:
                sensor_gains[eff] = (gain, lag)

        # Solar profile for this sensor
        solar: dict[int, float] = {}
        for (sens, hour), gain in params.solar.items():
            if sens == sensor_col:
                solar[hour] = gain

        solar_vec = np.array([
            solar.get(int((state.hour_of_day + step * _DT_HOURS) % 24), 0.0)
            for step in range(1, max_horizon + 1)
        ])

        # Current temp (try sensor col name, then room name)
        cur_temp = state.current_temps.get(sensor_col)
        if cur_temp is None:
            for room, sc in room_to_sensor.items():
                if sc == sensor_col:
                    cur_temp = state.current_temps.get(room)
                    break
        if cur_temp is None:
            cur_temp = 70.0

        # Pre-compute total effector forcing: (n_scenarios, max_horizon + 1)
        # Vectorized over both scenarios AND timesteps — one slice op per effector
        total_eff = np.zeros((n_scenarios, max_horizon + 1))
        for eff_name, (gain, lag_min) in sensor_gains.items():
            lag_s = int(round(lag_min / 5.0))
            mat = activity_matrices[eff_name]
            # Source: activity matrix columns for steps 1..max_horizon, shifted by lag
            src_start = _HISTORY_STEPS + 1 - lag_s
            src_end = _HISTORY_STEPS + max_horizon + 1 - lag_s
            # Destination: total_eff columns 1..max_horizon
            dst_start = 1
            dst_end = max_horizon + 1
            # Handle boundary clipping
            if src_start < 0:
                dst_start += -src_start
                src_start = 0
            if src_end > n_total:
                dst_end -= src_end - n_total
                src_end = n_total
            if dst_start < dst_end:
                total_eff[:, dst_start:dst_end] += gain * mat[:, src_start:src_end]

        # Euler integration: loop over timesteps, vectorized over scenarios
        T = np.full(n_scenarios, cur_temp)
        horizon_results: dict[int, np.ndarray] = {}

        for step_idx in range(max_horizon):
            step = step_idx + 1
            dTdt = (outdoor_vec[step_idx] - T) / tau + total_eff[:, step] + solar_vec[step_idx]
            T = T + _DT_HOURS * dTdt
            if step in horizon_set:
                horizon_results[step] = T.copy()

        # Fill result columns for this sensor
        for j, h in col_targets:
            result[:, j] = horizon_results.get(h, np.nan)

    return target_names, result


# ── Room name mapping ────────────────────────────────────────────────────


def _room_to_sensor_map(sensor_names: list[str]) -> dict[str, str]:
    """Map PREDICTION_ROOMS names to sensor column names from sysid.

    PREDICTION_ROOMS uses names like "bedroom", "upstairs".
    Sensor columns are like "bedroom_temp", "thermostat_upstairs_temp".
    """
    mapping: dict[str, str] = {}
    for room in PREDICTION_ROOMS:
        # Direct: "bedroom" -> "bedroom_temp"
        candidate = f"{room}_temp"
        if candidate in sensor_names:
            mapping[room] = candidate
            continue
        # Thermostat zone: "upstairs" -> "thermostat_upstairs_temp"
        candidate = f"thermostat_{room}_temp"
        if candidate in sensor_names:
            mapping[room] = candidate
            continue
        # Aggregate: "upstairs" -> "upstairs_aggregate_temp"
        candidate = f"{room}_aggregate_temp"
        if candidate in sensor_names:
            mapping[room] = candidate
    return mapping


# ── Recent history extraction from snapshot data ─────────────────────────


def extract_recent_history(df_raw: dict | object, params: SimParams) -> dict[str, list[float]]:
    """Extract recent effector activity from raw snapshot DataFrame.

    Returns effector_name -> list of activity values (oldest first),
    up to _HISTORY_STEPS entries.
    """
    import pandas as pd

    if not isinstance(df_raw, pd.DataFrame):
        return {}

    history: dict[str, list[float]] = {}
    # Use last _HISTORY_STEPS rows
    tail = df_raw.tail(_HISTORY_STEPS)

    for eff in params.effectors:
        col = eff["state_column"]
        encoding = eff["encoding"]
        cmd_col = eff.get("command_column")
        cmd_enc = eff.get("command_encoding") or encoding
        if col in tail.columns:
            vals = [encoding.get(str(v), cmd_enc.get(str(v), 0.0)) for v in tail[col]]
            history[eff["name"]] = vals
        elif cmd_col and cmd_col in tail.columns:
            history[eff["name"]] = [cmd_enc.get(str(v), 0.0) for v in tail[cmd_col]]
        else:
            history[eff["name"]] = [0.0] * len(tail)

    # Apply state confirmation — matches sysid preprocessing.
    # Intent-only signals (e.g., thermostat calling) are confirmed by a
    # paired device (e.g., boiler responding) to reflect actual delivery.
    for eff in params.effectors:
        state_eff = eff.get("state_effector")
        if state_eff:
            state_hist = history.get(state_eff, [])
            eff_name = eff["name"]
            eff_hist = history.get(eff_name, [])
            if state_hist and eff_hist:
                history[eff_name] = [a * s for a, s in zip(eff_hist, state_hist, strict=True)]

    return history
