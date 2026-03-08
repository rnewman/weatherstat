"""Grey-box forward simulator using sysid parameters.

Physics-based temperature prediction: Euler integration of Newton cooling
with effector gains, delays, and solar profiles from system identification.

Drop-in replacement for ML batch_predict in the control sweep:
same (target_names, np.ndarray) output format.

Usage:
  from weatherstat.simulator import load_sim_params, batch_simulate
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from weatherstat.config import DATA_DIR, PREDICTION_ROOMS
from weatherstat.types import BlowerDecision, HVACScenario, MiniSplitDecision

# ── SimParams: loaded once from thermal_params.json ──────────────────────


@dataclass(frozen=True)
class SimParams:
    """Lookup structures for fast simulation from sysid output."""

    taus: dict[str, tuple[float, float]]  # sensor -> (tau_sealed, tau_vent)
    gains: dict[tuple[str, str], tuple[float, float]]  # (effector, sensor) -> (gain_f/hr, lag_min)
    solar: dict[tuple[str, int], float]  # (sensor, hour) -> gain_f/hr
    sensors: list[str]  # sensor names with params
    effectors: list[dict]  # raw effector dicts (name, encoding, device_type)
    sensor_window_cols: dict[str, list[str]]  # sensor -> window column names


def load_sim_params(path: Path | None = None) -> SimParams:
    """Load sysid parameters from thermal_params.json."""
    p = path or DATA_DIR / "thermal_params.json"
    with open(p) as f:
        data = json.load(f)

    # Tau lookup
    taus: dict[str, tuple[float, float]] = {}
    for ft in data["fitted_taus"]:
        sensor = ft["sensor"]
        tau_s = ft["tau_sealed"]
        tau_v = ft["tau_ventilated"] if ft["tau_ventilated"] is not None else tau_s * 0.44
        taus[sensor] = (tau_s, tau_v)

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
    sensor_window_cols = {s["name"]: s["window_columns"] for s in data["sensors"]}

    return SimParams(
        taus=taus,
        gains=gains,
        solar=solar,
        sensors=sensors,
        effectors=data["effectors"],
        sensor_window_cols=sensor_window_cols,
    )


# ── Scenario → effector activity conversion ──────────────────────────────


def _scenario_to_activities(scenario: HVACScenario, params: SimParams) -> dict[str, float]:
    """Convert an HVACScenario to per-effector numeric activity levels."""
    activities: dict[str, float] = {}

    for eff in params.effectors:
        name = eff["name"]
        encoding = eff["encoding"]
        dtype = eff["device_type"]

        if dtype == "thermostat":
            # thermostat_upstairs / thermostat_downstairs
            zone = name.removeprefix("thermostat_")
            if zone == "upstairs":
                state = "heating" if scenario.upstairs_heating else "idle"
            elif zone == "downstairs":
                state = "heating" if scenario.downstairs_heating else "idle"
            else:
                state = "idle"
            activities[name] = encoding.get(state, 0.0)

        elif dtype == "boiler":
            # Navien fires when either thermostat is on
            if scenario.upstairs_heating or scenario.downstairs_heating:
                activities[name] = encoding.get("Space Heating", 1.0)
            else:
                activities[name] = encoding.get("Idle", 0.0)

        elif dtype == "mini_split":
            # mini_split_bedroom / mini_split_living_room
            split_name = name.removeprefix("mini_split_")
            sd = _find_split(scenario.mini_splits, split_name)
            mode = sd.mode if sd else "off"
            activities[name] = encoding.get(mode, 0.0)

        elif dtype == "blower":
            # blower_family_room / blower_office
            blower_name = name.removeprefix("blower_")
            bd = _find_blower(scenario.blowers, blower_name)
            mode = bd.mode if bd else "off"
            activities[name] = encoding.get(mode, 0.0)

        else:
            activities[name] = 0.0

    return activities


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


# ── Activity timeline: merge recent history + scenario ───────────────────

_HISTORY_STEPS = 18  # 90 minutes of 5-min history


def build_activity_timeline(
    scenario_activity: float,
    recent_history: list[float],
    n_future_steps: int,
) -> list[float]:
    """Build complete activity timeline: [...history, scenario, scenario, ...].

    Index 0 = oldest history step. Index len(history) = t=0 (start of scenario).
    Steps [len(history)..] use scenario_activity.

    If recent_history is shorter than _HISTORY_STEPS, pad with zeros at the front.
    """
    # Pad history to _HISTORY_STEPS
    padded = [0.0] * max(0, _HISTORY_STEPS - len(recent_history)) + recent_history[-_HISTORY_STEPS:]
    # Append future steps
    return padded + [scenario_activity] * n_future_steps


# ── Core simulation: single sensor, single scenario ─────────────────────

_DT_HOURS = 5.0 / 60.0  # 5-minute timestep in hours


def simulate_sensor(
    sensor: str,
    current_temp: float,
    outdoor_temp: float,
    forecast_temps: list[float],
    tau_sealed: float,
    tau_vent: float,
    is_ventilated: bool,
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
        tau_sealed: Envelope time constant, sealed (hours).
        tau_vent: Envelope time constant, ventilated (hours).
        is_ventilated: Whether windows are open for this sensor.
        effector_timelines: effector_name -> full timeline (history + future).
            Timeline index `history_len` corresponds to t=0.
        gains: effector_name -> (gain_f_per_hour, lag_minutes) for this sensor.
        solar_profile: hour_of_day -> gain_f_per_hour for this sensor.
        start_hour: Fractional hour of day at t=0.
        n_steps: Number of 5-min steps to simulate.

    Returns:
        Temperature at each step [t+1, t+2, ..., t+n_steps].
    """
    tau = tau_vent if is_ventilated else tau_sealed
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


# ── Batch simulate: all sensors, all scenarios ──────────────────────────


def batch_simulate(
    current_temps: dict[str, float],
    outdoor_temp: float,
    forecast_temps: list[float],
    window_states: dict[str, bool],
    scenarios: list[HVACScenario],
    params: SimParams,
    hour_of_day: float,
    horizons: list[int],
    recent_history: dict[str, list[float]] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Simulate all scenarios and return predictions in the same format as _batch_predict.

    Args:
        current_temps: sensor_name -> current temperature (F). Keys should use
            the sensor column names from sysid (e.g., "bedroom_temp").
        outdoor_temp: Current outdoor temperature (F).
        forecast_temps: Hourly forecast [h+1, h+2, ..., h+N].
        window_states: window_name -> is_open (e.g., "bedroom" -> True).
        scenarios: List of HVACScenario to evaluate.
        params: Loaded SimParams from thermal_params.json.
        hour_of_day: Current fractional hour of day (0-23.99).
        horizons: Prediction horizons in 5-min steps (e.g., [12, 24, 48, 72]).
        recent_history: effector_name -> list of recent activity values (oldest first).
            If None, assumes all effectors were off.

    Returns:
        (target_names, predictions) where predictions shape is (n_scenarios, n_targets).
        target_names like "bedroom_temp_t+12".
    """
    if recent_history is None:
        recent_history = {}

    max_horizon = max(horizons)
    n_scenarios = len(scenarios)

    # Build target names: room_temp_t+horizon for each room in PREDICTION_ROOMS
    # Map room names to sensor column names
    room_to_sensor = _room_to_sensor_map(params.sensors)

    target_names: list[str] = []
    target_info: list[tuple[str, int]] = []  # (sensor_col, horizon_step)
    for room in PREDICTION_ROOMS:
        sensor_col = room_to_sensor.get(room)
        if sensor_col is None:
            continue
        for h in horizons:
            target_names.append(f"{room}_temp_t+{h}")
            target_info.append((sensor_col, h))

    n_targets = len(target_names)
    result = np.empty((n_scenarios, n_targets))

    # Pre-compute per-sensor static info
    sensor_info: dict[str, dict] = {}
    for sensor_col in set(sc for sc, _ in target_info):
        tau_s, tau_v = params.taus.get(sensor_col, (40.0, 17.6))
        # Window state for this sensor
        win_cols = params.sensor_window_cols.get(sensor_col, [])
        is_vent = any(
            window_states.get(wc.removeprefix("window_").removesuffix("_open"), False)
            for wc in win_cols
        )
        # Gains for this sensor
        sensor_gains: dict[str, tuple[float, float]] = {}
        for (eff, sens), (gain, lag) in params.gains.items():
            if sens == sensor_col:
                sensor_gains[eff] = (gain, lag)
        # Solar profile
        solar: dict[int, float] = {}
        for (sens, hour), gain in params.solar.items():
            if sens == sensor_col:
                solar[hour] = gain

        sensor_info[sensor_col] = {
            "tau_sealed": tau_s,
            "tau_vent": tau_v,
            "is_vent": is_vent,
            "gains": sensor_gains,
            "solar": solar,
        }

    # Simulate each scenario
    for i, scenario in enumerate(scenarios):
        activities = _scenario_to_activities(scenario, params)

        # Build timelines for each effector
        timelines: dict[str, list[float]] = {}
        for eff in params.effectors:
            eff_name = eff["name"]
            hist = recent_history.get(eff_name, [])
            timelines[eff_name] = build_activity_timeline(
                activities.get(eff_name, 0.0),
                hist,
                max_horizon + 1,
            )

        # Simulate each needed sensor
        sensor_trajectories: dict[str, list[float]] = {}
        for sensor_col in sensor_info:
            if sensor_col in sensor_trajectories:
                continue
            info = sensor_info[sensor_col]
            cur_temp = current_temps.get(sensor_col)
            if cur_temp is None:
                # Try room name mapping
                for room, sc in room_to_sensor.items():
                    if sc == sensor_col:
                        cur_temp = current_temps.get(room)
                        break
            if cur_temp is None:
                cur_temp = 70.0  # fallback

            traj = simulate_sensor(
                sensor=sensor_col,
                current_temp=cur_temp,
                outdoor_temp=outdoor_temp,
                forecast_temps=forecast_temps,
                tau_sealed=info["tau_sealed"],
                tau_vent=info["tau_vent"],
                is_ventilated=info["is_vent"],
                effector_timelines=timelines,
                gains=info["gains"],
                solar_profile=info["solar"],
                start_hour=hour_of_day,
                n_steps=max_horizon,
            )
            sensor_trajectories[sensor_col] = traj

        # Extract horizon values
        for j, (sensor_col, h_step) in enumerate(target_info):
            traj = sensor_trajectories.get(sensor_col)
            if traj is not None and h_step - 1 < len(traj):
                result[i, j] = traj[h_step - 1]  # traj[0] = t+1, traj[11] = t+12
            else:
                result[i, j] = np.nan

    return target_names, result


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
        if col in tail.columns:
            vals = [encoding.get(str(v), 0.0) for v in tail[col]]
            history[eff["name"]] = vals
        else:
            history[eff["name"]] = [0.0] * len(tail)

    return history
