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

from weatherstat.config import DATA_DIR, EFFECTOR_MAP, PREDICTION_SENSORS, UNIT_SYMBOL, abs_temp, delta_temp
from weatherstat.types import EffectorDecision, Scenario

# Minimum |t-statistic| for an effector→sensor gain to be used in simulation.
# Gains below this threshold are likely confounded (e.g., bedroom split runs
# when the house is warming for other reasons → OLS attributes warming to split).
_MIN_T_STATISTIC = 1.5

# Maximum plausible gain magnitude (per hour, in configured unit). Gains larger
# than this are almost certainly confounded or from a sensor near a vent/register.
# Typical real gains (°F/hr): thermostats 0.3-1.0, mini splits 1.0-1.5, blowers <0.5.
_MAX_GAIN_MAGNITUDE = delta_temp(3.0)

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
class StateGateInfo:
    """A state sensor gate for confirming effector delivery."""

    column: str
    encoding: dict[str, float]


@dataclass(frozen=True)
class SimParams:
    """Lookup structures for fast simulation from sysid output."""

    taus: dict[str, TauModel]  # sensor -> TauModel
    gains: dict[tuple[str, str], tuple[float, float]]  # (effector, sensor) -> (gain_f/hr, lag_min)
    solar: dict[tuple[str, int], float]  # (sensor, hour) -> gain_f/hr (legacy per-hour)
    sensors: list[str]  # sensor names with params
    effectors: list[dict]  # raw effector dicts (name, encoding, device_type)
    state_gates: dict[str, StateGateInfo] = field(default_factory=dict)  # gate_name -> info
    mrt_weights: dict[str, float] = field(default_factory=dict)  # sensor -> derived MRT weight
    solar_elevation_gains: dict[str, float] = field(default_factory=dict)  # sensor -> gain per sin(elev)×frac


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
    solar_fractions: list[float] = field(default_factory=list)
    # Per-hour solar fractions [now, h+1, h+2, ...] from weather conditions.
    # Index 0 = current hour, index 1 = next hour, etc.
    solar_elevations: list[float] = field(default_factory=list)
    # sin+(elevation) at each 5-min step [step1, step2, ...] for the prediction
    # horizon. Precomputed from lat/lon/time.

    def __post_init__(self) -> None:
        if not self.solar_fractions:
            raise ValueError(
                "HouseState requires solar_fractions (per-hour cloud attenuation). "
                "Pass an all-zeros list for nighttime or test scenarios."
            )
        if not self.solar_elevations:
            raise ValueError(
                "HouseState requires solar_elevations (sin⁺(elev) at 5-min steps). "
                "Pass an all-zeros list for nighttime or test scenarios."
            )


def load_sim_params(path: Path | None = None) -> SimParams:
    """Load sysid parameters from thermal_params.json."""
    p = path or DATA_DIR / "thermal_params.json"
    with open(p) as f:
        data = json.load(f)

    # Tau lookup: build TauModel from fitted_taus
    taus: dict[str, TauModel] = {}
    for ft in data["fitted_taus"]:
        sensor = ft["sensor"]
        taus[sensor] = TauModel(
            tau_base=ft["tau_base"],
            window_betas=ft.get("window_betas", {}),
            interaction_betas=ft.get("interaction_betas", {}),
        )

    # Gain lookup: filter by negligible flag, t-statistic significance,
    # physical plausibility (gain magnitude), and mode-direction consistency
    # (heating-only effectors can't have negative gains).
    gains: dict[tuple[str, str], tuple[float, float]] = {}
    n_pruned_t = 0
    n_pruned_mag = 0
    n_pruned_sign = 0
    for g in data["effector_sensor_gains"]:
        if g["negligible"]:
            continue
        if abs(g.get("t_statistic", 999)) < _MIN_T_STATISTIC:
            n_pruned_t += 1
            continue
        if abs(g["gain_f_per_hour"]) > _MAX_GAIN_MAGNITUDE:
            n_pruned_mag += 1
            continue
        # Mode-direction filter: a heating-only effector can't cool (negative gain),
        # and a cooling-only effector can't heat (positive gain). Confounded OLS
        # can produce these nonsensical cross-coupling effects.
        eff_cfg = EFFECTOR_MAP.get(g["effector"])
        if eff_cfg is not None:
            modes = set(eff_cfg.supported_modes)
            gain_val = g["gain_f_per_hour"]
            if "heat" in modes and "cool" not in modes and gain_val < 0:
                n_pruned_sign += 1
                continue
            if "cool" in modes and "heat" not in modes and gain_val > 0:
                n_pruned_sign += 1
                continue
        gains[(g["effector"], g["sensor"])] = (g["gain_f_per_hour"], g["best_lag_minutes"])
    pruned_parts: list[str] = []
    if n_pruned_t:
        pruned_parts.append(f"{n_pruned_t} by |t| < {_MIN_T_STATISTIC:.1f}")
    if n_pruned_mag:
        pruned_parts.append(f"{n_pruned_mag} by |gain| > {_MAX_GAIN_MAGNITUDE:.1f}{UNIT_SYMBOL}/hr")
    if n_pruned_sign:
        pruned_parts.append(f"{n_pruned_sign} by wrong sign for mode")
    if pruned_parts:
        print(f"  [sim] Pruned gains: {', '.join(pruned_parts)}")

    # Solar lookup: elevation-based gains (preferred) and legacy per-hour
    solar_elevation_gains = {str(k): float(v) for k, v in data.get("solar_elevation_gains", {}).items()}
    solar: dict[tuple[str, int], float] = {}
    for sg in data.get("solar_gains", []):
        solar[(sg["sensor"], sg["hour_of_day"])] = sg["gain_f_per_hour"]

    sensors = [s["name"] for s in data["sensors"]]

    # State gates (for confirming effector delivery from history)
    state_gates: dict[str, StateGateInfo] = {}
    for gate_name, gate_data in data.get("state_gates", {}).items():
        state_gates[gate_name] = StateGateInfo(
            column=gate_data["column"],
            encoding={str(k): float(v) for k, v in gate_data["encoding"].items()},
        )

    # MRT weights (derived from solar gain profiles by sysid)
    mrt_weights = {str(k): float(v) for k, v in data.get("mrt_weights", {}).items()}

    return SimParams(
        taus=taus,
        gains=gains,
        solar=solar,
        sensors=sensors,
        effectors=data["effectors"],
        state_gates=state_gates,
        mrt_weights=mrt_weights,
        solar_elevation_gains=solar_elevation_gains,
    )


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
    solar_fractions: list[float] | None = None,
    solar_elev_gain: float = 0.0,
    solar_elevations: list[float] | None = None,
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
        solar_profile: hour_of_day -> gain_f_per_hour for this sensor (legacy).
        start_hour: Fractional hour of day at t=0.
        n_steps: Number of 5-min steps to simulate.
        solar_fractions: Per-hour solar fractions [now, h+1, h+2, ...].
            Index 0 = current hour. None or empty → 1.0 (full sun).
        solar_elev_gain: Elevation-based solar gain coefficient for this sensor.
        solar_elevations: sin+(elevation) at each 5-min step. If provided with
            solar_elev_gain > 0, uses elevation model instead of per-hour profile.

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
        if solar_elev_gain != 0.0 and solar_elevations:
            # Elevation-based: gain × sin+(elevation) × condition_fraction
            elev_idx = min(step - 1, len(solar_elevations) - 1)
            sin_elev = solar_elevations[max(0, elev_idx)]
            sf = 1.0
            if solar_fractions:
                hour_idx = min(int(hours_from_start), len(solar_fractions) - 1)
                sf = solar_fractions[max(0, hour_idx)]
            dTdt += solar_elev_gain * sin_elev * sf
        else:
            # Legacy per-hour profile
            current_hour = int((start_hour + hours_from_start) % 24)
            solar_gain = solar_profile.get(current_hour, 0.0)
            if solar_gain != 0.0 and solar_fractions:
                hour_idx = min(int(hours_from_start), len(solar_fractions) - 1)
                solar_gain *= solar_fractions[max(0, hour_idx)]
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


@dataclass(frozen=True)
class _RegulatingEffector:
    """Info for a regulating (target-based) effector, used in Euler loop."""

    effector_name: str
    split_name: str
    proportional_band: float
    targets: np.ndarray  # (n_scenarios,) — target temp per scenario
    modes: np.ndarray  # (n_scenarios,) — +1 for heat, -1 for cool, 0 for off


def _build_activity_matrices(
    scenarios: list[Scenario],
    params: SimParams,
    recent_history: dict[str, list[float]],
    n_future: int,
) -> tuple[dict[str, np.ndarray], list[_RegulatingEffector]]:
    """Build per-effector activity matrices for all scenarios.

    Regulating effectors (mini splits with control_type="regulating") are excluded
    from activity matrices and returned separately for dynamic Euler-loop computation.

    Returns:
        (matrices, regulating_effectors) where matrices maps
        effector_name -> np.ndarray of shape (n_scenarios, _HISTORY_STEPS + n_future),
        and regulating_effectors contains info for target-based effectors.
    """
    n = len(scenarios)
    n_total = _HISTORY_STEPS + n_future
    steps = np.arange(n_future)  # [0, 1, ..., n_future-1]

    # Build trajectory active masks for all trajectory effectors
    trajectory_active: dict[str, np.ndarray] = {}
    for eff in params.effectors:
        if eff["device_type"] != "thermostat":
            continue
        name = eff["name"]
        decisions = [s.effectors.get(name, EffectorDecision(name)) for s in scenarios]
        heating = np.array([d.mode != "off" for d in decisions])
        delay = np.array([d.delay_steps for d in decisions])
        dur = np.array([
            d.duration_steps if d.duration_steps is not None else (n_future - d.delay_steps)
            for d in decisions
        ])
        trajectory_active[name] = (
            heating[:, None]
            & (steps[None, :] >= delay[:, None])
            & (steps[None, :] < (delay + dur)[:, None])
        )

    matrices: dict[str, np.ndarray] = {}
    regulating: list[_RegulatingEffector] = []

    for eff in params.effectors:
        name = eff["name"]
        encoding = eff.get("command_encoding") or eff["encoding"]
        dtype = eff["device_type"]

        # History: shared across all scenarios
        hist = recent_history.get(name, [])
        padded = [0.0] * max(0, _HISTORY_STEPS - len(hist)) + hist[-_HISTORY_STEPS:]
        hist_arr = np.array(padded)  # (_HISTORY_STEPS,)

        # Future: varies by device type
        if dtype == "thermostat":
            active = trajectory_active.get(name, np.zeros((n, n_future), dtype=bool))
            future = active.astype(np.float64) * encoding.get("heating", 1.0)

        elif dtype == "blower":
            eff_cfg = EFFECTOR_MAP.get(name)
            dep_names = eff_cfg.depends_on if eff_cfg else ()
            if dep_names:
                # AND gate: blower needs ALL parents active
                dep_active = np.ones((n, n_future), dtype=bool)
                for dname in dep_names:
                    dep_active &= trajectory_active.get(dname, np.zeros((n, n_future), dtype=bool))
            else:
                dep_active = np.ones((n, n_future), dtype=bool)
            enc_vals = np.array([
                encoding.get(s.effectors.get(name, EffectorDecision(name)).mode, 0.0)
                for s in scenarios
            ])
            future = dep_active.astype(np.float64) * enc_vals[:, None]

        elif dtype == "mini_split":
            eff_cfg = EFFECTOR_MAP.get(name)

            if eff_cfg and eff_cfg.control_type == "regulating":
                # Regulating: extract targets/modes per scenario for Euler loop
                decisions = [s.effectors.get(name, EffectorDecision(name)) for s in scenarios]
                targets = np.array([
                    d.target if d.mode != "off" and d.target is not None else 0.0
                    for d in decisions
                ])
                modes = np.array([
                    1.0 if d.mode == "heat" else (-1.0 if d.mode == "cool" else 0.0)
                    for d in decisions
                ])
                regulating.append(_RegulatingEffector(
                    effector_name=name,
                    split_name=name.removeprefix("mini_split_"),
                    proportional_band=eff_cfg.proportional_band,
                    targets=targets,
                    modes=modes,
                ))
                # Still need history in the matrix for lag lookback
                future = np.zeros((n, n_future))
            else:
                # Binary mini split (original behavior)
                enc_vals = np.array([
                    encoding.get(s.effectors.get(name, EffectorDecision(name)).mode, 0.0)
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

    return matrices, regulating


def predict(
    state: HouseState,
    scenarios: list[Scenario],
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

    # Build target names: sensor_t+horizon for each sensor in PREDICTION_SENSORS
    target_names: list[str] = []
    target_info: list[tuple[str, int]] = []
    for sensor_col in PREDICTION_SENSORS:
        if sensor_col not in params.sensors:
            continue
        for h in horizons:
            target_names.append(f"{sensor_col}_t+{h}")
            target_info.append((sensor_col, h))

    n_targets = len(target_names)
    result = np.empty((n_scenarios, n_targets))

    # Build activity matrices for all effectors: (n_scenarios, n_total) each
    activity_matrices, regulating_effectors = _build_activity_matrices(
        scenarios, params, state.recent_history, n_future,
    )

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

        # Solar forcing for this sensor
        elev_gain = params.solar_elevation_gains.get(sensor_col, 0.0)
        if elev_gain != 0.0 and state.solar_elevations:
            # Elevation-based: gain × sin+(elev) × condition_fraction per step
            elev_arr = np.array([
                state.solar_elevations[min(step - 1, len(state.solar_elevations) - 1)]
                for step in range(1, max_horizon + 1)
            ])
            if state.solar_fractions:
                sf = state.solar_fractions
                sf_vec = np.array([
                    sf[min(int(step * _DT_HOURS), len(sf) - 1)]
                    for step in range(1, max_horizon + 1)
                ])
                solar_vec = elev_gain * elev_arr * sf_vec
            else:
                solar_vec = elev_gain * elev_arr
        else:
            # Legacy per-hour solar profile
            solar: dict[int, float] = {}
            for (sens, hour), gain in params.solar.items():
                if sens == sensor_col:
                    solar[hour] = gain
            base_solar = np.array([
                solar.get(int((state.hour_of_day + step * _DT_HOURS) % 24), 0.0)
                for step in range(1, max_horizon + 1)
            ])
            if state.solar_fractions:
                sf = state.solar_fractions
                sf_vec = np.array([
                    sf[min(int(step * _DT_HOURS), len(sf) - 1)]
                    for step in range(1, max_horizon + 1)
                ])
                solar_vec = base_solar * sf_vec
            else:
                solar_vec = base_solar

        cur_temp = state.current_temps.get(sensor_col, abs_temp(70.0))

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

        # Collect regulating effector info for this sensor
        sensor_reg: list[tuple[float, int, float, np.ndarray, np.ndarray]] = []
        for reg in regulating_effectors:
            gain_info = sensor_gains.get(reg.effector_name)
            if gain_info is None:
                continue
            gain, lag_min = gain_info
            lag_steps = int(round(lag_min / 5.0))
            sensor_reg.append((abs(gain), lag_steps, reg.proportional_band, reg.targets, reg.modes))

        # Euler integration: loop over timesteps, vectorized over scenarios
        T = np.full(n_scenarios, cur_temp)
        horizon_results: dict[int, np.ndarray] = {}

        for step_idx in range(max_horizon):
            step = step_idx + 1
            dTdt = (outdoor_vec[step_idx] - T) / tau + total_eff[:, step] + solar_vec[step_idx]

            # Regulating effector contributions (proportional to target - current temp)
            for abs_gain, lag_s, p_band, targets, modes in sensor_reg:
                if step - lag_s < 1:
                    continue
                # Heating: activity = clip((target - T) / band, 0, 1)
                # Cooling: activity = clip((T - target) / band, 0, 1)
                heat_mask = modes > 0  # (n_scenarios,)
                cool_mask = modes < 0
                activity = np.zeros(n_scenarios)
                if heat_mask.any():
                    activity = np.where(
                        heat_mask,
                        np.clip((targets - T) / p_band, 0.0, 1.0),
                        activity,
                    )
                if cool_mask.any():
                    activity = np.where(
                        cool_mask,
                        np.clip((T - targets) / p_band, 0.0, 1.0),
                        activity,
                    )
                # Sign: heating adds heat (+gain), cooling removes heat (-gain)
                signed_activity = np.where(heat_mask, activity, np.where(cool_mask, -activity, 0.0))
                contribution = abs_gain * signed_activity
                # Mode-direction clamp: heating can't cool, cooling can't warm.
                # Gains fitted from heating data may be confounded; inverting
                # them for cooling can produce nonsense cross-coupling effects.
                contribution = np.where(heat_mask, np.maximum(contribution, 0.0), contribution)
                contribution = np.where(cool_mask, np.minimum(contribution, 0.0), contribution)
                dTdt += contribution

            T = T + _DT_HOURS * dTdt
            if step in horizon_set:
                horizon_results[step] = T.copy()

        # Fill result columns for this sensor
        for j, h in col_targets:
            result[:, j] = horizon_results.get(h, np.nan)

    return target_names, result



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
    # state sensor gate (e.g., boiler firing) to reflect actual delivery.
    for eff in params.effectors:
        gate_name = eff.get("state_gate")
        if gate_name and gate_name in params.state_gates:
            gate_info = params.state_gates[gate_name]
            gate_col = gate_info.column
            gate_enc = gate_info.encoding
            eff_name = eff["name"]
            eff_hist = history.get(eff_name, [])
            if gate_col in tail.columns and eff_hist:
                gate_vals = [gate_enc.get(str(v), 0.0) for v in tail[gate_col]]
                history[eff_name] = [a * g for a, g in zip(eff_hist, gate_vals, strict=True)]

    return history
