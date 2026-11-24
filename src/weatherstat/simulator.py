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
from weatherstat.regulation import RegulationModel, build_models, responses_from_json
from weatherstat.types import EffectorDecision, Scenario

# Minimum |t-statistic| for an effector→sensor gain to be used in simulation.
# Gains below this threshold are likely confounded (e.g., bedroom split runs
# when the house is warming for other reasons → OLS attributes warming to split).
_MIN_T_STATISTIC = 1.5

# Maximum plausible gain magnitude (per hour, in configured unit). Gains larger
# than this are almost certainly confounded or from a sensor near a vent/register.
# Typical real gains (°F/hr): thermostats 0.3-1.0, mini splits 1.0-1.5, blowers 0.1-0.5.
_MAX_GAIN_MAGNITUDE = delta_temp(2.0)

# Maximum plausible solar gain magnitude (per hour at full sin⁺(elev)×fraction).
# Observed real elevation gains are 0.3-1.3°F/hr; sensors with sparse tau data
# have produced ±17°F/hr noise fits (docs/debugging-notes.md § "Ridge
# Regularization Effects"). Applies to both elevation gains and legacy per-hour
# solar profiles.
_MAX_SOLAR_GAIN_MAGNITUDE = delta_temp(5.0)

# A tau fitted from too few overnight segments that also lands implausibly slow
# is a regression artifact (sensors with <5 segments hit the 500h fit cap and
# lose all cooling behavior — docs/debugging-notes.md § "Prediction Sanity
# Failures"). Such sensors fall back to the YAML default tau, and their solar
# gains are dropped (they were fitted against the bad tau's Newton residuals).
_MIN_TAU_SEGMENTS = 5
_SUSPECT_TAU_HOURS = 250.0

# ── SimParams: loaded once from thermal_params.json ──────────────────────


@dataclass(frozen=True)
class TauModel:
    """Thermal envelope model with learned advisory effector effects.

    tau_base: sealed envelope time constant (hours, all advisories at default).
    environment_tau_betas: per-advisory-effector additional cooling rate coefficient.
        When device is active, effective 1/tau += beta.
    """

    tau_base: float
    environment_tau_betas: dict[str, float] = field(default_factory=dict)

    def effective_tau(self, environment_states: dict[str, bool]) -> float:
        """Compute effective tau given current environment factor states."""
        inv_tau = 1.0 / self.tau_base
        for dev, beta in self.environment_tau_betas.items():
            if environment_states.get(dev, False):
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
    solar_plane_gains: dict[str, dict[str, float]] = field(default_factory=dict)  # sensor -> {plane -> gain}
    correlational_couplings: dict[str, frozenset[tuple[str, str]]] = field(default_factory=dict)
    # family ("tau"/"solar") -> {(device, sensor)}: couplings tagged correlational
    # by transition-weighted identification. Used for observed-state prediction
    # but excluded from counterfactual advisory toggles (see PLAN-36).
    environment_solar_betas: dict[str, dict[str, float]] = field(default_factory=dict)  # device -> {sensor -> beta}
    regulation: dict[str, RegulationModel] = field(default_factory=dict)
    # regulating effector -> what it regulates on (virtual input weights) and how it
    # responds (learned curve). See regulation.py / PLAN-41.
    environment_support: dict[str, float] | None = None
    # environment entry -> minutes it has been held active in history (sysid
    # environment_support); None when thermal_params predates it (no limit).


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
    environment_states: dict[str, bool]  # environment factor name -> is_active (non-default)
    hour_of_day: float  # fractional hour of day (0-23.99)
    recent_history: dict[str, list[float]] = field(default_factory=dict)
    # effector_name -> recent activity values (oldest first)
    solar_fractions: list[float] = field(default_factory=list)
    # Per-hour solar fractions [now, h+1, h+2, ...] from weather conditions.
    # Index 0 = current hour, index 1 = next hour, etc.
    solar_elevations: list[float] = field(default_factory=list)
    # sin+(elevation) at each 5-min step [step1, step2, ...] for the prediction
    # horizon. Precomputed from lat/lon/time.
    solar_planes: dict[str, list[float]] = field(default_factory=dict)
    # Per-step wall-plane incidence factors keyed by plane name ("east",
    # "west"), same indexing as solar_elevations. Optional: empty dict means
    # no directional solar forcing (backward compatible).
    unit_states: dict[str, tuple[str, bool]] = field(default_factory=dict)
    # Regulating effector -> (hvac mode it is in, whether it is running) right now.
    # A plan that keeps the unit in that mode continues from it; any other start
    # is a mode change (regulation.py § Hysteresis). Absent: every start is one.

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

    # Tau lookup: build TauModel from fitted_taus. Sensors whose tau came from
    # too few segments AND landed implausibly slow fall back to the YAML
    # default tau; their solar gains are dropped below.
    yaml_tau_lookup = {s["name"]: s.get("yaml_tau_base") for s in data["sensors"]}
    unreliable_tau_sensors: set[str] = set()
    taus: dict[str, TauModel] = {}
    for ft in data["fitted_taus"]:
        sensor = ft["sensor"]
        # Backward compat: old files use advisory_tau_betas or window_betas
        tau_betas = ft.get("environment_tau_betas", ft.get("advisory_tau_betas", ft.get("window_betas", {})))
        tau_base = ft["tau_base"]
        n_segments = ft.get("n_segments")
        yaml_tau = yaml_tau_lookup.get(sensor)
        # n_segments == 0 already means "yaml default" upstream; None means an
        # old params file that predates the field — trust it rather than guess.
        if (
            n_segments is not None
            and 0 < n_segments < _MIN_TAU_SEGMENTS
            and tau_base > _SUSPECT_TAU_HOURS
            and yaml_tau
        ):
            print(
                f"  [sim] {sensor}: tau_base={tau_base:.0f}h from only {n_segments} "
                f"segment(s) — using YAML default {yaml_tau:.0f}h, dropping solar gain"
            )
            tau_base = yaml_tau
            unreliable_tau_sensors.add(sensor)
        taus[sensor] = TauModel(
            tau_base=tau_base,
            environment_tau_betas=tau_betas,
        )

    # Gain lookup: filter by negligible flag, t-statistic significance,
    # and mode-direction consistency (heating-only effectors can't have
    # negative gains). Gains exceeding _MAX_GAIN_MAGNITUDE are clamped
    # rather than pruned — a large gain may be real but exaggerated due
    # to sparse data or multicollinearity.
    gains: dict[tuple[str, str], tuple[float, float]] = {}
    n_pruned_t = 0
    n_clamped_mag = 0
    n_pruned_sign = 0
    for g in data["effector_sensor_gains"]:
        if g["negligible"]:
            continue
        if abs(g.get("t_statistic", 999)) < _MIN_T_STATISTIC:
            n_pruned_t += 1
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
        # Clamp magnitude to plausible range (preserve direction)
        gain_val = g["gain_f_per_hour"]
        if abs(gain_val) > _MAX_GAIN_MAGNITUDE:
            gain_val = _MAX_GAIN_MAGNITUDE if gain_val > 0 else -_MAX_GAIN_MAGNITUDE
            n_clamped_mag += 1
        gains[(g["effector"], g["sensor"])] = (gain_val, g["best_lag_minutes"])

    # Solar lookup: elevation-based gains (preferred) and legacy per-hour.
    # Solar gains get the same magnitude clamp as effector gains (at a higher
    # ceiling), and are dropped entirely for sensors whose tau fit was rejected
    # above — they were regressed against that tau's Newton residuals.
    n_solar_clamped = 0
    n_solar_dropped = 0

    def _filter_solar(sensor: str, gain: float) -> float | None:
        nonlocal n_solar_clamped, n_solar_dropped
        if sensor in unreliable_tau_sensors:
            n_solar_dropped += 1
            return None
        if abs(gain) > _MAX_SOLAR_GAIN_MAGNITUDE:
            n_solar_clamped += 1
            return _MAX_SOLAR_GAIN_MAGNITUDE if gain > 0 else -_MAX_SOLAR_GAIN_MAGNITUDE
        return gain

    solar_elevation_gains: dict[str, float] = {}
    for k, v in data.get("solar_elevation_gains", {}).items():
        filtered = _filter_solar(str(k), float(v))
        if filtered is not None:
            solar_elevation_gains[str(k)] = filtered
    solar: dict[tuple[str, int], float] = {}
    for sg in data.get("solar_gains", []):
        filtered = _filter_solar(sg["sensor"], sg["gain_f_per_hour"])
        if filtered is not None:
            solar[(sg["sensor"], sg["hour_of_day"])] = filtered

    # Directional plane gains (batch fit). Planes are additive heat inputs:
    # negative coefficients are confounded, drop them; clamp like other solar.
    solar_plane_gains: dict[str, dict[str, float]] = {}
    for sensor_name, planes in data.get("solar_plane_gains", {}).items():
        if sensor_name in unreliable_tau_sensors:
            continue
        kept = {}
        for plane_name, g in planes.items():
            g = float(g)
            if g <= 0.0:
                continue
            kept[str(plane_name)] = min(g, _MAX_SOLAR_GAIN_MAGNITUDE)
        if kept:
            solar_plane_gains[str(sensor_name)] = kept

    # Tracked solar coefficients (multi-timescale residual tracker) override
    # the batch values when fresh — the tracker follows slow directional
    # drift (canopy, structures) that a full-history batch fit averages over.
    from weatherstat.tracker import load_tracked_solar_gains

    tracked = load_tracked_solar_gains()
    n_tracked = 0
    for sensor_name, coefs in tracked.items():
        if sensor_name in unreliable_tau_sensors:
            continue
        elev = coefs.get("_solar_elev")
        if elev is not None and sensor_name in solar_elevation_gains:
            solar_elevation_gains[sensor_name] = min(max(float(elev), 0.0), _MAX_SOLAR_GAIN_MAGNITUDE)
        planes = {
            name.removeprefix("_solar_"): min(max(float(v), 0.0), _MAX_SOLAR_GAIN_MAGNITUDE)
            for name, v in coefs.items()
            if name.startswith("_solar_") and name != "_solar_elev"
        }
        planes = {k: v for k, v in planes.items() if v > 0.0}
        if planes:
            solar_plane_gains[sensor_name] = planes
        n_tracked += 1
    if n_tracked:
        print(f"  [sim] Tracked solar gains active for {n_tracked} sensor(s)")

    filter_parts: list[str] = []
    if n_pruned_t:
        filter_parts.append(f"{n_pruned_t} pruned by |t| < {_MIN_T_STATISTIC:.1f}")
    if n_clamped_mag:
        filter_parts.append(f"{n_clamped_mag} clamped to ±{_MAX_GAIN_MAGNITUDE:.1f}{UNIT_SYMBOL}/hr")
    if n_pruned_sign:
        filter_parts.append(f"{n_pruned_sign} pruned by wrong sign for mode")
    if n_solar_clamped:
        filter_parts.append(f"{n_solar_clamped} solar clamped to ±{_MAX_SOLAR_GAIN_MAGNITUDE:.1f}{UNIT_SYMBOL}/hr")
    if n_solar_dropped:
        filter_parts.append(f"{n_solar_dropped} solar dropped (unreliable tau)")
    if filter_parts:
        print(f"  [sim] Gain filters: {', '.join(filter_parts)}")

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

    # Correlational couplings: identified by transition-weighted refit at fit
    # time. These stay active for observed-state prediction but must not be
    # credited when the sweep counterfactually toggles the device.
    correlational_couplings: dict[str, frozenset[tuple[str, str]]] = {}
    for family, devs in data.get("coupling_identification", {}).items():
        pairs = {
            (dev, sensor)
            for dev, sensors_ in devs.items()
            for sensor, record in sensors_.items()
            if record.get("tag") == "correlational"
        }
        if pairs:
            correlational_couplings[family] = frozenset(pairs)
    n_correlational = sum(len(v) for v in correlational_couplings.values())
    if n_correlational:
        print(f"  [sim] {n_correlational} correlational coupling(s): prediction-only, excluded from advisory counterfactuals")

    # Environment solar betas (device modulates solar gain for sensor)
    raw_solar_betas = data.get("environment_solar_betas", data.get("advisory_solar_betas", {}))
    environment_solar_betas: dict[str, dict[str, float]] = {}
    for dev, sensor_betas in raw_solar_betas.items():
        environment_solar_betas[dev] = {str(k): float(v) for k, v in sensor_betas.items()}

    # Regulating effectors: what each regulates on (adopted virtual-input weights,
    # else computed from these filtered gains) and its learned response curve.
    regulation = build_models(gains, responses_from_json(data.get("regulation_responses", {})), sensors)

    return SimParams(
        taus=taus,
        gains=gains,
        solar=solar,
        sensors=sensors,
        effectors=data["effectors"],
        state_gates=state_gates,
        mrt_weights=mrt_weights,
        solar_elevation_gains=solar_elevation_gains,
        solar_plane_gains=solar_plane_gains,
        environment_solar_betas=environment_solar_betas,
        correlational_couplings=correlational_couplings,
        regulation=regulation,
        environment_support=(
            {str(k): float(v) for k, v in data["environment_support"].items()}
            if "environment_support" in data else None
        ),
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
            TauModel.effective_tau(environment_states).
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
    proportional_band: float
    targets: np.ndarray  # (n_scenarios,) — target temp per scenario
    modes: np.ndarray  # (n_scenarios,) — +1 for heat, -1 for cool, 0 for off
    delay_steps: np.ndarray  # (n_scenarios,)
    duration_steps: np.ndarray  # (n_scenarios,)


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

    # Pre-extract all effector decisions once (avoid repeated list comps over 100K+ scenarios)
    eff_names = {eff["name"] for eff in params.effectors}
    _off_cache: dict[str, EffectorDecision] = {}
    decisions_by_eff: dict[str, list[EffectorDecision]] = {}
    for name in eff_names:
        _off_cache[name] = EffectorDecision(name)
        off = _off_cache[name]
        decisions_by_eff[name] = [s.effectors.get(name, off) for s in scenarios]

    # Build trajectory active masks for all trajectory effectors
    trajectory_active: dict[str, np.ndarray] = {}
    for eff in params.effectors:
        if eff["device_type"] != "thermostat":
            continue
        name = eff["name"]
        decisions = decisions_by_eff[name]
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
        decisions = decisions_by_eff[name]

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
                dep_active = np.ones((n, n_future), dtype=bool)
                for dname in dep_names:
                    dep_active &= trajectory_active.get(dname, np.zeros((n, n_future), dtype=bool))
            else:
                dep_active = np.ones((n, n_future), dtype=bool)
            enc_vals = np.array([encoding.get(d.mode, 0.0) for d in decisions])
            delay_arr = np.array([d.delay_steps for d in decisions])
            dur_arr = np.array([
                d.duration_steps if d.duration_steps is not None else (n_future - d.delay_steps)
                for d in decisions
            ])
            time_active = (
                (steps[None, :] >= delay_arr[:, None])
                & (steps[None, :] < (delay_arr + dur_arr)[:, None])
            )
            future = dep_active.astype(np.float64) * enc_vals[:, None] * time_active.astype(np.float64)

        elif dtype == "mini_split":
            eff_cfg = EFFECTOR_MAP.get(name)

            if eff_cfg and eff_cfg.control_type == "regulating":
                targets = np.array([
                    d.target if d.mode != "off" and d.target is not None else 0.0
                    for d in decisions
                ])
                modes = np.array([
                    1.0 if d.mode == "heat" else (-1.0 if d.mode == "cool" else 0.0)
                    for d in decisions
                ])
                delays = np.array([d.delay_steps for d in decisions])
                durations = np.array([
                    d.duration_steps if d.duration_steps is not None else (n_future - d.delay_steps)
                    for d in decisions
                ])
                regulating.append(_RegulatingEffector(
                    effector_name=name,
                    proportional_band=eff_cfg.proportional_band,
                    targets=targets,
                    modes=modes,
                    delay_steps=delays,
                    duration_steps=durations,
                ))
                future = np.zeros((n, n_future))
            else:
                enc_vals = np.array([encoding.get(d.mode, 0.0) for d in decisions])
                ms_delays = np.array([d.delay_steps for d in decisions])
                ms_durs = np.array([
                    d.duration_steps if d.duration_steps is not None else (n_future - d.delay_steps)
                    for d in decisions
                ])
                ms_active = (
                    (steps[None, :] >= ms_delays[:, None])
                    & (steps[None, :] < (ms_delays + ms_durs)[:, None])
                )
                future = enc_vals[:, None] * ms_active.astype(np.float64)

        else:
            future = np.zeros((n, n_future))

        # Combine history + future into (n, n_total)
        matrix = np.empty((n, n_total))
        matrix[:, :_HISTORY_STEPS] = hist_arr  # broadcasts across rows
        matrix[:, _HISTORY_STEPS:] = future
        matrices[name] = matrix

    return matrices, regulating


def _build_environment_timelines(
    scenarios: list[Scenario],
    environment_states: dict[str, bool],
    n_future: int,
) -> dict[str, np.ndarray]:
    """Build per-environment-factor state timelines for all scenarios.

    Returns dict mapping device name -> ndarray of shape
    (n_scenarios, _HISTORY_STEPS + n_future) with float values
    (0.0 = default/inactive, 1.0 = active).

    History prefix is constant at the device's current state, enabling
    lag-based gain lookback.  Future portion reflects AdvisoryDecision
    transitions: hold keeps current state; close/open flips at
    transition_step.
    """
    all_devices: set[str] = set()
    for s in scenarios:
        all_devices.update(s.advisories.keys())
    if not all_devices:
        return {}

    n = len(scenarios)
    n_total = _HISTORY_STEPS + n_future

    timelines: dict[str, np.ndarray] = {}
    for device in sorted(all_devices):
        current = 1.0 if environment_states.get(device, False) else 0.0
        matrix = np.full((n, n_total), current)

        for i, scenario in enumerate(scenarios):
            adv = scenario.advisories.get(device)
            if adv is None or adv.action == "hold":
                continue
            if adv.action in ("close", "turn_off", "lower"):
                target = 0.0
            elif adv.action in ("open", "turn_on", "raise"):
                target = 1.0
            else:
                continue
            transition_idx = _HISTORY_STEPS + adv.transition_step
            if transition_idx < n_total:
                matrix[i, transition_idx:] = target
            # return_step: device reverts to original state (proactive advice)
            if adv.return_step is not None:
                return_idx = _HISTORY_STEPS + adv.return_step
                if return_idx < n_total:
                    matrix[i, return_idx:] = current

        timelines[device] = matrix

    return timelines


# Scenarios per simulation chunk: per-sensor arrays are (n, horizon); chunking keeps
# the regulating joint pass (all member sensors held at once) bounded in memory.
_PREDICT_CHUNK = 8192


@dataclass
class _SensorPrep:
    """Per-sensor inputs for the Euler loop, vectorized over scenarios."""

    cur_temp: float
    tau_scalar: float
    tau_matrix: np.ndarray | None  # (n, H) when advisory timelines modulate tau
    solar_vec: np.ndarray  # (H,)
    solar_arr: np.ndarray | None  # (n, H) when advisory devices modulate solar
    total_eff: np.ndarray  # (n, H + 1): non-regulating effector forcing (+ regulating history)
    reg_gains: list[tuple[str, float, int]]  # (regulating effector, positive gain, lag steps)

    def base_dTdt(self, T: np.ndarray, step_idx: int, outdoor: float) -> np.ndarray:
        tau = self.tau_matrix[:, step_idx] if self.tau_matrix is not None else self.tau_scalar
        solar = self.solar_arr[:, step_idx] if self.solar_arr is not None else self.solar_vec[step_idx]
        return (outdoor - T) / tau + self.total_eff[:, step_idx + 1] + solar


def predict(
    state: HouseState,
    scenarios: list[Scenario],
    params: SimParams,
    horizons: list[int],
    activity_out: dict[str, np.ndarray] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Predict room temperatures for multiple HVAC scenarios.

    Vectorized Euler integration: loops over timesteps (72) but processes
    all scenarios simultaneously via numpy. Activity matrices are built
    once using numpy broadcasting, and effector forcing is pre-computed
    per sensor as a single slice operation per effector.

    Regulating effectors (mini splits) are delegated: each runs its own
    thermostat on its virtual input V = Σ wₛ·Tₛ (regulation.py), so its activity
    is one trace per scenario — computed in a joint pass over V's member
    sensors — that then reaches every sensor through its learned gain.

    Args:
        state: Current house state (temps, weather, windows, effector history).
        scenarios: HVAC trajectory scenarios to evaluate.
        params: Thermal model parameters from sysid.
        horizons: Prediction horizons in 5-min steps (e.g., [12, 24, 48, 72]).
        activity_out: If given, filled with regulating effector name ->
            (n_scenarios, max_horizon) signed activity (+heat, −cool).

    Returns:
        (target_names, predictions) where predictions shape is (n_scenarios, n_targets).
        target_names like "bedroom_temp_t+12".
    """
    if len(scenarios) <= _PREDICT_CHUNK:
        return _predict_chunk(state, scenarios, params, horizons, activity_out)
    names: list[str] = []
    parts: list[np.ndarray] = []
    act_parts: dict[str, list[np.ndarray]] = defaultdict(list)
    for start in range(0, len(scenarios), _PREDICT_CHUNK):
        chunk_act: dict[str, np.ndarray] | None = {} if activity_out is not None else None
        names, part = _predict_chunk(state, scenarios[start:start + _PREDICT_CHUNK], params, horizons, chunk_act)
        parts.append(part)
        for k, v in (chunk_act or {}).items():
            act_parts[k].append(v)
    if activity_out is not None:
        for k, v in act_parts.items():
            activity_out[k] = np.concatenate(v, axis=0)
    return names, np.concatenate(parts, axis=0)


def _predict_chunk(
    state: HouseState,
    scenarios: list[Scenario],
    params: SimParams,
    horizons: list[int],
    activity_out: dict[str, np.ndarray] | None,
) -> tuple[list[str], np.ndarray]:
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
    regulating_names = {reg.effector_name for reg in regulating_effectors}

    # Build advisory timelines and merge into activity matrices for gain lookup.
    # Advisory timelines encode per-step advisory effector state (0/1) for each
    # scenario, enabling mid-horizon state transitions (e.g., window closes at step 12).
    has_advisories = any(s.advisories for s in scenarios)
    advisory_future: dict[str, np.ndarray] = {}
    if has_advisories:
        advisory_timelines = _build_environment_timelines(scenarios, state.environment_states, n_future)
        # Merge so advisory gain effects are picked up by the total_eff computation
        activity_matrices.update(advisory_timelines)
        # Pre-extract future portion (steps 1..max_horizon) for tau/solar computation
        advisory_future = {
            dev: tl[:, _HISTORY_STEPS + 1:_HISTORY_STEPS + 1 + max_horizon]
            for dev, tl in advisory_timelines.items()
        }

    # Group targets by sensor for efficient per-sensor simulation
    sensor_targets: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for j, (sc, h) in enumerate(target_info):
        sensor_targets[sc].append((j, h))

    # Pre-compute outdoor temperature and solar gain vectors (shared across sensors)
    outdoor_vec = np.array([
        _outdoor_at(state.outdoor_temp, state.forecast_temps, step * _DT_HOURS)
        for step in range(1, max_horizon + 1)
    ])

    def _prep(sensor_col: str) -> _SensorPrep:
        tau_model = params.taus.get(sensor_col, TauModel(tau_base=40.0))

        # Per-step tau when advisory timelines modulate this sensor's envelope.
        # Fast path: scalar tau when no advisory transitions affect tau.
        per_step_tau = bool(
            advisory_future and tau_model.environment_tau_betas
        )
        tau_scalar = 40.0
        tau_matrix: np.ndarray | None = None
        if per_step_tau:
            inv_tau = np.full((n_scenarios, max_horizon), 1.0 / tau_model.tau_base)
            corr_tau = params.correlational_couplings.get("tau", frozenset())
            for dev, beta in tau_model.environment_tau_betas.items():
                af = advisory_future.get(dev)
                if af is not None and (dev, sensor_col) not in corr_tau:
                    inv_tau += beta * af
                elif state.environment_states.get(dev, False):
                    # Passive device — or a correlational coupling, which is
                    # pinned to the observed state: toggling it in a scenario
                    # earns no credit (identification says the effect is
                    # habit correlation, not causation).
                    inv_tau += beta
            tau_matrix = 1.0 / np.maximum(inv_tau, 0.01)
        else:
            tau_scalar = tau_model.effective_tau(state.environment_states)
            if tau_scalar <= 0:
                tau_scalar = 40.0

        # Gains for this sensor
        sensor_gains: dict[str, tuple[float, float]] = {}
        for (eff, sens), (gain, lag) in params.gains.items():
            if sens == sensor_col:
                sensor_gains[eff] = (gain, lag)

        # Solar forcing for this sensor. Cloud fraction is shared by all
        # solar channels (horizontal + wall planes).
        if state.solar_fractions:
            sf = state.solar_fractions
            sf_vec = np.array([
                sf[min(int(step * _DT_HOURS), len(sf) - 1)]
                for step in range(1, max_horizon + 1)
            ])
        else:
            sf_vec = np.ones(max_horizon)

        elev_gain = params.solar_elevation_gains.get(sensor_col, 0.0)
        if elev_gain != 0.0 and state.solar_elevations:
            # Elevation-based: gain × sin+(elev) × condition_fraction per step
            elev_arr = np.array([
                state.solar_elevations[min(step - 1, len(state.solar_elevations) - 1)]
                for step in range(1, max_horizon + 1)
            ])
            solar_vec = elev_gain * elev_arr * sf_vec
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
            solar_vec = base_solar * sf_vec

        # Directional wall-plane forcing (batch or tracked coefficients).
        # Additive alongside the horizontal channel; scenario-independent, so
        # superposition is unaffected.
        plane_gains = params.solar_plane_gains.get(sensor_col, {})
        if plane_gains and state.solar_planes:
            for plane_name, plane_gain in plane_gains.items():
                factors = state.solar_planes.get(plane_name)
                if not factors or plane_gain == 0.0:
                    continue
                plane_arr = np.array([
                    factors[min(step - 1, len(factors) - 1)]
                    for step in range(1, max_horizon + 1)
                ])
                solar_vec = solar_vec + plane_gain * plane_arr * sf_vec

        # Advisory solar modulation: advisory effectors can amplify/attenuate
        # solar gain (e.g., blinds reduce solar, open window changes airflow).
        # Multiplicative: solar_vec *= (1 + Σ(beta × timeline)).
        solar_arr: np.ndarray | None = None
        if advisory_future and params.environment_solar_betas:
            solar_mod = np.zeros((n_scenarios, max_horizon))
            has_solar_mod = False
            corr_solar = params.correlational_couplings.get("solar", frozenset())
            for dev, sensor_betas in params.environment_solar_betas.items():
                beta = sensor_betas.get(sensor_col, 0.0)
                if beta == 0.0:
                    continue
                af = advisory_future.get(dev)
                if af is not None and (dev, sensor_col) not in corr_solar:
                    solar_mod += beta * af
                    has_solar_mod = True
                elif state.environment_states.get(dev, False):
                    # Correlational couplings pinned to observed state (see tau)
                    solar_mod += beta
                    has_solar_mod = True
            if has_solar_mod:
                solar_arr = solar_vec[None, :] * (1.0 + solar_mod)

        cur_temp = state.current_temps.get(sensor_col, abs_temp(70.0))

        # Pre-compute total effector forcing: (n_scenarios, max_horizon + 1)
        # Vectorized over both scenarios AND timesteps — one slice op per effector.
        # A regulating effector's matrix holds only its recorded history (its future
        # comes from the joint pass below), and only its warming side counts:
        # heating can't cool a sensor and cooling can't warm one, so a negative
        # cross gain contributes nothing (it is confounding, not physics).
        total_eff = np.zeros((n_scenarios, max_horizon + 1))
        reg_gains: list[tuple[str, float, int]] = []
        for eff_name, (gain, lag_min) in sensor_gains.items():
            lag_s = int(round(lag_min / 5.0))
            if eff_name in regulating_names:
                gain = max(gain, 0.0)
                if gain == 0.0:
                    continue
                reg_gains.append((eff_name, gain, lag_s))
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

        return _SensorPrep(
            cur_temp=cur_temp, tau_scalar=tau_scalar, tau_matrix=tau_matrix,
            solar_vec=solar_vec, solar_arr=solar_arr, total_eff=total_eff, reg_gains=reg_gains,
        )

    preps = {sensor_col: _prep(sensor_col) for sensor_col in sensor_targets}

    # ── Regulating effectors: one activity trace per unit (joint pass) ──
    activity = _regulating_activity(
        regulating_effectors, params, preps, state, outdoor_vec, n_scenarios, max_horizon,
    )
    if activity_out is not None:
        for reg in regulating_effectors:
            act = activity.get(reg.effector_name)
            activity_out[reg.effector_name] = act[:, 1:] if act is not None else np.zeros((n_scenarios, max_horizon))
    for prep in preps.values():
        for eff_name, gain, lag_s in prep.reg_gains:
            act = activity.get(eff_name)
            if act is None or lag_s >= max_horizon:
                continue
            # Step k feels the unit's activity at step k − lag (none before step 1).
            prep.total_eff[:, 1 + lag_s:] += gain * act[:, 1:max_horizon + 1 - lag_s]

    # ── Euler integration per sensor, vectorized over scenarios ──
    for sensor_col, col_targets in sensor_targets.items():
        prep = preps[sensor_col]
        T = np.full(n_scenarios, prep.cur_temp)
        horizon_results: dict[int, np.ndarray] = {}
        for step_idx in range(max_horizon):
            T = T + _DT_HOURS * prep.base_dTdt(T, step_idx, outdoor_vec[step_idx])
            step = step_idx + 1
            if step in horizon_set:
                horizon_results[step] = T.copy()
        for j, h in col_targets:
            result[:, j] = horizon_results.get(h, np.nan)

    return target_names, result


def _regulating_activity(
    regulating_effectors: list[_RegulatingEffector],
    params: SimParams,
    preps: dict[str, _SensorPrep],
    state: HouseState,
    outdoor_vec: np.ndarray,
    n_scenarios: int,
    max_horizon: int,
) -> dict[str, np.ndarray]:
    """Signed activity (n, max_horizon + 1; column 0 unused) per regulating effector.

    Each unit runs on its virtual input V = Σ wₛ·Tₛ over member sensors that
    have a reading (renormalized, as the published value is). V's members
    are integrated jointly, because a unit's activity depends on V and V on
    every active unit's activity. Activity at step k is decided from the
    temperatures at the start of step k from err = sign·(target − V), gated
    by the plan's [delay, delay + duration) window.

    With fitted transitions the level is the unit's running probability a,
    a two-state chain: a·stay(err) + (1 − a)·start(err) while the plan keeps
    the unit in one mode, enter(err) on the step it switches mode — from
    ``state.unit_states`` on step 1, from off after a delay. Without them it
    is the mode's single curve, response(err).
    """
    from weatherstat.regulation import MIN_COVERAGE

    units: list[tuple[_RegulatingEffector, list[str], np.ndarray, RegulationModel]] = []
    for reg in regulating_effectors:
        model = params.regulation.get(reg.effector_name)
        if model is None or not np.any(reg.modes != 0):
            continue
        members = [s for s in model.weights if s in preps and s in state.current_temps]
        total = sum(model.weights.values())
        present = sum(model.weights[s] for s in members)
        if not members or total <= 0 or present / total < MIN_COVERAGE:
            continue  # no usable input: the real unit falls back to its own thermistor
        w = np.array([model.weights[s] for s in members]) / present
        units.append((reg, members, w, model))
    if not units:
        return {}

    member_set = sorted({s for _, members, _, _ in units for s in members})
    T = {s: np.full(n_scenarios, preps[s].cur_temp) for s in member_set}
    activity = {reg.effector_name: np.zeros((n_scenarios, max_horizon + 1)) for reg, *_ in units}
    # Hysteresis state per unit and scenario: whether the unit is already in the
    # scenario's mode, and its running probability.
    in_mode: dict[str, np.ndarray] = {}
    running: dict[str, np.ndarray] = {}
    for reg, *_ in units:
        mode_now, running_now = state.unit_states.get(reg.effector_name, ("off", False))
        sign_now = {"heat": 1.0, "cool": -1.0}.get(mode_now, 0.0)
        in_mode[reg.effector_name] = (reg.modes == sign_now) & (sign_now != 0.0)
        running[reg.effector_name] = np.where(in_mode[reg.effector_name], float(running_now), 0.0)
    for step in range(1, max_horizon + 1):
        for reg, members, w, model in units:
            name = reg.effector_name
            V = sum(wi * T[s] for wi, s in zip(w, members, strict=True))
            active = (step >= reg.delay_steps) & (step < reg.delay_steps + reg.duration_steps) & (reg.modes != 0)
            sign = np.where(active, reg.modes, 0.0)
            err = sign * (reg.targets - V)
            response = model.response
            if response.has_transitions:
                a = running[name]
                kept = response.level_for(err, sign, "stay") * a + response.level_for(err, sign, "start") * (1.0 - a)
                level = np.where(
                    active, np.where(in_mode[name], kept, response.level_for(err, sign, "enter")), 0.0,
                )
                running[name] = level
                in_mode[name] = active  # off before a delayed start and after the window
            else:
                level = response.level_for(err, sign)
            activity[name][:, step] = sign * level
        for s in member_set:
            prep = preps[s]
            dTdt = prep.base_dTdt(T[s], step - 1, outdoor_vec[step - 1])
            for eff_name, gain, lag_s in prep.reg_gains:
                act = activity.get(eff_name)
                if act is not None and step - lag_s >= 1:
                    dTdt = dTdt + gain * act[:, step - lag_s]
            T[s] = T[s] + _DT_HOURS * dTdt
    return activity


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
