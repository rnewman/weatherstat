"""System identification: extract thermal parameters from collector data.

Fits physical parameters from observed temperature data:
- Tau (envelope loss rate) per sensor, sealed and ventilated
- Effector × sensor gain matrix (how much each HVAC device heats each sensor)
- Solar gain profiles per sensor

Two-stage approach:
  Stage 1: Fit tau from nighttime HVAC-off periods (Newton cooling)
  Stage 2: Linear regression of Newton residuals on effector activity + solar

Config-driven: reads sensors, effectors, and windows from weatherstat.yaml.
Adding a device = YAML edit + rerun.

Usage:
  python -m weatherstat.sysid [--output PATH] [--verbose]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from weatherstat.config import DATA_DIR, UNIT_SYMBOL, delta_temp
from weatherstat.extract import load_collector_snapshots
from weatherstat.yaml_config import load_config

if TYPE_CHECKING:
    from weatherstat.validate import RegressionDiagnostics

_CFG = load_config()

# ── Output data structures ────────────────────────────────────────────────


@dataclass(frozen=True)
class EffectorSpec:
    """An effector (HVAC device) derived from config.

    Each effector has a command/state pair:
    - state_column + encoding: what the device actually did (for sysid training)
    - command_column + command_encoding: what we told it to do (for control scenarios)

    When command_column is None, state_column serves both roles.
    When state_gate is set, effective activity = self × encoded(state_gate_column)
    (e.g., thermostat calling × boiler firing = actual heat delivery).
    """

    name: str
    state_column: str  # column for measured state (what it did)
    encoding: dict[str, float]  # encoding for state_column
    max_lag_minutes: int
    device_type: str
    state_gate: str | None = None  # state sensor column that gates this effector
    command_column: str | None = None  # column for command (what we told it)
    command_encoding: dict[str, float] | None = None  # encoding for command_column


@dataclass(frozen=True)
class SensorSpec:
    """A temperature sensor derived from config."""

    name: str
    temp_column: str
    yaml_tau_base: float


@dataclass(frozen=True)
class FittedTau:
    """Envelope loss rate fitted from overnight cooling data.

    tau_base: sealed envelope time constant (all windows/advisories at default).
    environment_tau_betas: per-advisory-effector additional cooling rate coefficients,
        learned by regression in Stage 2. Keyed by device name.
    """

    sensor: str
    tau_base: float
    n_segments: int
    environment_tau_betas: dict[str, float] = field(default_factory=dict)



@dataclass(frozen=True)
class EffectorSensorGain:
    """How one sensor responds to one effector."""

    effector: str
    sensor: str
    gain_f_per_hour: float
    best_lag_minutes: float
    t_statistic: float
    negligible: bool


@dataclass(frozen=True)
class SolarGainProfile:
    sensor: str
    hour_of_day: int
    gain_f_per_hour: float
    std_error: float
    t_statistic: float


@dataclass(frozen=True)
class StateGate:
    """A state sensor gate referenced by effectors for delivery confirmation."""

    column: str
    encoding: dict[str, float]


@dataclass(frozen=True)
class SysIdResult:
    timestamp: str
    data_start: str
    data_end: str
    n_snapshots: int
    effectors: list[EffectorSpec]
    sensors: list[SensorSpec]
    fitted_taus: list[FittedTau]
    effector_sensor_gains: list[EffectorSensorGain]
    solar_gains: list[SolarGainProfile]  # legacy per-hour (empty for new fits)
    solar_elevation_gains: dict[str, float] = field(default_factory=dict)  # sensor -> gain
    state_gates: dict[str, StateGate] = field(default_factory=dict)
    mrt_weights: dict[str, float] = field(default_factory=dict)
    environment_solar_betas: dict[str, dict[str, float]] = field(default_factory=dict)  # device -> {sensor -> beta}


# ── Stage 0: Enumerate effectors and sensors from config ──────────────────


def _resolve_gate(device_name: str) -> str | None:
    """Resolve a state_device name to the state sensor column, or None."""
    if device_name in _CFG.state_sensors:
        return device_name
    return None


def _enumerate_effectors() -> list[EffectorSpec]:
    """Build effector list from YAML config.

    Single loop over the flat effectors dict. All sysid-relevant properties
    (encoding, max_lag_minutes, state_gate) come from YAML config.

    Column naming convention (determined by HA entity domain):
    - climate + mode_control=manual: state from {name}_action (hvac_action attribute)
    - climate + mode_control=automatic: state from {name}_action if state_encoding,
      else {name}_mode; command from {name}_mode
    - fan: state from {name}_mode (preset_mode attribute)

    device_type for simulator compatibility:
    - trajectory → "thermostat", regulating → "mini_split", binary → "blower"
    """
    device_type_map = {"trajectory": "thermostat", "regulating": "mini_split", "binary": "blower"}
    effectors: list[EffectorSpec] = []

    for name, cfg in _CFG.effectors.items():
        dtype = device_type_map.get(cfg.control_type, cfg.control_type)

        if cfg.domain == "climate" and cfg.mode_control == "manual":
            # Manual-mode climate (thermostat): state = hvac_action attribute
            effectors.append(EffectorSpec(
                name=name,
                state_column=f"{name}_action",
                encoding=cfg.state_encoding,
                max_lag_minutes=cfg.max_lag_minutes,
                device_type=dtype,
                state_gate=_resolve_gate(cfg.state_device) if cfg.state_device else None,
            ))
        elif cfg.domain == "climate":
            # Automatic-mode climate (mini-split): separate command/state
            if cfg.command_encoding and cfg.state_encoding != cfg.command_encoding:
                effectors.append(EffectorSpec(
                    name=name,
                    state_column=f"{name}_action",
                    encoding=cfg.state_encoding,
                    max_lag_minutes=cfg.max_lag_minutes,
                    device_type=dtype,
                    command_column=f"{name}_mode",
                    command_encoding=cfg.command_encoding,
                ))
            else:
                enc = cfg.command_encoding or cfg.state_encoding
                effectors.append(EffectorSpec(
                    name=name,
                    state_column=f"{name}_mode",
                    encoding=enc,
                    max_lag_minutes=cfg.max_lag_minutes,
                    device_type=dtype,
                ))
        else:
            # Fan entity (blower): state from preset_mode
            effectors.append(EffectorSpec(
                name=name,
                state_column=f"{name}_mode",
                encoding=cfg.state_encoding,
                max_lag_minutes=cfg.max_lag_minutes,
                device_type=dtype,
            ))

    return effectors


def _enumerate_sensors() -> list[SensorSpec]:
    """Build sensor list from YAML config.

    Only includes sensors that have comfort constraints (optimization targets).
    Unconstrained sensors (outdoor, basement, aggregates, etc.) are collected
    but not regressed — their gains would be unused by control and are a source
    of confounded coefficients.

    Tau uses defaults.tau as the initial guess for curve_fit. Sysid fits
    the actual value from data. Window effects are learned in Stage 2
    regression — no configured window→sensor mapping needed.
    """
    constrained = set(_CFG.prediction_sensors)
    sensors: list[SensorSpec] = []

    for col_name in _CFG.temp_sensors:
        if col_name not in constrained:
            continue

        sensors.append(SensorSpec(
            name=col_name,
            temp_column=col_name,
            yaml_tau_base=_CFG.default_tau,
        ))

    return sensors


# ── Stage 1: Fit tau per sensor ───────────────────────────────────────────

_MIN_SEGMENT_STEPS = 12  # 1 hour at 5-min intervals
# Maximum solar elevation (degrees) for tau segment qualification.
# Below this threshold, solar gain is negligible and Newton cooling
# dominates. 5° captures civil twilight and the brief period after
# sunrise / before sunset when the sun is too low to heat the house.
# At Seattle (47.7°N), this adds ~1-2 hours per day vs the old
# nighttime-only approach (which used 10pm-6am hardcoded hours).
_MAX_SOLAR_ELEV_FOR_TAU = 5.0


def _fit_tau_curve(t_hours: np.ndarray, temps: np.ndarray, t_outdoor: float) -> float | None:
    """Fit tau from a single cooling segment. Returns None on failure."""
    t_0 = temps[0]
    if abs(t_0 - t_outdoor) < 1.0:
        return None
    normalized = (temps - t_outdoor) / (t_0 - t_outdoor)
    try:
        popt, _ = curve_fit(
            lambda t, tau: np.exp(-t / tau), t_hours, normalized, p0=[40.0], bounds=(1.0, 500.0)
        )
        return float(popt[0])
    except RuntimeError:
        return None


def _find_uncontrolled_segments(
    df: pd.DataFrame,
    effectors: list[EffectorSpec],
    sensor: SensorSpec,
) -> list[pd.DataFrame]:
    """Find contiguous low-solar segments with no active HVAC control.

    These segments show passive convergence toward outdoor temperature —
    cooling when indoors is warmer, warming when outdoors is warmer.
    The exponential time constant (tau) is symmetric either way.

    Qualifying rows: solar elevation < 5° (negligible solar gain) AND all
    HVAC effectors off. This replaces the old nighttime-only approach
    (10pm-6am), gaining ~1-2 hours per day from twilight periods and
    adapting automatically to season and latitude.

    Window/door states must be constant within each segment (no transitions)
    but need not all be closed. The fitted tau includes whatever window
    effects are present; Stage 2 regression decomposes them.
    """
    # Identify low-solar rows: sun below threshold elevation.
    # Replaces hardcoded nighttime hours — adapts to season and latitude.
    if "_solar_elev_deg" in df.columns:
        is_dark = df["_solar_elev_deg"] < _MAX_SOLAR_ELEV_FOR_TAU
    else:
        # Fallback if preprocessing didn't compute elevation (e.g., tests)
        local_hour = df["_local_hour"]
        is_dark = (local_hour >= 22) | (local_hour < 6)

    # Identify all-HVAC-off rows
    all_off = pd.Series(True, index=df.index)
    for eff in effectors:
        if eff.state_column not in df.columns:
            continue
        enc_col = f"_eff_{eff.name}"
        if enc_col in df.columns:
            all_off &= df[enc_col] == 0.0

    mask = is_dark & all_off

    # Drop rows where this sensor's temp is NaN
    temp_col = sensor.temp_column
    if temp_col not in df.columns:
        return []
    mask &= df[temp_col].notna() & df["_outdoor_best"].notna()

    qualifying = df[mask].copy()
    if len(qualifying) < _MIN_SEGMENT_STEPS:
        return []

    # Split into contiguous segments (gaps > 10 min break segments)
    dt = qualifying["_ts"].diff()
    breaks = dt > pd.Timedelta(minutes=10)
    seg_ids = breaks.cumsum()

    # Only keep segments where all window/door states are constant
    # (no transitions mid-segment). NaN (sensor didn't exist) is treated
    # as a constant unknown state and does not disqualify the segment.
    all_env_cols = [cfg.column for cfg in _CFG.environment.values()]
    existing_env_cols = [c for c in all_env_cols if c in df.columns]

    segments: list[pd.DataFrame] = []
    for _, seg_df in qualifying.groupby(seg_ids):
        if len(seg_df) < _MIN_SEGMENT_STEPS:
            continue
        # Check window stability: each column must have at most 1 unique
        # non-null value within the segment
        stable = True
        for wc in existing_env_cols:
            n_unique = seg_df[wc].dropna().nunique()
            if n_unique > 1:
                stable = False
                break
        if stable:
            segments.append(seg_df)

    return segments


def _fit_tau(
    df: pd.DataFrame,
    effectors: list[EffectorSpec],
    sensors: list[SensorSpec],
    verbose: bool = False,
) -> list[FittedTau]:
    """Stage 1: Fit tau per sensor from uncontrolled low-solar segments.

    Uses segments with stable window state (no transitions). The fitted
    tau includes any window effects present; Stage 2 decomposes them.
    """
    results: list[FittedTau] = []

    for sensor in sensors:
        segments = _find_uncontrolled_segments(df, effectors, sensor)
        if not segments:
            if verbose:
                print(f"  {sensor.name}: no qualifying uncontrolled overnight segments")
            results.append(FittedTau(
                sensor=sensor.name,
                tau_base=sensor.yaml_tau_base,
                n_segments=0,
            ))
            continue

        # Fit tau from each segment
        fitted_taus: list[tuple[float, int]] = []  # (tau, segment_length)
        for seg_df in segments:
            t_hours = (seg_df["_ts"] - seg_df["_ts"].iloc[0]).dt.total_seconds().values / 3600.0
            temps = seg_df[sensor.temp_column].values
            t_outdoor = seg_df["_outdoor_best"].mean()
            tau = _fit_tau_curve(t_hours, temps, t_outdoor)
            if tau is not None:
                fitted_taus.append((tau, len(seg_df)))

        tau_base = _weighted_median(fitted_taus) if fitted_taus else sensor.yaml_tau_base

        results.append(FittedTau(
            sensor=sensor.name,
            tau_base=tau_base,
            n_segments=len(fitted_taus),
        ))

    return results


def _weighted_median(values_weights: list[tuple[float, int]]) -> float:
    """Compute weighted median."""
    if len(values_weights) == 1:
        return round(values_weights[0][0], 1)
    vals = np.array([v for v, _ in values_weights])
    weights = np.array([w for _, w in values_weights], dtype=float)
    sorted_idx = np.argsort(vals)
    vals = vals[sorted_idx]
    weights = weights[sorted_idx]
    cumw = np.cumsum(weights)
    mid = cumw[-1] / 2.0
    idx = np.searchsorted(cumw, mid)
    return round(float(vals[min(idx, len(vals) - 1)]), 1)


# ── Preprocessing ─────────────────────────────────────────────────────────


def _preprocess(
    df: pd.DataFrame,
    effectors: list[EffectorSpec],
    sensors: list[SensorSpec],
) -> pd.DataFrame:
    """Preprocess snapshot data for sysid."""
    df = df.copy()

    # Parse timestamps
    df["_ts"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df = df.sort_values("_ts").reset_index(drop=True)

    # Local hour for nighttime detection and solar features
    tz = _CFG.location.timezone
    df["_local_hour"] = df["_ts"].dt.tz_convert(tz).dt.hour

    # Best outdoor temp: prefer configured outdoor sensor, fall back to
    # met.no weather data. Interpolate to fill gaps in hourly met data.
    outdoor_col = _CFG.outdoor_sensor  # None if not configured
    met_available = "met_outdoor_temp" in df.columns and df["met_outdoor_temp"].notna().sum() > 100
    sensor_available = outdoor_col and outdoor_col in df.columns and df[outdoor_col].notna().sum() > 100

    if sensor_available and met_available:
        sensor = pd.to_numeric(df[outdoor_col], errors="coerce")
        met = pd.to_numeric(df["met_outdoor_temp"], errors="coerce").interpolate(method="linear", limit=24)
        df["_outdoor_best"] = sensor.fillna(met)
        n_sensor = sensor.notna().sum()
        n_fallback = sensor.isna().sum() - df["_outdoor_best"].isna().sum()
        print(f"  Outdoor temp: {outdoor_col} for {n_sensor}/{len(df)} rows, met.no fallback for {n_fallback}")
    elif sensor_available:
        df["_outdoor_best"] = pd.to_numeric(df[outdoor_col], errors="coerce")
        print(f"  Outdoor temp: {outdoor_col} (no met.no data)")
    elif met_available:
        met = pd.to_numeric(df["met_outdoor_temp"], errors="coerce").interpolate(method="linear", limit=24)
        df["_outdoor_best"] = met
        print(f"  Outdoor temp: met.no for {met.notna().sum()}/{len(df)} rows")
    else:
        raise ValueError("No outdoor temperature source available (configure a sensor or ensure weather entity data)")

    # Outdoor temp rate-of-change (°F/hr, for weather control feature)
    dt_hours = 5.0 / 60.0
    out_vals = df["_outdoor_best"].values.astype(float)
    dT_out = np.full_like(out_vals, np.nan)
    if len(out_vals) > 2:
        dT_out[1:-1] = (out_vals[2:] - out_vals[:-2]) / (2 * dt_hours)
        dT_out[0] = (out_vals[1] - out_vals[0]) / dt_hours
        dT_out[-1] = (out_vals[-1] - out_vals[-2]) / dt_hours
    df["_dTdt_outdoor"] = dT_out

    # Encode effector states to numeric.
    # Each effector has a state column (measured action) and optionally a
    # command column (intent) as fallback for rows where state isn't available.
    for eff in effectors:
        col = eff.state_column
        if col in df.columns:
            encoded = df[col].map(eff.encoding)
            # Fall back to command column where state is missing (e.g., old data
            # before action column was captured)
            if eff.command_column and eff.command_column in df.columns:
                cmd_encoded = df[eff.command_column].map(eff.command_encoding or eff.encoding)
                encoded = encoded.fillna(cmd_encoded)
            df[f"_eff_{eff.name}"] = encoded.fillna(0.0).astype(float)
        elif eff.command_column and eff.command_column in df.columns:
            df[f"_eff_{eff.name}"] = (
                df[eff.command_column].map(eff.command_encoding or eff.encoding).fillna(0.0).astype(float)
            )
        else:
            df[f"_eff_{eff.name}"] = 0.0

    # Encode state sensor columns (categorical → numeric via encoding).
    for col, scfg in _CFG.state_sensors.items():
        if col in df.columns:
            df[f"_state_{col}"] = df[col].map(scfg.encoding).fillna(0.0).astype(float)

    # Apply state confirmation: effective activity requires state sensor gate.
    # E.g., a thermostat "calling for heat" only delivers heat when the boiler
    # confirms it's responding. Without this, fault periods show intent-on
    # with no temperature rise, diluting fitted gains.
    for eff in effectors:
        if eff.state_gate:
            gate_col = f"_state_{eff.state_gate}"
            if gate_col in df.columns:
                df[f"_eff_{eff.name}"] *= df[gate_col]

    # Compute dT/dt for each sensor (°F/hr).
    #
    # Naive 5-minute central differences amplify sensor noise: ±0.1°F jitter
    # becomes ±1.2°F/hr in the derivative, producing a residual std of ~10°F/hr
    # that drowns effector signals of ~0.3°F/hr. The lag-2 autocorrelation of
    # -0.6 confirms this is differentiation noise, not real temperature dynamics.
    #
    # Fix: smooth temperature with a centered rolling mean, then differentiate
    # over the wider window. With a 3-step (15-min) half-window, noise drops
    # ~5× while preserving signals on the timescale of effector lags (≥15 min).
    _SMOOTH_HALF_WINDOW = 3  # steps; total smoothing = (2w+1)×5 = 35 min
    dt_hours = 5.0 / 60.0  # 5-minute intervals
    w = _SMOOTH_HALF_WINDOW
    for sensor in sensors:
        col = sensor.temp_column
        if col not in df.columns:
            continue
        temps = df[col].values.astype(float)
        # Smooth with centered rolling mean to suppress sensor jitter
        temps_smooth = pd.Series(temps).rolling(
            2 * w + 1, center=True, min_periods=w + 1,
        ).mean().values
        # Central differences on smoothed series (span = 2w steps = 2w×5 min)
        dT = np.full_like(temps, np.nan)
        if len(temps) > 2 * w:
            dT[w:-w] = (temps_smooth[2 * w:] - temps_smooth[:-2 * w]) / (2 * w * dt_hours)
        df[f"_dTdt_{sensor.name}"] = dT

    # Generate lagged effector features in coarse bins derived from max_lag_minutes
    for eff in effectors:
        eff_col = f"_eff_{eff.name}"
        for (start, end), label in _lag_bins(eff.max_lag_minutes):
            # Mean activity in the lag bin [start, end) steps back
            lagged_sum = pd.Series(0.0, index=df.index)
            count = 0
            for lag in range(start, end):
                shifted = df[eff_col].shift(lag)
                lagged_sum += shifted.fillna(0.0)
                count += 1
            df[f"_lag_{eff.name}_{label}"] = lagged_sum / max(count, 1)

    # Solar elevation: raw degrees (for tau segment selection) and
    # sin⁺(elevation) × condition_fraction (for regression feature).
    from weatherstat.weather import condition_to_solar_fraction, solar_elevation, solar_sin_elevation

    lat = _CFG.location.latitude
    lon = _CFG.location.longitude

    # Raw elevation in degrees — used by _find_uncontrolled_segments() to
    # replace hardcoded nighttime hours. Negative = sun below horizon.
    df["_solar_elev_deg"] = df["_ts"].apply(lambda dt: solar_elevation(lat, lon, dt))

    # Regression feature: sin⁺(elevation) × weather condition fraction.
    solar_frac = df["weather_condition"].map(condition_to_solar_fraction).fillna(0.3) if "weather_condition" in df.columns else pd.Series(0.3, index=df.index)
    df["_solar_elev"] = df["_ts"].apply(lambda dt: solar_sin_elevation(lat, lon, dt)) * solar_frac

    return df


# ── Stage 2: Regression per sensor ────────────────────────────────────────

_GAIN_THRESHOLD = delta_temp(0.05)  # per hour — below this, gain is negligible
_T_STAT_THRESHOLD = 2.0
_MIN_REGRESSION_ROWS = 500  # need substantial data for reliable regression

def _lag_bins(max_lag_minutes: int) -> list[tuple[tuple[int, int], str]]:
    """Derive lag bins from max_lag_minutes.

    Returns (start_step, end_step) pairs with labels, at 5-min resolution.
    Reproduces current behavior:
      max_lag=5  → [(0,1)] "0_5"
      max_lag=15 → [(0,1),(1,3)] "0_5", "5_15"
      max_lag=90 → [(0,3),(3,6),(6,12),(12,18)] "0_15", "15_30", "30_60", "60_90"
    """
    steps = max(1, max_lag_minutes // 5)
    if steps <= 1:
        bins = [(0, 1)]
    elif steps <= 3:
        bins = [(0, 1), (1, steps)]
    elif steps <= 6:
        bins = [(0, 3), (3, steps)]
    else:
        # Logarithmic-ish bins for long-lag effectors
        bins = [(0, 3), (3, 6), (6, 12), (12, steps)]
    return [((s, e), f"{s * 5}_{e * 5}") for s, e in bins]


def _get_lag_columns(eff: EffectorSpec) -> list[str]:
    """Get lag column names for an effector, derived from max_lag_minutes."""
    return [f"_lag_{eff.name}_{label}" for _, label in _lag_bins(eff.max_lag_minutes)]


def _lag_label_to_minutes(label: str) -> float:
    """Convert lag bin label (e.g. '0_15') to midpoint in minutes."""
    parts = label.split("_")
    if len(parts) == 2:
        start, end = int(parts[0]), int(parts[1])
        return (start + end) / 2.0
    return 0.0


def _fit_sensor_model(
    df: pd.DataFrame,
    sensor: SensorSpec,
    effectors: list[EffectorSpec],
    tau_base: float,
    verbose: bool = False,
) -> tuple[list[EffectorSensorGain], float, float, dict[str, float], dict[str, float], RegressionDiagnostics | None]:
    """Stage 2: Fit regression for one sensor.

    Returns (gains, solar_elev_gain, solar_elev_t, environment_tau_betas,
    environment_solar_betas, diagnostics).
    solar_elev_gain is °F/hr per unit sin(elevation)×solar_fraction.
    Advisory tau betas are per-device additional cooling rate coefficients,
    learned as regression coefficients on advisory_state × (T_out - T).
    Advisory solar betas model how advisory state modulates solar gain.
    diagnostics is None for early returns (missing columns, insufficient data).
    """
    temp_col = sensor.temp_column
    dTdt_col = f"_dTdt_{sensor.name}"

    if temp_col not in df.columns or dTdt_col not in df.columns:
        return [], 0.0, 0.0, {}, {}, None
    if "_outdoor_best" not in df.columns:
        return [], 0.0, 0.0, {}, {}, None

    outdoor = df["_outdoor_best"].values
    sensor_temp = df[temp_col].values

    # Newton residual using tau_base (all rows, not split by window state)
    newton_dTdt = (outdoor - sensor_temp) / tau_base
    residual = df[dTdt_col].values - newton_dTdt

    # Build feature matrix
    feature_names: list[str] = []
    feature_cols: list[np.ndarray] = []

    # Effector lag features
    effector_feature_ranges: dict[str, tuple[int, int]] = {}  # eff_name -> (start_idx, end_idx)
    for eff in effectors:
        lag_cols = _get_lag_columns(eff)
        start = len(feature_names)
        for lc in lag_cols:
            if lc in df.columns:
                feature_names.append(lc)
                feature_cols.append(df[lc].values.astype(float))
        end = len(feature_names)
        if end > start:
            effector_feature_ranges[eff.name] = (start, end)

    # Weather control features — absorb weather-driven variance that would
    # otherwise be attributed to correlated HVAC activity.
    delta_t = outdoor - sensor_temp

    # ΔT² — captures nonlinear heat loss (stack effect, radiation increase
    # at large indoor-outdoor difference). The Newton model assumes linear
    # heat loss; this term lets the regression correct for the nonlinearity.
    feature_names.append("_weather_dt2")
    feature_cols.append(delta_t ** 2 * np.sign(delta_t))

    # wind × ΔT — wind-driven convective heat loss. A windy cold night
    # causes more heat loss than a calm one at the same ΔT.
    if "wind_speed" in df.columns:
        wind = pd.to_numeric(df["wind_speed"], errors="coerce").fillna(0.0).values
        feature_names.append("_weather_wind_dt")
        feature_cols.append(wind * delta_t)

    # dT_outdoor/dt — rapid outdoor temp drops cause more heat loss than
    # the instantaneous ΔT suggests (interior mass hasn't equilibrated).
    if "_dTdt_outdoor" in df.columns:
        feature_names.append("_weather_dTout_dt")
        feature_cols.append(df["_dTdt_outdoor"].values.astype(float))

    # Time-of-day Fourier features — absorb diurnal patterns in internal
    # heat (occupancy, cooking, stored wall heat) that correlate with HVAC
    # activity but aren't caused by it.
    if "_ts" in df.columns:
        _tz = _CFG.location.timezone
        ts_local = df["_ts"].dt.tz_convert(_tz)
        hour_frac = (ts_local.dt.hour + ts_local.dt.minute / 60.0).values
        hour_rad = hour_frac * (2 * np.pi / 24)
        feature_names.extend(["_tod_sin1", "_tod_cos1", "_tod_sin2", "_tod_cos2"])
        feature_cols.extend([
            np.sin(hour_rad),
            np.cos(hour_rad),
            np.sin(2 * hour_rad),
            np.cos(2 * hour_rad),
        ])

    # Environment factor × ΔT features: state × (T_outdoor - T_sensor)
    # The coefficient β gives the additional cooling/heating rate when active.
    # Skip devices with too few active rows — near-zero-variance columns blow up
    # the condition number and the coefficient can't be reliably estimated anyway.
    # Skip kind=shade entirely: shades affect solar gain (modeled separately as
    # _adv_solar_*), not envelope conduction. A shade has no physical mechanism
    # for changing tau, and "closed shade" tends to correlate with HVAC-off
    # nighttime hours, producing strongly confounded β > 0 (apparent fast cooling).
    _MIN_ACTIVE_ROWS = 50  # ~4h of 5-min data
    _TAU_SKIP_KINDS = {"shade"}
    env_col_to_name = {cfg.column: name for name, cfg in _CFG.environment.items()}
    all_adv_cols = [
        cfg.column for cfg in _CFG.environment.values() if cfg.kind not in _TAU_SKIP_KINDS
    ]
    existing_adv_cols = [c for c in all_adv_cols if c in df.columns]

    adv_tau_feature_start = len(feature_names)
    adv_tau_feature_names: list[str] = []  # bare device names in order
    adv_state_arrays: dict[str, np.ndarray] = {}  # device_name -> state array (for reuse)
    for wc in existing_adv_cols:
        dev_name = env_col_to_name[wc]
        state_arr = df[wc].astype("float64").fillna(0.0).values
        n_active = int(state_arr.sum())
        if n_active < _MIN_ACTIVE_ROWS:
            if verbose:
                print(f"    advisory_tau {dev_name}: skipped ({n_active} active rows < {_MIN_ACTIVE_ROWS})")
            continue
        adv_state_arrays[dev_name] = state_arr
        feature_names.append(f"_adv_tau_{dev_name}")
        feature_cols.append(state_arr * delta_t)
        adv_tau_feature_names.append(dev_name)

    # Solar features still need state arrays for skipped tau-kinds (shades).
    # Build them now, since the tau loop above didn't.
    for wc in [c for c in (cfg.column for cfg in _CFG.environment.values()) if c in df.columns]:
        dev_name = env_col_to_name[wc]
        if dev_name in adv_state_arrays:
            continue
        state_arr = df[wc].astype("float64").fillna(0.0).values
        if int(state_arr.sum()) < _MIN_ACTIVE_ROWS:
            continue
        adv_state_arrays[dev_name] = state_arr

    # Solar elevation feature (single continuous feature per sensor)
    solar_start = len(feature_names)
    solar_elev_arr: np.ndarray | None = None
    if "_solar_elev" in df.columns:
        solar_elev_arr = df["_solar_elev"].values.astype(float)
        feature_names.append("_solar_elev")
        feature_cols.append(solar_elev_arr)

    # Advisory × solar interaction: advisory state modulates solar gain.
    # Only for kinds that physically modulate solar (shades, blinds) — not
    # windows/doors, which affect tau (already modeled) but not solar gain.
    # Including windows/doors here produces confounded coefficients (windows
    # are open on sunny days → regression attributes sunshine to window state).
    _SOLAR_MOD_KINDS = {"shade"}
    adv_solar_feature_start = len(feature_names)
    adv_solar_feature_names: list[str] = []  # device names in order
    if solar_elev_arr is not None:
        for dev_name, state_arr in adv_state_arrays.items():
            env_cfg = _CFG.environment.get(dev_name)
            if env_cfg is None or env_cfg.kind not in _SOLAR_MOD_KINDS:
                continue
            feature_names.append(f"_adv_solar_{dev_name}")
            feature_cols.append(state_arr * solar_elev_arr)
            adv_solar_feature_names.append(dev_name)

    if not feature_cols:
        return [], 0.0, 0.0, {}, {}, None

    X = np.column_stack(feature_cols)

    # Drop rows with NaN in residual or any feature
    valid = ~np.isnan(residual)
    for i in range(X.shape[1]):
        valid &= ~np.isnan(X[:, i])
    X = X[valid]
    y = residual[valid]

    if len(y) < max(X.shape[1] + 10, _MIN_REGRESSION_ROWS):
        if verbose:
            print(f"  {sensor.name}: insufficient data ({len(y)} rows, need {_MIN_REGRESSION_ROWS}+)")
        gains = [
            EffectorSensorGain(
                effector=e.name, sensor=sensor.name,
                gain_f_per_hour=0.0, best_lag_minutes=0.0,
                t_statistic=0.0, negligible=True,
            )
            for e in effectors
        ]
        return gains, 0.0, 0.0, {}, {}, None

    # Selectively standardized ridge regularization.
    # OLS on observational HVAC data produces confounded gains: heating
    # correlates with cold weather, inflating gain estimates. Ridge shrinks
    # poorly-identified coefficients toward zero, improving t-statistics
    # for genuine effects and reducing false large gains.
    #
    # Solar and window features are selectively standardized: divided by
    # their std so the penalty falls proportionally. Without this, solar
    # indicators (active ~1/24 of the time, low variance) are over-penalized,
    # suppressing legitimate solar gain estimates. Effector and weather
    # features are left in raw scale to maintain full regularization against
    # confounded gains (they have high variance from frequent HVAC cycling).
    scale = np.ones(X.shape[1])
    for j, name in enumerate(feature_names):
        if name.startswith("_solar_") or name.startswith("_adv_"):
            s = np.std(X[:, j])
            if s > 0:
                scale[j] = s
    X_s = X / scale

    cond = np.linalg.cond(X_s)
    lam = 0.01 * len(y)
    if cond > 1e6:
        lam *= 10  # stronger regularization for ill-conditioned problems

    XsXs = X_s.T @ X_s + lam * np.eye(X_s.shape[1])
    beta_s = np.linalg.solve(XsXs, X_s.T @ y)

    # Transform back to original feature scale
    beta = beta_s / scale
    resid = y - X @ beta
    s2 = np.sum(resid**2) / max(len(y) - X.shape[1], 1)
    try:
        cov_s = s2 * np.linalg.inv(XsXs)
        se = np.sqrt(np.abs(np.diag(cov_s))) / scale
    except np.linalg.LinAlgError:
        se = np.full(len(beta), np.nan)

    t_stats = np.where(se > 0, beta / se, 0.0)

    # Per-sensor regression diagnostics (R², DW, VIF, holdout, bootstrap stability)
    from weatherstat.validate import validate_sysid_regression

    diagnostics = validate_sysid_regression(
        sensor.name, X, y, feature_names, scale, lam, beta, resid,
    )
    bootstrap_cvs = diagnostics.bootstrap_cvs
    for issue in diagnostics.issues:
        prefix = "ERROR" if issue.severity.value == "error" else "WARNING"
        print(f"  {prefix} [{sensor.name}]: {issue.message}")

    # Report weather/time-of-day control feature coefficients (diagnostic, not stored)
    if verbose:
        for i, fname in enumerate(feature_names):
            if fname.startswith("_weather_"):
                label = fname.removeprefix("_weather_")
                print(f"    weather {label}: β={beta[i]:.6f}, t={t_stats[i]:.1f}")
            elif fname.startswith("_tod_"):
                label = fname.removeprefix("_tod_")
                print(f"    tod {label}: β={beta[i]:.6f}, t={t_stats[i]:.1f}")

    # Extract effector gains
    gains: list[EffectorSensorGain] = []
    for eff in effectors:
        if eff.name not in effector_feature_ranges:
            gains.append(EffectorSensorGain(
                effector=eff.name, sensor=sensor.name,
                gain_f_per_hour=0.0, best_lag_minutes=0.0,
                t_statistic=0.0, negligible=True,
            ))
            continue

        start, end = effector_feature_ranges[eff.name]
        lag_betas = beta[start:end]
        lag_tstats = t_stats[start:end]
        lag_cols_used = feature_names[start:end]

        best_idx = int(np.argmax(np.abs(lag_betas)))
        best_lag_label = lag_cols_used[best_idx].split("_")[-2] + "_" + lag_cols_used[best_idx].split("_")[-1]
        best_lag_min = _lag_label_to_minutes(best_lag_label)

        total_gain = float(np.sum(lag_betas))
        max_t = float(lag_tstats[best_idx])

        negligible = abs(total_gain) < _GAIN_THRESHOLD and abs(max_t) < _T_STAT_THRESHOLD

        gains.append(EffectorSensorGain(
            effector=eff.name,
            sensor=sensor.name,
            gain_f_per_hour=round(total_gain, 4),
            best_lag_minutes=best_lag_min,
            t_statistic=round(max_t, 2),
            negligible=negligible,
        ))

    # Extract advisory tau betas
    # β > 0 is physical (device increases heat exchange rate when active).
    # β < 0 is unphysical for window-type devices — flag as zero.
    # Only warn about bootstrap instability for features we actually keep.
    n_unstable_kept = 0
    environment_tau_betas: dict[str, float] = {}
    for i, dev_name in enumerate(adv_tau_feature_names):
        idx = adv_tau_feature_start + i
        feature_key = f"_adv_tau_{dev_name}"
        cv = bootstrap_cvs.get(feature_key)
        if idx < len(beta):
            b = float(beta[idx])
            t = float(t_stats[idx])
            cv_str = f", CV={cv:.1f}" if cv is not None else ""
            if b > 0 and abs(t) >= _T_STAT_THRESHOLD:
                stable = cv is None or cv <= 1.0
                tag = "stable" if stable else "UNSTABLE"
                environment_tau_betas[dev_name] = round(b, 6)
                if not stable:
                    n_unstable_kept += 1
                    print(f"  WARNING [{sensor.name}]: advisory_tau {dev_name} kept but unstable (CV={cv:.1f})")
                if verbose:
                    print(f"    advisory_tau {dev_name}: β={b:.6f}, t={t:.1f}{cv_str} → kept ({tag})")
            elif verbose:
                reason = "β≤0" if b <= 0 else f"|t|={abs(t):.1f}<{_T_STAT_THRESHOLD}"
                print(f"    advisory_tau {dev_name}: β={b:.6f}, t={t:.1f}{cv_str} → dropped ({reason})")

    # Extract advisory × solar betas (how advisory state modulates solar gain)
    environment_solar_betas: dict[str, float] = {}
    for i, dev_name in enumerate(adv_solar_feature_names):
        idx = adv_solar_feature_start + i
        feature_key = f"_adv_solar_{dev_name}"
        cv = bootstrap_cvs.get(feature_key)
        if idx < len(beta):
            b = float(beta[idx])
            t = float(t_stats[idx])
            cv_str = f", CV={cv:.1f}" if cv is not None else ""
            if abs(t) >= _T_STAT_THRESHOLD:
                stable = cv is None or cv <= 1.0
                tag = "stable" if stable else "UNSTABLE"
                environment_solar_betas[dev_name] = round(b, 4)
                if not stable:
                    n_unstable_kept += 1
                    print(f"  WARNING [{sensor.name}]: advisory_solar {dev_name} kept but unstable (CV={cv:.1f})")
                if verbose:
                    print(f"    advisory_solar {dev_name}: β={b:.6f}, t={t:.1f}{cv_str} → kept ({tag})")
            elif verbose:
                print(f"    advisory_solar {dev_name}: β={b:.6f}, t={t:.1f}{cv_str} → dropped (|t|<{_T_STAT_THRESHOLD})")

    # Extract solar elevation gain (single coefficient per sensor)
    solar_elev_gain: float = 0.0
    solar_elev_t: float = 0.0
    if solar_start < len(beta):
        solar_elev_gain = round(float(beta[solar_start]), 4)
        solar_elev_t = round(float(t_stats[solar_start]), 2)
        if verbose:
            solar_se = round(float(se[solar_start]), 4) if not np.isnan(se[solar_start]) else 0.0
            print(f"    solar_elev: β={solar_elev_gain:.4f}, se={solar_se:.4f}, t={solar_elev_t:.1f}")

    return gains, solar_elev_gain, solar_elev_t, environment_tau_betas, environment_solar_betas, diagnostics


# ── Derived MRT weights ──────────────────────────────────────────────────


# ── Main pipeline ─────────────────────────────────────────────────────────


def run_sysid(output_path: Path | None = None, verbose: bool = False) -> tuple[SysIdResult, dict]:
    """Run full system identification pipeline: fit + write.

    Convenience wrapper around fit_sysid() + save_sysid_result().
    Returns (result, diagnostics) for caller use.
    """
    result, diagnostics = fit_sysid(verbose=verbose)
    save_sysid_result(result, output_path, sensor_diagnostics=diagnostics)
    return result, diagnostics


def save_sysid_result(
    result: SysIdResult,
    output_path: Path | None = None,
    sensor_diagnostics: dict[str, RegressionDiagnostics] | None = None,
) -> Path:
    """Write a SysIdResult to disk as JSON.

    Runs validate_sysid_result() before writing. Prints health summary
    with per-sensor grades and gain stability comparison.
    Errors are printed but do NOT block saving — the caller (TUI quality
    gate) decides whether to use the result.

    Returns the path written to.
    """
    from weatherstat.validate import (
        format_issues,
        has_errors,
        validate_sysid_result,
    )

    issues = validate_sysid_result(result)
    if issues:
        print("\n── Sysid Validation ──")
        print(format_issues(issues))
        if has_errors(issues):
            print("  ⚠ Errors found — parameters may produce unreliable predictions.")

    out = output_path or DATA_DIR / "thermal_params.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    def _serialize(obj: object) -> object:
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)  # type: ignore[arg-type]
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(out, "w") as f:
        json.dump(asdict(result), f, indent=2, default=_serialize)
    print(f"\nResults written to {out}")
    return out


def _print_health_summary(
    result: SysIdResult,
    sensor_diagnostics: dict[str, RegressionDiagnostics],
    output_path: Path | None = None,
) -> None:
    """Compute and print per-sensor health grades + gain stability."""
    from weatherstat.validate import (
        SensorHealth,
        compute_sensor_health,
        format_health_summary,
    )

    # Build per-sensor gain counts
    sensor_gain_counts: dict[str, int] = {}
    for g in result.effector_sensor_gains:
        if not g.negligible:
            sensor_gain_counts[g.sensor] = sensor_gain_counts.get(g.sensor, 0) + 1

    # Build tau segment lookup and advisory beta counts
    tau_lookup: dict[str, int] = {}
    adv_beta_counts: dict[str, int] = {}
    for ft in result.fitted_taus:
        tau_lookup[ft.sensor] = ft.n_segments
        adv_beta_counts[ft.sensor] = len(ft.environment_tau_betas)

    n_effectors = len(result.effectors)

    # Compute n_unstable_kept per sensor from diagnostics + kept betas
    def _count_unstable(sensor_name: str, diag: RegressionDiagnostics) -> int:
        n = 0
        ft = next((f for f in result.fitted_taus if f.sensor == sensor_name), None)
        if ft:
            for dev_name in ft.environment_tau_betas:
                cv = diag.bootstrap_cvs.get(f"_adv_tau_{dev_name}")
                if cv is not None and cv > 1.0:
                    n += 1
        # Check solar betas
        for dev_name, sensor_betas in result.environment_solar_betas.items():
            if sensor_name in sensor_betas:
                cv = diag.bootstrap_cvs.get(f"_adv_solar_{dev_name}")
                if cv is not None and cv > 1.0:
                    n += 1
        return n

    # Per-sensor validation errors
    from weatherstat.validate import validate_sysid_result
    post_issues = validate_sysid_result(result)
    sensor_has_errors: dict[str, bool] = {}
    for issue in post_issues:
        if issue.severity.value == "error" and issue.sensor:
            sensor_has_errors[issue.sensor] = True

    healths: list[SensorHealth] = []
    for sensor in result.sensors:
        diag = sensor_diagnostics.get(sensor.name)
        r2 = diag.r_squared if diag else 0.0
        dw = diag.durbin_watson if diag else 0.0
        holdout = diag.holdout_degradation if diag else None
        n_unstable = _count_unstable(sensor.name, diag) if diag else 0
        # Also count regression-level errors (VIF errors)
        reg_errors = any(i.severity.value == "error" for i in diag.issues) if diag else False

        health = compute_sensor_health(
            sensor.name,
            r_squared=r2,
            durbin_watson=dw,
            n_segments=tau_lookup.get(sensor.name, 0),
            n_gains=sensor_gain_counts.get(sensor.name, 0),
            n_effectors=n_effectors,
            n_advisory_betas=adv_beta_counts.get(sensor.name, 0),
            n_unstable_kept=n_unstable,
            holdout_degradation=holdout,
            has_validation_errors=sensor_has_errors.get(sensor.name, False) or reg_errors,
        )
        healths.append(health)

    # Gain stability: compare with previous thermal_params.json
    gain_changes = _compute_gain_drift(result, output_path)

    print(format_health_summary(healths, gain_changes))


def _compute_gain_drift(
    result: SysIdResult,
    output_path: Path | None = None,
) -> dict[tuple[str, str], tuple[float, float]]:
    """Compare current gains with previous thermal_params.json.

    Returns {(effector, sensor): (old_gain, new_gain)} for gains that
    changed by >50% and were non-negligible in at least one version.
    """
    prev_path = output_path or DATA_DIR / "thermal_params.json"
    if not prev_path.exists():
        return {}

    try:
        with open(prev_path) as f:
            prev = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    # Build old gain lookup
    old_gains: dict[tuple[str, str], float] = {}
    for g in prev.get("effector_sensor_gains", []):
        if not g.get("negligible", True):
            old_gains[(g["effector"], g["sensor"])] = g["gain_f_per_hour"]

    # Build new gain lookup
    new_gains: dict[tuple[str, str], float] = {}
    for g in result.effector_sensor_gains:
        if not g.negligible:
            new_gains[(g.effector, g.sensor)] = g.gain_f_per_hour

    # Find significant changes (>50% change, non-negligible in at least one)
    changes: dict[tuple[str, str], tuple[float, float]] = {}
    all_keys = set(old_gains) | set(new_gains)
    for key in all_keys:
        old_g = old_gains.get(key, 0.0)
        new_g = new_gains.get(key, 0.0)
        if abs(old_g) < 0.05 and abs(new_g) < 0.05:
            continue  # both negligible
        if abs(old_g) > 1e-6:
            pct_change = abs(new_g - old_g) / abs(old_g)
        else:
            pct_change = float("inf") if abs(new_g) > 0.05 else 0.0
        if pct_change > 0.5:
            changes[key] = (old_g, new_g)

    return changes


def fit_sysid(verbose: bool = False) -> tuple[SysIdResult, dict[str, RegressionDiagnostics]]:
    """Run system identification and return the result + per-sensor diagnostics.

    Use save_sysid_result() to persist.
    """
    print("Loading collector snapshots...")
    df = load_collector_snapshots()
    print(f"  {len(df)} snapshots loaded")

    effectors = _enumerate_effectors()
    sensors = _enumerate_sensors()

    if verbose:
        print(f"\nEffectors ({len(effectors)}):")
        for e in effectors:
            print(f"  {e.name} ({e.device_type}): {e.state_column}")
        print(f"\nSensors ({len(sensors)}):")
        for s in sensors:
            print(f"  {s.name}: tau_default={s.yaml_tau_base}")

    print("\nPreprocessing...")
    df = _preprocess(df, effectors, sensors)

    ts = pd.to_datetime(df["timestamp"], format="ISO8601")
    data_start = str(ts.iloc[0])
    data_end = str(ts.iloc[-1])
    print(f"  Data range: {data_start} to {data_end}")

    # Stage 1: Fit tau_base (sealed envelope)
    print("\n── Stage 1: Fitting tau_base (sealed envelope loss) ──")
    fitted_taus = _fit_tau(df, effectors, sensors, verbose)
    for ft in fitted_taus:
        src = f"fitted ({ft.n_segments} seg)" if ft.n_segments > 0 else "yaml default"
        print(f"  {ft.sensor:30s}: tau_base={ft.tau_base:5.1f}h ({src})")

    # Build tau lookup
    tau_lookup: dict[str, float] = {}
    for ft in fitted_taus:
        tau_lookup[ft.sensor] = ft.tau_base

    # Stage 2: Regression per sensor (effector gains, solar, advisory effects)
    print("\n── Stage 2: Fitting effector gains, solar, and advisory effects ──")
    all_gains: list[EffectorSensorGain] = []
    solar_elevation_gains: dict[str, float] = {}  # sensor -> elevation-based gain
    all_adv_tau_betas: dict[str, dict[str, float]] = {}  # sensor -> {device -> beta}
    all_adv_solar_betas: dict[str, dict[str, float]] = {}  # sensor -> {device -> beta}
    sensor_diagnostics: dict[str, RegressionDiagnostics] = {}  # for health summary

    for sensor in sensors:
        tau_base = tau_lookup.get(sensor.name, sensor.yaml_tau_base)
        gains, solar_elev_gain, solar_elev_t, adv_tau_b, adv_solar_b, diag = _fit_sensor_model(
            df, sensor, effectors, tau_base, verbose,
        )
        all_gains.extend(gains)
        if diag is not None:
            sensor_diagnostics[sensor.name] = diag
        if solar_elev_gain != 0.0:
            solar_elevation_gains[sensor.name] = solar_elev_gain
            print(f"  {sensor.name}: solar_elev β={solar_elev_gain:+.3f}, t={solar_elev_t:.1f}")
        if adv_tau_b:
            all_adv_tau_betas[sensor.name] = adv_tau_b
        if adv_solar_b:
            all_adv_solar_betas[sensor.name] = adv_solar_b

        if verbose and gains:
            sig_gains = [g for g in gains if not g.negligible]
            if sig_gains:
                print(f"  {sensor.name}: {len(sig_gains)} significant effectors")
                for g in sig_gains:
                    print(
                        f"    {g.effector}: {g.gain_f_per_hour:+.3f} {UNIT_SYMBOL}/hr,"
                        f" lag={g.best_lag_minutes:.0f}min, t={g.t_statistic:.1f}"
                    )

    # Merge advisory tau betas into FittedTau
    for i, ft in enumerate(fitted_taus):
        tau_b = all_adv_tau_betas.get(ft.sensor, {})
        if tau_b:
            fitted_taus[i] = FittedTau(
                sensor=ft.sensor,
                tau_base=ft.tau_base,
                n_segments=ft.n_segments,
                environment_tau_betas=tau_b,
            )

    # Restructure environment_solar_betas: sensor->{device->beta} → device->{sensor->beta}
    environment_solar_betas: dict[str, dict[str, float]] = {}
    for sensor_name, dev_betas in all_adv_solar_betas.items():
        for dev_name, beta_val in dev_betas.items():
            environment_solar_betas.setdefault(dev_name, {})[sensor_name] = beta_val

    # Build state_gates from config
    state_gates: dict[str, StateGate] = {}
    for col, scfg in _CFG.state_sensors.items():
        state_gates[col] = StateGate(column=col, encoding=scfg.encoding)

    # MRT weights: no longer derived from solar gains — the MRT correction
    # now uses solar_elevation_gains dynamically (per current sun state).
    # Static mrt_weights are only set via manual YAML config, not sysid.

    sysid_result = SysIdResult(
        timestamp=datetime.now(UTC).isoformat(),
        data_start=data_start,
        data_end=data_end,
        n_snapshots=len(df),
        effectors=effectors,
        sensors=sensors,
        fitted_taus=fitted_taus,
        effector_sensor_gains=all_gains,
        solar_gains=[],  # legacy per-hour format, empty for elevation-based fits
        solar_elevation_gains=solar_elevation_gains,
        state_gates=state_gates,
        environment_solar_betas=environment_solar_betas,
    )
    return sysid_result, sensor_diagnostics


# ── Report printing ───────────────────────────────────────────────────────


def print_report(result: SysIdResult) -> None:
    """Print formatted sysid report."""
    print("\n" + "=" * 80)
    print("SYSTEM IDENTIFICATION REPORT")
    print("=" * 80)

    # Data summary
    print(f"\nData: {result.data_start} to {result.data_end} ({result.n_snapshots} snapshots)")

    # Tau fits
    print("\n── Envelope Loss (tau_base, hours) ──")
    sensor_map = {s.name: s for s in result.sensors}
    hdr = f"  {'Sensor':<30s}  {'tau_base':>8s}  {'Default':>8s}  {'Seg':>4s}  {'Advisory':>8s}"
    print(hdr)
    print("  " + "-" * 66)
    for ft in result.fitted_taus:
        s = sensor_map.get(ft.sensor)
        default_str = f"{s.yaml_tau_base:.1f}" if s else "?"
        n_adv = len(ft.environment_tau_betas)
        adv_str = f"{n_adv} β" if n_adv > 0 else "–"
        print(f"  {ft.sensor:<30s}  {ft.tau_base:8.1f}  {default_str:>8s}  {ft.n_segments:4d}  {adv_str:>8s}")

    # Advisory tau couplings (if any)
    any_couplings = any(ft.environment_tau_betas for ft in result.fitted_taus)
    if any_couplings:
        print("\n── Advisory Tau Couplings (β: additional cooling rate when active) ──")
        for ft in result.fitted_taus:
            if not ft.environment_tau_betas:
                continue
            print(f"\n  {ft.sensor}:")
            for dev, beta in sorted(ft.environment_tau_betas.items()):
                eff_tau = 1.0 / (1.0 / ft.tau_base + beta) if beta > 0 else ft.tau_base
                print(f"    {dev:20s}: β={beta:.6f}  (eff tau={eff_tau:.1f}h)")

    # Advisory solar couplings (if any)
    if result.environment_solar_betas:
        print("\n── Advisory Solar Couplings (β: solar gain modulation) ──")
        for dev, sensor_betas in sorted(result.environment_solar_betas.items()):
            print(f"\n  {dev}:")
            for sensor_name, beta in sorted(sensor_betas.items()):
                print(f"    {sensor_name:30s}: β={beta:+.4f}")

    # Effector x sensor gain matrix
    print(f"\n── Effector × Sensor Gain Matrix ({UNIT_SYMBOL}/hr, delay in parens) ──")
    sensor_names = [s.name for s in result.sensors]
    eff_names = list(dict.fromkeys(g.effector for g in result.effector_sensor_gains))

    # Build lookup
    gain_lookup: dict[tuple[str, str], EffectorSensorGain] = {}
    for g in result.effector_sensor_gains:
        gain_lookup[(g.effector, g.sensor)] = g

    # Abbreviate sensor names for display
    def _short(name: str) -> str:
        return name.removesuffix("_temp")[:12]

    # Print header
    header = f"  {'Effector':<25s}"
    for s in sensor_names:
        header += f"  {_short(s):>14s}"
    print(header)
    print("  " + "-" * (25 + 16 * len(sensor_names)))

    for eff in eff_names:
        row = f"  {eff:<25s}"
        for s in sensor_names:
            g = gain_lookup.get((eff, s))
            if g is None or g.negligible:
                row += f"  {'–':>14s}"
            else:
                cell = f"{g.gain_f_per_hour:+.2f}({g.best_lag_minutes:.0f}m)"
                row += f"  {cell:>14s}"
        print(row)

    # Solar elevation gains
    if result.solar_elevation_gains:
        print(f"\n── Solar Elevation Gains ({UNIT_SYMBOL}/hr per unit sin(elev)×fraction) ──")
        for sensor_name, gain in sorted(result.solar_elevation_gains.items(), key=lambda x: -x[1]):
            bar = "█" * max(0, int(gain * 3))
            print(f"  {sensor_name:<30s}  {gain:+.3f}  {bar}")

    # Legacy per-hour solar profiles (for old fits loaded from disk)
    if result.solar_gains:
        print(f"\n── Solar Gain Profiles (legacy per-hour, {UNIT_SYMBOL}/hr) ──")
        solar_by_sensor: dict[str, list[SolarGainProfile]] = {}
        for sg in result.solar_gains:
            solar_by_sensor.setdefault(sg.sensor, []).append(sg)
        for sensor_name, profiles in solar_by_sensor.items():
            sig = [p for p in profiles if abs(p.t_statistic) >= _T_STAT_THRESHOLD]
            if not sig:
                continue
            print(f"\n  {sensor_name}:")
            for p in sorted(profiles, key=lambda x: x.hour_of_day):
                marker = "*" if abs(p.t_statistic) >= _T_STAT_THRESHOLD else " "
                bar = "█" * max(0, int(p.gain_f_per_hour * 10))
                print(
                    f"    {p.hour_of_day:2d}:00  {p.gain_f_per_hour:+.3f} ±{p.std_error:.3f}"
                    f"  t={p.t_statistic:5.1f} {marker} {bar}"
                )

    # Effector activity summary
    print("\n── Effector Activity Summary ──")
    for eff in result.effectors:
        # Count non-negligible gains for this effector
        eff_gains = [g for g in result.effector_sensor_gains if g.effector == eff.name and not g.negligible]
        print(f"  {eff.name} ({eff.device_type}): {len(eff_gains)} significant sensor effects")


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="System identification: extract thermal parameters")
    parser.add_argument("--output", type=Path, help="Output JSON path (default: data/thermal_params.json)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    result, diagnostics = run_sysid(output_path=args.output, verbose=args.verbose)
    print_report(result)
    if diagnostics:
        _print_health_summary(result, diagnostics, args.output)


if __name__ == "__main__":
    main()
