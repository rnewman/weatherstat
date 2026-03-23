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

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from weatherstat.config import DATA_DIR
from weatherstat.extract import load_collector_snapshots
from weatherstat.yaml_config import load_config

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

    tau_base: sealed envelope time constant (all windows closed).
    window_betas: per-window additional cooling rate coefficients,
        learned by regression in Stage 2.
    interaction_betas: cross-breeze coefficients for window pairs.
    """

    sensor: str
    tau_base: float
    n_segments: int
    window_betas: dict[str, float] = field(default_factory=dict)
    interaction_betas: dict[str, float] = field(default_factory=dict)


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
    solar_gains: list[SolarGainProfile]
    state_gates: dict[str, StateGate] = field(default_factory=dict)
    mrt_weights: dict[str, float] = field(default_factory=dict)


# ── Stage 0: Enumerate effectors and sensors from config ──────────────────


def _resolve_gate(device_name: str) -> str | None:
    """Resolve a state_device name to the state sensor column, or None."""
    if device_name in _CFG.state_sensors:
        return device_name
    return None


def _enumerate_effectors() -> list[EffectorSpec]:
    """Build effector list from YAML config."""
    effectors: list[EffectorSpec] = []

    for name, therm_cfg in _CFG.thermostats.items():
        state_gate = _resolve_gate(therm_cfg.state_device) if therm_cfg.state_device else None
        effectors.append(EffectorSpec(
            name=f"thermostat_{name}",
            state_column=f"thermostat_{name}_action",
            encoding={"heating": 1.0, "idle": 0.0, "off": 0.0},
            max_lag_minutes=90,
            device_type="thermostat",
            state_gate=state_gate,
        ))

    for name, cfg in _CFG.mini_splits.items():
        if cfg.state_encoding:
            # Prefer measured state (hvac_action) over command (hvac_mode)
            effectors.append(EffectorSpec(
                name=f"mini_split_{name}",
                state_column=f"mini_split_{name}_action",
                encoding=cfg.state_encoding,
                max_lag_minutes=15,
                device_type="mini_split",
                command_column=f"mini_split_{name}_mode",
                command_encoding=cfg.command_encoding,
            ))
        else:
            effectors.append(EffectorSpec(
                name=f"mini_split_{name}",
                state_column=f"mini_split_{name}_mode",
                encoding={str(k): float(v) for k, v in cfg.command_encoding.items()},
                max_lag_minutes=15,
                device_type="mini_split",
            ))

    for name, cfg in _CFG.blowers.items():
        effectors.append(EffectorSpec(
            name=f"blower_{name}",
            state_column=f"blower_{name}_mode",
            encoding={str(k): float(v) for k, v in cfg.level_encoding.items()},
            max_lag_minutes=5,
            device_type="blower",
        ))

    return effectors


def _enumerate_sensors() -> list[SensorSpec]:
    """Build sensor list from YAML config.

    Tau uses defaults.tau as the initial guess for curve_fit. Sysid fits
    the actual value from data. Window effects are learned in Stage 2
    regression — no configured window→sensor mapping needed.
    """
    sensors: list[SensorSpec] = []

    for col_name in _CFG.temp_sensors:
        if col_name == "outdoor_temp":
            continue

        sensors.append(SensorSpec(
            name=col_name,
            temp_column=col_name,
            yaml_tau_base=_CFG.default_tau,
        ))

    return sensors


# ── Stage 1: Fit tau per sensor ───────────────────────────────────────────

_MIN_SEGMENT_STEPS = 12  # 1 hour at 5-min intervals
_NIGHTTIME_START_HOUR = 22  # 10pm local
_NIGHTTIME_END_HOUR = 6  # 6am local


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
    """Find contiguous nighttime segments with no active HVAC control.

    These segments show passive convergence toward outdoor temperature —
    cooling when indoors is warmer, warming when outdoors is warmer.
    The exponential time constant (tau) is symmetric either way.

    Requires all HVAC effectors to be off. Window/door states must be
    constant within each segment (no transitions) but need not all be closed.
    The fitted tau includes whatever window effects are present; Stage 2
    regression decomposes them via window×ΔT features.
    """
    # Identify nighttime rows
    local_hour = df["_local_hour"]
    is_night = (local_hour >= _NIGHTTIME_START_HOUR) | (local_hour < _NIGHTTIME_END_HOUR)

    # Identify all-HVAC-off rows
    all_off = pd.Series(True, index=df.index)
    for eff in effectors:
        if eff.state_column not in df.columns:
            continue
        enc_col = f"_eff_{eff.name}"
        if enc_col in df.columns:
            all_off &= df[enc_col] == 0.0

    mask = is_night & all_off

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
    all_win_cols = [f"window_{name}_open" for name in _CFG.windows]
    existing_win_cols = [c for c in all_win_cols if c in df.columns]

    segments: list[pd.DataFrame] = []
    for _, seg_df in qualifying.groupby(seg_ids):
        if len(seg_df) < _MIN_SEGMENT_STEPS:
            continue
        # Check window stability: each column must have at most 1 unique
        # non-null value within the segment
        stable = True
        for wc in existing_win_cols:
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
    """Stage 1: Fit tau per sensor from uncontrolled overnight segments.

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

        # Fit tau from each sealed segment
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

    # Best outdoor temp: use met.no (ERA5/NWP) exclusively when available.
    # The house-mounted side sensor reads ~3°F warm at all hours (house heat
    # radiation through the wall) plus solar spikes of +10-12°F in the
    # afternoon. Met.no data has no such biases.
    # Backfill data is hourly; interpolate to fill 5-min gaps.
    # Fall back to side sensor only for rows with no met coverage at all.
    if "met_outdoor_temp" in df.columns and df["met_outdoor_temp"].notna().sum() > 100:
        met = pd.to_numeric(df["met_outdoor_temp"], errors="coerce")
        met = met.interpolate(method="linear", limit=24)  # fill up to 2h gaps
        n_met = met.notna().sum()
        n_fallback = met.isna().sum()
        df["_outdoor_best"] = met.fillna(pd.to_numeric(df["outdoor_temp"], errors="coerce"))
        if n_fallback > 0:
            print(f"  Outdoor temp: met.no for {n_met}/{len(df)} rows, side sensor fallback for {n_fallback}")
        else:
            print(f"  Outdoor temp: met.no for all {n_met} rows")
    else:
        df["_outdoor_best"] = pd.to_numeric(df["outdoor_temp"], errors="coerce")

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

    # Compute dT/dt for each sensor (°F/hr, central differences)
    dt_hours = 5.0 / 60.0  # 5-minute intervals
    for sensor in sensors:
        col = sensor.temp_column
        if col not in df.columns:
            continue
        temps = df[col].values.astype(float)
        dT = np.full_like(temps, np.nan)
        # Central differences for interior points
        dT[1:-1] = (temps[2:] - temps[:-2]) / (2 * dt_hours)
        # Forward/backward at edges
        dT[0] = (temps[1] - temps[0]) / dt_hours if len(temps) > 1 else 0.0
        dT[-1] = (temps[-1] - temps[-2]) / dt_hours if len(temps) > 1 else 0.0
        df[f"_dTdt_{sensor.name}"] = dT

    # Generate lagged effector features in coarse bins
    for eff in effectors:
        eff_col = f"_eff_{eff.name}"
        if eff.device_type == "thermostat":
            # Floor heat: 0-15min, 15-30min, 30-60min, 60-90min
            bins = [(0, 3), (3, 6), (6, 12), (12, 18)]  # in 5-min steps
            bin_labels = ["0_15", "15_30", "30_60", "60_90"]
        elif eff.device_type == "mini_split":
            # Mini splits: 0-5min, 5-15min
            bins = [(0, 1), (1, 3)]
            bin_labels = ["0_5", "5_15"]
        else:
            # Blowers: 0-5min (immediate)
            bins = [(0, 1)]
            bin_labels = ["0_5"]

        for (start, end), label in zip(bins, bin_labels, strict=True):
            # Mean activity in the lag bin [start, end) steps back
            lagged_sum = pd.Series(0.0, index=df.index)
            count = 0
            for lag in range(start, end):
                shifted = df[eff_col].shift(lag)
                lagged_sum += shifted.fillna(0.0)
                count += 1
            df[f"_lag_{eff.name}_{label}"] = lagged_sum / max(count, 1)

    # Weather-conditioned solar features (hours 7-17).
    # Each feature = (is_hour_H) × solar_fraction, so the regression learns
    # the gain per unit of clear sky at each hour. On a cloudy day the feature
    # is 0.15 instead of 1.0; on a sunny day it's 1.0.
    from weatherstat.weather import condition_to_solar_fraction

    if "weather_condition" in df.columns:
        solar_frac = df["weather_condition"].map(condition_to_solar_fraction).fillna(0.3)
    else:
        solar_frac = pd.Series(0.3, index=df.index)

    for h in range(7, 18):
        df[f"_solar_h{h}"] = ((df["_local_hour"] == h).astype(float) * solar_frac)

    return df


# ── Stage 2: Regression per sensor ────────────────────────────────────────

_GAIN_THRESHOLD = 0.05  # °F/hr — below this, gain is negligible
_T_STAT_THRESHOLD = 2.0
_MIN_REGRESSION_ROWS = 500  # need substantial data for reliable regression
_MIN_INTERACTION_ROWS = 100  # min co-open rows for window interaction terms

def _get_lag_columns(eff: EffectorSpec) -> list[str]:
    """Get lag column names for an effector."""
    if eff.device_type == "thermostat":
        return [f"_lag_{eff.name}_{b}" for b in ("0_15", "15_30", "30_60", "60_90")]
    elif eff.device_type == "mini_split":
        return [f"_lag_{eff.name}_{b}" for b in ("0_5", "5_15")]
    else:
        return [f"_lag_{eff.name}_0_5"]


def _lag_label_to_minutes(label: str) -> float:
    """Convert lag bin label to midpoint in minutes."""
    mapping = {
        "0_15": 7.5, "15_30": 22.5, "30_60": 45.0, "60_90": 75.0,
        "0_5": 2.5, "5_15": 10.0,
    }
    return mapping.get(label, 0.0)


def _fit_sensor_model(
    df: pd.DataFrame,
    sensor: SensorSpec,
    effectors: list[EffectorSpec],
    tau_base: float,
    verbose: bool = False,
) -> tuple[list[EffectorSensorGain], list[SolarGainProfile], dict[str, float], dict[str, float]]:
    """Stage 2: Fit regression for one sensor.

    Returns (gains, solar_profiles, window_betas, interaction_betas).
    Window betas are per-window additional cooling rate coefficients,
    learned as regression coefficients on window_state × (T_out - T).
    """
    temp_col = sensor.temp_column
    dTdt_col = f"_dTdt_{sensor.name}"

    if temp_col not in df.columns or dTdt_col not in df.columns:
        return [], [], {}, {}
    if "_outdoor_best" not in df.columns:
        return [], [], {}, {}

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

    # Window × ΔT features: window_state × (T_outdoor - T_sensor)
    # The coefficient β_w gives the additional cooling rate when the window is open.
    all_win_cols = [f"window_{name}_open" for name in _CFG.windows]
    existing_win_cols = [c for c in all_win_cols if c in df.columns]

    window_feature_start = len(feature_names)
    window_feature_names: list[str] = []  # bare window names in order
    for wc in existing_win_cols:
        win_name = wc.removeprefix("window_").removesuffix("_open")
        feature_names.append(f"_win_{win_name}")
        feature_cols.append(df[wc].fillna(False).values.astype(float) * delta_t)
        window_feature_names.append(win_name)

    # Window interaction features: pairs with enough co-open data
    interaction_feature_start = len(feature_names)
    interaction_feature_names: list[str] = []  # "w1+w2" in order
    for i, wc1 in enumerate(existing_win_cols):
        for wc2 in existing_win_cols[i + 1:]:
            co_open = df[wc1].fillna(False).values.astype(bool) & df[wc2].fillna(False).values.astype(bool)
            if co_open.sum() >= _MIN_INTERACTION_ROWS:
                w1 = wc1.removeprefix("window_").removesuffix("_open")
                w2 = wc2.removeprefix("window_").removesuffix("_open")
                feature_names.append(f"_winx_{w1}+{w2}")
                feature_cols.append(co_open.astype(float) * delta_t)
                interaction_feature_names.append(f"{w1}+{w2}")

    # Solar hour indicators
    solar_start = len(feature_names)
    for h in range(7, 18):
        col_name = f"_solar_h{h}"
        if col_name in df.columns:
            feature_names.append(col_name)
            feature_cols.append(df[col_name].values.astype(float))
    if not feature_cols:
        return [], [], {}, {}

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
        return gains, [], {}, {}

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
        if name.startswith("_solar_") or name.startswith("_win_") or name.startswith("_winx_"):
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

    # Report weather control feature coefficients (diagnostic, not stored)
    if verbose:
        for i, fname in enumerate(feature_names):
            if fname.startswith("_weather_"):
                label = fname.removeprefix("_weather_")
                print(f"    weather {label}: β={beta[i]:.6f}, t={t_stats[i]:.1f}")

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

    # Extract window betas
    # β_w > 0 is physical (window increases heat exchange rate).
    # β_w < 0 is unphysical — flag as zero.
    window_betas: dict[str, float] = {}
    for i, win_name in enumerate(window_feature_names):
        idx = window_feature_start + i
        if idx < len(beta):
            b = float(beta[idx])
            t = float(t_stats[idx])
            # Only keep physically sensible (positive) and statistically significant
            if b > 0 and abs(t) >= _T_STAT_THRESHOLD:
                window_betas[win_name] = round(b, 6)
            elif verbose:
                print(f"    window {win_name}: β={b:.6f}, t={t:.1f} (dropped)")

    interaction_betas: dict[str, float] = {}
    for i, pair_name in enumerate(interaction_feature_names):
        idx = interaction_feature_start + i
        if idx < len(beta):
            b = float(beta[idx])
            t = float(t_stats[idx])
            if b > 0 and abs(t) >= _T_STAT_THRESHOLD:
                interaction_betas[pair_name] = round(b, 6)

    # Extract solar profile
    solar_profiles: list[SolarGainProfile] = []
    for i, h in enumerate(range(7, 18)):
        idx = solar_start + i
        if idx < len(beta):
            solar_profiles.append(SolarGainProfile(
                sensor=sensor.name,
                hour_of_day=h,
                gain_f_per_hour=round(float(beta[idx]), 4),
                std_error=round(float(se[idx]), 4) if not np.isnan(se[idx]) else 0.0,
                t_statistic=round(float(t_stats[idx]), 2),
            ))

    return gains, solar_profiles, window_betas, interaction_betas


# ── Derived MRT weights ──────────────────────────────────────────────────


def _compute_mrt_weights(
    solar_gains: list[SolarGainProfile],
    constrained_sensors: list[str],
) -> dict[str, float]:
    """Derive per-sensor MRT weights from solar gain profiles.

    Sensors with high total daily solar gain get weight < 1 (less MRT correction
    needed because sun warms surfaces). Sensors with zero solar gain get weight > 1
    (cold surfaces dominate, more MRT correction needed).

    Weight is centered around 1.0: sensor at mean solar → 1.0.
    Clamped to [0.3, 2.0].
    """
    # Sum significant solar gains per sensor (hours 7-17)
    totals: dict[str, float] = {}
    for sg in solar_gains:
        if sg.sensor not in constrained_sensors:
            continue
        if abs(sg.t_statistic) < _T_STAT_THRESHOLD:
            continue
        if sg.gain_f_per_hour > 0:  # only positive (warming) solar gains
            totals[sg.sensor] = totals.get(sg.sensor, 0.0) + sg.gain_f_per_hour

    # Mean across sensors with nonzero gain
    nonzero = [v for v in totals.values() if v > 0]
    if not nonzero:
        return {}  # no solar data → no derived weights

    mean_solar = sum(nonzero) / len(nonzero)

    weights: dict[str, float] = {}
    for sensor in constrained_sensors:
        total = totals.get(sensor, 0.0)
        ratio = total / mean_solar if mean_solar > 0 else 0.0
        # Invert: high solar → low weight, zero solar → high weight
        raw_weight = 2.0 - ratio
        weights[sensor] = max(0.3, min(2.0, raw_weight))

    return weights


# ── Main pipeline ─────────────────────────────────────────────────────────


def run_sysid(output_path: Path | None = None, verbose: bool = False) -> SysIdResult:
    """Run full system identification pipeline."""
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

    # Stage 2: Regression per sensor (effector gains, solar, window effects)
    print("\n── Stage 2: Fitting effector gains, solar, and window effects ──")
    all_gains: list[EffectorSensorGain] = []
    all_solar: list[SolarGainProfile] = []
    all_window_betas: dict[str, dict[str, float]] = {}  # sensor -> {window -> beta}
    all_interaction_betas: dict[str, dict[str, float]] = {}  # sensor -> {"w1+w2" -> beta}

    for sensor in sensors:
        tau_base = tau_lookup.get(sensor.name, sensor.yaml_tau_base)
        gains, solar, win_betas, int_betas = _fit_sensor_model(df, sensor, effectors, tau_base, verbose)
        all_gains.extend(gains)
        all_solar.extend(solar)
        if win_betas:
            all_window_betas[sensor.name] = win_betas
        if int_betas:
            all_interaction_betas[sensor.name] = int_betas

        if verbose and gains:
            sig_gains = [g for g in gains if not g.negligible]
            if sig_gains:
                print(f"  {sensor.name}: {len(sig_gains)} significant effectors")
                for g in sig_gains:
                    print(
                        f"    {g.effector}: {g.gain_f_per_hour:+.3f} °F/hr,"
                        f" lag={g.best_lag_minutes:.0f}min, t={g.t_statistic:.1f}"
                    )

    # Merge window betas into FittedTau
    for i, ft in enumerate(fitted_taus):
        win_b = all_window_betas.get(ft.sensor, {})
        int_b = all_interaction_betas.get(ft.sensor, {})
        if win_b or int_b:
            fitted_taus[i] = FittedTau(
                sensor=ft.sensor,
                tau_base=ft.tau_base,
                n_segments=ft.n_segments,
                window_betas=win_b,
                interaction_betas=int_b,
            )

    # Build state_gates from config
    state_gates: dict[str, StateGate] = {}
    for col, scfg in _CFG.state_sensors.items():
        state_gates[col] = StateGate(column=col, encoding=scfg.encoding)

    # Derive per-sensor MRT weights from solar gain profiles
    constrained_sensors = [c.sensor for c in _CFG.constraints]
    mrt_weights = _compute_mrt_weights(all_solar, constrained_sensors)

    result = SysIdResult(
        timestamp=datetime.now(UTC).isoformat(),
        data_start=data_start,
        data_end=data_end,
        n_snapshots=len(df),
        effectors=effectors,
        sensors=sensors,
        fitted_taus=fitted_taus,
        effector_sensor_gains=all_gains,
        solar_gains=all_solar,
        state_gates=state_gates,
        mrt_weights=mrt_weights,
    )

    # Save output
    out = output_path or DATA_DIR / "thermal_params.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    def _serialize(obj: object) -> object:
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)  # type: ignore[arg-type]
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(out, "w") as f:
        json.dump(asdict(result), f, indent=2, default=_serialize)
    print(f"\nResults written to {out}")

    return result


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
    hdr = f"  {'Sensor':<30s}  {'tau_base':>8s}  {'Default':>8s}  {'Seg':>4s}  {'Windows':>8s}"
    print(hdr)
    print("  " + "-" * 66)
    for ft in result.fitted_taus:
        s = sensor_map.get(ft.sensor)
        default_str = f"{s.yaml_tau_base:.1f}" if s else "?"
        n_win = len(ft.window_betas)
        win_str = f"{n_win} β" if n_win > 0 else "–"
        print(f"  {ft.sensor:<30s}  {ft.tau_base:8.1f}  {default_str:>8s}  {ft.n_segments:4d}  {win_str:>8s}")

    # Window couplings (if any)
    any_couplings = any(ft.window_betas for ft in result.fitted_taus)
    if any_couplings:
        print("\n── Window Couplings (β_w: additional cooling rate when open) ──")
        for ft in result.fitted_taus:
            if not ft.window_betas and not ft.interaction_betas:
                continue
            print(f"\n  {ft.sensor}:")
            for win, beta in sorted(ft.window_betas.items()):
                # Show effective tau when this window is open alone
                eff_tau = 1.0 / (1.0 / ft.tau_base + beta) if beta > 0 else ft.tau_base
                print(f"    {win:20s}: β={beta:.6f}  (eff tau={eff_tau:.1f}h)")
            for pair, beta in sorted(ft.interaction_betas.items()):
                print(f"    {pair:20s}: β={beta:.6f}  (interaction)")

    # Effector x sensor gain matrix
    print("\n── Effector × Sensor Gain Matrix (°F/hr, delay in parens) ──")
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

    # Solar profiles (only sensors with significant solar gain)
    print("\n── Solar Gain Profiles (°F/hr by hour, significant only) ──")
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

    # MRT weights (derived from solar profiles)
    if result.mrt_weights:
        print("\n── Derived MRT Weights (per-sensor, from solar gain) ──")
        for sensor_name, weight in sorted(result.mrt_weights.items()):
            label = sensor_name.removeprefix("thermostat_").removesuffix("_temp")
            bar = "◀" if weight < 1.0 else ("▶" if weight > 1.0 else "=")
            print(f"  {label:<25s}  {weight:.2f}  {bar}")

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

    result = run_sysid(output_path=args.output, verbose=args.verbose)
    print_report(result)


if __name__ == "__main__":
    main()
