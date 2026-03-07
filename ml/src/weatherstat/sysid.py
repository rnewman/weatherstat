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
from dataclasses import asdict, dataclass
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
    """An effector (HVAC device) derived from config."""

    name: str
    state_column: str
    encoding: dict[str, float]
    max_lag_minutes: int
    device_type: str


@dataclass(frozen=True)
class SensorSpec:
    """A temperature sensor derived from config."""

    name: str
    temp_column: str
    window_columns: list[str]
    yaml_tau_sealed: float
    yaml_tau_ventilated: float


@dataclass(frozen=True)
class FittedTau:
    """Envelope loss rate fitted from overnight cooling data."""

    sensor: str
    tau_sealed: float
    tau_ventilated: float | None
    n_segments_sealed: int
    n_segments_ventilated: int


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


# ── Stage 0: Enumerate effectors and sensors from config ──────────────────


def _enumerate_effectors() -> list[EffectorSpec]:
    """Build effector list from YAML config."""
    effectors: list[EffectorSpec] = []

    for name in _CFG.thermostats:
        effectors.append(EffectorSpec(
            name=f"thermostat_{name}",
            state_column=f"thermostat_{name}_action",
            encoding={"heating": 1.0, "idle": 0.0, "off": 0.0},
            max_lag_minutes=90,
            device_type="thermostat",
        ))

    for name, cfg in _CFG.mini_splits.items():
        effectors.append(EffectorSpec(
            name=f"mini_split_{name}",
            state_column=f"mini_split_{name}_mode",
            encoding={str(k): float(v) for k, v in cfg.mode_encoding.items()},
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

    effectors.append(EffectorSpec(
        name="navien",
        state_column="navien_heating_mode",
        encoding={str(k): float(v) for k, v in _CFG.boiler.mode_encoding.items()},
        max_lag_minutes=90,
        device_type="boiler",
    ))

    return effectors


def _enumerate_sensors() -> list[SensorSpec]:
    """Build sensor list from YAML config."""
    # Build room -> window columns mapping
    room_windows: dict[str, list[str]] = {}
    for win_name, win_cfg in _CFG.windows.items():
        col = f"window_{win_name}_open"
        for room in win_cfg.rooms:
            room_windows.setdefault(room, []).append(col)

    thermal = _CFG.thermal
    sensors: list[SensorSpec] = []

    for col_name in _CFG.temp_sensors:
        if col_name == "outdoor_temp":
            continue

        # Try to find a room name for this sensor (for tau and window lookup)
        room_name = None
        for rname, rcfg in _CFG.rooms.items():
            if rcfg.temp_column == col_name:
                room_name = rname
                break
        # Also check if the column name itself matches a room key pattern
        if room_name is None:
            # e.g., "bedroom_temp" -> "bedroom"
            candidate = col_name.removesuffix("_temp")
            if candidate in _CFG.rooms:
                room_name = candidate

        window_cols = room_windows.get(room_name, []) if room_name else []
        if room_name:
            tau_s = thermal.tau_sealed.get(room_name, thermal.default_tau_sealed)
            tau_v = thermal.tau_ventilated.get(room_name, thermal.default_tau_ventilated)
        else:
            tau_s = thermal.default_tau_sealed
            tau_v = thermal.default_tau_ventilated

        sensors.append(SensorSpec(
            name=col_name,
            temp_column=col_name,
            window_columns=window_cols,
            yaml_tau_sealed=tau_s,
            yaml_tau_ventilated=tau_v,
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


def _find_nighttime_hvac_off_segments(
    df: pd.DataFrame,
    effectors: list[EffectorSpec],
    sensor: SensorSpec,
    verbose: bool = False,
) -> tuple[list[tuple[pd.DataFrame, bool]], int]:
    """Find contiguous nighttime HVAC-off segments for a sensor.

    Returns:
        (segments, total_count) where each segment is (sub_df, is_ventilated)
        and total_count is number of qualifying segments found.
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
        return [], 0
    mask &= df[temp_col].notna() & df["outdoor_temp"].notna()

    qualifying = df[mask].copy()
    if len(qualifying) < _MIN_SEGMENT_STEPS:
        return [], 0

    # Split into contiguous segments (gaps > 10 min break segments)
    dt = qualifying["_ts"].diff()
    breaks = dt > pd.Timedelta(minutes=10)

    # Also break on window state changes
    window_cols = [c for c in sensor.window_columns if c in df.columns]
    any_ventilated_col = None
    if window_cols:
        any_ventilated_col = "_vent_" + sensor.name
        qualifying[any_ventilated_col] = qualifying[window_cols].any(axis=1)
        vent_changes = qualifying[any_ventilated_col].diff().abs() > 0
        breaks |= vent_changes

    seg_ids = breaks.cumsum()
    segments: list[tuple[pd.DataFrame, bool]] = []
    for _, seg_df in qualifying.groupby(seg_ids):
        if len(seg_df) < _MIN_SEGMENT_STEPS:
            continue
        is_vent = False
        if any_ventilated_col and any_ventilated_col in seg_df.columns:
            is_vent = bool(seg_df[any_ventilated_col].iloc[0])
        segments.append((seg_df, is_vent))

    return segments, len(segments)


def _fit_tau(
    df: pd.DataFrame,
    effectors: list[EffectorSpec],
    sensors: list[SensorSpec],
    verbose: bool = False,
) -> list[FittedTau]:
    """Stage 1: Fit tau per sensor from overnight cooling data."""
    results: list[FittedTau] = []

    for sensor in sensors:
        segments, n_total = _find_nighttime_hvac_off_segments(df, effectors, sensor, verbose)
        if not segments:
            if verbose:
                print(f"  {sensor.name}: no qualifying overnight segments")
            # Use YAML defaults
            results.append(FittedTau(
                sensor=sensor.name,
                tau_sealed=sensor.yaml_tau_sealed,
                tau_ventilated=sensor.yaml_tau_ventilated,
                n_segments_sealed=0,
                n_segments_ventilated=0,
            ))
            continue

        # Fit sealed and ventilated separately
        sealed_taus: list[tuple[float, int]] = []  # (tau, segment_length)
        vent_taus: list[tuple[float, int]] = []

        for seg_df, is_vent in segments:
            t_hours = (seg_df["_ts"] - seg_df["_ts"].iloc[0]).dt.total_seconds().values / 3600.0
            temps = seg_df[sensor.temp_column].values
            t_outdoor = seg_df["outdoor_temp"].mean()
            tau = _fit_tau_curve(t_hours, temps, t_outdoor)
            if tau is None:
                continue
            target = vent_taus if is_vent else sealed_taus
            target.append((tau, len(seg_df)))

        # Weighted median by segment length
        tau_sealed = _weighted_median(sealed_taus) if sealed_taus else sensor.yaml_tau_sealed
        tau_vent = _weighted_median(vent_taus) if vent_taus else None

        results.append(FittedTau(
            sensor=sensor.name,
            tau_sealed=tau_sealed,
            tau_ventilated=tau_vent,
            n_segments_sealed=len(sealed_taus),
            n_segments_ventilated=len(vent_taus),
        ))

    # Sanity check: ventilated tau must be less than sealed tau.
    # If not (e.g., window sensor stuck open), discard the ventilated fit.
    for i, ft in enumerate(results):
        if ft.tau_ventilated is not None and ft.tau_sealed > 0 and ft.tau_ventilated > ft.tau_sealed:
            results[i] = FittedTau(
                sensor=ft.sensor,
                tau_sealed=ft.tau_sealed,
                tau_ventilated=None,
                n_segments_sealed=ft.n_segments_sealed,
                n_segments_ventilated=0,
            )

    # Estimate ventilated tau for sensors without measured data using ratio
    # from a sensor that has both
    ratio = _estimate_vent_ratio(results)
    for i, ft in enumerate(results):
        if ft.tau_ventilated is None:
            results[i] = FittedTau(
                sensor=ft.sensor,
                tau_sealed=ft.tau_sealed,
                tau_ventilated=round(ft.tau_sealed * ratio, 1),
                n_segments_sealed=ft.n_segments_sealed,
                n_segments_ventilated=0,
            )

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


def _estimate_vent_ratio(fitted: list[FittedTau]) -> float:
    """Estimate ventilated/sealed ratio from sensors that have both."""
    ratios: list[float] = []
    for ft in fitted:
        if ft.tau_ventilated is not None and ft.n_segments_ventilated > 0 and ft.tau_sealed > 0:
            ratios.append(ft.tau_ventilated / ft.tau_sealed)
    return float(np.median(ratios)) if ratios else 0.44


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

    # Encode effector states to numeric
    for eff in effectors:
        col = eff.state_column
        if col in df.columns:
            df[f"_eff_{eff.name}"] = df[col].map(eff.encoding).fillna(0.0).astype(float)
        else:
            df[f"_eff_{eff.name}"] = 0.0

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
        if eff.device_type == "thermostat" or eff.device_type == "boiler":
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

    # Hour-of-day indicators for solar (hours 7-17)
    for h in range(7, 18):
        df[f"_solar_h{h}"] = (df["_local_hour"] == h).astype(float)

    return df


# ── Stage 2: Regression per sensor ────────────────────────────────────────

_GAIN_THRESHOLD = 0.05  # °F/hr — below this, gain is negligible
_T_STAT_THRESHOLD = 2.0
_MIN_REGRESSION_ROWS = 500  # need substantial data for reliable regression


def _get_lag_columns(eff: EffectorSpec) -> list[str]:
    """Get lag column names for an effector."""
    if eff.device_type in ("thermostat", "boiler"):
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
    tau: float,
    tau_vent: float,
    verbose: bool = False,
) -> tuple[list[EffectorSensorGain], list[SolarGainProfile]]:
    """Stage 2: Fit regression for one sensor."""
    temp_col = sensor.temp_column
    dTdt_col = f"_dTdt_{sensor.name}"

    if temp_col not in df.columns or dTdt_col not in df.columns:
        return [], []
    if "outdoor_temp" not in df.columns:
        return [], []

    # Determine tau per row (sealed vs ventilated)
    window_cols = [c for c in sensor.window_columns if c in df.columns]
    if window_cols:
        is_vent = df[window_cols].any(axis=1)
        tau_per_row = np.where(is_vent, tau_vent, tau)
    else:
        tau_per_row = np.full(len(df), tau)

    # Newton residual: observed dT/dt minus expected passive cooling
    newton_dTdt = (df["outdoor_temp"].values - df[temp_col].values) / tau_per_row
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

    # Solar hour indicators
    solar_start = len(feature_names)
    for h in range(7, 18):
        col_name = f"_solar_h{h}"
        if col_name in df.columns:
            feature_names.append(col_name)
            feature_cols.append(df[col_name].values.astype(float))
    if not feature_cols:
        return [], []

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
        # Return all-negligible gains
        gains = [
            EffectorSensorGain(
                effector=e.name, sensor=sensor.name,
                gain_f_per_hour=0.0, best_lag_minutes=0.0,
                t_statistic=0.0, negligible=True,
            )
            for e in effectors
        ]
        return gains, []

    # Check condition number for collinearity
    cond = np.linalg.cond(X)
    use_ridge = cond > 1e6

    if use_ridge:
        # Ridge regression: (X'X + λI)^-1 X'y
        lam = 0.01 * len(y)
        XtX = X.T @ X + lam * np.eye(X.shape[1])
        beta = np.linalg.solve(XtX, X.T @ y)
        # Approximate standard errors
        resid = y - X @ beta
        s2 = np.sum(resid**2) / max(len(y) - X.shape[1], 1)
        try:
            cov = s2 * np.linalg.inv(XtX)
            se = np.sqrt(np.abs(np.diag(cov)))
        except np.linalg.LinAlgError:
            se = np.full(len(beta), np.nan)
    else:
        # OLS: numpy lstsq
        beta, residuals_sum, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        s2 = np.sum(resid**2) / max(len(y) - X.shape[1], 1)
        try:
            cov = s2 * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.abs(np.diag(cov)))
        except np.linalg.LinAlgError:
            se = np.full(len(beta), np.nan)

    t_stats = np.where(se > 0, beta / se, 0.0)

    # Extract effector gains
    gains: list[EffectorSensorGain] = []
    for eff in effectors:
        if eff.name not in effector_feature_ranges:
            # Effector had no data columns
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

        # Best lag = bin with largest |beta|
        best_idx = int(np.argmax(np.abs(lag_betas)))
        best_lag_label = lag_cols_used[best_idx].split("_")[-2] + "_" + lag_cols_used[best_idx].split("_")[-1]
        best_lag_min = _lag_label_to_minutes(best_lag_label)

        # Total gain = sum of betas across lag bins
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

    return gains, solar_profiles


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
            print(f"  {s.name}: windows={s.window_columns}")

    print("\nPreprocessing...")
    df = _preprocess(df, effectors, sensors)

    ts = pd.to_datetime(df["timestamp"], format="ISO8601")
    data_start = str(ts.iloc[0])
    data_end = str(ts.iloc[-1])
    print(f"  Data range: {data_start} to {data_end}")

    # Stage 1: Fit tau
    print("\n── Stage 1: Fitting tau (envelope loss) ──")
    fitted_taus = _fit_tau(df, effectors, sensors, verbose)
    for ft in fitted_taus:
        src_s = f"fitted ({ft.n_segments_sealed} seg)" if ft.n_segments_sealed > 0 else "yaml default"
        src_v = f"fitted ({ft.n_segments_ventilated} seg)" if ft.n_segments_ventilated > 0 else "estimated"
        v_str = f"{ft.tau_ventilated:.1f}" if ft.tau_ventilated is not None else "N/A"
        print(f"  {ft.sensor:30s}: sealed={ft.tau_sealed:5.1f}h ({src_s}), vent={v_str}h ({src_v})")

    # Build tau lookup
    tau_lookup: dict[str, tuple[float, float]] = {}
    for ft in fitted_taus:
        tau_lookup[ft.sensor] = (ft.tau_sealed, ft.tau_ventilated or ft.tau_sealed * 0.44)

    # Stage 2: Regression per sensor
    print("\n── Stage 2: Fitting effector gains and solar profiles ──")
    all_gains: list[EffectorSensorGain] = []
    all_solar: list[SolarGainProfile] = []

    for sensor in sensors:
        tau_s, tau_v = tau_lookup.get(sensor.name, (sensor.yaml_tau_sealed, sensor.yaml_tau_ventilated))
        gains, solar = _fit_sensor_model(df, sensor, effectors, tau_s, tau_v, verbose)
        all_gains.extend(gains)
        all_solar.extend(solar)

        if verbose and gains:
            sig_gains = [g for g in gains if not g.negligible]
            if sig_gains:
                print(f"  {sensor.name}: {len(sig_gains)} significant effectors")
                for g in sig_gains:
                    print(
                        f"    {g.effector}: {g.gain_f_per_hour:+.3f} °F/hr,"
                        f" lag={g.best_lag_minutes:.0f}min, t={g.t_statistic:.1f}"
                    )

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
    print("\n── Envelope Loss (tau, hours) ──")
    hdr = f"  {'Sensor':<30s}  {'Sealed':>8s}  {'Vent':>8s}  {'YAML Sealed':>12s}  {'YAML Vent':>10s}  {'Seg':>8s}"
    print(hdr)
    print("  " + "-" * 82)
    sensor_map = {s.name: s for s in result.sensors}
    for ft in result.fitted_taus:
        s = sensor_map.get(ft.sensor)
        yaml_s = f"{s.yaml_tau_sealed:.1f}" if s else "?"
        yaml_v = f"{s.yaml_tau_ventilated:.1f}" if s else "?"
        vent_str = f"{ft.tau_ventilated:.1f}" if ft.tau_ventilated is not None else "N/A"
        segs = f"{ft.n_segments_sealed}s/{ft.n_segments_ventilated}v"
        print(f"  {ft.sensor:<30s}  {ft.tau_sealed:8.1f}  {vent_str:>8s}  {yaml_s:>12s}  {yaml_v:>10s}  {segs:>8s}")

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
