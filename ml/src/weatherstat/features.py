"""Feature engineering pipeline shared by training and inference.

Both train.py and inference.py call build_features() to ensure
consistent feature sets (no training/serving skew).
"""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from astral import LocationInfo
from astral.sun import azimuth, elevation

from weatherstat.weather import add_weather_features, encode_weather_condition
from weatherstat.yaml_config import load_config

if TYPE_CHECKING:
    from weatherstat.forecast import ForecastEntry

_CFG = load_config()

# Location for solar calculations (from YAML config)
_LOCATION = LocationInfo(
    name="Home",
    region="US",
    timezone=_CFG.location.timezone,
    latitude=_CFG.location.latitude,
    longitude=_CFG.location.longitude,
)

# ── Column groupings (from YAML config) ──────────────────────────────────────

# All temperature columns available in the full snapshot schema
TEMP_COLUMNS_FULL = _CFG.all_temp_columns

# Temperature columns available in hourly statistics (same set — all have statistics)
TEMP_COLUMNS_HOURLY = _CFG.all_temp_columns

# Room temperature columns used as prediction targets
ROOM_TEMP_COLUMNS = _CFG.room_temp_columns

# Backward-compatible alias
ZONE_TEMP_COLUMNS = ROOM_TEMP_COLUMNS

# Lag and rolling parameters — adjusted per data frequency
LAG_PERIODS_5MIN = [1, 3, 6, 12]  # 5min, 15min, 30min, 1hr
LAG_PERIODS_HOURLY = [1, 2, 4, 6]  # 1hr, 2hr, 4hr, 6hr
ROLLING_WINDOWS_5MIN = [6, 12, 24]  # 30min, 1hr, 2hr
ROLLING_WINDOWS_HOURLY = [3, 6, 12]  # 3hr, 6hr, 12hr


# ── Time and solar features ─────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features from the timestamp column.

    Adds: hour, day_of_week, month, hour_sin, hour_cos, month_sin, month_cos.
    """
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])

    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    return df


def add_solar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add solar position features (elevation and azimuth).

    Uses the astral library with configured location.
    """
    df = df.copy()
    timestamps = pd.to_datetime(df["timestamp"])

    elevations: list[float] = []
    azimuths: list[float] = []

    for ts in timestamps:
        dt = ts.to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=None)  # astral handles naive as UTC
        try:
            elev = elevation(_LOCATION.observer, dt)
            azi = azimuth(_LOCATION.observer, dt)
        except ValueError:
            elev = 0.0
            azi = 0.0
        elevations.append(elev)
        azimuths.append(azi)

    df["solar_elevation"] = elevations
    df["solar_azimuth"] = azimuths

    return df


# ── HVAC encoding ────────────────────────────────────────────────────────────

def encode_hvac_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode HVAC categorical features as numeric values.

    - Thermostat actions: heating=1, idle=0
    - Mini split modes: off=0, heat=1, cool=-1, fan_only=0.5, dry=0.25, auto=0.5
    - Blower modes: off=0, low=1, high=2
    - Navien heating mode: Space Heating=1, Idle=0

    Column lists and encoding maps are driven by YAML config.
    """
    df = df.copy()

    # Thermostat actions
    action_map = {"heating": 1, "idle": 0, "off": 0}
    for col in _CFG.thermostat_action_columns:
        if col in df.columns:
            df[f"{col}_enc"] = df[col].map(action_map).fillna(0).astype(float)

    # Mini split modes (encoding from YAML)
    split_mode_map = next(iter(_CFG.mini_splits.values())).mode_encoding
    for col in _CFG.mini_split_mode_columns:
        if col in df.columns:
            df[f"{col}_enc"] = df[col].map(split_mode_map).fillna(0).astype(float)

    # Blower modes (encoding from YAML)
    blower_map = next(iter(_CFG.blowers.values())).level_encoding
    for col in _CFG.blower_mode_columns:
        if col in df.columns:
            df[f"{col}_enc"] = df[col].map(blower_map).fillna(0).astype(float)

    # Navien heating mode (encoding from YAML)
    if "navien_heating_mode" in df.columns:
        df["navien_heating_mode_enc"] = (
            df["navien_heating_mode"]
            .map(_CFG.boiler.mode_encoding)
            .fillna(0)
            .astype(float)
        )

    return df


# ── Retrospective HVAC features ──────────────────────────────────────────


# HVAC devices to track, keyed by encoded column name.
# Each value describes how to detect "device is active" from the encoded column.
# "positive" means any value > 0 is active (blowers on at any level).
# "nonzero" means any value != 0 is active (mini splits in heat OR cool).
# "heating" means value == 1 is active (thermostats heating, navien space heating).
_HVAC_RETRO_DEVICES: dict[str, str] = {
    "thermostat_upstairs_action_enc": "heating",
    "thermostat_downstairs_action_enc": "heating",
    "navien_heating_mode_enc": "heating",
    "mini_split_bedroom_mode_enc": "nonzero",
    "mini_split_living_room_mode_enc": "nonzero",
    "blower_family_room_mode_enc": "positive",
    "blower_office_mode_enc": "positive",
}


def add_retrospective_hvac_features(df: pd.DataFrame, mode: str = "full") -> pd.DataFrame:
    """Add retrospective HVAC features: cumulative runtime, duty cycle, time-since-transition.

    For each HVAC device, computes:
    - Rolling ON-minutes at 1h/2h/4h windows
    - Duty cycle at 1h/2h windows
    - Minutes since last OFF→ON and ON→OFF transitions

    Must be called after encode_hvac_features() (needs _enc columns).
    """
    df = df.copy()
    dt_minutes = 5 if mode == "full" else 60
    windows_minutes = [60, 120, 240]

    for col, active_type in _HVAC_RETRO_DEVICES.items():
        if col not in df.columns:
            continue

        # Build binary ON signal
        if active_type == "heating":
            on_signal = (df[col] == 1.0).astype(float)
        elif active_type == "nonzero":
            on_signal = (df[col] != 0).astype(float)
        else:  # "positive"
            on_signal = (df[col] > 0).astype(float)

        base = col.replace("_enc", "")

        # Rolling ON-minutes and duty cycle
        for window_min in windows_minutes:
            window_periods = window_min // dt_minutes
            if window_periods < 1:
                continue
            label = f"{window_min // 60}h"
            rolling = on_signal.rolling(window_periods, min_periods=1)
            df[f"{base}_on_minutes_{label}"] = rolling.sum() * dt_minutes
            if window_min <= 120:
                df[f"{base}_duty_cycle_{label}"] = rolling.mean()

        # Time since last transition (ON→OFF and OFF→ON)
        diff = on_signal.diff()
        # diff == 1 means OFF→ON, diff == -1 means ON→OFF
        turned_on = (diff == 1)
        turned_off = (diff == -1)

        # Cumulative time since each transition type
        since_on = pd.Series(np.nan, index=df.index)
        since_off = pd.Series(np.nan, index=df.index)
        last_on_idx = np.nan
        last_off_idx = np.nan
        for i in range(len(df)):
            if turned_on.iloc[i]:
                last_on_idx = i
            if turned_off.iloc[i]:
                last_off_idx = i
            if not np.isnan(last_on_idx):
                since_on.iloc[i] = (i - last_on_idx) * dt_minutes
            if not np.isnan(last_off_idx):
                since_off.iloc[i] = (i - last_off_idx) * dt_minutes

        df[f"{base}_since_on"] = since_on
        df[f"{base}_since_off"] = since_off

    return df


# ── Delta features ───────────────────────────────────────────────────────────

def add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temperature delta features.

    Indoor-outdoor delta captures heat loss driving force.
    Mini split target-current delta captures the proportional control gap.

    Thermostat target deltas are NOT included — thermostats are binary (on/off)
    controllers, so the setpoint number is not a meaningful model input.
    """
    df = df.copy()

    # Indoor-outdoor deltas for all rooms (heat loss driving force, from YAML config)
    if "outdoor_temp" in df.columns:
        for room, col in _CFG.room_temp_columns.items():
            if col in df.columns:
                df[f"{room}_outdoor_delta"] = df[col] - df["outdoor_temp"]

    # Mini split target-current deltas (proportional controllers — setpoint matters)
    for target_col, temp_col, delta_name in _CFG.mini_split_delta_pairs:
        if target_col in df.columns and temp_col in df.columns:
            df[delta_name] = df[target_col] - df[temp_col]

    return df


# ── Physics-informed features ──────────────────────────────────────────────


def add_physics_features(df: pd.DataFrame, mode: str = "full") -> pd.DataFrame:
    """Add physics-informed features capturing thermal dynamics.

    - dT/dt: temperature rate of change (°F/hr) at multiple windows
    - d²T/dt²: thermal acceleration (rate of rate change)
    - Heating power interaction: action × outdoor_delta

    These help the model learn control-relevant dynamics — "the room is warming
    at 1°F/hr" predicts future temps better than just the current reading.
    """
    df = df.copy()

    # Time step in hours (for normalizing rates to °F/hr)
    if mode == "full":
        dt_hours = 5.0 / 60.0  # 5 minutes
        rate_periods = [1, 3, 6, 12]  # 5min, 15min, 30min, 1hr
    else:
        dt_hours = 1.0  # 1 hour
        rate_periods = [1, 2, 3, 6]  # 1hr, 2hr, 3hr, 6hr

    temp_cols = [c for c in _CFG.room_temp_columns.values() if c in df.columns]
    if "outdoor_temp" in df.columns:
        temp_cols.append("outdoor_temp")

    # dT/dt at multiple windows (°F/hr)
    for col in temp_cols:
        for period in rate_periods:
            window_hours = period * dt_hours
            rate = df[col].diff(period) / window_hours
            df[f"{col}_rate_{period}"] = rate

    # d²T/dt² — thermal acceleration (only for the shortest window to avoid noise)
    for col in temp_cols:
        rate_col = f"{col}_rate_{rate_periods[0]}"
        if rate_col in df.columns:
            df[f"{col}_accel"] = df[rate_col].diff(1)

    # Heating power interaction: action × outdoor_delta
    # Captures "how hard is the heating system working against the cold?"
    for name in _CFG.thermostats:
        action_col = f"thermostat_{name}_action_enc"
        delta_col = f"{name}_outdoor_delta"
        if action_col in df.columns and delta_col in df.columns:
            df[f"{name}_heating_power"] = df[action_col] * df[delta_col]

    return df


# ── Forecast features ─────────────────────────────────────────────────────

# Horizon labels → shift periods for "perfect forecast" columns
_FORECAST_HORIZONS = {"1h": 1, "2h": 2, "4h": 4, "6h": 6, "12h": 12}


def add_forecast_features(
    df: pd.DataFrame,
    mode: str,
    forecast_data: list[ForecastEntry] | None = None,
) -> pd.DataFrame:
    """Add forecast outdoor temp, condition, and wind at each prediction horizon.

    Two modes of operation:
    - Training (forecast_data=None): shift outdoor_temp forward to create
      "perfect forecast" columns. The model learns the relationship between
      future outdoor temp and room behavior. At inference, real forecast substitutes.
    - Inference (forecast_data provided): inject actual HA forecast values
      into the last row.

    Features per horizon (1h, 2h, 4h, 6h, 12h):
      - forecast_outdoor_temp_{label}     (°F)
      - forecast_condition_code_{label}   (encoded)
      - forecast_wind_speed_{label}       (mph)

    Also adds hourly forecast_outdoor_temp_{N}h for N=1..12, used by
    piecewise Newton integration.
    """
    df = df.copy()

    # Period multiplier: how many DataFrame rows per hour
    periods_per_hour = 12 if mode == "full" else 1

    if forecast_data is not None:
        # ── Inference mode: inject real forecast into last row ──
        from datetime import datetime, timedelta

        from weatherstat.forecast import forecast_at_horizons

        # Parse reference time from last row
        ref_str = df["timestamp"].iloc[-1]
        try:
            ref_time = datetime.fromisoformat(ref_str)
            if ref_time.tzinfo is None:
                ref_time = ref_time.replace(tzinfo=UTC)
        except (ValueError, TypeError):
            ref_time = datetime.now(UTC)

        # Get forecast at key horizons
        horizon_hours = [float(h) for h in _FORECAST_HORIZONS.values()]
        at_horizons = forecast_at_horizons(forecast_data, ref_time, horizon_hours)

        last_idx = df.index[-1]
        for label, _hours in _FORECAST_HORIZONS.items():
            entry = at_horizons.get(label)
            if entry is not None:
                df.loc[last_idx, f"forecast_outdoor_temp_{label}"] = entry.temperature
                df.loc[last_idx, f"forecast_condition_code_{label}"] = encode_weather_condition(
                    entry.condition
                )
                df.loc[last_idx, f"forecast_wind_speed_{label}"] = (
                    entry.wind_speed if entry.wind_speed is not None else np.nan
                )
            else:
                df.loc[last_idx, f"forecast_outdoor_temp_{label}"] = np.nan
                df.loc[last_idx, f"forecast_condition_code_{label}"] = np.nan
                df.loc[last_idx, f"forecast_wind_speed_{label}"] = np.nan

        # Hourly forecast temps for piecewise Newton (1h through 12h)
        sorted_entries = sorted(forecast_data, key=lambda e: e.datetime)
        for h in range(1, 13):
            target_time = ref_time + timedelta(hours=h)
            best: ForecastEntry | None = None
            best_diff = float("inf")
            for entry in sorted_entries:
                try:
                    edt = datetime.fromisoformat(entry.datetime)
                    if edt.tzinfo is None:
                        edt = edt.replace(tzinfo=UTC)
                    diff = abs((edt - target_time).total_seconds())
                    if diff < best_diff:
                        best_diff = diff
                        best = entry
                except (ValueError, TypeError):
                    continue
            if best is not None and best_diff <= 5400:
                df.loc[last_idx, f"forecast_outdoor_temp_{h}h"] = best.temperature
            else:
                df.loc[last_idx, f"forecast_outdoor_temp_{h}h"] = np.nan

    else:
        # ── Training mode: shift outdoor_temp forward ("perfect forecast") ──
        if "outdoor_temp" in df.columns:
            for label, hours in _FORECAST_HORIZONS.items():
                shift = hours * periods_per_hour
                df[f"forecast_outdoor_temp_{label}"] = df["outdoor_temp"].shift(-shift)

            if "weather_condition_code" in df.columns:
                for label, hours in _FORECAST_HORIZONS.items():
                    shift = hours * periods_per_hour
                    df[f"forecast_condition_code_{label}"] = df["weather_condition_code"].shift(-shift)

            if "wind_speed" in df.columns:
                for label, hours in _FORECAST_HORIZONS.items():
                    shift = hours * periods_per_hour
                    df[f"forecast_wind_speed_{label}"] = df["wind_speed"].shift(-shift)

            # Hourly forecast temps for piecewise Newton (1h through 12h)
            for h in range(1, 13):
                shift = h * periods_per_hour
                df[f"forecast_outdoor_temp_{h}h"] = df["outdoor_temp"].shift(-shift)

    return df


# ── Newton's law of cooling features ───────────────────────────────────────

# Hours ahead for each horizon label
HORIZON_HOURS = {"1h": 1.0, "2h": 2.0, "4h": 4.0, "6h": 6.0, "12h": 12.0}


def add_newton_cooling_features(df: pd.DataFrame, mode: str = "full") -> pd.DataFrame:
    """Add Newton's law of cooling predictions as features.

    For each room and horizon, computes sealed and ventilated variants:
      T_newton = T_outdoor + (T_room - T_outdoor) * exp(-hours / tau)
      delta    = T_newton - T_room  (expected passive change, ≤ 0 when warmer)

    When forecast_outdoor_temp_{N}h columns are present (from add_forecast_features),
    uses piecewise Newton integration that chains hourly segments with different
    outdoor temps — much more accurate at 4h+ horizons where outdoor temp changes
    significantly (e.g., sunrise warming from 40°F to 55°F).

    Two τ values per room (sealed = windows closed, ventilated = windows open)
    let LightGBM split on window state to pick the right physics prediction.

    LightGBM can't compute exp(-t/τ) from raw temperatures (trees do axis-aligned
    splits). Pre-computing gives the model direct access to physics predictions.
    During HVAC sweep, Newton features stay unchanged (passive prediction);
    HVAC features capture the heating effect. LightGBM combines them.
    """
    from weatherstat.forecast import piecewise_newton_prediction

    thermal = _CFG.thermal
    new_cols: dict[str, pd.Series] = {}

    # Check which hourly forecast columns are available for piecewise integration
    max_forecast_hour = 0
    for h in range(1, 13):
        if f"forecast_outdoor_temp_{h}h" in df.columns:
            max_forecast_hour = h

    for room, temp_col in _CFG.room_temp_columns.items():
        if temp_col not in df.columns or "outdoor_temp" not in df.columns:
            continue

        tau_s = thermal.tau_sealed.get(room, thermal.default_tau_sealed)
        tau_v = thermal.tau_ventilated.get(room, thermal.default_tau_ventilated)
        t_room = df[temp_col]
        t_outdoor = df["outdoor_temp"]

        for label, hours in HORIZON_HOURS.items():
            hours_int = int(hours)

            # Check if we have enough forecast data for piecewise integration
            if max_forecast_hour >= hours_int and hours_int >= 2:
                # Piecewise Newton: chain hourly segments with forecast outdoor temps
                # Vectorized over DataFrame rows
                forecast_cols = [f"forecast_outdoor_temp_{h}h" for h in range(1, hours_int + 1)]

                pred_s = pd.Series(np.nan, index=df.index)
                pred_v = pd.Series(np.nan, index=df.index)
                for idx in df.index:
                    room_t = t_room.loc[idx]
                    outdoor_temps = [df.loc[idx, c] for c in forecast_cols]
                    if np.isnan(room_t) or any(np.isnan(t) for t in outdoor_temps):
                        continue
                    pred_s.loc[idx] = piecewise_newton_prediction(room_t, outdoor_temps, tau_s, hours)
                    pred_v.loc[idx] = piecewise_newton_prediction(room_t, outdoor_temps, tau_v, hours)
            else:
                # Fallback: constant outdoor temp (existing behavior)
                decay_s = np.exp(-hours / tau_s)
                pred_s = t_outdoor + (t_room - t_outdoor) * decay_s
                decay_v = np.exp(-hours / tau_v)
                pred_v = t_outdoor + (t_room - t_outdoor) * decay_v

            new_cols[f"{room}_newton_sealed_{label}"] = pred_s
            new_cols[f"{room}_newton_sealed_delta_{label}"] = pred_s - t_room
            new_cols[f"{room}_newton_vent_{label}"] = pred_v
            new_cols[f"{room}_newton_vent_delta_{label}"] = pred_v - t_room

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ── Lag and rolling features ────────────────────────────────────────────────

def add_lag_features(
    df: pd.DataFrame, columns: list[str], lags: list[int]
) -> pd.DataFrame:
    """Add lag features for specified columns.

    Args:
        df: DataFrame sorted by timestamp.
        columns: Column names to create lags for.
        lags: List of lag periods (in snapshot intervals).
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame, columns: list[str], windows: list[int]
) -> pd.DataFrame:
    """Add rolling mean features for specified columns.

    Args:
        df: DataFrame sorted by timestamp.
        columns: Column names to create rolling features for.
        windows: List of window sizes (in snapshot intervals).
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        for window in windows:
            rolling = df[col].rolling(window, min_periods=1)
            df[f"{col}_rolling_{window}"] = rolling.mean()
    return df


# ── Future temperature targets ───────────────────────────────────────────────

def add_future_targets(
    df: pd.DataFrame,
    zone_columns: dict[str, str],
    horizons: list[int],
) -> pd.DataFrame:
    """Create future temperature columns for multi-horizon prediction.

    Shifts temperature columns BACKWARD (negative shift) to align future values
    with current rows. Rows at the end where future is unknown become NaN.

    Args:
        df: DataFrame sorted by timestamp.
        zone_columns: {zone_name: temp_column_name} mapping.
        horizons: List of forward steps to create targets for.

    Returns:
        DataFrame with additional columns: {zone}_temp_t+{horizon}
    """
    df = df.copy()
    for zone, col in zone_columns.items():
        if col not in df.columns:
            continue
        for h in horizons:
            df[f"{zone}_temp_t+{h}"] = df[col].shift(-h)
    return df


# ── Main pipeline ────────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    mode: str = "full",
    forecast_data: list[ForecastEntry] | None = None,
) -> pd.DataFrame:
    """Full feature engineering pipeline.

    Takes a raw snapshot DataFrame and returns a feature-enriched DataFrame.
    Used by both training and inference to ensure consistency.

    Args:
        df: Raw snapshot or statistics DataFrame.
        mode: "full" for 5-min full-feature data, "baseline" for hourly temp-only.
        forecast_data: Optional forecast entries from HA. When provided (inference),
            injects real forecast values into the last row. When None (training),
            creates "perfect forecast" columns by shifting outdoor_temp forward.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Time and solar features (always)
    df = add_time_features(df)
    df = add_solar_features(df)

    # Weather features (if weather columns present)
    if "weather_condition" in df.columns:
        df = add_weather_features(df)

    if mode == "full":
        # HVAC encoding (only in full mode)
        df = encode_hvac_features(df)

        # Retrospective HVAC features (cumulative runtime, duty cycle, time-since-transition)
        df = add_retrospective_hvac_features(df, mode="full")

        # Delta features (only in full mode)
        df = add_delta_features(df)

        # Physics-informed features (dT/dt, acceleration, heating power)
        df = add_physics_features(df, mode="full")

        # Forecast features (before Newton so piecewise integration can use them)
        df = add_forecast_features(df, mode="full", forecast_data=forecast_data)

        # Newton's law of cooling predictions (passive thermal decay)
        # Uses forecast columns when available for piecewise integration
        df = add_newton_cooling_features(df, mode="full")

        # Temperature lags and rolling at 5-min intervals
        temp_cols = [c for c in TEMP_COLUMNS_FULL if c in df.columns]
        df = add_lag_features(df, temp_cols, LAG_PERIODS_5MIN)
        df = add_rolling_features(df, temp_cols, ROLLING_WINDOWS_5MIN)

    elif mode == "baseline":
        # HVAC encoding (when HVAC columns are present from merged data)
        df = encode_hvac_features(df)

        # Retrospective HVAC features (cumulative runtime, duty cycle, time-since-transition)
        df = add_retrospective_hvac_features(df, mode="baseline")

        # Delta features (target-current, indoor-outdoor, room-zone)
        df = add_delta_features(df)

        # Physics-informed features (dT/dt, acceleration, heating power)
        df = add_physics_features(df, mode="baseline")

        # Forecast features (before Newton so piecewise integration can use them)
        df = add_forecast_features(df, mode="baseline", forecast_data=forecast_data)

        # Newton's law of cooling predictions (passive thermal decay)
        df = add_newton_cooling_features(df, mode="baseline")

        # Temperature lags and rolling at hourly intervals
        temp_cols = [c for c in TEMP_COLUMNS_HOURLY if c in df.columns]
        df = add_lag_features(df, temp_cols, LAG_PERIODS_HOURLY)
        df = add_rolling_features(df, temp_cols, ROLLING_WINDOWS_HOURLY)

    return df
