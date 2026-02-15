"""Feature engineering pipeline shared by training and inference.

Both train.py and inference.py call build_features() to ensure
consistent feature sets (no training/serving skew).
"""

import numpy as np
import pandas as pd
from astral import LocationInfo
from astral.sun import azimuth, elevation

from weatherstat.weather import add_weather_features
from weatherstat.yaml_config import load_config

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


# ── Newton's law of cooling features ───────────────────────────────────────

# Hours ahead for each horizon label
HORIZON_HOURS = {"1h": 1.0, "2h": 2.0, "4h": 4.0, "6h": 6.0, "12h": 12.0}


def add_newton_cooling_features(df: pd.DataFrame, mode: str = "full") -> pd.DataFrame:
    """Add Newton's law of cooling predictions as features.

    For each room and horizon, computes sealed and ventilated variants:
      T_newton = T_outdoor + (T_room - T_outdoor) * exp(-hours / tau)
      delta    = T_newton - T_room  (expected passive change, ≤ 0 when warmer)

    Two τ values per room (sealed = windows closed, ventilated = windows open)
    let LightGBM split on window state to pick the right physics prediction.

    LightGBM can't compute exp(-t/τ) from raw temperatures (trees do axis-aligned
    splits). Pre-computing gives the model direct access to physics predictions.
    During HVAC sweep, Newton features stay unchanged (passive prediction);
    HVAC features capture the heating effect. LightGBM combines them.
    """
    thermal = _CFG.thermal
    new_cols: dict[str, pd.Series] = {}

    for room, temp_col in _CFG.room_temp_columns.items():
        if temp_col not in df.columns or "outdoor_temp" not in df.columns:
            continue

        tau_s = thermal.tau_sealed.get(room, thermal.default_tau_sealed)
        tau_v = thermal.tau_ventilated.get(room, thermal.default_tau_ventilated)
        t_room = df[temp_col]
        t_outdoor = df["outdoor_temp"]

        for label, hours in HORIZON_HOURS.items():
            # Sealed (windows closed)
            decay_s = np.exp(-hours / tau_s)
            pred_s = t_outdoor + (t_room - t_outdoor) * decay_s
            new_cols[f"{room}_newton_sealed_{label}"] = pred_s
            new_cols[f"{room}_newton_sealed_delta_{label}"] = pred_s - t_room

            # Ventilated (windows open)
            decay_v = np.exp(-hours / tau_v)
            pred_v = t_outdoor + (t_room - t_outdoor) * decay_v
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
) -> pd.DataFrame:
    """Full feature engineering pipeline.

    Takes a raw snapshot DataFrame and returns a feature-enriched DataFrame.
    Used by both training and inference to ensure consistency.

    Args:
        df: Raw snapshot or statistics DataFrame.
        mode: "full" for 5-min full-feature data, "baseline" for hourly temp-only.
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

        # Delta features (only in full mode)
        df = add_delta_features(df)

        # Physics-informed features (dT/dt, acceleration, heating power)
        df = add_physics_features(df, mode="full")

        # Newton's law of cooling predictions (passive thermal decay)
        df = add_newton_cooling_features(df, mode="full")

        # Temperature lags and rolling at 5-min intervals
        temp_cols = [c for c in TEMP_COLUMNS_FULL if c in df.columns]
        df = add_lag_features(df, temp_cols, LAG_PERIODS_5MIN)
        df = add_rolling_features(df, temp_cols, ROLLING_WINDOWS_5MIN)

    elif mode == "baseline":
        # HVAC encoding (when HVAC columns are present from merged data)
        df = encode_hvac_features(df)

        # Delta features (target-current, indoor-outdoor, room-zone)
        df = add_delta_features(df)

        # Physics-informed features (dT/dt, acceleration, heating power)
        df = add_physics_features(df, mode="baseline")

        # Newton's law of cooling predictions (passive thermal decay)
        df = add_newton_cooling_features(df, mode="baseline")

        # Temperature lags and rolling at hourly intervals
        temp_cols = [c for c in TEMP_COLUMNS_HOURLY if c in df.columns]
        df = add_lag_features(df, temp_cols, LAG_PERIODS_HOURLY)
        df = add_rolling_features(df, temp_cols, ROLLING_WINDOWS_HOURLY)

    return df
