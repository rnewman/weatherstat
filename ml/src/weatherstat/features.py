"""Feature engineering pipeline shared by training and inference.

Both train.py and inference.py call build_features() to ensure
consistent feature sets (no training/serving skew).
"""

import numpy as np
import pandas as pd
from astral import LocationInfo
from astral.sun import azimuth, elevation

from weatherstat.config import LATITUDE, LONGITUDE
from weatherstat.weather import add_weather_features

# Location for solar calculations
_LOCATION = LocationInfo(
    name="Home",
    region="US",
    timezone="America/Los_Angeles",
    latitude=LATITUDE,
    longitude=LONGITUDE,
)

# ── Column groupings ────────────────────────────────────────────────────────

# All temperature columns available in the full snapshot schema
TEMP_COLUMNS_FULL = [
    "thermostat_upstairs_temp",
    "thermostat_downstairs_temp",
    "upstairs_aggregate_temp",
    "downstairs_aggregate_temp",
    "family_room_temp",
    "office_temp",
    "kitchen_temp",
    "bedroom_temp",
    "piano_temp",
    "bathroom_temp",
    "living_room_temp",
    "outdoor_temp",
]

# Temperature columns available in hourly statistics (subset of full)
TEMP_COLUMNS_HOURLY = [
    "thermostat_upstairs_temp",
    "thermostat_downstairs_temp",
    "upstairs_aggregate_temp",
    "downstairs_aggregate_temp",
    "family_room_temp",
    "office_temp",
    "kitchen_temp",
    "bedroom_temp",
    "piano_temp",
    "bathroom_temp",
    "living_room_temp",
    "outdoor_temp",
]

# Room temperature columns used as prediction targets
ROOM_TEMP_COLUMNS = {
    "upstairs": "thermostat_upstairs_temp",
    "downstairs": "thermostat_downstairs_temp",
    "bedroom": "bedroom_temp",
    "kitchen": "kitchen_temp",
    "piano": "piano_temp",
    "bathroom": "bathroom_temp",
    "family_room": "family_room_temp",
    "office": "office_temp",
}

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
    """
    df = df.copy()

    # Thermostat actions
    action_map = {"heating": 1, "idle": 0, "off": 0}
    for col in ["thermostat_upstairs_action", "thermostat_downstairs_action"]:
        if col in df.columns:
            df[f"{col}_enc"] = df[col].map(action_map).fillna(0).astype(float)

    # Mini split modes
    split_mode_map = {
        "off": 0, "heat": 1, "cool": -1, "fan_only": 0.5,
        "dry": 0.25, "auto": 0.5, "heat_cool": 0.5,
    }
    for col in ["mini_split_bedroom_mode", "mini_split_living_room_mode"]:
        if col in df.columns:
            df[f"{col}_enc"] = df[col].map(split_mode_map).fillna(0).astype(float)

    # Blower modes
    blower_map = {"off": 0, "low": 1, "high": 2}
    for col in ["blower_family_room_mode", "blower_office_mode"]:
        if col in df.columns:
            df[f"{col}_enc"] = df[col].map(blower_map).fillna(0).astype(float)

    # Navien heating mode
    if "navien_heating_mode" in df.columns:
        df["navien_heating_mode_enc"] = (
            df["navien_heating_mode"]
            .map({"Space Heating": 1, "Idle": 0})
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

    # Indoor-outdoor deltas for all rooms (heat loss driving force)
    room_outdoor_cols = {
        "upstairs": "thermostat_upstairs_temp",
        "downstairs": "thermostat_downstairs_temp",
        "bedroom": "bedroom_temp",
        "kitchen": "kitchen_temp",
        "piano": "piano_temp",
        "bathroom": "bathroom_temp",
        "family_room": "family_room_temp",
        "office": "office_temp",
    }
    if "outdoor_temp" in df.columns:
        for room, col in room_outdoor_cols.items():
            if col in df.columns:
                df[f"{room}_outdoor_delta"] = df[col] - df["outdoor_temp"]

    # Mini split target-current deltas (proportional controllers — setpoint matters)
    mini_split_pairs = [
        ("mini_split_bedroom_target", "mini_split_bedroom_temp", "bedroom_target_delta"),
        ("mini_split_living_room_target", "mini_split_living_room_temp", "living_room_target_delta"),
    ]
    for target_col, temp_col, delta_name in mini_split_pairs:
        if target_col in df.columns and temp_col in df.columns:
            df[delta_name] = df[target_col] - df[temp_col]

    return df


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

        # Temperature lags and rolling at 5-min intervals
        temp_cols = [c for c in TEMP_COLUMNS_FULL if c in df.columns]
        df = add_lag_features(df, temp_cols, LAG_PERIODS_5MIN)
        df = add_rolling_features(df, temp_cols, ROLLING_WINDOWS_5MIN)

    elif mode == "baseline":
        # HVAC encoding (when HVAC columns are present from merged data)
        df = encode_hvac_features(df)

        # Delta features (target-current, indoor-outdoor, room-zone)
        df = add_delta_features(df)

        # Temperature lags and rolling at hourly intervals
        temp_cols = [c for c in TEMP_COLUMNS_HOURLY if c in df.columns]
        df = add_lag_features(df, temp_cols, LAG_PERIODS_HOURLY)
        df = add_rolling_features(df, temp_cols, ROLLING_WINDOWS_HOURLY)

    return df
