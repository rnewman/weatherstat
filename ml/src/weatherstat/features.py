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
            # Sun below horizon or calculation error
            elev = 0.0
            azi = 0.0
        elevations.append(elev)
        azimuths.append(azi)

    df["solar_elevation"] = elevations
    df["solar_azimuth"] = azimuths

    return df


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
        for window in windows:
            rolling = df[col].rolling(window, min_periods=1)
            df[f"{col}_rolling_{window}"] = rolling.mean()
    return df


# Columns to compute lags and rolling means for
TEMP_COLUMNS = [
    "thermostat_upstairs_temp",
    "thermostat_downstairs_temp",
    "mini_split_1_temp",
    "mini_split_2_temp",
    "outdoor_temp",
]

LAG_PERIODS = [1, 3, 6, 12]  # 5min, 15min, 30min, 1hr at 5-min intervals
ROLLING_WINDOWS = [6, 12, 24]  # 30min, 1hr, 2hr


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline.

    Takes a raw snapshot DataFrame and returns a feature-enriched DataFrame.
    Used by both training and inference to ensure consistency.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    df = add_time_features(df)
    df = add_solar_features(df)
    df = add_weather_features(df)
    df = add_lag_features(df, TEMP_COLUMNS, LAG_PERIODS)
    df = add_rolling_features(df, TEMP_COLUMNS, ROLLING_WINDOWS)

    return df
