"""Parse HA weather entity data from snapshot rows.

Weather data comes from HA's weather.* entity (met.no integration),
ensuring training and inference use the same source — no feature skew.
"""

import pandas as pd

from weatherstat.types import WeatherCondition

# Map all met.no weather conditions to numeric codes for ML features.
# Ordered roughly from clear to severe.
CONDITION_CODES: dict[str, int] = {
    condition.value: i for i, condition in enumerate(WeatherCondition)
}


def encode_weather_condition(condition: str) -> int:
    """Encode a weather condition string to a numeric code.

    Handles all met.no conditions via WeatherCondition enum.
    Unknown conditions map to the UNKNOWN code.
    """
    return CONDITION_CODES.get(condition, CONDITION_CODES[WeatherCondition.UNKNOWN])


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weather-derived features to a snapshot DataFrame.

    Expects columns: outdoor_temp, weather_condition.
    Optional columns: outdoor_humidity, wind_speed.
    Adds: weather_condition_code, wind_chill_approx, heat_index_approx.
    """
    df = df.copy()

    if "weather_condition" in df.columns:
        df["weather_condition_code"] = df["weather_condition"].map(encode_weather_condition)

    # Wind chill approximation (valid below ~10C with wind)
    if "outdoor_temp" in df.columns and "wind_speed" in df.columns:
        temp = df["outdoor_temp"]
        wind = df["wind_speed"]
        df["wind_chill_approx"] = temp - (0.4 * wind)

    # Heat index approximation
    if "outdoor_temp" in df.columns and "outdoor_humidity" in df.columns:
        temp = df["outdoor_temp"]
        humidity = df["outdoor_humidity"]
        df["heat_index_approx"] = temp + (0.33 * humidity / 100.0 * temp) - 4.0

    return df
