"""Parse HA weather entity data from snapshot rows.

This is a helper for extracting weather features from snapshot data,
not an external API client. Weather data comes from HA's weather.* entity,
ensuring training and inference use the same source.
"""

import pandas as pd

from weatherstat.types import WeatherCondition

# Map HA weather conditions to numeric codes for ML features
CONDITION_CODES: dict[str, int] = {
    condition.value: i for i, condition in enumerate(WeatherCondition)
}


def encode_weather_condition(condition: str) -> int:
    """Encode a weather condition string to a numeric code."""
    return CONDITION_CODES.get(condition, CONDITION_CODES[WeatherCondition.UNKNOWN])


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weather-derived features to a snapshot DataFrame.

    Expects columns: outdoor_temp, outdoor_humidity, wind_speed, weather_condition.
    Adds: weather_condition_code, wind_chill_approx, heat_index_approx.
    """
    df = df.copy()

    df["weather_condition_code"] = df["weather_condition"].map(encode_weather_condition)

    # Simple wind chill approximation (valid below ~10°C with wind)
    temp = df["outdoor_temp"]
    wind = df["wind_speed"]
    df["wind_chill_approx"] = temp - (0.4 * wind)

    # Simple heat index approximation
    humidity = df["outdoor_humidity"]
    df["heat_index_approx"] = temp + (0.33 * humidity / 100.0 * temp) - 4.0

    return df
