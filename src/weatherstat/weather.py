"""Parse HA weather entity data from snapshot rows.

Weather data comes from HA's weather.* entity (met.no integration),
ensuring training and inference use the same source — no feature skew.
"""

from __future__ import annotations

from weatherstat.types import WeatherCondition

# Map all met.no weather conditions to numeric codes for ML features.
# Ordered roughly from clear to severe.
CONDITION_CODES: dict[str, int] = {
    condition.value: i for i, condition in enumerate(WeatherCondition)
}

# Fraction of maximum solar irradiance reaching the surface for each
# met.no weather condition. Used to modulate per-hour solar gains in
# both sysid (training) and the forward simulator (inference).
#
# Values are approximate — the key property is monotonicity: sunny > partly
# cloudy > overcast > precipitation. Exact values will be absorbed into the
# per-hour regression coefficients during sysid fitting.
SOLAR_FRACTION: dict[str, float] = {
    "sunny": 1.0,
    "clear-night": 0.0,  # no sun at night
    "partlycloudy": 0.5,
    "cloudy": 0.15,
    "fog": 0.15,
    "rainy": 0.05,
    "pouring": 0.02,
    "snowy": 0.1,
    "snowy-rainy": 0.05,
    "lightning": 0.1,
    "lightning-rainy": 0.05,
    "hail": 0.05,
    "windy": 0.7,  # wind alone doesn't imply clouds
    "windy-variant": 0.5,
    "exceptional": 0.3,
    "unknown": 0.3,
    "unavailable": 0.3,
}

# Default for conditions not in the map (e.g., future met.no additions)
_DEFAULT_SOLAR_FRACTION = 0.3


def condition_to_solar_fraction(condition: str) -> float:
    """Map a met.no weather condition string to a solar fraction (0–1)."""
    return SOLAR_FRACTION.get(condition, _DEFAULT_SOLAR_FRACTION)


def encode_weather_condition(condition: str) -> int:
    """Encode a weather condition string to a numeric code.

    Handles all met.no conditions via WeatherCondition enum.
    Unknown conditions map to the UNKNOWN code.
    """
    return CONDITION_CODES.get(condition, CONDITION_CODES[WeatherCondition.UNKNOWN])


