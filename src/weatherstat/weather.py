"""Parse HA weather entity data from snapshot rows.

Weather data comes from HA's weather.* entity (met.no integration),
ensuring training and inference use the same source — no feature skew.
"""

from __future__ import annotations

import math
from datetime import datetime

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


def kasten_czeplak(cloud_coverage_pct: float) -> float:
    """Kasten-Czeplak (1980) clear-sky attenuation as a function of cloud cover.

    Returns the fraction of clear-sky irradiance reaching the surface for a
    given cloud cover percentage (0–100). Empirically: clear-sky stays near
    1.0 for cc ≲ 30%, drops sharply above cc ≳ 60%. This non-linear shape
    matches observed sunny→partly-cloudy→overcast attenuation much better
    than a linear model and avoids over-attenuating lightly hazy days.
    """
    cc = max(0.0, min(100.0, float(cloud_coverage_pct))) / 100.0
    return max(0.0, 1.0 - 0.75 * (cc ** 3.4))


def effective_solar_fraction(
    cloud_coverage: float | None,
    condition: str | None,
) -> float:
    """Continuous solar fraction in [0, 1].

    Prefers ``cloud_coverage`` (Kasten-Czeplak transfer); falls back to the
    categorical condition map when cloud_coverage is missing — e.g., snapshots
    from before April 2026 (when the column was introduced) and forecast entries
    whose source did not provide a cloud_coverage attribute.
    """
    if cloud_coverage is not None:
        return kasten_czeplak(cloud_coverage)
    if condition:
        return condition_to_solar_fraction(condition)
    return _DEFAULT_SOLAR_FRACTION


def encode_weather_condition(condition: str) -> int:
    """Encode a weather condition string to a numeric code.

    Handles all met.no conditions via WeatherCondition enum.
    Unknown conditions map to the UNKNOWN code.
    """
    return CONDITION_CODES.get(condition, CONDITION_CODES[WeatherCondition.UNKNOWN])


def solar_elevation(lat_deg: float, lon_deg: float, dt: datetime) -> float:
    """Solar elevation angle in degrees for a given location and UTC time.

    Uses the standard astronomical formula with Spencer (1971) declination.
    Accuracy ~1° — sufficient for thermal modeling since the per-sensor
    regression coefficient absorbs systematic error.

    Args:
        lat_deg: Latitude in degrees (positive north).
        lon_deg: Longitude in degrees (positive east).
        dt: UTC-aware datetime.

    Returns:
        Elevation angle in degrees. Negative means sun is below horizon.
    """
    day_of_year = dt.timetuple().tm_yday
    declination = 23.45 * math.sin(math.radians(360.0 / 365.0 * (day_of_year - 81)))

    # Solar hour angle: UTC hour + longitude correction
    utc_hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    solar_hour = utc_hour + lon_deg / 15.0
    hour_angle = 15.0 * (solar_hour - 12.0)

    lat_r = math.radians(lat_deg)
    dec_r = math.radians(declination)
    ha_r = math.radians(hour_angle)
    sin_elev = math.sin(lat_r) * math.sin(dec_r) + math.cos(lat_r) * math.cos(dec_r) * math.cos(ha_r)
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))


def solar_sin_elevation(lat_deg: float, lon_deg: float, dt: datetime) -> float:
    """max(0, sin(elevation)) — the solar forcing feature for regression.

    Returns 0 when the sun is below the horizon, otherwise sin(elevation)
    which is proportional to the horizontal-surface irradiance (Lambert's
    cosine law). For a complex house, the per-sensor regression coefficient
    absorbs the effective geometry.
    """
    elev = solar_elevation(lat_deg, lon_deg, dt)
    return max(0.0, math.sin(math.radians(elev)))
