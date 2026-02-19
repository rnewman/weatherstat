"""Weather forecast integration — fetch from HA + piecewise Newton cooling.

Fetches hourly forecast from HA via REST service call (weather.get_forecasts),
provides piecewise Newton cooling that chains hourly segments with changing
outdoor temps, and helpers to extract forecast at prediction horizons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

import requests

from weatherstat.config import HA_TOKEN, HA_URL


@dataclass(frozen=True)
class ForecastEntry:
    """One hourly forecast point from HA (met.no via weather.forecast_home)."""

    datetime: str  # ISO 8601
    temperature: float  # °F
    condition: str  # "sunny", "cloudy", "rainy", etc.
    wind_speed: float | None
    cloud_coverage: float | None  # 0-100%
    precipitation: float | None


def fetch_forecast(entity_id: str = "weather.forecast_home") -> list[ForecastEntry]:
    """Fetch hourly forecast from HA via REST service call.

    Calls POST /api/services/weather/get_forecasts with
    {"entity_id": entity_id, "type": "hourly"}.

    Returns list of ForecastEntry sorted by datetime (next 24-48h).
    """
    url = f"{HA_URL}/api/services/weather/get_forecasts"
    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "entity_id": entity_id,
        "type": "hourly",
    }

    resp = requests.post(url, headers=headers, json=payload, params={"return_response": ""}, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # HA returns {"weather.forecast_home": {"forecast": [...]}}
    entity_data = data.get(entity_id, {})
    raw_entries = entity_data.get("forecast", [])

    entries: list[ForecastEntry] = []
    for raw in raw_entries:
        entries.append(ForecastEntry(
            datetime=raw["datetime"],
            temperature=float(raw["temperature"]),
            condition=str(raw.get("condition", "unknown")),
            wind_speed=_opt_float(raw.get("wind_speed")),
            cloud_coverage=_opt_float(raw.get("cloud_coverage")),
            precipitation=_opt_float(raw.get("precipitation")),
        ))

    return sorted(entries, key=lambda e: e.datetime)


def _opt_float(val: object) -> float | None:
    """Convert to float if not None."""
    if val is None:
        return None
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def piecewise_newton_prediction(
    current_temp: float,
    outdoor_temps: list[float],
    tau: float,
    hours_ahead: float,
) -> float:
    """Chain hourly Newton cooling segments using forecast outdoor temps.

    For each hour segment:
        T(h+1) = T_out[0] + (T(h) - T_out[0]) * exp(-1/τ)
        T(h+2) = T_out[1] + (T(h+1) - T_out[1]) * exp(-1/τ)
        ...

    For fractional final segments (e.g., hours_ahead=1.5): apply partial
    decay exp(-0.5/τ) for the remaining 30 minutes.

    If outdoor_temps has fewer entries than needed, the last value is reused.

    Args:
        current_temp: Starting room temperature (°F).
        outdoor_temps: Hourly outdoor temps [h+1, h+2, ..., h+N].
        tau: Time constant in hours.
        hours_ahead: Target prediction horizon in hours.

    Returns:
        Predicted temperature after hours_ahead.
    """
    if not outdoor_temps or hours_ahead <= 0:
        return current_temp

    t = current_temp
    full_hours = int(hours_ahead)
    frac = hours_ahead - full_hours

    for i in range(full_hours):
        # Use outdoor temp for this hour segment (clamp to available data)
        t_out = outdoor_temps[min(i, len(outdoor_temps) - 1)]
        decay = math.exp(-1.0 / tau)
        t = t_out + (t - t_out) * decay

    # Fractional remaining segment
    if frac > 0:
        t_out = outdoor_temps[min(full_hours, len(outdoor_temps) - 1)]
        decay = math.exp(-frac / tau)
        t = t_out + (t - t_out) * decay

    return t


def forecast_at_horizons(
    entries: list[ForecastEntry],
    reference_time: datetime,
    horizons_hours: list[float],
) -> dict[str, ForecastEntry | None]:
    """Extract forecast entries closest to each prediction horizon.

    Args:
        entries: Sorted list of ForecastEntry from fetch_forecast().
        reference_time: Current time (UTC-aware).
        horizons_hours: List of hours ahead to extract (e.g., [1, 2, 4, 6, 12]).

    Returns:
        {"1h": ForecastEntry, "2h": ForecastEntry, ...} mapping.
        None for horizons beyond available forecast range.
    """
    if not entries:
        return {f"{int(h)}h": None for h in horizons_hours}

    # Parse forecast datetimes
    parsed: list[tuple[datetime, ForecastEntry]] = []
    for entry in entries:
        dt = _parse_iso(entry.datetime)
        if dt is not None:
            parsed.append((dt, entry))

    result: dict[str, ForecastEntry | None] = {}
    for h in horizons_hours:
        label = f"{int(h)}h"
        from datetime import timedelta

        target_time = reference_time + timedelta(hours=h)

        # Find the entry closest to target_time
        best: ForecastEntry | None = None
        best_diff = float("inf")
        for dt, entry in parsed:
            diff = abs((dt - target_time).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best = entry

        # Only accept if within 90 minutes of target
        if best_diff <= 5400:
            result[label] = best
        else:
            result[label] = None

    return result


def _parse_iso(s: str) -> datetime | None:
    """Parse an ISO 8601 datetime string to UTC-aware datetime."""
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except (ValueError, TypeError):
        return None
