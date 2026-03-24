"""Tests for the forecast module: piecewise Newton + forecast helpers."""

import math
from datetime import UTC

from weatherstat.forecast import (
    ForecastEntry,
    forecast_at_horizons,
    piecewise_newton_prediction,
)

# ── piecewise_newton_prediction ────────────────────────────────────────────


class TestPiecewiseNewton:
    def test_constant_outdoor_matches_standard_newton(self) -> None:
        """With constant outdoor temps, piecewise should match standard Newton."""
        current = 72.0
        outdoor = 40.0
        tau = 24.0
        hours = 6.0

        # Standard Newton: T = T_out + (T_room - T_out) * exp(-hours/tau)
        expected = outdoor + (current - outdoor) * math.exp(-hours / tau)

        # Piecewise with constant outdoor temps
        outdoor_temps = [outdoor] * 12
        result = piecewise_newton_prediction(current, outdoor_temps, tau, hours)

        assert abs(result - expected) < 0.01, (
            f"Piecewise {result:.4f} should match standard Newton {expected:.4f}"
        )

    def test_warming_outdoor_predicts_warmer(self) -> None:
        """Rising outdoor temps should predict warmer than constant."""
        current = 72.0
        tau = 24.0
        hours = 6.0

        # Constant outdoor at 40°F
        constant_result = piecewise_newton_prediction(
            current, [40.0] * 6, tau, hours
        )

        # Rising outdoor from 40°F to 55°F
        rising_temps = [42.0, 44.0, 47.0, 50.0, 53.0, 55.0]
        rising_result = piecewise_newton_prediction(
            current, rising_temps, tau, hours
        )

        assert rising_result > constant_result, (
            f"Rising outdoor ({rising_result:.2f}) should predict warmer than "
            f"constant ({constant_result:.2f})"
        )

    def test_zero_hours_returns_current(self) -> None:
        """Zero hours ahead should return current temperature."""
        result = piecewise_newton_prediction(72.0, [40.0, 40.0], 24.0, 0.0)
        assert result == 72.0

    def test_fractional_hours(self) -> None:
        """1.5 hours: 1 full segment + 0.5 fractional."""
        current = 72.0
        outdoor = [40.0, 40.0]
        tau = 24.0

        result = piecewise_newton_prediction(current, outdoor, tau, 1.5)

        # Manual: after 1h: T1 = 40 + 32*exp(-1/24) ≈ 70.68
        t1 = 40 + (current - 40) * math.exp(-1 / 24)
        # After 0.5h more: T = 40 + (T1-40)*exp(-0.5/24)
        expected = 40 + (t1 - 40) * math.exp(-0.5 / 24)

        assert abs(result - expected) < 0.01

    def test_empty_outdoor_temps_returns_current(self) -> None:
        """Empty outdoor temps list should return current temperature."""
        result = piecewise_newton_prediction(72.0, [], 24.0, 6.0)
        assert result == 72.0

    def test_fewer_entries_than_hours(self) -> None:
        """When fewer forecast entries than hours, last value is reused."""
        current = 72.0
        outdoor = [45.0]  # only one entry for 3-hour forecast
        tau = 24.0

        result = piecewise_newton_prediction(current, outdoor, tau, 3.0)

        # Should be equivalent to constant outdoor at 45°F
        constant_result = piecewise_newton_prediction(
            current, [45.0] * 3, tau, 3.0
        )
        assert abs(result - constant_result) < 0.01

    def test_summer_cooling_direction(self) -> None:
        """When outdoor is warmer, temperature should rise toward outdoor."""
        current = 72.0
        outdoor = [95.0] * 6
        tau = 24.0

        result = piecewise_newton_prediction(current, outdoor, tau, 6.0)
        assert result > current, "Temperature should rise toward warmer outdoor"
        assert result < 95.0, "Temperature should not exceed outdoor"


# ── forecast_at_horizons ──────────────────────────────────────────────────


class TestForecastAtHorizons:
    def _make_entries(self, base_hour: int = 14) -> list[ForecastEntry]:
        """Create forecast entries for testing (hourly for 24h)."""
        entries = []
        for h in range(24):
            hour = (base_hour + h + 1) % 24
            entries.append(ForecastEntry(
                datetime=f"2024-01-15T{hour:02d}:00:00+00:00",
                temperature=40.0 + h * 0.5,
                condition="cloudy" if h < 12 else "sunny",
                wind_speed=5.0 + h * 0.2,
                cloud_coverage=80.0 - h * 3,
                precipitation=0.1 if h < 6 else 0.0,
            ))
        return entries

    def test_extracts_closest_entries(self) -> None:
        """Should find the closest forecast entry to each horizon."""
        from datetime import datetime

        entries = self._make_entries(14)
        ref = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)

        result = forecast_at_horizons(entries, ref, [1.0, 2.0, 4.0])

        assert result["1h"] is not None
        assert result["2h"] is not None
        assert result["4h"] is not None

    def test_empty_entries_returns_none(self) -> None:
        """Empty forecast list should return None for all horizons."""
        from datetime import datetime

        ref = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)
        result = forecast_at_horizons([], ref, [1.0, 2.0])

        assert result["1h"] is None
        assert result["2h"] is None

    def test_beyond_range_returns_none(self) -> None:
        """Horizons beyond forecast range should return None."""
        from datetime import datetime

        # Only 2 entries (1h and 2h ahead)
        entries = [
            ForecastEntry(
                datetime="2024-01-15T15:00:00+00:00",
                temperature=41.0,
                condition="cloudy",
                wind_speed=5.0,
                cloud_coverage=80.0,
                precipitation=0.0,
            ),
            ForecastEntry(
                datetime="2024-01-15T16:00:00+00:00",
                temperature=42.0,
                condition="cloudy",
                wind_speed=5.0,
                cloud_coverage=80.0,
                precipitation=0.0,
            ),
        ]
        ref = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)
        result = forecast_at_horizons(entries, ref, [1.0, 12.0])

        assert result["1h"] is not None
        # 12h is way beyond the 2 available entries (but within 90min of last)
        # The closest entry at 12h ahead would be 2am next day, but we only have 3pm and 4pm
        # 14:00 + 12h = 02:00 next day, closest is 16:00 = 10h off > 90min → None
        assert result["12h"] is None
