"""Unit tests for weather.py: solar fraction transfer functions."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from weatherstat.weather import (
    condition_to_solar_fraction,
    effective_solar_fraction,
    kasten_czeplak,
    plane_incidence,
    solar_azimuth,
    solar_sin_elevation,
)


class TestKastenCzeplak:
    """Kasten-Czeplak (1980) clear-sky attenuation."""

    def test_zero_clouds_full_irradiance(self) -> None:
        assert kasten_czeplak(0.0) == pytest.approx(1.0)

    def test_full_overcast_lower_bound(self) -> None:
        # f = 1 - 0.75 = 0.25 at cc=100
        assert kasten_czeplak(100.0) == pytest.approx(0.25)

    def test_partly_cloudy_above_linear(self) -> None:
        # At 50%, 1 - 0.75*(0.5)^3.4 ≈ 0.929 — far above naive linear (0.5)
        # and matches the empirical "sunny enough to still warm a south wall"
        # observation that justified the curve choice.
        v = kasten_czeplak(50.0)
        assert 0.9 < v < 0.95
        assert v > 0.5  # confirm we're above the naive linear midpoint

    def test_thirty_percent_close_to_clear(self) -> None:
        # Most "sunny" days in Seattle have 5-25% cc; output should stay >0.97
        # for the regression to behave like the old sunny=1.0 calibration.
        assert kasten_czeplak(20.0) > 0.97
        assert kasten_czeplak(30.0) > 0.95

    def test_sharp_dropoff_above_sixty(self) -> None:
        # Above 60% cc the curve drops faster than at low cc; by 100% we're at
        # 0.25. Exact values: cc=60→0.87, cc=80→0.65, cc=90→0.48.
        assert kasten_czeplak(60.0) == pytest.approx(0.868, abs=0.01)
        assert kasten_czeplak(80.0) == pytest.approx(0.649, abs=0.01)
        assert kasten_czeplak(60.0) < kasten_czeplak(50.0)
        assert kasten_czeplak(80.0) < kasten_czeplak(60.0)

    def test_clamps_negative(self) -> None:
        # Defensive: HA could send a negative value via attribute glitches.
        assert kasten_czeplak(-10.0) == pytest.approx(1.0)

    def test_clamps_above_hundred(self) -> None:
        assert kasten_czeplak(150.0) == pytest.approx(0.25)

    def test_monotonic_decreasing(self) -> None:
        # Sanity: any time more clouds means less sun.
        prev = math.inf
        for cc in range(0, 101, 5):
            v = kasten_czeplak(float(cc))
            assert v <= prev
            prev = v


class TestEffectiveSolarFraction:
    """Unified solar fraction helper used by sysid + control + TUI + scripts."""

    def test_cloud_coverage_takes_precedence(self) -> None:
        # When both are present, cloud_coverage wins — the categorical map
        # would have given 1.0 here ("sunny"), but at cc=80 we're attenuated.
        v = effective_solar_fraction(80.0, "sunny")
        assert v == pytest.approx(kasten_czeplak(80.0))
        # Sanity: the kasten path returns something materially different from
        # the categorical sunny=1.0 result — confirming precedence.
        assert v < 0.7
        assert v < condition_to_solar_fraction("sunny")

    def test_clear_cloud_coverage_matches_sunny_categorical(self) -> None:
        # cc=0 → 1.0, identical to the old "sunny" categorical result.
        assert effective_solar_fraction(0.0, "sunny") == pytest.approx(1.0)
        assert effective_solar_fraction(0.0, None) == pytest.approx(1.0)

    def test_falls_back_to_categorical_when_cc_missing(self) -> None:
        # Pre-April 2026 snapshots and forecasts without cloud_coverage.
        assert effective_solar_fraction(None, "sunny") == pytest.approx(1.0)
        assert effective_solar_fraction(None, "cloudy") == pytest.approx(0.15)
        assert effective_solar_fraction(None, "partlycloudy") == pytest.approx(0.5)

    def test_categorical_matches_legacy_helper(self) -> None:
        # Drop-in equivalence on the fallback path.
        for cond in ["sunny", "cloudy", "partlycloudy", "rainy", "fog", "unknown"]:
            assert effective_solar_fraction(None, cond) == condition_to_solar_fraction(cond)

    def test_default_when_neither_provided(self) -> None:
        # Total absence of weather data — use the neutral default rather than
        # raising or returning a misleading 0.
        assert effective_solar_fraction(None, None) == pytest.approx(0.3)
        assert effective_solar_fraction(None, "") == pytest.approx(0.3)

    def test_cloud_coverage_with_unknown_condition(self) -> None:
        # Even if condition is bogus, cloud_coverage drives the result.
        assert effective_solar_fraction(20.0, "moonshine") == pytest.approx(kasten_czeplak(20.0))


class TestSolarAzimuthAndPlanes:
    """Solar azimuth and wall-plane incidence (Seattle: 47.66N, -122.40W)."""

    LAT, LON = 47.66, -122.40

    def _utc(self, month: int, day: int, hour: int, minute: int = 0) -> datetime:
        return datetime(2026, month, day, hour, minute, tzinfo=UTC)

    def test_azimuth_south_at_solar_noon(self) -> None:
        # Solar noon at lon -122.4 is ~12:00 + 122.4/15 h = ~20:10 UTC
        az = solar_azimuth(self.LAT, self.LON, self._utc(6, 21, 20, 10))
        assert 170 <= az <= 190

    def test_azimuth_east_in_morning_west_in_evening(self) -> None:
        morning = solar_azimuth(self.LAT, self.LON, self._utc(6, 21, 15, 0))  # ~7am local
        evening = solar_azimuth(self.LAT, self.LON, self._utc(6, 22, 1, 0))  # ~6pm local
        assert 60 <= morning <= 120
        assert 240 <= evening <= 300

    def test_horizontal_plane_matches_sin_elevation(self) -> None:
        dt = self._utc(6, 21, 20, 10)
        assert plane_incidence(self.LAT, self.LON, dt, None) == pytest.approx(
            solar_sin_elevation(self.LAT, self.LON, dt)
        )

    def test_west_wall_dark_in_morning_lit_in_evening(self) -> None:
        morning = self._utc(6, 21, 15, 0)
        evening = self._utc(6, 22, 1, 0)
        assert plane_incidence(self.LAT, self.LON, morning, 270.0) == 0.0
        assert plane_incidence(self.LAT, self.LON, evening, 270.0) > 0.3
        # And the east wall mirrors it
        assert plane_incidence(self.LAT, self.LON, morning, 90.0) > 0.3
        assert plane_incidence(self.LAT, self.LON, evening, 90.0) == 0.0

    def test_night_is_zero_everywhere(self) -> None:
        midnight = self._utc(6, 21, 9, 0)  # ~1am local (PDT = UTC-7)
        for plane in (None, 90.0, 180.0, 270.0):
            assert plane_incidence(self.LAT, self.LON, midnight, plane) == 0.0
