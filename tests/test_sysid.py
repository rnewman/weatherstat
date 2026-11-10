"""Tests for sysid internals whose semantics must survive performance work."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

import weatherstat.sysid as sysid
from weatherstat.sysid import _MIN_SEGMENT_STEPS, SensorSpec, _find_uncontrolled_segments

_SENSOR = SensorSpec(name="bedroom_temp", temp_column="bedroom_temp", yaml_tau_base=45.0)


def _night(n: int, **cols) -> pd.DataFrame:
    """n dark 5-min rows (sun well below 5°), every row qualifying unless overridden."""
    df = pd.DataFrame({
        "_ts": pd.date_range("2026-01-10 08:00", periods=n, freq="5min", tz="UTC"),
        "_solar_elev_deg": np.full(n, -20.0),
        "bedroom_temp": np.linspace(68.0, 64.0, n),
        "_outdoor_best": np.full(n, 40.0),
        "bedroom_open": pd.array([0] * n, dtype="boolean"),
    })
    for name, values in cols.items():
        df[name] = values
    return df


def _ranges(segments: list[pd.DataFrame]) -> list[tuple[int, int]]:
    return [(int(s.index[0]), int(s.index[-1])) for s in segments]


class TestUncontrolledSegments:
    def test_single_skipped_row_does_not_split(self) -> None:
        """A 10-minute step (one non-qualifying row) stays inside the segment."""
        temps = np.linspace(68.0, 64.0, 30)
        temps[15] = np.nan
        segs = _find_uncontrolled_segments(_night(30, bedroom_temp=temps), [], _SENSOR)
        assert _ranges(segs) == [(0, 29)]
        assert 15 not in segs[0].index
        assert len(segs[0]) == 29

    def test_longer_gap_splits(self) -> None:
        temps = np.linspace(68.0, 64.0, 40)
        temps[[18, 19]] = np.nan  # 15-minute step → break
        segs = _find_uncontrolled_segments(_night(40, bedroom_temp=temps), [], _SENSOR)
        assert _ranges(segs) == [(0, 17), (20, 39)]

    def test_short_segments_dropped(self) -> None:
        temps = np.linspace(68.0, 64.0, 30)
        temps[[_MIN_SEGMENT_STEPS - 1, _MIN_SEGMENT_STEPS]] = np.nan  # first piece is 11 rows
        segs = _find_uncontrolled_segments(_night(30, bedroom_temp=temps), [], _SENSOR)
        assert _ranges(segs) == [(_MIN_SEGMENT_STEPS + 1, 29)]

    def test_daylight_rows_excluded(self) -> None:
        elev = np.full(36, -20.0)
        elev[12:24] = 10.0  # sun up in the middle
        segs = _find_uncontrolled_segments(_night(36, _solar_elev_deg=elev), [], _SENSOR)
        assert _ranges(segs) == [(0, 11), (24, 35)]

    def test_environment_transition_disqualifies_segment(self) -> None:
        state = pd.array([0] * 20 + [1] * 10 + [0] * 2 + [1] * 30, dtype="boolean")
        temps = np.linspace(68.0, 64.0, 62)
        temps[[30, 31]] = np.nan  # split: rows 0–29 (has a 0→1 transition), 32–61 (constant 1)
        segs = _find_uncontrolled_segments(_night(62, bedroom_temp=temps, bedroom_open=state), [], _SENSOR)
        assert _ranges(segs) == [(32, 61)]

    def test_missing_environment_state_is_not_a_transition(self) -> None:
        """NaN means the sensor didn't exist yet — constant unknown, not a change."""
        state = pd.array([None] * 10 + [1] * 20, dtype="boolean")
        segs = _find_uncontrolled_segments(_night(30, bedroom_open=state), [], _SENSOR)
        assert _ranges(segs) == [(0, 29)]

    def test_segment_frames_carry_fit_columns_in_order(self) -> None:
        segs = _find_uncontrolled_segments(_night(20), [], _SENSOR)
        assert list(segs[0].columns) == ["_ts", "bedroom_temp", "_outdoor_best"]
        assert segs[0]["_ts"].is_monotonic_increasing


class TestTauCurveMemo:
    def _segment(self, tau: float = 42.0):
        t = np.arange(24) * (5 / 60)
        return t, 40.0 + 25.0 * np.exp(-t / tau)

    def test_memo_matches_fit_and_skips_refit(self) -> None:
        sysid._TAU_CURVE_MEMO.clear()
        t, temps = self._segment()
        expected = sysid._fit_tau_curve_uncached(t, temps, 40.0)
        assert sysid._fit_tau_curve(t, temps, 40.0) == expected
        with patch.object(sysid, "curve_fit", side_effect=AssertionError("refit")):
            assert sysid._fit_tau_curve(t.copy(), temps.copy(), 40.0) == expected  # equal content → hit

    def test_changed_samples_refit(self) -> None:
        sysid._TAU_CURVE_MEMO.clear()
        t, temps = self._segment()
        first = sysid._fit_tau_curve(t, temps, 40.0)
        temps2 = temps.copy()
        temps2[-1] += 0.3
        assert sysid._fit_tau_curve(t, temps2, 40.0) == sysid._fit_tau_curve_uncached(t, temps2, 40.0) != first
        assert len(sysid._TAU_CURVE_MEMO) == 2

    def test_failures_are_memoized_too(self) -> None:
        sysid._TAU_CURVE_MEMO.clear()
        t, temps = self._segment()
        assert sysid._fit_tau_curve(t, temps, temps[0] - 0.5) is None  # too close to outdoor: no fit
        assert len(sysid._TAU_CURVE_MEMO) == 1
