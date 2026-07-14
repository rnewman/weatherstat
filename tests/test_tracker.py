"""Tests for the multi-timescale residual tracker (tracker.py)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from weatherstat import tracker as tracker_mod
from weatherstat.tracker import SOLAR_BLOCK, TrackedBlock, load_tracked_solar_gains, solve_tracked_block

_RNG = np.random.default_rng(42)


def _sunny_window(n: int = 1500, k: int = 3) -> np.ndarray:
    """Synthetic block features with solar-like scale (nonneg, mean ~0.4)."""
    F = np.abs(_RNG.normal(0.4, 0.25, size=(n, k)))
    return np.clip(F, 0.0, 1.0)


class TestSolveTrackedBlock:
    def test_converges_toward_truth_under_strong_signal(self) -> None:
        F = _sunny_window()
        gamma_true = np.array([0.8, 0.1, 0.6])
        r = F @ gamma_true + _RNG.normal(0, 0.05, size=len(F))
        gamma_prev = np.zeros(3)

        g1, info, updated = solve_tracked_block(F, r, gamma_prev, SOLAR_BLOCK)
        assert updated and info > SOLAR_BLOCK.min_information
        # prior_ratio=1.0 → roughly halfway per update on full signal
        assert 0.3 <= g1[0] <= 0.7

        g2, _, _ = solve_tracked_block(F, r, g1, SOLAR_BLOCK)
        assert np.allclose(g2, gamma_true, atol=0.25)
        assert abs(g2[0] - gamma_true[0]) < abs(g1[0] - gamma_true[0])

    def test_step_change_moves_gamma_down(self) -> None:
        """Canopy leaf-out: transmission drops; γ follows within a few updates."""
        F = _sunny_window()
        gamma_leafed = np.array([0.5, 0.05, 0.1])  # west response collapsed
        r = F @ gamma_leafed + _RNG.normal(0, 0.05, size=len(F))
        gamma = np.array([0.5, 0.05, 0.7])  # bare-season estimate

        trajectory = []
        for _ in range(6):
            gamma, _, _ = solve_tracked_block(F, r, gamma, SOLAR_BLOCK)
            trajectory.append(gamma[2])
        # Monotone decay toward the new regime; hourly updates make this
        # transient ~half a day of sunny data in production.
        assert all(b < a for a, b in zip(trajectory, trajectory[1:], strict=False))
        assert gamma[2] < 0.25

    def test_dark_window_holds_previous(self) -> None:
        """A dark fortnight (features ~0) must not move the estimate."""
        F = np.full((1500, 3), 0.01)
        r = _RNG.normal(0, 0.5, size=1500)
        gamma_prev = np.array([0.5, 0.1, 0.7])

        gamma, info, updated = solve_tracked_block(F, r, gamma_prev, SOLAR_BLOCK)
        assert not updated
        assert info < SOLAR_BLOCK.min_information
        assert np.array_equal(gamma, gamma_prev)

    def test_bounds_clamp_sign_and_magnitude(self) -> None:
        F = _sunny_window()
        # Confounded window: residual anti-correlates with feature 1
        r = F @ np.array([0.5, -2.0, 8.0])
        gamma = np.zeros(3)
        for _ in range(6):
            gamma, _, _ = solve_tracked_block(F, r, gamma, SOLAR_BLOCK)
        assert gamma[1] >= SOLAR_BLOCK.lower  # no negative solar response
        assert gamma[2] <= SOLAR_BLOCK.upper  # magnitude cap

    def test_custom_block_gate(self) -> None:
        block = TrackedBlock(name="x", prefixes=("_x_",), min_information=1e9)
        F = _sunny_window()
        gamma_prev = np.array([1.0, 1.0, 1.0])
        gamma, _, updated = solve_tracked_block(F, F @ gamma_prev, gamma_prev, block)
        assert not updated and np.array_equal(gamma, gamma_prev)


class TestLoadTrackedSolarGains:
    def _write_state(self, tmp_path, monkeypatch, updated_at: datetime) -> None:
        state = {
            "blocks": {
                "solar": {
                    "sensors": {
                        "kitchen_temp": {
                            "coefficients": {"_solar_elev": 0.3, "_solar_west": 0.5},
                            "updated_at": updated_at.isoformat(),
                            "information": 80.0,
                            "n_rows": 3000,
                        }
                    }
                }
            }
        }
        p = tmp_path / "tracker_state.json"
        p.write_text(json.dumps(state))
        monkeypatch.setattr(tracker_mod, "TRACKER_STATE_FILE", p)

    def test_fresh_state_loaded(self, tmp_path, monkeypatch) -> None:
        self._write_state(tmp_path, monkeypatch, datetime.now(UTC) - timedelta(hours=2))
        gains = load_tracked_solar_gains()
        assert gains["kitchen_temp"]["_solar_west"] == pytest.approx(0.5)

    def test_stale_state_excluded(self, tmp_path, monkeypatch) -> None:
        self._write_state(tmp_path, monkeypatch, datetime.now(UTC) - timedelta(days=5))
        assert load_tracked_solar_gains(max_age_hours=48.0) == {}

    def test_missing_state_file(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(tracker_mod, "TRACKER_STATE_FILE", tmp_path / "nope.json")
        assert load_tracked_solar_gains() == {}
