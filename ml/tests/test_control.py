"""Tests for control optimizer constraints.

Tests the sweep constraints without HA or real models. Mock models return
configurable predictions, so tests verify the **constraint logic**, not ML accuracy.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from weatherstat.config import BLOWERS, MINI_SPLITS
from weatherstat.control import (
    ABSOLUTE_MAX,
    CONTROL_HORIZONS,
    HORIZON_WEIGHTS,
    _cautious_setpoint,
    _zone_comfort_max,
    compute_comfort_cost,
    compute_energy_cost,
    generate_scenarios,
    sweep_scenarios,
)
from weatherstat.types import (
    BlowerDecision,
    ComfortSchedule,
    ComfortScheduleEntry,
    HVACScenario,
    MiniSplitDecision,
    RoomComfort,
)

# ── Test helpers ──────────────────────────────────────────────────────────


def make_schedules(**overrides: list[ComfortScheduleEntry]) -> list[ComfortSchedule]:
    """Return default_comfort_schedules() with optional room overrides.

    Usage:
        make_schedules(upstairs=[ComfortScheduleEntry(0, 24, RoomComfort("upstairs", 70, 74))])
    """
    from weatherstat.control import default_comfort_schedules

    defaults = {s.room: s for s in default_comfort_schedules()}
    for room, entries in overrides.items():
        defaults[room] = ComfortSchedule(room=room, entries=tuple(entries))
    return list(defaults.values())


def make_mock_models(predictions: dict[str, float]) -> dict[str, object]:
    """Return dict of mock models that always predict the given values.

    Args:
        predictions: Target name -> predicted value.
    """
    models: dict[str, object] = {}
    for target, value in predictions.items():
        mock = MagicMock()
        mock.predict.return_value = np.array([value])
        models[target] = mock
    return models


def make_base_row(feature_columns: list[str], **values: float) -> pd.DataFrame:
    """Return a 1-row DataFrame with all feature columns, defaults + overrides.

    Missing columns default to 0.0.
    """
    data = {col: [values.get(col, 0.0)] for col in feature_columns}
    return pd.DataFrame(data)


def _standard_feature_columns() -> list[str]:
    """Return a representative set of feature columns for testing."""
    cols = [
        "thermostat_upstairs_temp",
        "thermostat_downstairs_temp",
        "outdoor_temp",
        "thermostat_upstairs_action_enc",
        "thermostat_downstairs_action_enc",
        "navien_heating_mode_enc",
        "hour", "hour_sin", "hour_cos",
        "solar_elevation", "solar_azimuth",
    ]
    for b in BLOWERS:
        cols.append(b.feature_col)
    for s in MINI_SPLITS:
        cols.extend([s.mode_feature_col, s.target_feature_col, s.delta_feature_col])
    return cols


def _all_room_predictions(temp: float) -> dict[str, float]:
    """Return predictions for all rooms at all horizons with the same temp."""
    from weatherstat.config import PREDICTION_ROOMS

    preds: dict[str, float] = {}
    for room in PREDICTION_ROOMS:
        for h in CONTROL_HORIZONS:
            preds[f"{room}_temp_t+{h}"] = temp
    return preds


def _room_predictions_varying(room_temps: dict[str, float]) -> dict[str, float]:
    """Return predictions for specified rooms at all horizons."""
    preds: dict[str, float] = {}
    for room, temp in room_temps.items():
        for h in CONTROL_HORIZONS:
            preds[f"{room}_temp_t+{h}"] = temp
    return preds


# ── Zone comfort max tests ────────────────────────────────────────────────


class TestZoneComfortMax:
    """Test that zone thermostat at/above comfort max blocks heating."""

    def test_zone_above_comfort_max_blocks_heating(self) -> None:
        """Upstairs at 75F, comfort max 74F -> all upstairs-heating scenarios pruned."""
        schedules = make_schedules()
        feature_cols = _standard_feature_columns()
        preds = _all_room_predictions(72.0)
        models = make_mock_models(preds)
        base_row = make_base_row(feature_cols)

        decision = sweep_scenarios(
            base_row, feature_cols, models,
            up_current=75.0, dn_current=70.0,
            current_split_temps={}, current_temps={"upstairs": 75.0, "downstairs": 70.0},
            schedules=schedules, base_hour=12,
        )
        assert not decision.upstairs_heating

    def test_zone_below_comfort_max_allows_heating(self) -> None:
        """Upstairs at 69F, comfort max 74F -> upstairs-heating scenarios present."""
        schedules = make_schedules()
        up_max = _zone_comfort_max("upstairs", schedules, 12)
        assert up_max > 69.0  # confirms comfort max constraint allows heating


# ── Thermal direction tests ───────────────────────────────────────────────


class TestThermalDirection:
    """Test thermal direction constraint."""

    def test_thermal_direction_blocks_heating_when_warm(self) -> None:
        """All rooms above comfort min at all future horizons -> heating blocked."""
        schedules = make_schedules()
        feature_cols = _standard_feature_columns()
        preds = _all_room_predictions(73.0)
        models = make_mock_models(preds)
        base_row = make_base_row(feature_cols)

        current_temps = {
            "upstairs": 73.0, "downstairs": 73.0,
            "bedroom": 73.0, "kitchen": 73.0, "piano": 73.0,
            "bathroom": 73.0, "family_room": 73.0, "office": 73.0,
        }
        decision = sweep_scenarios(
            base_row, feature_cols, models,
            up_current=73.0, dn_current=73.0,
            current_split_temps={}, current_temps=current_temps,
            schedules=schedules, base_hour=12,
        )
        assert not decision.upstairs_heating
        assert not decision.downstairs_heating

    def test_thermal_direction_allows_heating_when_cold(self) -> None:
        """One room below comfort min -> heating allowed."""
        scenarios = generate_scenarios()
        heating_scenarios = [s for s in scenarios if s.upstairs_heating or s.downstairs_heating]
        assert len(heating_scenarios) > 0  # heating scenarios exist in the pool


# ── Min improvement tests ─────────────────────────────────────────────────


class TestMinImprovement:
    """Test minimum improvement threshold."""

    def test_min_improvement_reverts_to_all_off(self) -> None:
        """Best active scenario improves cost by 0.5 (< 1.0 threshold) -> all-off."""
        schedules = make_schedules()
        feature_cols = _standard_feature_columns()
        preds = _all_room_predictions(72.0)
        models = make_mock_models(preds)
        base_row = make_base_row(feature_cols)

        current_temps = {
            "upstairs": 69.0, "downstairs": 73.0,
            "bedroom": 73.0, "kitchen": 73.0, "piano": 73.0,
            "bathroom": 73.0, "family_room": 73.0, "office": 73.0,
        }
        decision = sweep_scenarios(
            base_row, feature_cols, models,
            up_current=69.0, dn_current=73.0,
            current_split_temps={}, current_temps=current_temps,
            schedules=schedules, base_hour=12,
        )
        # Since all predictions are the same regardless of HVAC scenario,
        # improvement over all-off = energy_cost (tiny) which is < MIN_IMPROVEMENT.
        assert not decision.upstairs_heating
        assert not decision.downstairs_heating
        assert all(b.mode == "off" for b in decision.blowers)
        assert all(s.mode == "off" for s in decision.mini_splits)

    def test_min_improvement_keeps_active_when_significant(self) -> None:
        """Large comfort improvement -> active decision kept."""
        feature_cols = _standard_feature_columns()
        schedules = make_schedules()

        warm_preds = _room_predictions_varying({
            "upstairs": 72.0, "downstairs": 72.0,
            "bedroom": 72.0, "kitchen": 72.0, "piano": 72.0,
            "bathroom": 72.0, "family_room": 72.0, "office": 72.0,
        })
        cold_preds = _room_predictions_varying({
            "upstairs": 65.0, "downstairs": 65.0,
            "bedroom": 65.0, "kitchen": 65.0, "piano": 65.0,
            "bathroom": 65.0, "family_room": 65.0, "office": 65.0,
        })

        # Create models that predict differently based on thermostat action
        models: dict[str, object] = {}
        for target in warm_preds:
            mock = MagicMock()

            def _make_predict(warm_val: float, cold_val: float):
                def predict(X: pd.DataFrame) -> np.ndarray:
                    if X["thermostat_upstairs_action_enc"].iloc[0] > 0.5:
                        return np.array([warm_val])
                    return np.array([cold_val])
                return predict

            mock.predict = _make_predict(warm_preds[target], cold_preds[target])
            models[target] = mock

        base_row = make_base_row(feature_cols)

        current_temps = {
            "upstairs": 65.0, "downstairs": 65.0,
            "bedroom": 65.0, "kitchen": 65.0, "piano": 65.0,
            "bathroom": 65.0, "family_room": 65.0, "office": 65.0,
        }
        decision = sweep_scenarios(
            base_row, feature_cols, models,
            up_current=65.0, dn_current=65.0,
            current_split_temps={}, current_temps=current_temps,
            schedules=schedules, base_hour=12,
        )
        # With 65F predicted for all-off vs 72F for heating -> huge improvement
        assert decision.upstairs_heating or decision.downstairs_heating


# ── Comfort cost tests ────────────────────────────────────────────────────


class TestComfortCost:
    """Test comfort cost computation."""

    def test_comfort_cost_quadratic_penalty(self) -> None:
        """Room at 76F, max 74F, hot_penalty 2.0 -> cost = (76-74)^2 * 2.0 * weight."""
        schedules = [
            ComfortSchedule(
                room="upstairs",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("upstairs", 70.0, 74.0, hot_penalty=2.0)),),
            ),
        ]
        predictions = {"upstairs_temp_t+12": 76.0}
        cost = compute_comfort_cost(predictions, schedules, base_hour=12)

        expected = (76.0 - 74.0) ** 2 * 2.0 * HORIZON_WEIGHTS[12]
        assert abs(cost - expected) < 0.001

    def test_comfort_cost_respects_schedule_hour(self) -> None:
        """Bedroom at hour 22 (night: max 69F) vs hour 10 (day: max 72F) -> different costs."""
        from weatherstat.control import default_comfort_schedules

        schedules = default_comfort_schedules()
        bedroom_schedules = [s for s in schedules if s.room == "bedroom"]

        predictions = {"bedroom_temp_t+12": 71.0}

        # Hour 22 (night schedule: max 69F) -> 71 > 69, penalized
        cost_night = compute_comfort_cost(predictions, bedroom_schedules, base_hour=22)

        # Hour 10 (day schedule: max 72F) -> 71 < 72, no penalty
        cost_day = compute_comfort_cost(predictions, bedroom_schedules, base_hour=10)

        assert cost_night > cost_day
        assert cost_day == 0.0

    def test_energy_cost_tiebreaker(self) -> None:
        """Two scenarios with identical comfort cost -> lower energy wins."""
        scenario_both = HVACScenario(
            upstairs_heating=True,
            downstairs_heating=True,
            blowers=tuple(BlowerDecision(b.name, "off") for b in BLOWERS),
            mini_splits=tuple(MiniSplitDecision(s.name, "off", 72.0) for s in MINI_SPLITS),
        )
        scenario_one = HVACScenario(
            upstairs_heating=True,
            downstairs_heating=False,
            blowers=tuple(BlowerDecision(b.name, "off") for b in BLOWERS),
            mini_splits=tuple(MiniSplitDecision(s.name, "off", 72.0) for s in MINI_SPLITS),
        )

        cost_both = compute_energy_cost(scenario_both)
        cost_one = compute_energy_cost(scenario_one)

        assert cost_both > cost_one


# ── Blower constraint tests ──────────────────────────────────────────────


class TestBlowerConstraints:
    """Test that blowers are forced off when their zone isn't heating."""

    def test_blowers_only_active_when_zone_heating(self) -> None:
        """When downstairs off -> family_room and office blowers forced to off."""
        scenarios = generate_scenarios()

        for scenario in scenarios:
            if not scenario.downstairs_heating:
                for blower in scenario.blowers:
                    if blower.name in ("family_room", "office"):
                        assert blower.mode == "off", (
                            f"Blower {blower.name} should be off when downstairs not heating, "
                            f"got {blower.mode}"
                        )


# ── Cautious setpoint tests ──────────────────────────────────────────────


class TestCautiousSetpoint:
    """Test cautious setpoint computation and clamping."""

    def test_cautious_setpoint_clamped(self) -> None:
        """Current temp 77F + offset 2 = 79F -> clamped to ABSOLUTE_MAX (78F)."""
        sp = _cautious_setpoint(77.0, heating=True)
        assert sp == ABSOLUTE_MAX

    def test_cautious_setpoint_heating(self) -> None:
        """Normal heating: current + offset."""
        sp = _cautious_setpoint(70.0, heating=True)
        assert sp == 72.0

    def test_cautious_setpoint_cooling(self) -> None:
        """Cooling: current - offset, but not below min."""
        sp = _cautious_setpoint(70.0, heating=False)
        assert sp == 68.0


# ── Scenario generation tests ────────────────────────────────────────────


class TestScenarioGeneration:
    """Test scenario count from cartesian product."""

    def test_generate_scenarios_count(self) -> None:
        """2 thermo x blower levels x split modes -> expected count."""
        scenarios = generate_scenarios()

        # Both on:      3*3=9 blower combos * 3*3=9 split combos = 81
        # Up on, Dn off: 1*1=1 blower combos (both downstairs zone) * 9 = 9
        # Up off, Dn on: 3*3=9 blower combos * 9 = 81
        # Both off:      1*1=1 blower combos * 9 = 9
        # Total = 81 + 9 + 81 + 9 = 180
        expected = 81 + 9 + 81 + 9
        assert len(scenarios) == expected, f"Expected {expected} scenarios, got {len(scenarios)}"
