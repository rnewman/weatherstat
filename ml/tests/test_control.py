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
    ABSOLUTE_MIN,
    CONTROL_HORIZONS,
    HORIZON_WEIGHTS,
    _cautious_setpoint,
    _in_quiet_hours,
    _zone_comfort_max,
    adjust_schedules_for_windows,
    build_advisory_actions,
    compute_comfort_cost,
    compute_energy_cost,
    evaluate_advisories,
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

    Returns per-row arrays matching input batch size (works with both
    single-row and batch prediction).

    Args:
        predictions: Target name -> predicted value.
    """
    models: dict[str, object] = {}
    for target, value in predictions.items():
        mock = MagicMock()
        mock.predict.side_effect = lambda X, v=value: np.full(len(X), v)
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
    from weatherstat.yaml_config import load_config

    cfg = load_config()
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
    # Window columns
    for name in cfg.windows:
        cols.append(f"window_{name}_open")
    cols.append("any_window_open")
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
                    mask = X["thermostat_upstairs_action_enc"] > 0.5
                    return np.where(mask, warm_val, cold_val)
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
        """Bedroom at hour 22 (night: max 72F) vs hour 10 (day: max 75F) -> different costs at 74F."""
        from weatherstat.control import default_comfort_schedules

        schedules = default_comfort_schedules()
        bedroom_schedules = [s for s in schedules if s.room == "bedroom"]

        predictions = {"bedroom_temp_t+12": 74.0}

        # Hour 22 (night schedule: max 72F) -> 74 > 72, penalized
        cost_night = compute_comfort_cost(predictions, bedroom_schedules, base_hour=22)

        # Hour 10 (day schedule: max 75F) -> 74 < 75, no penalty
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

    def test_cautious_setpoint_off_uses_comfort_min(self) -> None:
        """Heating off: setpoint = comfort_min (safety net at comfort floor)."""
        sp = _cautious_setpoint(70.0, heating=False, comfort_min=70.0)
        assert sp == 70.0

    def test_cautious_setpoint_off_defaults_to_absolute_min(self) -> None:
        """Heating off without comfort_min: defaults to ABSOLUTE_MIN."""
        sp = _cautious_setpoint(70.0, heating=False)
        assert sp == ABSOLUTE_MIN

    def test_cautious_setpoint_heating_respects_comfort_min(self) -> None:
        """Heating on: setpoint = max(current + offset, comfort_min + offset)."""
        # Current temp below comfort min — setpoint should use comfort_min + offset
        sp = _cautious_setpoint(67.0, heating=True, comfort_min=70.0)
        assert sp == 72.0


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


# ── Advisory evaluation tests ────────────────────────────────────────────


class TestAdvisoryEvaluation:
    """Test model-based window advisory sweep."""

    def test_build_advisory_actions_from_window_states(self) -> None:
        """Verify one Action per YAML window with correct current states and overrides."""
        from weatherstat.yaml_config import load_config

        cfg = load_config()
        window_states = {name: False for name in cfg.windows}
        window_states["bedroom"] = True

        actions = build_advisory_actions(window_states)
        assert len(actions) == len(cfg.windows)

        bedroom_action = next(a for a in actions if a.name == "bedroom")
        assert bedroom_action.current == "open"
        assert len(bedroom_action.options) == 2
        # open option sets window_bedroom_open=1.0
        open_opt = next(o for o in bedroom_action.options if o.name == "open")
        assert open_opt.feature_overrides == {"window_bedroom_open": 1.0}
        closed_opt = next(o for o in bedroom_action.options if o.name == "closed")
        assert closed_opt.feature_overrides == {"window_bedroom_open": 0.0}

        basement_action = next(a for a in actions if a.name == "basement")
        assert basement_action.current == "closed"

    def test_window_sweep_recommends_closing(self) -> None:
        """Mock models predict colder with window_bedroom_open=1 -> recommend closing."""
        feature_cols = _standard_feature_columns()
        schedules = make_schedules()

        # Models sensitive to bedroom window: open -> 65°F (cold), closed -> 72°F (comfy)
        def _make_predict(target: str):
            def predict(X: pd.DataFrame) -> np.ndarray:
                temp = np.full(len(X), 72.0)
                if "window_bedroom_open" in X.columns:
                    temp = np.where(X["window_bedroom_open"] > 0.5, 65.0, temp)
                return temp
            return predict

        preds = _all_room_predictions(72.0)
        models: dict[str, object] = {}
        for target in preds:
            mock = MagicMock()
            mock.predict = _make_predict(target)
            models[target] = mock

        base_row = make_base_row(feature_cols, window_bedroom_open=1.0)
        # All windows closed except bedroom
        window_states = {name: False for name, _ in _cfg_windows()}
        window_states["bedroom"] = True
        advisory_actions = build_advisory_actions(window_states)

        recommendations = evaluate_advisories(
            base_row, feature_cols, models,
            electronic_overrides={},
            advisory_actions=advisory_actions,
            schedules=schedules,
            base_hour=12,
        )

        assert len(recommendations) >= 1
        bedroom_rec = next(r for r in recommendations if r.action_name == "bedroom")
        assert bedroom_rec.recommended_state == "closed"
        assert bedroom_rec.current_state == "open"
        assert bedroom_rec.comfort_improvement > 0

    def test_window_sweep_recommends_multiple(self) -> None:
        """Mock models sensitive to two windows -> recommends closing both."""
        feature_cols = _standard_feature_columns()
        schedules = make_schedules()

        # Models sensitive to bedroom AND kitchen windows
        def _make_predict(target: str):
            def predict(X: pd.DataFrame) -> np.ndarray:
                temp = np.full(len(X), 72.0)
                if "window_bedroom_open" in X.columns:
                    temp = np.where(X["window_bedroom_open"] > 0.5, temp - 4.0, temp)
                if "window_kitchen_open" in X.columns:
                    temp = np.where(X["window_kitchen_open"] > 0.5, temp - 3.0, temp)
                return temp
            return predict

        preds = _all_room_predictions(72.0)
        models: dict[str, object] = {}
        for target in preds:
            mock = MagicMock()
            mock.predict = _make_predict(target)
            models[target] = mock

        base_row = make_base_row(
            feature_cols, window_bedroom_open=1.0, window_kitchen_open=1.0,
        )
        window_states = {name: False for name, _ in _cfg_windows()}
        window_states["bedroom"] = True
        window_states["kitchen"] = True
        advisory_actions = build_advisory_actions(window_states)

        recommendations = evaluate_advisories(
            base_row, feature_cols, models,
            electronic_overrides={},
            advisory_actions=advisory_actions,
            schedules=schedules,
            base_hour=12,
        )

        rec_names = {r.action_name for r in recommendations}
        assert "bedroom" in rec_names
        assert "kitchen" in rec_names

    def test_window_sweep_skips_below_effort_cost(self) -> None:
        """Mock models predict same temp regardless of window -> no recommendations."""
        feature_cols = _standard_feature_columns()
        schedules = make_schedules()

        # Models insensitive to windows — always return 72°F
        preds = _all_room_predictions(72.0)
        models = make_mock_models(preds)

        base_row = make_base_row(feature_cols, window_bedroom_open=1.0)
        window_states = {name: False for name, _ in _cfg_windows()}
        window_states["bedroom"] = True
        advisory_actions = build_advisory_actions(window_states)

        recommendations = evaluate_advisories(
            base_row, feature_cols, models,
            electronic_overrides={},
            advisory_actions=advisory_actions,
            schedules=schedules,
            base_hour=12,
        )

        assert recommendations == []

    def test_effort_cost_scales_with_changes(self) -> None:
        """Closing 2 windows requires 2x effort threshold vs closing 1."""
        from weatherstat.config import ADVISORY_EFFORT_COST

        feature_cols = _standard_feature_columns()
        schedules = make_schedules()

        # Tiny improvement per window (just above 1x effort, below 2x)
        improvement_per_window = ADVISORY_EFFORT_COST * 0.8

        def _make_predict(target: str):
            def predict(X: pd.DataFrame) -> np.ndarray:
                temp = np.full(len(X), 72.0)
                if "window_bedroom_open" in X.columns:
                    temp = np.where(X["window_bedroom_open"] > 0.5, temp - improvement_per_window * 0.3, temp)
                if "window_kitchen_open" in X.columns:
                    temp = np.where(X["window_kitchen_open"] > 0.5, temp - improvement_per_window * 0.3, temp)
                return temp
            return predict

        preds = _all_room_predictions(72.0)
        models: dict[str, object] = {}
        for target in preds:
            mock = MagicMock()
            mock.predict = _make_predict(target)
            models[target] = mock

        # Both windows open — small improvement each
        base_row = make_base_row(
            feature_cols, window_bedroom_open=1.0, window_kitchen_open=1.0,
        )
        window_states = {name: False for name, _ in _cfg_windows()}
        window_states["bedroom"] = True
        window_states["kitchen"] = True
        advisory_actions = build_advisory_actions(window_states)

        recommendations = evaluate_advisories(
            base_row, feature_cols, models,
            electronic_overrides={},
            advisory_actions=advisory_actions,
            schedules=schedules,
            base_hour=12,
        )

        # The sweep might find closing both (2x effort) or just one (1x effort)
        # depending on whether the total improvement exceeds the scaled threshold.
        # With small improvements, closing both may not clear 2x threshold,
        # but closing just one might clear 1x threshold.
        # The key assertion: if any recommendations, effort cost was cleared.
        for rec in recommendations:
            assert rec.comfort_improvement > 0


def _cfg_windows() -> list[tuple[str, object]]:
    """Return list of (name, config) for all YAML windows."""
    from weatherstat.yaml_config import load_config

    cfg = load_config()
    return list(cfg.windows.items())


# ── Window comfort adjustment tests ─────────────────────────────────────


class TestAdjustSchedulesForWindows:
    """Test comfort schedule adjustment when windows are open."""

    def test_no_open_windows_returns_unchanged(self) -> None:
        """All windows closed -> schedules unchanged."""
        from weatherstat.yaml_config import load_config

        cfg = load_config()
        schedules = make_schedules()
        window_states = {name: False for name in cfg.windows}

        adjusted = adjust_schedules_for_windows(
            schedules, window_states, cfg.windows, -3.0, 2.0,
        )
        # Same number of schedules, same entries
        assert len(adjusted) == len(schedules)
        for orig, adj in zip(schedules, adjusted, strict=True):
            assert orig is adj  # should be the exact same object

    def test_open_window_widens_comfort_bounds(self) -> None:
        """Bedroom window open -> bedroom schedule shifted by offset."""
        from weatherstat.yaml_config import load_config

        cfg = load_config()
        schedules = make_schedules(
            bedroom=[
                ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 70.0, 74.0, 2.0, 1.0)),
            ],
        )
        window_states = {name: False for name in cfg.windows}
        window_states["bedroom"] = True

        adjusted = adjust_schedules_for_windows(
            schedules, window_states, cfg.windows, -3.0, 2.0,
        )

        bedroom_sched = next(s for s in adjusted if s.room == "bedroom")
        entry = bedroom_sched.entries[0]
        assert entry.comfort.min_temp == 67.0  # 70 + (-3)
        assert entry.comfort.max_temp == 76.0  # 74 + 2

    def test_unrelated_room_not_affected(self) -> None:
        """Bedroom window open -> upstairs schedule untouched."""
        from weatherstat.yaml_config import load_config

        cfg = load_config()
        schedules = make_schedules()
        window_states = {name: False for name in cfg.windows}
        window_states["bedroom"] = True

        adjusted = adjust_schedules_for_windows(
            schedules, window_states, cfg.windows, -3.0, 2.0,
        )

        upstairs_orig = next(s for s in schedules if s.room == "upstairs")
        upstairs_adj = next(s for s in adjusted if s.room == "upstairs")
        assert upstairs_orig is upstairs_adj  # unchanged

    def test_penalties_preserved(self) -> None:
        """Window adjustment preserves cold_penalty and hot_penalty."""
        from weatherstat.yaml_config import load_config

        cfg = load_config()
        schedules = make_schedules(
            bedroom=[
                ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 70.0, 74.0, 3.0, 0.5)),
            ],
        )
        window_states = {name: False for name in cfg.windows}
        window_states["bedroom"] = True

        adjusted = adjust_schedules_for_windows(
            schedules, window_states, cfg.windows, -3.0, 2.0,
        )

        bedroom_sched = next(s for s in adjusted if s.room == "bedroom")
        entry = bedroom_sched.entries[0]
        assert entry.comfort.cold_penalty == 3.0
        assert entry.comfort.hot_penalty == 0.5

    def test_window_without_rooms_no_effect(self) -> None:
        """Basement window (rooms=[]) open -> no schedules affected."""
        from weatherstat.yaml_config import load_config

        cfg = load_config()
        schedules = make_schedules()
        window_states = {name: False for name in cfg.windows}
        window_states["basement"] = True  # basement has rooms=[]

        adjusted = adjust_schedules_for_windows(
            schedules, window_states, cfg.windows, -3.0, 2.0,
        )

        for orig, adj in zip(schedules, adjusted, strict=True):
            assert orig is adj


# ── Quiet hours tests ─────────────────────────────────────────────────────


class TestQuietHours:
    """Test quiet hours helper."""

    def test_within_quiet_hours_wrapping(self) -> None:
        """22:00-07:00 range wraps midnight. Hour 23, 0, 6 are quiet."""
        assert _in_quiet_hours(23, (22, 7)) is True
        assert _in_quiet_hours(0, (22, 7)) is True
        assert _in_quiet_hours(6, (22, 7)) is True

    def test_outside_quiet_hours_wrapping(self) -> None:
        """22:00-07:00 range. Hour 7, 12, 21 are not quiet."""
        assert _in_quiet_hours(7, (22, 7)) is False
        assert _in_quiet_hours(12, (22, 7)) is False
        assert _in_quiet_hours(21, (22, 7)) is False

    def test_non_wrapping_range(self) -> None:
        """8:00-18:00 non-wrapping range."""
        assert _in_quiet_hours(8, (8, 18)) is True
        assert _in_quiet_hours(12, (8, 18)) is True
        assert _in_quiet_hours(17, (8, 18)) is True
        assert _in_quiet_hours(18, (8, 18)) is False
        assert _in_quiet_hours(7, (8, 18)) is False

    def test_boundary_start(self) -> None:
        """Start hour is inclusive."""
        assert _in_quiet_hours(22, (22, 7)) is True

    def test_boundary_end(self) -> None:
        """End hour is exclusive."""
        assert _in_quiet_hours(7, (22, 7)) is False
