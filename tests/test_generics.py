"""Tests for generic N-effector scenarios, dependency resolution, and energy costs.

Verifies that the unified effector model handles:
- Dependency constraints (single and multi-parent OR gate)
- Scenario count scaling with effector count
- Per-effector energy costs (scalar and per-mode dict)
- Simulator dependency gating for blowers
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from weatherstat.config import EFFECTOR_MAP, EFFECTORS
from weatherstat.control import (
    compute_energy_cost,
    generate_trajectory_scenarios,
)
from weatherstat.simulator import HouseState, load_sim_params, predict
from weatherstat.types import EffectorDecision, Scenario

# ── Dependency constraint tests ──────────────────────────────────────────


class TestDependencyConstraints:
    """Verify dependent effectors are pruned/expanded based on parent state."""

    def test_blowers_off_when_no_thermostat_heating(self) -> None:
        """All blowers should be off in scenarios where their parent thermostat is off."""
        scenarios = generate_trajectory_scenarios()
        for s in scenarios:
            dn = s.effectors.get("thermostat_downstairs")
            dn_off = dn is None or dn.mode == "off"
            dn_delayed = dn is not None and dn.mode != "off" and dn.delay_steps > 0
            if dn_off or dn_delayed:
                # All blowers depending on thermostat_downstairs should be off
                for bname in ("blower_family_room", "blower_office", "blower_gym"):
                    bed = s.effectors.get(bname)
                    assert bed is None or bed.mode == "off", (
                        f"{bname} should be off when thermostat_downstairs is off/delayed, "
                        f"but got mode={bed.mode if bed else 'missing'}"
                    )

    def test_blowers_swept_when_thermostat_immediate(self) -> None:
        """Blowers should have multiple modes when parent thermostat is on with delay=0."""
        scenarios = generate_trajectory_scenarios()
        # Find scenarios where thermostat_downstairs is on with no delay
        immediate = [
            s for s in scenarios
            if (dn := s.effectors.get("thermostat_downstairs")) is not None
            and dn.mode != "off" and dn.delay_steps == 0
        ]
        assert len(immediate) > 0, "Should have scenarios with immediate downstairs heating"

        # Among those, blowers should have multiple modes
        blower_modes: set[str] = set()
        for s in immediate:
            bfr = s.effectors.get("blower_family_room")
            if bfr is not None:
                blower_modes.add(bfr.mode)
        assert len(blower_modes) >= 2, f"Blower should have multiple modes, got {blower_modes}"
        assert "off" in blower_modes

    def test_multi_parent_and_gate_swept_when_all_active(self) -> None:
        """A dependent effector with multiple parents should be swept only when ALL parents are active."""
        from weatherstat.config import EffectorConfig

        # Create a mock effector that depends on both thermostats
        mock_eff = EffectorConfig(
            name="blower_hallway",
            entity_id="fan.blower_hallway",
            control_type="binary",
            mode_control="automatic",
            supported_modes=("off", "low", "high"),
            command_keys={"mode": "blowerHallwayMode"},
            depends_on=("thermostat_upstairs", "thermostat_downstairs"),
            mode_encoding={"off": 0, "low": 1, "high": 2},
            energy_cost={"off": 0.0, "low": 0.001, "high": 0.002},
        )
        mock_effectors = EFFECTORS + (mock_eff,)
        mock_map = {e.name: e for e in mock_effectors}

        with (
            patch("weatherstat.control.EFFECTORS", mock_effectors),
            patch("weatherstat.control.EFFECTOR_MAP", mock_map),
        ):
            scenarios = generate_trajectory_scenarios()

        # Find scenarios where BOTH thermostats are immediately on
        both_on = [
            s for s in scenarios
            if (up := s.effectors.get("thermostat_upstairs")) is not None
            and up.mode != "off" and up.delay_steps == 0
            and (dn := s.effectors.get("thermostat_downstairs")) is not None
            and dn.mode != "off" and dn.delay_steps == 0
        ]
        assert len(both_on) > 0
        # Hallway blower should be swept (multiple modes) when both parents active
        hallway_modes = {
            s.effectors.get("blower_hallway", EffectorDecision("blower_hallway")).mode
            for s in both_on
        }
        assert len(hallway_modes) >= 2, (
            f"Hallway blower should be swept when both parents active, got {hallway_modes}"
        )

    def test_multi_parent_and_gate_off_when_one_parent_off(self) -> None:
        """Multi-parent dependent effector should be off when ANY parent is off (AND gate)."""
        from weatherstat.config import EffectorConfig

        mock_eff = EffectorConfig(
            name="blower_hallway",
            entity_id="fan.blower_hallway",
            control_type="binary",
            mode_control="automatic",
            supported_modes=("off", "low", "high"),
            command_keys={"mode": "blowerHallwayMode"},
            depends_on=("thermostat_upstairs", "thermostat_downstairs"),
            mode_encoding={"off": 0, "low": 1, "high": 2},
            energy_cost={"off": 0.0, "low": 0.001, "high": 0.002},
        )
        mock_effectors = EFFECTORS + (mock_eff,)
        mock_map = {e.name: e for e in mock_effectors}

        with (
            patch("weatherstat.control.EFFECTORS", mock_effectors),
            patch("weatherstat.control.EFFECTOR_MAP", mock_map),
        ):
            scenarios = generate_trajectory_scenarios()

        # Find scenarios where only upstairs is on (downstairs off)
        up_only = [
            s for s in scenarios
            if (up := s.effectors.get("thermostat_upstairs")) is not None
            and up.mode != "off" and up.delay_steps == 0
            and (dn := s.effectors.get("thermostat_downstairs")) is not None
            and dn.mode == "off"
        ]
        assert len(up_only) > 0
        # Hallway blower must be off — downstairs (one parent) is not active
        for s in up_only:
            bh = s.effectors.get("blower_hallway")
            assert bh is None or bh.mode == "off", (
                f"Hallway blower should be off when downstairs is off (AND gate), "
                f"got {bh.mode if bh else 'missing'}"
            )


# ── Scenario count and structure ─────────────────────────────────────────


class TestScenarioStructure:
    """Verify scenario generation produces expected structure."""

    def test_all_effectors_present_in_every_scenario(self) -> None:
        """Every scenario should have a decision for every configured effector."""
        scenarios = generate_trajectory_scenarios()
        effector_names = {e.name for e in EFFECTORS}
        for i, s in enumerate(scenarios[:20]):  # spot-check first 20
            assert set(s.effectors.keys()) == effector_names, (
                f"Scenario {i} missing effectors: {effector_names - set(s.effectors.keys())}"
            )

    def test_scenario_count_substantial(self) -> None:
        """With 2 trajectory × 2 regulating × 3 binary, expect thousands of scenarios."""
        scenarios = generate_trajectory_scenarios()
        # Minimum: must be > 100 (nontrivial sweep)
        assert len(scenarios) > 100
        # Should be in the thousands for full config
        assert len(scenarios) > 1000, f"Expected >1000 scenarios, got {len(scenarios)}"

    def test_ineligible_reduces_count(self) -> None:
        """Making a trajectory effector ineligible should reduce scenario count."""
        full = generate_trajectory_scenarios()
        reduced = generate_trajectory_scenarios(ineligible_effectors={"thermostat_upstairs"})
        assert len(reduced) < len(full)

    def test_effector_types_from_config(self) -> None:
        """Verify EFFECTORS tuple has expected control_type distribution."""
        types = {e.control_type for e in EFFECTORS}
        assert "trajectory" in types
        assert "regulating" in types
        assert "binary" in types

        trajectory_count = sum(1 for e in EFFECTORS if e.control_type == "trajectory")
        regulating_count = sum(1 for e in EFFECTORS if e.control_type == "regulating")
        binary_count = sum(1 for e in EFFECTORS if e.control_type == "binary")
        assert trajectory_count == 2
        assert regulating_count == 2
        assert binary_count == 3


# ── Per-effector energy cost tests ───────────────────────────────────────


class TestPerEffectorEnergyCost:
    """Verify energy cost computation uses per-effector config values."""

    def test_trajectory_energy_cost_from_config(self) -> None:
        """Trajectory effector energy cost should come from EffectorConfig, not constants."""
        eff = EFFECTOR_MAP["thermostat_upstairs"]
        assert isinstance(eff.energy_cost, int | float)
        assert eff.energy_cost > 0

        scenario = Scenario(effectors={
            "thermostat_upstairs": EffectorDecision("thermostat_upstairs", mode="heating"),
        })
        cost = compute_energy_cost(scenario)
        assert cost > 0

    def test_binary_per_mode_energy_cost(self) -> None:
        """Binary effector should use per-mode energy cost dict."""
        eff = EFFECTOR_MAP["blower_family_room"]
        assert isinstance(eff.energy_cost, dict)
        assert eff.energy_cost.get("off", 0.0) == 0.0
        assert eff.energy_cost.get("high", 0.0) > eff.energy_cost.get("low", 0.0)

        off_scenario = Scenario(effectors={
            "blower_family_room": EffectorDecision("blower_family_room", mode="off"),
        })
        high_scenario = Scenario(effectors={
            "blower_family_room": EffectorDecision("blower_family_room", mode="high"),
        })
        assert compute_energy_cost(off_scenario) == 0.0
        assert compute_energy_cost(high_scenario) > 0.0

    def test_regulating_energy_cost_from_config(self) -> None:
        """Regulating effector should use scalar energy cost."""
        eff = EFFECTOR_MAP["mini_split_bedroom"]
        assert isinstance(eff.energy_cost, int | float)
        assert eff.energy_cost > 0

    def test_off_effectors_no_cost(self) -> None:
        """All-off scenario should have zero energy cost."""
        scenario = Scenario(effectors={
            name: EffectorDecision(name, mode="off")
            for name in EFFECTOR_MAP
        })
        assert compute_energy_cost(scenario) == 0.0


# ── Simulator dependency gating tests ────────────────────────────────────


class TestSimulatorDependencyGating:
    """Verify the simulator gates dependent effectors by parent activity."""

    @pytest.fixture
    def sim_params(self):
        return load_sim_params()

    def _make_state(self, outdoor: float = 42.0, hour: float = 14.5) -> HouseState:
        temps = {
            "thermostat_upstairs_temp": 70.0, "thermostat_downstairs_temp": 69.0, "bedroom_temp": 68.5,
            "office_temp": 67.0, "family_room_temp": 69.5,
            "kitchen_temp": 68.0, "piano_temp": 67.5, "bathroom_temp": 68.0, "living_room_temp": 69.0,
        }
        return HouseState(
            current_temps=temps,
            outdoor_temp=outdoor,
            forecast_temps=[outdoor] * 12,
            window_states={},
            hour_of_day=hour,
            solar_fractions=[0.0] * 12,
            solar_elevations=[0.0] * 72,
        )

    def test_blower_with_thermostat_warms_more(self, sim_params) -> None:
        """Blower + thermostat should warm family room more than thermostat alone."""
        thermostat_only = Scenario(effectors={
            "thermostat_downstairs": EffectorDecision("thermostat_downstairs", mode="heating"),
        })
        thermostat_plus_blower = Scenario(effectors={
            "thermostat_downstairs": EffectorDecision("thermostat_downstairs", mode="heating"),
            "blower_family_room": EffectorDecision("blower_family_room", mode="high"),
        })
        targets, preds = predict(
            self._make_state(hour=2.0),
            [thermostat_only, thermostat_plus_blower],
            sim_params, [48],
        )
        # Find family_room prediction
        fr_idx = next(j for j, t in enumerate(targets) if "family_room" in t and "t+48" in t)
        # Blower should add heat delivery to family room
        assert preds[1, fr_idx] >= preds[0, fr_idx], (
            "Blower + thermostat should warm family_room at least as much as thermostat alone"
        )

    def test_blower_without_thermostat_no_effect(self, sim_params) -> None:
        """Blower alone (no thermostat) should have no heating effect."""
        all_off = Scenario(effectors={})
        blower_only = Scenario(effectors={
            "blower_family_room": EffectorDecision("blower_family_room", mode="high"),
        })
        targets, preds = predict(
            self._make_state(hour=2.0),
            [all_off, blower_only],
            sim_params, [48],
        )
        # Blower without thermostat heating should produce same temps as all-off
        # (dependency gate zeroes the blower activity)
        for j in range(len(targets)):
            assert preds[0, j] == pytest.approx(preds[1, j], abs=0.01), (
                f"{targets[j]}: blower alone should equal all-off"
            )

    def test_delayed_thermostat_gates_blower(self, sim_params) -> None:
        """Blower activity should be gated by thermostat trajectory timing."""
        # Thermostat delayed 2h, blower on from start
        delayed = Scenario(effectors={
            "thermostat_downstairs": EffectorDecision(
                "thermostat_downstairs", mode="heating", delay_steps=24,
            ),
            "blower_family_room": EffectorDecision("blower_family_room", mode="high"),
        })
        # Thermostat immediate, blower on from start
        immediate = Scenario(effectors={
            "thermostat_downstairs": EffectorDecision(
                "thermostat_downstairs", mode="heating", delay_steps=0,
            ),
            "blower_family_room": EffectorDecision("blower_family_room", mode="high"),
        })
        targets, preds = predict(
            self._make_state(hour=2.0),
            [delayed, immediate],
            sim_params, [12],  # 1h horizon (before delayed thermostat starts)
        )
        # At 1h, immediate thermostat should have warmed more (blower contributing)
        fr_idx = next(j for j, t in enumerate(targets) if "family_room" in t)
        assert preds[1, fr_idx] > preds[0, fr_idx], (
            "Immediate thermostat+blower should warm more at 1h than delayed thermostat+blower"
        )

    def test_multiple_blowers_same_parent(self, sim_params) -> None:
        """Multiple blowers depending on same thermostat should all be gated together."""
        all_blowers = Scenario(effectors={
            "thermostat_downstairs": EffectorDecision("thermostat_downstairs", mode="heating"),
            "blower_family_room": EffectorDecision("blower_family_room", mode="high"),
            "blower_office": EffectorDecision("blower_office", mode="high"),
            "blower_gym": EffectorDecision("blower_gym", mode="high"),
        })
        targets, preds = predict(
            self._make_state(hour=2.0),
            [Scenario(effectors={}), all_blowers],
            sim_params, [48],
        )
        # Multiple blowers should produce warmer temps than all-off
        dn_idx = next(j for j, t in enumerate(targets) if "downstairs" in t and "t+48" in t)
        assert preds[1, dn_idx] > preds[0, dn_idx]


# ── Config-driven properties ─────────────────────────────────────────────


class TestConfigDriven:
    """Verify effector properties are correctly derived from YAML config."""

    def test_depends_on_is_tuple(self) -> None:
        """All depends_on fields should be tuples (possibly empty)."""
        for eff in EFFECTORS:
            assert isinstance(eff.depends_on, tuple), (
                f"{eff.name}.depends_on should be tuple, got {type(eff.depends_on)}"
            )

    def test_blowers_depend_on_thermostat(self) -> None:
        """All blowers in example config depend on thermostat_downstairs."""
        for eff in EFFECTORS:
            if eff.control_type == "binary":
                assert len(eff.depends_on) > 0, f"{eff.name} should have depends_on"
                assert "thermostat_downstairs" in eff.depends_on

    def test_independent_effectors_no_depends(self) -> None:
        """Trajectory and regulating effectors should not have depends_on."""
        for eff in EFFECTORS:
            if eff.control_type in ("trajectory", "regulating"):
                assert len(eff.depends_on) == 0, (
                    f"{eff.name} ({eff.control_type}) should not have depends_on"
                )

    def test_max_lag_minutes_from_config(self) -> None:
        """Max lag should vary by control type."""
        from weatherstat.yaml_config import load_config
        cfg = load_config()
        for _name, eff in cfg.effectors.items():
            if eff.control_type == "trajectory":
                assert eff.max_lag_minutes == 90
            elif eff.control_type == "regulating":
                assert eff.max_lag_minutes == 15
            elif eff.control_type == "binary":
                assert eff.max_lag_minutes == 5

    def test_energy_cost_type_matches_control_type(self) -> None:
        """Trajectory/regulating should have scalar cost, binary should have dict cost."""
        for eff in EFFECTORS:
            if eff.control_type in ("trajectory", "regulating"):
                assert isinstance(eff.energy_cost, int | float), (
                    f"{eff.name}: {eff.control_type} should have scalar energy_cost"
                )
            elif eff.control_type == "binary":
                assert isinstance(eff.energy_cost, dict), (
                    f"{eff.name}: binary should have dict energy_cost"
                )


# ── Temperature unit conversion tests ────────────────────────────────────


class TestTemperatureUnits:
    """Verify unit conversion helpers produce correct results."""

    def _make_config(self, unit: str) -> "WeatherstatConfig":
        from weatherstat.yaml_config import LocationConfig, WeatherstatConfig

        loc = LocationConfig(latitude=0, longitude=0, elevation=0, timezone="UTC", unit=unit)
        return WeatherstatConfig(
            location=loc,
            temp_sensors={},
            humidity_sensors={},
            effectors={},
            state_sensors={},
            power_sensors={},
            health_checks=[],
            windows={},
            weather_entity="weather.test",
            constraints=[],
            notification_target="notify.test",
        )

    def test_fahrenheit_identity(self) -> None:
        """Fahrenheit config should return values unchanged."""
        cfg = self._make_config("F")
        assert cfg.abs_temp(62) == 62.0
        assert cfg.delta_temp(5.0) == 5.0
        assert cfg.delta_scale == 1.0
        assert cfg.unit_symbol == "°F"

    def test_celsius_abs_temp(self) -> None:
        """Absolute temp conversion: 32°F = 0°C, 212°F = 100°C."""
        cfg = self._make_config("C")
        assert cfg.abs_temp(32) == pytest.approx(0.0)
        assert cfg.abs_temp(212) == pytest.approx(100.0)
        assert cfg.abs_temp(62) == pytest.approx(16.667, abs=0.01)
        assert cfg.abs_temp(78) == pytest.approx(25.556, abs=0.01)

    def test_celsius_delta_temp(self) -> None:
        """Delta conversion: 9°F delta = 5°C delta."""
        cfg = self._make_config("C")
        assert cfg.delta_temp(9) == pytest.approx(5.0)
        assert cfg.delta_temp(1.0) == pytest.approx(5.0 / 9.0)
        assert cfg.delta_scale == pytest.approx(5.0 / 9.0)
        assert cfg.unit_symbol == "°C"

    def test_default_unit_is_fahrenheit(self) -> None:
        """Config without explicit unit should default to Fahrenheit."""
        from weatherstat.yaml_config import load_config

        cfg = load_config()
        assert cfg.location.unit == "F"
        assert cfg.unit_symbol == "°F"
