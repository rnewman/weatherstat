"""Tests for causal identification of behavior-driven couplings."""

from __future__ import annotations

import numpy as np
import pytest

from weatherstat.identification import (
    IdentificationTag,
    classify_coupling,
    transition_weights,
)

_RNG = np.random.default_rng(7)


def _scenario(n_days: int = 60, toggles_per_day: float = 1.0):
    """Synthetic 5-min series: device state with daytime toggles + solar carrier.

    Returns (ts_hours, state, solar, interaction) where interaction is the
    state x solar feature (the design column under test).
    """
    steps_per_day = 288
    n = n_days * steps_per_day
    ts_hours = np.arange(n) * (5.0 / 60.0)
    hour_of_day = ts_hours % 24.0
    solar = np.clip(np.sin((hour_of_day - 6.0) / 12.0 * np.pi), 0.0, None)  # day bump

    # Device toggles at random daytime moments
    state = np.zeros(n, dtype=bool)
    current = False
    toggle_times = np.sort(_RNG.uniform(0, n_days * 24, size=int(n_days * toggles_per_day)))
    ti = 0
    for i, t in enumerate(ts_hours):
        while ti < len(toggle_times) and toggle_times[ti] <= t:
            current = not current
            ti += 1
        state[i] = current

    interaction = state.astype(float) * solar
    return ts_hours, state, solar, interaction


def _design(solar: np.ndarray, interaction: np.ndarray) -> tuple[np.ndarray, list[str]]:
    X = np.column_stack([solar, interaction])
    return X, ["_solar_elev", "_adv_solar_test_dev"]


class TestClassifyCoupling:
    def test_causal_effect_holds(self) -> None:
        ts, state, solar, interaction = _scenario()
        X, names = _design(solar, interaction)
        # True causal effect: y responds to the interaction everywhere,
        # including immediately across transitions.
        y = 0.5 * solar - 0.4 * interaction + _RNG.normal(0, 0.05, len(ts))
        res = classify_coupling(
            X, y, names, "_adv_solar_test_dev", state, ts, beta_batch=-0.4, t_batch=-5.0,
        )
        assert res.tag == IdentificationTag.CAUSAL
        for rec in res.by_sigma.values():
            assert rec["beta"] < 0

    def test_habit_correlation_collapses(self) -> None:
        ts, state, solar, interaction = _scenario()
        X, names = _design(solar, interaction)
        # Confound: y depends on a slow daily pattern that CORRELATES with the
        # state (same slow component drives both), but does not respond to the
        # state itself — near transitions the relationship vanishes.
        w_near = transition_weights(state, ts, 4.0)
        habit_component = interaction * (1.0 - w_near)  # effect absent near transitions
        y = 0.5 * solar - 0.4 * habit_component + _RNG.normal(0, 0.05, len(ts))
        res = classify_coupling(
            X, y, names, "_adv_solar_test_dev", state, ts, beta_batch=-0.4, t_batch=-3.0,
        )
        assert res.tag == IdentificationTag.CORRELATIONAL

    def test_no_transitions_is_underpowered(self) -> None:
        ts, state, solar, interaction = _scenario(toggles_per_day=0.0)
        state[:] = True  # always active: zero transitions
        interaction = state.astype(float) * solar
        X, names = _design(solar, interaction)
        y = 0.5 * solar + _RNG.normal(0, 0.05, len(ts))
        res = classify_coupling(
            X, y, names, "_adv_solar_test_dev", state, ts, beta_batch=-0.2, t_batch=-2.1,
        )
        assert res.tag == IdentificationTag.UNDERPOWERED

    def test_night_only_transitions_underpowered(self) -> None:
        """Toggles at night give no solar contrast — no verdict, not a false one."""
        ts, state, solar, _ = _scenario(toggles_per_day=0.0)
        # Toggle at 2am each day
        hour = ts % 24.0
        day = (ts // 24.0).astype(int)
        state = (day % 2 == 0)  # flips at midnight boundaries only
        state = state & True
        interaction = state.astype(float) * solar
        X, names = _design(solar, interaction)
        y = 0.5 * solar + _RNG.normal(0, 0.05, len(ts))
        res = classify_coupling(
            X, y, names, "_adv_solar_test_dev", state, ts, beta_batch=-0.2, t_batch=-2.1,
        )
        # Midnight transitions have near-zero solar energy within tight windows
        assert res.tag in (IdentificationTag.UNDERPOWERED, IdentificationTag.CORRELATIONAL)


class TestTransitionWeights:
    def test_peak_at_transition(self) -> None:
        state = np.array([False] * 50 + [True] * 50)
        ts = np.arange(100) * (5.0 / 60.0)
        w = transition_weights(state, ts, sigma_hours=1.0)
        assert w[50] == pytest.approx(1.0)
        assert w[0] < 0.01 and w[-1] < 0.01

    def test_no_transitions_zero_weights(self) -> None:
        state = np.ones(100, dtype=bool)
        ts = np.arange(100) * (5.0 / 60.0)
        assert transition_weights(state, ts, 2.0).sum() == 0.0


class TestSimulatorEnforcement:
    """Correlational couplings: kept for observed-state prediction, pinned
    (no credit) under counterfactual advisory toggles."""

    def _params(self) -> SimParams:
        from weatherstat.simulator import SimParams, TauModel

        return SimParams(
            taus={"piano_temp": TauModel(tau_base=45.0), "bedroom_temp": TauModel(tau_base=45.0)},
            gains={},
            solar={},
            sensors=["piano_temp", "bedroom_temp"],
            effectors=[],
            solar_elevation_gains={"piano_temp": 1.0, "bedroom_temp": 1.0},
            environment_solar_betas={"test_shade": {"piano_temp": -0.5, "bedroom_temp": -0.5}},
            correlational_couplings={"solar": frozenset({("test_shade", "piano_temp")})},
        )

    def _state(self, shade_active: bool) -> HouseState:
        from weatherstat.simulator import HouseState

        return HouseState(
            current_temps={"piano_temp": 70.0, "bedroom_temp": 70.0},
            outdoor_temp=60.0,
            forecast_temps=[60.0] * 12,
            environment_states={"test_shade": shade_active},
            hour_of_day=12.0,
            solar_fractions=[1.0] * 12,
            solar_elevations=[0.6] * 72,
        )

    def test_counterfactual_toggle_credits_causal_not_correlational(self) -> None:
        from weatherstat.simulator import predict
        from weatherstat.types import AdvisoryDecision, Scenario

        params = self._params()
        state = self._state(shade_active=False)
        hold = Scenario({}, advisories={"test_shade": AdvisoryDecision("test_shade", action="hold")})
        # Timeline verbs map to raw state: "open"/"raise" -> 1.0 (active).
        # active_state for this synthetic shade is "closed", where its solar
        # modulation (beta = -0.5) applies.
        activate = Scenario({}, advisories={"test_shade": AdvisoryDecision("test_shade", action="raise", transition_step=0)})

        targets, preds = predict(state, [hold, activate], params, [24])
        idx = {t: j for j, t in enumerate(targets)}

        # b (causal): closing the shade reduces solar gain → cooler prediction
        assert preds[1, idx["bedroom_temp_t+24"]] < preds[0, idx["bedroom_temp_t+24"]] - 0.05
        # a (correlational): the toggle earns no credit — identical prediction
        assert preds[1, idx["piano_temp_t+24"]] == pytest.approx(preds[0, idx["piano_temp_t+24"]], abs=1e-9)

    def test_observed_state_still_predicts_correlational(self) -> None:
        """With the shade actually closed, sensor a's conditional prediction
        keeps the (correlational) modulation — habit still predicts."""
        from weatherstat.simulator import predict
        from weatherstat.types import AdvisoryDecision, Scenario

        params = self._params()
        hold = Scenario({}, advisories={"test_shade": AdvisoryDecision("test_shade", action="hold")})

        _, p_open = predict(self._state(shade_active=False), [hold], params, [24])
        targets, p_closed = predict(self._state(shade_active=True), [hold], params, [24])
        idx = {t: j for j, t in enumerate(targets)}

        assert p_closed[0, idx["piano_temp_t+24"]] < p_open[0, idx["piano_temp_t+24"]] - 0.05
