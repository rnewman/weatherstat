"""Virtual regulation sensors (PLAN-41): weights, the virtual input, learned response,
and the simulator's one-activity-per-unit model."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from weatherstat.regulation import (
    RegulationModel,
    ResponseCurve,
    compute_weights,
    fit_response,
    virtual_reading,
)
from weatherstat.simulator import HouseState, SimParams, TauModel, load_sim_params, predict
from weatherstat.types import EffectorDecision, Scenario

UNIT = "mini_split_bedroom"

# ── Weights and the virtual input ────────────────────────────────────────


def test_weights_proportional_to_positive_gains() -> None:
    gains = {
        (UNIT, "bedroom_temp"): (0.8, 10.0),
        (UNIT, "piano_temp"): (0.4, 10.0),
        (UNIT, "office_temp"): (-0.3, 10.0),  # negative: not a member
        (UNIT, "outdoor_temp"): (0.5, 10.0),  # not constrained: not a member
        ("mini_split_living_room", "kitchen_temp"): (0.9, 10.0),  # other unit
    }
    w = compute_weights(gains, UNIT, ["bedroom_temp", "piano_temp", "office_temp", "kitchen_temp"])
    assert w == pytest.approx({"bedroom_temp": 2 / 3, "piano_temp": 1 / 3}, abs=1e-4)


def test_weights_empty_without_positive_gain() -> None:
    assert compute_weights({(UNIT, "bedroom_temp"): (-0.1, 0.0)}, UNIT, ["bedroom_temp"]) == {}


def test_virtual_reading_drops_and_renormalizes() -> None:
    weights = {"bedroom_temp": 0.6, "piano_temp": 0.3, "kitchen_temp": 0.1}
    temps = {"bedroom_temp": 70.0, "piano_temp": 76.0, "kitchen_temp": 72.0}
    full = virtual_reading(weights, temps)
    assert full.value == pytest.approx(0.6 * 70 + 0.3 * 76 + 0.1 * 72)
    assert full.coverage == 1.0

    frozen = virtual_reading(weights, temps, unavailable={"piano_temp": "frozen"})
    assert frozen.value == pytest.approx((0.6 * 70 + 0.1 * 72) / 0.7, abs=0.01)
    assert frozen.dropped == {"piano_temp": "frozen"}
    assert frozen.coverage == pytest.approx(0.7)


def test_virtual_reading_unavailable_below_coverage() -> None:
    weights = {"bedroom_temp": 0.5, "piano_temp": 0.5}
    r = virtual_reading(weights, {"piano_temp": 75.0}, min_coverage=0.6)
    assert r.value is None
    assert r.dropped == {"bedroom_temp": "no reading"}


# ── Learned response ─────────────────────────────────────────────────────


def test_fit_response_recovers_known_curve() -> None:
    """Simulated unit: P(running) = clip((err + 0.5) / 2.0, 0, 1) → fit finds it."""
    rng = np.random.default_rng(0)
    n = 4000
    target = np.full(n, 72.0)
    temp = 72.0 + rng.uniform(-3.0, 3.0, n)
    mode = np.where(rng.random(n) < 0.5, "heat", "cool")
    sign = np.where(mode == "heat", 1.0, -1.0)
    p_run = np.clip((sign * (target - temp) + 0.5) / 2.0, 0.0, 1.0)
    on = rng.random(n) < p_run
    action = np.where(on, np.where(mode == "heat", "heating", "cooling"), "idle")
    ts = pd.date_range("2026-11-01", periods=n, freq="5min", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.000Z")
    df = pd.DataFrame({
        "timestamp": ts, f"{UNIT}_temp": temp, f"{UNIT}_target": target,
        f"{UNIT}_mode": mode, f"{UNIT}_action": action,
    })
    curve = fit_response(df, UNIT)
    assert curve is not None
    assert curve.offset == pytest.approx(0.5, abs=0.25)
    assert curve.band == pytest.approx(2.0, abs=0.5)


def _response_rows(mode: str, n: int, running_given_past, rng: np.random.Generator, whole: bool = True) -> pd.DataFrame:
    """History rows for one mode: reading relative to a 72° setpoint, running by a rule on
    how far the reading is PAST the setpoint (+ = the unit has gone beyond it)."""
    sign = 1.0 if mode == "heat" else -1.0
    past = rng.integers(-3, 6, n).astype(float) if whole else rng.uniform(-3.0, 5.0, n)
    p_run = np.clip(running_given_past(past), 0.0, 1.0)
    on = rng.random(n) < p_run
    return pd.DataFrame({
        f"{UNIT}_temp": 72.0 + sign * past,
        f"{UNIT}_target": 72.0,
        f"{UNIT}_mode": mode,
        f"{UNIT}_action": np.where(on, "heating" if mode == "heat" else "cooling", "idle"),
    })


def _with_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df.insert(0, "timestamp", pd.date_range("2026-10-01", periods=len(df), freq="5min", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.000Z"))
    return df


def test_fit_response_separately_per_mode() -> None:
    """A unit that holds 2.5° past its heat setpoint but cools to just past its cool
    setpoint: each mode gets its own curve, and the pooled one is right for neither.

    Regression: one pooled fit, on a grid capped at 2°, put the heat switch point
    more than a degree short of where the unit switches, so every heat setpoint
    held the room warmer than planned.
    """
    rng = np.random.default_rng(1)
    heat = _response_rows("heat", 3000, lambda past: (3.0 - past) / 1.0, rng)  # on to +2, off from +3
    cool = _response_rows("cool", 3000, lambda past: (0.75 - past) / 1.5, rng, whole=False)  # half-on at 0
    curve = fit_response(_with_timestamps(pd.concat([heat, cool]).sample(frac=1.0, random_state=0)), UNIT)
    assert curve is not None and set(curve.modes) == {"heat", "cool"}
    assert curve.for_mode("heat").half_on == pytest.approx(2.5, abs=0.25)
    assert curve.for_mode("cool").half_on == pytest.approx(0.0, abs=0.25)
    assert 0.5 < curve.half_on < 2.0  # pooled: in between
    assert curve.for_mode("heat").brier < curve.brier  # type: ignore[operator]
    # To hold 70°: heat is told 2.5° less, cool about what it should hold.
    assert curve.for_mode("heat").setpoint_for(70.0, 1.0) == pytest.approx(67.5, abs=0.25)
    assert curve.for_mode("cool").setpoint_for(70.0, -1.0) == pytest.approx(70.0, abs=0.25)


def test_fit_response_whole_degree_gap_takes_the_middle() -> None:
    """Readings in whole degrees: always on at +1, never at +2. The switch point is
    somewhere in between; the fit says +1.5, not the first grid value past +1."""
    rng = np.random.default_rng(2)
    rows = _response_rows("heat", 2000, lambda past: (past <= 1).astype(float), rng)
    curve = fit_response(_with_timestamps(rows), UNIT)
    assert curve is not None
    assert curve.for_mode("heat").half_on == pytest.approx(1.5)
    assert curve.for_mode("heat").brier == 0.0


def test_fit_response_mode_without_data_uses_pooled() -> None:
    rng = np.random.default_rng(3)
    heat = _response_rows("heat", 2000, lambda past: (2.0 - past) / 1.0, rng)
    cool = _response_rows("cool", 100, lambda past: (0.5 - past) / 1.0, rng)
    curve = fit_response(_with_timestamps(pd.concat([heat, cool])), UNIT)
    assert curve is not None and set(curve.modes) == {"heat"}
    assert curve.for_mode("cool") is curve


def test_responses_json_round_trip_and_old_files() -> None:
    from weatherstat.regulation import responses_from_json, responses_to_json

    curve = ResponseCurve(1.25, 0.25, 5000, 0.17, modes={"heat": ResponseCurve(3.25, 2.0, 2800, 0.06)})
    data = responses_to_json({UNIT: curve})
    assert data[UNIT]["modes"]["heat"]["offset"] == 3.25
    assert responses_from_json(data) == {UNIT: curve}
    # Written before per-mode fits: no "modes"; the pooled curve serves both.
    old = responses_from_json({UNIT: {"offset": 1.25, "band": 0.25, "n_rows": 5000, "brier": 0.17}})
    assert old[UNIT].modes == {} and old[UNIT].for_mode("heat") is old[UNIT]


def test_fit_response_needs_data() -> None:
    df = pd.DataFrame({"timestamp": ["2026-11-01T00:00:00.000Z"], f"{UNIT}_temp": [70.0],
                       f"{UNIT}_target": [71.0], f"{UNIT}_mode": ["heat"], f"{UNIT}_action": ["heating"]})
    assert fit_response(df, UNIT) is None


# ── Simulator: one activity per unit ─────────────────────────────────────

_SENSORS = ["bedroom_temp", "piano_temp", "office_temp"]


def _params(weights: dict[str, float], response: ResponseCurve | None = None) -> SimParams:
    """Sandbox params with the bedroom split as the only effect, no envelope loss, no sun."""
    base = load_sim_params()
    return replace(
        base,
        taus={s: TauModel(tau_base=1e9) for s in base.sensors},
        gains={
            (UNIT, "bedroom_temp"): (0.8, 0.0),
            (UNIT, "piano_temp"): (0.4, 0.0),
            (UNIT, "office_temp"): (-0.3, 0.0),  # confounded negative cross gain
        },
        solar={}, solar_elevation_gains={}, solar_plane_gains={},
        regulation={UNIT: RegulationModel(UNIT, weights, response or ResponseCurve(offset=0.0, band=1.0))},
    )


def _state(temps: dict[str, float], outdoor: float = 50.0) -> HouseState:
    """tau is floored at 100 h (TauModel), so rooms still drift toward ``outdoor``."""
    return HouseState(
        current_temps=temps, outdoor_temp=outdoor, forecast_temps=[outdoor] * 12,
        environment_states={}, hour_of_day=2.0,
        solar_fractions=[0.0] * 12, solar_elevations=[0.0] * 72,
    )


def _split(mode: str, target: float | None = None) -> Scenario:
    return Scenario(effectors={UNIT: EffectorDecision(UNIT, mode=mode, target=target)})


def _deltas(params: SimParams, state: HouseState, scenario: Scenario) -> tuple[dict[str, float], np.ndarray]:
    act: dict[str, np.ndarray] = {}
    names, preds = predict(state, [Scenario(effectors={}), scenario], params, [72], activity_out=act)
    idx = {n.removesuffix("_t+72"): j for j, n in enumerate(names)}
    return {s: float(preds[1, idx[s]] - preds[0, idx[s]]) for s in _SENSORS}, act[UNIT][1]


def test_one_activity_reaches_every_sensor_by_gain() -> None:
    """Every sensor gets gain × the same trace; a negative cross gain contributes nothing."""
    params = _params({"bedroom_temp": 2 / 3, "piano_temp": 1 / 3})
    temps = {"bedroom_temp": 68.0, "piano_temp": 70.0, "office_temp": 70.0}
    d, act = _deltas(params, _state(temps), _split("heat", 80.0))  # far above V: saturated
    assert np.allclose(act, 1.0)
    # 6 h at full activity, less a little extra envelope loss from being warmer (tau 100 h)
    assert d["bedroom_temp"] == pytest.approx(0.8 * 6.0, rel=0.05)
    assert d["bedroom_temp"] / d["piano_temp"] == pytest.approx(0.8 / 0.4, rel=1e-9)
    assert d["office_temp"] == 0.0  # heating can't cool: a negative gain is confounding


def test_cool_with_cold_room_is_off() -> None:
    """Warm piano (75.2) + cold room (69.8): cool @ 73.3 on V ≈ 72.5 never runs —
    it simulates exactly as off (it used to cool only the warm room)."""
    params = _params({"bedroom_temp": 0.5, "piano_temp": 0.5})
    temps = {"bedroom_temp": 69.8, "piano_temp": 75.2, "office_temp": 70.0}
    d, act = _deltas(params, _state(temps), _split("cool", 73.3))
    assert np.all(act == 0.0)
    assert all(v == 0.0 for v in d.values())

    d_heat, act_heat = _deltas(params, _state(temps), _split("heat", 73.0))
    assert act_heat[0] > 0
    assert d_heat["bedroom_temp"] > 0 and d_heat["piano_temp"] > 0
    assert d_heat["bedroom_temp"] / d_heat["piano_temp"] == pytest.approx(0.8 / 0.4, rel=1e-6)


def test_unit_backs_off_as_its_input_reaches_target() -> None:
    """Activity follows V: heat @ V + 0.5 starts at half and falls as the room warms,
    settling where delivery balances envelope loss (0.8·a ≈ (T − 50) / 100)."""
    params = _params({"bedroom_temp": 1.0})
    _, act = _deltas(params, _state({"bedroom_temp": 70.0, "piano_temp": 70.0, "office_temp": 70.0}),
                     _split("heat", 70.5))
    assert act[0] == pytest.approx(0.5)
    assert np.all(np.diff(act) <= 1e-12)
    assert act[-1] == pytest.approx((70.3 - 50.0) / 100.0 / 0.8, abs=0.03)


def test_missing_member_renormalizes_in_simulation() -> None:
    """A member without a reading drops out of V, as in the published value."""
    params = _params({"bedroom_temp": 0.7, "piano_temp": 0.3})
    temps = {"bedroom_temp": 70.0, "office_temp": 70.0}  # piano has no reading
    _, act = _deltas(params, _state(temps), _split("heat", 70.5))
    assert act[0] == pytest.approx(0.5)  # V = bedroom alone = 70.0


def test_low_coverage_means_no_activity() -> None:
    params = _params({"bedroom_temp": 0.3, "piano_temp": 0.7})
    temps = {"bedroom_temp": 60.0, "office_temp": 70.0}  # 30% of the weight available
    d, act = _deltas(params, _state(temps), _split("heat", 80.0))
    assert np.all(act == 0.0)
    assert all(v == 0.0 for v in d.values())


def test_learned_response_shifts_switch_point() -> None:
    """With offset +1, the unit already runs at V = target (P = 0.5 at band 2)."""
    params = _params({"bedroom_temp": 1.0}, ResponseCurve(offset=1.0, band=2.0))
    _, act = _deltas(params, _state({"bedroom_temp": 70.0, "piano_temp": 70.0, "office_temp": 70.0}),
                     _split("heat", 70.0))
    assert act[0] == pytest.approx(0.5)


def test_simulator_uses_the_curve_for_the_mode() -> None:
    """Heat and cool follow their own fits: the same distance from the setpoint is
    half-on in one mode and idle in the other."""
    curve = ResponseCurve(0.0, 1.0, modes={"heat": ResponseCurve(3.0, 1.0)})  # heat: half-on 2.5° past
    params = _params({"bedroom_temp": 1.0}, curve)
    state = _state({"bedroom_temp": 70.0, "piano_temp": 70.0, "office_temp": 70.0})
    _, heat = _deltas(params, state, _split("heat", 67.5))  # reading is 2.5° past the setpoint
    assert heat[0] == pytest.approx(0.5)
    _, cool = _deltas(params, state, _split("cool", 72.5))  # 2.5° past, pooled curve: idle
    assert cool[0] == 0.0
    _, cool_on = _deltas(params, state, _split("cool", 69.5))  # 0.5° short of it: half-on (pooled)
    assert cool_on[0] == pytest.approx(-0.5)


def test_setpoint_for_holds_the_level_asked() -> None:
    """The unit is half-on at the hold level, idle a band above it (heating) and
    flat out a band below; ``holds_at`` is the inverse."""
    for curve in (ResponseCurve(1.25, 0.25), ResponseCurve(1.25, 3.0), ResponseCurve(0.0, 1.0)):
        for sign in (1.0, -1.0):
            sp = curve.setpoint_for(70.0, sign)
            assert float(curve.level(sign * (sp - 70.0))) == pytest.approx(0.5)
            assert float(curve.level(sign * (sp - (70.0 + sign * curve.band)))) == 0.0
            assert float(curve.level(sign * (sp - (70.0 - sign * curve.band)))) == 1.0
            assert curve.holds_at(sp, sign) == pytest.approx(70.0)
    assert ResponseCurve(1.25, 0.25).setpoint_for(69.0, 1.0) == pytest.approx(67.875)  # overshoots: told less


def test_setpoint_below_the_input_is_a_floor() -> None:
    """A heat setpoint below what the unit reads does nothing until the room falls to
    the level it holds, then keeps it there: the unit acts when the temperature gets
    there, whichever way the house is drifting. Off just keeps falling."""
    curve = ResponseCurve(offset=1.25, band=0.25)
    params = _params({"bedroom_temp": 1.0}, curve)
    params = replace(params, taus={s: TauModel(tau_base=40.0) for s in params.sensors})  # losing heat: window open
    state = _state({"bedroom_temp": 70.3, "piano_temp": 70.0, "office_temp": 70.0}, outdoor=45.0)
    floor = _split("heat", curve.setpoint_for(68.0, 1.0))

    act: dict[str, np.ndarray] = {}
    names, preds = predict(state, [Scenario(effectors={}), floor], params, [12, 24, 48, 72], activity_out=act)
    bedroom = {int(n.rsplit("+", 1)[1]): (float(preds[0, j]), float(preds[1, j]))
               for j, n in enumerate(names) if n.startswith("bedroom_temp")}
    trace = act[UNIT][1]
    assert np.all(trace[:6] == 0.0)  # idle while the room is above the floor
    assert trace[-1] > 0.5  # working to hold it by the end
    off_6h, held_6h = bedroom[72]
    assert off_6h < 67.0  # 45 + 25.3·e^(−6/40)
    assert held_6h == pytest.approx(68.0, abs=0.3)
    assert bedroom[12][1] == pytest.approx(bedroom[12][0])  # the first hour is the same as off


def test_chunked_prediction_matches_single_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    import weatherstat.simulator as sim

    params = _params({"bedroom_temp": 0.5, "piano_temp": 0.5})
    state = _state({"bedroom_temp": 69.0, "piano_temp": 71.0, "office_temp": 70.0})
    scenarios = [_split("heat", 69.0 + 0.5 * i) for i in range(10)] + [_split("cool", 70.0)]
    act_a: dict[str, np.ndarray] = {}
    _, a = predict(state, scenarios, params, [12, 72], activity_out=act_a)
    monkeypatch.setattr(sim, "_PREDICT_CHUNK", 3)
    act_b: dict[str, np.ndarray] = {}
    _, b = predict(state, scenarios, params, [12, 72], activity_out=act_b)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(act_a[UNIT], act_b[UNIT])


# ── Planner: exact dedupe ────────────────────────────────────────────────


def test_dedupe_folds_idle_options_into_off() -> None:
    """With the room warm, heat options that never run collapse into off, and
    surviving options are priced from their activity."""
    from weatherstat.config import EFFECTOR_MAP
    from weatherstat.control import (
        CONTROL_HORIZONS,
        _build_option_groups,
        _compute_marginals,
        _dedupe_regulating,
        _regulating_sweep_options,
    )
    from weatherstat.types import ComfortSchedule, ComfortScheduleEntry, RoomComfort

    params = _params({"bedroom_temp": 1.0})
    temps = {"bedroom_temp": 72.0, "piano_temp": 70.0, "office_temp": 70.0}
    sched = [ComfortSchedule(sensor="bedroom_temp", label="bedroom", entries=(
        ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 69.0, 71.0, 68.0, 74.0, 65.0, 77.0)),
    ))]
    opts = _regulating_sweep_options(EFFECTOR_MAP[UNIT], sched, 12, params.gains, None, temps,
                                     regulation=params.regulation[UNIT])
    groups = _build_option_groups({UNIT: opts}, [], temps)
    marginals = _compute_marginals(_state(temps, outdoor=72.0), groups, params, CONTROL_HORIZONS)
    new_groups, new_m, removed = _dedupe_regulating(groups, marginals)

    assert removed > 0
    acts = new_m.activity[0][UNIT]
    assert new_groups[0].options[0][UNIT].mode == "off"
    # No surviving option (besides off) is idle, and no two simulate the same.
    assert np.all(np.abs(acts[1:]).max(axis=1) > 0)
    keys = {np.round(a, 3).tobytes() for a in acts}
    assert len(keys) == len(acts)
    # Energy is ec × mean |activity|
    ec = EFFECTOR_MAP[UNIT].energy_cost
    np.testing.assert_allclose(new_groups[0].energy_costs, ec * np.abs(acts).mean(axis=1))
    # Outdoors = room, so V stays 72, above the whole preferred band: every heat
    # setpoint is a floor that never engages and is gone; cooling to the band remains.
    modes = {o[UNIT].mode for o in new_groups[0].options}
    assert modes == {"off", "cool"}
