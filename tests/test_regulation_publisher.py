"""Virtual regulation sensors, publishing half (PLAN-41 §3, §7, §8).

MQTT discovery + payloads for ``sensor.weatherstat_<effector>_input``, the 60 s
publisher loop (cadence, single-publisher lock, membership-change cycle
requests), weight adoption, and the follows-V health check.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from weatherstat import regulation
from weatherstat.mqtt_bridge import (
    REGULATION_EXPIRE_AFTER,
    BridgeTopics,
    MqttBridge,
    discovery_messages,
    regulation_payload,
    regulation_summary,
    status_payload,
)
from weatherstat.regulation_publisher import (
    FollowChecker,
    FollowStatus,
    PublisherLock,
    RegulationPublisher,
    RegulationStatus,
    UnitInput,
    evaluate_unit,
    follows_text,
    member_values,
)
from weatherstat.yaml_config import MqttConfig

NOW = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
LR = "mini_split_living_room"

# column -> (entity_id, attribute)
SOURCES = {
    "piano_temp": ("sensor.piano", None),
    "living_room_temp": ("sensor.living_room", None),
    "kitchen_temp": ("sensor.kitchen", None),
}
WEIGHTS = {"piano_temp": 0.4, "living_room_temp": 0.4, "kitchen_temp": 0.2}


def _states(piano: object = 75.2, lr: object = 69.8, kitchen: object = 70.0, unit: float | None = 72.0) -> dict:
    from weatherstat.config import EFFECTOR_MAP

    out = {
        "sensor.piano": {"entity_id": "sensor.piano", "state": str(piano), "attributes": {}},
        "sensor.living_room": {"entity_id": "sensor.living_room", "state": str(lr), "attributes": {}},
        "sensor.kitchen": {"entity_id": "sensor.kitchen", "state": str(kitchen), "attributes": {}},
    }
    eid = EFFECTOR_MAP[LR].entity_id
    out[eid] = {"entity_id": eid, "state": "heat", "attributes": {"current_temperature": unit}}
    return out


class FakeClient:
    def __init__(self) -> None:
        self.published: list[tuple[str, str | None, bool]] = []

    def publish(self, topic, payload=None, qos=0, retain=False):
        self.published.append((topic, payload, retain))

    def subscribe(self, topic, qos=0):
        pass

    def on(self, topic: str) -> list[tuple[str | None, bool]]:
        return [(p, r) for t, p, r in self.published if t == topic]


@pytest.fixture
def bridge() -> tuple[MqttBridge, FakeClient]:
    b = MqttBridge(
        MqttConfig(),
        apply=lambda _cmd: None,
        log=lambda _m: None,
        labels={"piano_temp": "piano"},
        step=1.0,
        max_delta=5.0,
        unit="°F",
        source_entities={},
        load_overlays=dict,
        regulating=[LR],
    )
    client = FakeClient()
    b.attach(client)
    b._connected = True
    return b, client


def _unit(value: float | None = 71.5, **kw) -> UnitInput:
    base = {
        "effector": LR,
        "value": value,
        "weights": dict(WEIGHTS),
        "used": {"piano_temp": 0.5, "living_room_temp": 0.5} if value is not None else {},
        "dropped": {"kitchen_temp": "frozen (70°F for 20h)"},
        "coverage": 0.8,
        "adopted_at": "2026-11-19T18:00:00+00:00",
        "members": {"piano_temp": 75.2, "living_room_temp": 69.8},
    }
    base.update(kw)
    return UnitInput(**base)


# ── Discovery and payloads ────────────────────────────────────────────────


class TestDiscovery:
    def test_regulation_entity_config(self) -> None:
        topics = BridgeTopics()
        msgs = dict(discovery_messages(topics, {"piano_temp": "piano"}, step=1.0, max_delta=5.0, unit="°F", regulating=[LR]))
        cfg = msgs[f"homeassistant/sensor/weatherstat/{LR}_input/config"]
        assert cfg["default_entity_id"] == f"sensor.weatherstat_{LR}_input"
        assert cfg["unique_id"] == f"weatherstat_{LR}_input"
        assert cfg["device_class"] == "temperature"
        assert cfg["state_class"] == "measurement"
        assert cfg["unit_of_measurement"] == "°F"
        assert cfg["force_update"] is True
        assert cfg["expire_after"] == REGULATION_EXPIRE_AFTER == 180
        assert cfg["state_topic"] == cfg["json_attributes_topic"] == f"weatherstat/regulation/{LR}"
        # Both the bridge's last will and the entity's own coverage availability must be online.
        assert "availability_topic" not in cfg
        assert cfg["availability"] == [
            {"topic": "weatherstat/status"},
            {"topic": f"weatherstat/regulation/{LR}/availability"},
        ]
        assert cfg["availability_mode"] == "all"
        assert cfg["device"]["identifiers"] == ["weatherstat"]

    def test_no_regulating_effectors_no_entities(self) -> None:
        msgs = discovery_messages(BridgeTopics(), {"piano_temp": "piano"}, step=1.0, max_delta=5.0, unit="°F")
        assert not any("_input/config" in t for t, _ in msgs)

    def test_unique_ids_stay_unique(self) -> None:
        msgs = discovery_messages(
            BridgeTopics(), {"piano_temp": "piano"}, step=1.0, max_delta=5.0, unit="°F", regulating=[LR, "mini_split_bedroom"]
        )
        ids = [p["unique_id"] for _, p in msgs]
        assert len(ids) == len(set(ids))

    def test_bridge_announces_regulation_entities_on_connect(self, bridge) -> None:
        b, client = bridge
        b.on_connected()
        assert f"homeassistant/sensor/weatherstat/{LR}_input/config" in [t for t, _, _ in client.published]


class TestPayload:
    def test_attributes(self) -> None:
        p = regulation_payload(_unit(), NOW)
        assert p is not None
        assert p["value"] == 71.5
        assert p["weights"] == {"piano_temp": 0.5, "living_room_temp": 0.5}
        assert p["weight_set"] == WEIGHTS
        assert p["dropped"] == {"kitchen_temp": "frozen (70°F for 20h)"}
        assert p["coverage"] == 0.8
        assert p["members"] == {"piano_temp": 75.2, "living_room_temp": 69.8}
        assert p["adopted_at"] == "2026-11-19T18:00:00+00:00"
        assert p["updated"] == NOW.isoformat()

    def test_no_value_no_payload(self) -> None:
        assert regulation_payload(_unit(None), NOW) is None

    def test_publish_every_call_unretained(self, bridge) -> None:
        b, client = bridge
        topics = b.topics
        for _ in range(3):
            assert b.publish_regulation({LR: _unit()}, NOW) == 1
        states = client.on(topics.regulation_state(LR))
        assert len(states) == 3  # identical values still go out: force_update needs every publish
        assert all(not retained for _, retained in states)
        avail = client.on(topics.regulation_availability(LR))
        assert avail == [("online", True)]  # retained, sent on change only
        # The device availability is re-asserted each minute (another process's clean stop sends "offline").
        assert client.on(topics.availability) == [("online", True)] * 3

    def test_low_coverage_publishes_unavailable(self, bridge) -> None:
        b, client = bridge
        topics = b.topics
        b.publish_regulation({LR: _unit()}, NOW)
        assert b.publish_regulation({LR: _unit(None, reason="coverage 40% < 60%")}, NOW) == 0
        assert client.on(topics.regulation_availability(LR))[-1] == ("offline", True)
        assert len(client.on(topics.regulation_state(LR))) == 1  # no number while unavailable
        b.publish_regulation({LR: _unit()}, NOW)
        assert client.on(topics.regulation_availability(LR))[-1] == ("online", True)

    def test_reconnect_resends_availability(self, bridge) -> None:
        b, client = bridge
        b.publish_regulation({LR: _unit()}, NOW)
        b.on_connected()
        b.publish_regulation({LR: _unit()}, NOW)
        assert client.on(b.topics.regulation_availability(LR)) == [("online", True), ("online", True)]

    def test_disconnected_publishes_nothing(self, bridge) -> None:
        b, client = bridge
        b._connected = False
        assert b.publish_regulation({LR: _unit()}, NOW) == 0
        assert client.published == []

    def test_status_attribute(self) -> None:
        status = RegulationStatus("publisher", None, NOW, None, {LR: _unit(follows=FollowStatus("ok", 0.2))}, (LR,))
        summary = regulation_summary(status)
        assert summary is not None
        u = summary["units"][LR]
        assert u["value"] == 71.5 and u["coverage"] == 0.8
        assert u["dropped"] == {"kitchen_temp": "frozen (70°F for 20h)"}
        assert u["follows"]["state"] == "ok"
        assert regulation_summary(None) is None

    def test_status_payload_carries_regulation(self) -> None:
        from tests.test_mqtt_bridge import _snapshot

        status = RegulationStatus("publisher", None, NOW, None, {LR: _unit()}, ())
        assert status_payload(_snapshot(regulation=status))["regulation"]["units"][LR]["value"] == 71.5
        assert status_payload(_snapshot())["regulation"] is None


# ── Evaluating V from HA states ───────────────────────────────────────────


class TestEvaluate:
    def test_member_values(self) -> None:
        values, missing = member_values(_states(kitchen="unavailable"), list(SOURCES) + ["nowhere_temp"], SOURCES)
        assert values == {"piano_temp": 75.2, "living_room_temp": 69.8}
        assert missing == {"kitchen_temp": "unavailable", "nowhere_temp": "not configured"}

    def test_climate_attribute_source(self) -> None:
        from weatherstat.config import EFFECTOR_MAP

        src = {f"{LR}_temp": (EFFECTOR_MAP[LR].entity_id, "current_temperature")}
        values, _ = member_values(_states(unit=71.6), [f"{LR}_temp"], src)
        assert values == {f"{LR}_temp": 71.6}

    def test_unavailable_member_renormalized(self) -> None:
        u = evaluate_unit(LR, WEIGHTS, None, {"piano_temp": 75.0, "living_room_temp": 70.0}, {"kitchen_temp": "unavailable"})
        assert u.value == pytest.approx(72.5)
        assert u.used == {"piano_temp": 0.5, "living_room_temp": 0.5}
        assert u.dropped == {"kitchen_temp": "unavailable"}
        assert u.coverage == pytest.approx(0.8)

    def test_low_coverage_unavailable(self) -> None:
        u = evaluate_unit(LR, WEIGHTS, None, {"kitchen_temp": 70.0}, {}, min_coverage=0.6)
        assert u.value is None
        assert u.reason is not None and "coverage" in u.reason

    def test_no_weights(self) -> None:
        u = evaluate_unit(LR, {}, None, {"piano_temp": 70.0}, {})
        assert u.value is None and u.reason is not None


# ── The publisher loop ────────────────────────────────────────────────────


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _publisher(tmp_path: Path, **kw) -> tuple[RegulationPublisher, dict]:
    rec: dict = {"published": [], "cycles": [], "logs": [], "frozen": {}, "states": _states()}
    defaults = {
        "log": rec["logs"].append,
        "live": lambda: False,
        "publish": rec["published"].append,
        "request_cycle": rec["cycles"].append,
        "fetch_states": lambda: rec["states"],
        "frozen": lambda _values: dict(rec["frozen"]),
        "weight_sets": lambda: {LR: (dict(WEIGHTS), "2026-11-19T18:00:00+00:00")},
        "lock": PublisherLock(tmp_path / "pub.lock"),
        "live_effectors": (),
        "sources": SOURCES,
        "sensors": list(SOURCES),
        "now": lambda: NOW,
    }
    defaults.update(kw)
    return RegulationPublisher(**defaults), rec


class TestPublisher:
    def test_tick_publishes_v(self, tmp_path: Path) -> None:
        pub, rec = _publisher(tmp_path)
        status = pub.tick()
        assert status.role == "publisher"
        u = status.units[LR]
        assert u.value == pytest.approx(0.4 * 75.2 + 0.4 * 69.8 + 0.2 * 70.0, abs=0.01)
        assert u.unit_reading == 72.0 and u.unit_mode == "heat"
        assert rec["published"] == [{LR: u}]

    def test_frozen_member_dropped(self, tmp_path: Path) -> None:
        pub, rec = _publisher(tmp_path)
        rec["frozen"] = {"living_room_temp": "frozen (69.8°F for 23h)"}
        u = pub.tick().units[LR]
        assert u.dropped == {"living_room_temp": "frozen (69.8°F for 23h)"}
        assert u.value == pytest.approx((0.4 * 75.2 + 0.2 * 70.0) / 0.6, abs=0.01)

    def test_membership_change_requests_cycle(self, tmp_path: Path) -> None:
        pub, rec = _publisher(tmp_path)
        pub.tick()
        pub.tick()
        assert rec["cycles"] == []  # first tick is the baseline; nothing changed after
        rec["frozen"] = {"living_room_temp": "frozen"}
        pub.tick()
        assert len(rec["cycles"]) == 1 and "living_room_temp dropped" in rec["cycles"][0]
        pub.tick()
        assert len(rec["cycles"]) == 1  # still frozen: no new request
        rec["frozen"] = {}
        pub.tick()
        assert len(rec["cycles"]) == 2 and "living_room_temp back" in rec["cycles"][1]

    def test_value_change_alone_does_not_request_cycle(self, tmp_path: Path) -> None:
        pub, rec = _publisher(tmp_path)
        pub.tick()
        rec["states"] = _states(piano=76.0)
        pub.tick()
        assert rec["cycles"] == []
        assert len(rec["published"]) == 2

    def test_coverage_loss_publishes_unavailable_and_requests_cycle(self, tmp_path: Path) -> None:
        pub, rec = _publisher(tmp_path)
        pub.tick()
        rec["states"] = _states(piano="unavailable", lr="unknown")
        u = pub.tick().units[LR]
        assert u.value is None and u.dropped["piano_temp"] == "unavailable"
        assert rec["published"][-1][LR].value is None  # the bridge turns this into availability offline
        assert len(rec["cycles"]) == 1

    def test_ha_failure_publishes_nothing(self, tmp_path: Path) -> None:
        def boom() -> dict:
            raise RuntimeError("connection refused")

        pub, rec = _publisher(tmp_path, fetch_states=boom)
        status = pub.tick()
        assert status.error is not None and "connection refused" in status.error
        assert rec["published"] == []  # expire_after makes the entity unavailable after 3 minutes

    def test_single_publisher_lock(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "pub.lock"
        a, rec_a = _publisher(tmp_path, lock=PublisherLock(lock_path))
        b, rec_b = _publisher(tmp_path, lock=PublisherLock(lock_path))
        assert a.tick().role == "publisher"
        sb = b.tick()
        assert sb.role == "standby"
        assert sb.units[LR].value is not None  # standby still evaluates for its own display
        assert len(rec_a["published"]) == 1 and rec_b["published"] == []
        # Standby never requests cycles or publishes, even across a membership change.
        rec_b["frozen"] = {"living_room_temp": "frozen"}
        b.tick()
        b.tick()
        assert rec_b["cycles"] == [] and rec_b["published"] == []
        # The publisher exits: the standby takes over on its next tick.
        a.stop()
        assert b.tick().role == "publisher"
        assert len(rec_b["published"]) == 1

    def test_lock_contends_and_records_pid(self, tmp_path: Path) -> None:
        import os

        p = tmp_path / "x.lock"
        one, two = PublisherLock(p), PublisherLock(p)
        assert one.acquire() and one.acquire()  # idempotent for the holder
        assert not two.acquire()
        assert two.holder() == str(os.getpid())
        one.release()
        assert two.acquire()
        two.release()

    def test_sixty_second_cadence(self, tmp_path: Path) -> None:
        clock = FakeClock()
        pub, rec = _publisher(tmp_path, clock=clock)
        ticks: list[float] = []
        real_tick = pub.tick

        def tick():
            ticks.append(clock.t)
            clock.t += 7.0 if len(ticks) == 2 else 0.5  # a slow tick must not shift the schedule
            return real_tick()

        pub.tick = tick  # type: ignore[method-assign]
        waits: list[float] = []

        def wait(seconds: float) -> bool:
            waits.append(seconds)
            clock.t += seconds
            return len(waits) >= 5

        pub.run(wait=wait)
        assert ticks == [0.0, 60.0, 120.0, 180.0, 240.0]
        assert waits[0] == pytest.approx(59.5) and waits[1] == pytest.approx(53.0)
        assert len(rec["published"]) == 5

    def test_overrunning_tick_skips_missed_slots(self, tmp_path: Path) -> None:
        clock = FakeClock()
        pub, _rec = _publisher(tmp_path, clock=clock)
        ticks: list[float] = []

        def tick():
            ticks.append(clock.t)
            clock.t += 150.0 if len(ticks) == 1 else 1.0
            return None

        pub.tick = tick  # type: ignore[method-assign]
        n = [0]

        def wait(seconds: float) -> bool:
            clock.t += seconds
            n[0] += 1
            return n[0] >= 2

        pub.run(wait=wait)
        assert ticks == [0.0, 180.0]

    def test_tick_exception_does_not_kill_loop(self, tmp_path: Path) -> None:
        clock = FakeClock()
        pub, rec = _publisher(tmp_path, clock=clock)

        def tick():
            raise ValueError("bad")

        pub.tick = tick  # type: ignore[method-assign]
        calls = [0]

        def wait(seconds: float) -> bool:
            clock.t += seconds
            calls[0] += 1
            return calls[0] >= 3

        pub.run(wait=wait)
        assert sum("Tick failed: bad" in m for m in rec["logs"]) == 3


# ── Follows-V ─────────────────────────────────────────────────────────────


class TestFollowsV:
    def _checker(self, **kw):
        sent: list = []
        dismissed: list[str] = []
        logs: list[str] = []
        fc = FollowChecker(
            log=logs.append,
            tolerance=1.0,
            send=lambda alert: sent.append(alert) or True,
            dismiss=dismissed.append,
            **kw,
        )
        return fc, sent, dismissed, logs

    def test_within_tolerance_ok(self) -> None:
        fc, sent, _, _ = self._checker()
        st = fc.check(LR, 71.4, 71.6, NOW, live=True)  # within the unit's 0.5°C rounding
        assert st.state == "ok" and st.diff == pytest.approx(0.2)
        assert sent == []

    def test_alert_after_ten_minutes_then_clear(self) -> None:
        fc, sent, dismissed, logs = self._checker()
        assert fc.check(LR, 71.4, 73.0, NOW, live=True).state == "mismatch"
        assert fc.check(LR, 71.4, 73.0, NOW + timedelta(minutes=9), live=True).state == "mismatch"
        assert sent == []
        st = fc.check(LR, 71.4, 73.0, NOW + timedelta(minutes=10), live=True)
        assert st.state == "alert" and st.since == NOW
        assert len(sent) == 1 and sent[0].key == f"regulation_follow_{LR}"
        assert sent[0].category == "regulation_follow"
        # Still mismatched within the cooldown: no repeat notification.
        fc.check(LR, 71.4, 73.0, NOW + timedelta(minutes=20), live=True)
        assert len(sent) == 1
        # Recovery dismisses the notification.
        assert fc.check(LR, 71.4, 71.5, NOW + timedelta(minutes=21), live=True).state == "ok"
        assert dismissed == [LR]
        assert any("following again" in m for m in logs)

    def test_brief_mismatch_resets(self) -> None:
        fc, sent, _, _ = self._checker()
        fc.check(LR, 71.4, 73.0, NOW, live=True)
        fc.check(LR, 71.4, 71.4, NOW + timedelta(minutes=5), live=True)
        assert fc.check(LR, 71.4, 73.0, NOW + timedelta(minutes=12), live=True).state == "mismatch"
        assert sent == []

    def test_dry_run_logs_but_does_not_notify(self) -> None:
        fc, sent, _, logs = self._checker()
        fc.check(LR, 71.4, 73.0, NOW, live=False)
        assert fc.check(LR, 71.4, 73.0, NOW + timedelta(minutes=11), live=False).state == "alert"
        assert sent == []
        assert sum("not following" in m for m in logs) == 1

    def test_no_input_is_not_a_mismatch(self) -> None:
        fc, sent, dismissed, _ = self._checker(sent={LR: NOW.timestamp() - 60})
        assert fc.check(LR, None, 73.0, NOW, live=True).state == "no input"
        assert dismissed == [LR]  # an outstanding alert from before a restart clears
        assert fc.check(LR, 71.0, None, NOW, live=True).state == "no reading"
        assert sent == []

    def test_cooldown_renotifies(self) -> None:
        fc, sent, _, _ = self._checker(cooldown_seconds=3600)
        fc.check(LR, 71.4, 73.0, NOW, live=True)
        fc.check(LR, 71.4, 73.0, NOW + timedelta(minutes=10), live=True)
        fc.check(LR, 71.4, 73.0, NOW + timedelta(minutes=71), live=True)
        assert len(sent) == 2

    def test_publisher_runs_check_for_live_effectors_only(self, tmp_path: Path) -> None:
        fc, _, _, _ = self._checker()
        pub, _ = _publisher(tmp_path, follow=fc, live_effectors=(LR,))
        assert pub.tick().units[LR].follows is not None
        pub2, _ = _publisher(tmp_path / "other", follow=fc, live_effectors=())
        assert pub2.tick().units[LR].follows is None

    def test_follows_text(self) -> None:
        assert follows_text(None) is None
        assert follows_text(FollowStatus("ok", 0.1)) == "follows V"
        assert follows_text(FollowStatus("alert", 1.6, NOW), NOW + timedelta(minutes=12)) == "NOT FOLLOWING (+1.6 for 12m)"


# ── Weight adoption ───────────────────────────────────────────────────────


@pytest.fixture
def reg_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    state, history = tmp_path / "regulation_state.json", tmp_path / "regulation_history.jsonl"
    monkeypatch.setattr(regulation, "REGULATION_STATE_FILE", state)
    monkeypatch.setattr(regulation, "REGULATION_HISTORY_FILE", history)
    return state, history


class TestAdoption:
    GAINS = {
        (LR, "piano_temp"): (0.8, 5.0),
        (LR, "kitchen_temp"): (0.4, 5.0),
        (LR, "bedroom_temp"): (-0.1, 5.0),  # negative: never a member
    }

    def test_adopt_then_tolerance(self, reg_files) -> None:
        state, history = reg_files
        changed = regulation.adopt_weights(self.GAINS, now=NOW, log=lambda _m: None)
        assert LR in changed
        w = regulation.load_adopted()[LR]["weights"]
        assert w == pytest.approx({"piano_temp": 0.6667, "kitchen_temp": 0.3333}, abs=1e-4)
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-3)
        n_hist = len(history.read_text().splitlines())

        # Refit noise under the tolerance: not adopted, no history.
        small = {**self.GAINS, (LR, "piano_temp"): (0.85, 5.0)}
        assert LR not in regulation.adopt_weights(small, now=NOW + timedelta(hours=1), log=lambda _m: None)
        assert regulation.load_adopted()[LR]["weights"] == w
        assert len(history.read_text().splitlines()) == n_hist

        # A real move: adopted and appended to history with the previous set.
        big = {**self.GAINS, (LR, "kitchen_temp"): (0.8, 5.0)}
        assert LR in regulation.adopt_weights(big, now=NOW + timedelta(hours=2), log=lambda _m: None)
        lines = [json.loads(x) for x in history.read_text().splitlines()]
        assert len(lines) == n_hist + 1
        assert lines[-1]["effector"] == LR and lines[-1]["previous"] == w
        assert regulation.load_adopted()[LR]["weights"] == {"piano_temp": 0.5, "kitchen_temp": 0.5}

    def test_runtime_hook_adopts_from_sim_gains(self, reg_files, monkeypatch: pytest.MonkeyPatch) -> None:
        from weatherstat import runtime, simulator

        class P:
            gains = TestAdoption.GAINS

        monkeypatch.setattr(simulator, "load_sim_params", lambda *a, **k: P())
        msg = runtime.adopt_regulation_weights(log=lambda _m: None)
        assert LR in msg
        assert runtime.adopt_regulation_weights(log=lambda _m: None) == ""  # unchanged: nothing to report

    def test_runtime_hook_never_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from weatherstat import runtime, simulator

        def boom(*_a, **_k):
            raise OSError("no params")

        monkeypatch.setattr(simulator, "load_sim_params", boom)
        assert "error" in runtime.adopt_regulation_weights(log=lambda _m: None)


# ── Display ───────────────────────────────────────────────────────────────


class TestDisplay:
    def test_tui_lines(self) -> None:
        from weatherstat.tui.widgets import regulation_lines

        status = RegulationStatus(
            "publisher", None, NOW, None,
            {LR: _unit(unit_reading=73.0, follows=FollowStatus("alert", 1.5, NOW - timedelta(minutes=12)))},
            (LR,),
        )
        text = "\n".join(regulation_lines(status, NOW))
        assert "living room split" in text and "71.5" in text
        assert "NOT FOLLOWING (+1.5 for 12m)" in text
        assert "kitchen" in text and "frozen" in text  # dropped member shown

    def test_tui_lines_standby_and_unavailable(self) -> None:
        from weatherstat.tui.widgets import regulation_lines

        status = RegulationStatus("standby", "4242", NOW, None, {LR: _unit(None, reason="coverage 40% < 60%")}, ())
        text = "\n".join(regulation_lines(status, NOW))
        assert "pid 4242" in text and "unavailable" in text and "coverage 40%" in text

    def test_web_section_and_json(self) -> None:
        from tests.test_mqtt_bridge import _snapshot
        from weatherstat.web import _regulation_html, _snapshot_to_dict

        status = RegulationStatus("publisher", None, NOW, None, {LR: _unit(unit_reading=71.0)}, ())
        snap = _snapshot(regulation=status)
        html = _regulation_html(snap)
        assert "Regulation inputs" in html and "71.5" in html and "unit reads 71.0" in html
        assert _snapshot_to_dict(snap)["regulation"]["units"][LR]["value"] == 71.5
        assert _regulation_html(_snapshot()) == ""

    def test_controller_snapshot_and_request_cycle(self) -> None:
        from weatherstat.tui.controller import TUIController

        logs: list[str] = []
        spawned: list = []
        c = TUIController(live=False, log=logs.append, worker=spawned.append)
        status = RegulationStatus("publisher", None, NOW, None, {LR: _unit()}, ())
        c.set_regulation_source(lambda: status)
        assert c._read_regulation() is status

        assert c.request_cycle("Regulation input changed").status == "started"
        assert len(spawned) == 1
        # A request during a running cycle queues one more.
        assert c.request_cycle("again").status == "already_running"
        assert c._cycle_requested
