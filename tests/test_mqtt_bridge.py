"""MQTT bridge: discovery, overlay state, debounced commands, snapshot publishing."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from weatherstat.mqtt_bridge import (
    BridgeTopics,
    MqttBridge,
    comfort_status,
    command_for,
    discovery_messages,
    overlay_payload,
    parse_set_payload,
    room_payload,
    status_payload,
)
from weatherstat.overlay import ComfortOverlay, OverlayCommand
from weatherstat.tui.controller import ComfortBounds, EffectorView, OverlayView, StatusSnapshot
from weatherstat.yaml_config import MqttConfig

NOW = datetime(2026, 10, 22, 20, 0, tzinfo=UTC)
LABELS = {"bedroom_temp": "bedroom", "family_room_temp": "family_room"}


class FakeClient:
    def __init__(self) -> None:
        self.published: list[tuple[str, str | None, bool]] = []
        self.subscribed: list[str] = []

    def publish(self, topic, payload=None, qos=0, retain=False):
        self.published.append((topic, payload, retain))

    def subscribe(self, topic, qos=0):
        self.subscribed.append(topic)

    def last(self, topic: str) -> dict:
        for t, payload, _ in reversed(self.published):
            if t == topic:
                return json.loads(payload)
        raise AssertionError(f"nothing published to {topic}")

    def topics(self) -> list[str]:
        return [t for t, _, _ in self.published]


class ManualTimer:
    """Debounce timer the test fires by hand."""

    created: list[ManualTimer] = []

    def __init__(self, secs: float, fn) -> None:
        self.secs, self.fn = secs, fn
        self.started = self.cancelled = False
        ManualTimer.created.append(self)

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        assert self.started and not self.cancelled
        self.fn()


@pytest.fixture
def bridge():
    ManualTimer.created = []
    applied: list[OverlayCommand] = []
    logs: list[str] = []
    overlays: dict[str, ComfortOverlay] = {}
    b = MqttBridge(
        MqttConfig(debounce_seconds=2.0),
        apply=applied.append,
        log=logs.append,
        labels=LABELS,
        step=1.0,
        max_delta=5.0,
        unit="°F",
        source_entities={"bedroom_temp": "sensor.climate_bedroom_air_temperature"},
        load_overlays=lambda: dict(overlays),
        timer_factory=ManualTimer,
    )
    client = FakeClient()
    b.attach(client)
    b._connected = True
    return b, client, applied, overlays, logs


def _snapshot(**overrides) -> StatusSnapshot:
    base = {
        "timestamp": NOW,
        "profile": "Home",
        "live": True,
        "cycle_running": False,
        "sysid_running": False,
        "next_cycle_at": NOW + timedelta(minutes=3),
        "sysid_age": timedelta(minutes=30),
        "collector_age": timedelta(minutes=2),
        "collector_rows": 100,
        "local_tz": "America/Los_Angeles",
        "outdoor_temp": 54.0,
        "weather_condition": "cloudy",
        "temps": {"bedroom_temp": 70.0, "family_room_temp": 72.0},
        "sensor_labels": dict(LABELS),
        "comfort": {
            "bedroom": ComfortBounds(69.0, 71.0, 73.0, 74.0),
            "family_room": ComfortBounds(69.0, 70.0, 73.0, 74.0),
        },
        "mrt_offsets": {"bedroom_temp": 0.5},
        "environment": (),
        "forecast": {"1h": 53.0},
        "effectors": (
            EffectorView("mini_split_bedroom", "heat", 71.5, 0, 24, None),
            EffectorView("thermostat_upstairs", "off", 69.0, 0, None, None),
        ),
        "command_targets": {"mini_split_bedroom": 71.5},
        "rationale": {"mini_split_bedroom": "bedroom 1.5° below preferred"},
        "costs": None,
        "sensor_costs": {"bedroom_temp": 1.234},
        "baseline_sensor_costs": {"bedroom_temp": 4.5},
        "predictions": {"bedroom_temp": {"1h": 70.6, "2h": 71.2}},
        "opportunities": [],
        "warnings": [],
        "ha_error": None,
        "overlays": {"bedroom_temp": OverlayView(2.0, NOW + timedelta(hours=2), "ha")},
    }
    base.update(overrides)
    return StatusSnapshot(**base)


# ── Pure helpers ──────────────────────────────────────────────────────────


class TestTopicsAndDiscovery:
    def test_label_from_set_topic(self) -> None:
        t = BridgeTopics()
        assert t.label_from_set_topic("weatherstat/overlay/bedroom/set") == "bedroom"
        assert t.label_from_set_topic("weatherstat/overlay/bedroom") is None
        assert t.label_from_set_topic("weatherstat/overlay/a/b/set") is None
        assert t.label_from_set_topic("other/overlay/bedroom/set") is None

    def test_discovery_entities(self) -> None:
        msgs = dict(discovery_messages(BridgeTopics(), LABELS, step=1.0, max_delta=5.0, unit="°F"))
        assert len(msgs) == 3 * len(LABELS) + 1

        number = msgs["homeassistant/number/weatherstat/bedroom_comfort/config"]
        assert number["default_entity_id"] == "number.weatherstat_bedroom_comfort"
        assert number["command_topic"] == "weatherstat/overlay/bedroom/set"
        assert (number["min"], number["max"], number["step"]) == (-5.0, 5.0, 1.0)
        assert number["availability_topic"] == "weatherstat/status"
        assert number["device"]["identifiers"] == ["weatherstat"]

        until = msgs["homeassistant/sensor/weatherstat/family_room_comfort_until/config"]
        assert until["device_class"] == "timestamp"
        assert until["name"] == "Family room comfort until"

        room = msgs["homeassistant/sensor/weatherstat/bedroom/config"]
        assert room["default_entity_id"] == "sensor.weatherstat_bedroom"
        assert room["state_topic"] == room["json_attributes_topic"] == "weatherstat/room/bedroom"
        assert room["device_class"] == "temperature"

        status = msgs["homeassistant/sensor/weatherstat/status/config"]
        assert status["default_entity_id"] == "sensor.weatherstat_status"

    def test_unique_ids_are_unique(self) -> None:
        msgs = discovery_messages(BridgeTopics(), LABELS, step=1.0, max_delta=5.0, unit="°F")
        ids = [p["unique_id"] for _, p in msgs]
        assert len(ids) == len(set(ids))


class TestPayloads:
    def test_parse_set_payload_clamps(self) -> None:
        assert parse_set_payload(b"2", max_delta=5.0) == 2.0
        assert parse_set_payload("-1.5", max_delta=5.0) == -1.5
        assert parse_set_payload("9", max_delta=5.0) == 5.0
        assert parse_set_payload("-9", max_delta=5.0) == -5.0

    @pytest.mark.parametrize("bad", ["", "warm", "nan"])
    def test_parse_set_payload_rejects(self, bad: str) -> None:
        with pytest.raises(ValueError):
            parse_set_payload(bad, max_delta=5.0)

    def test_command_for(self) -> None:
        assert command_for("bedroom_temp", 0.0) == OverlayCommand("bedroom_temp", "clear")
        assert command_for("bedroom_temp", 2.0) == OverlayCommand("bedroom_temp", "set", 2.0)

    def test_overlay_payload(self) -> None:
        assert overlay_payload(None, NOW) == {"delta": 0.0, "expires_at": None, "remaining": None, "source": None}
        o = ComfortOverlay("bedroom_temp", 2.0, NOW, NOW + timedelta(hours=2), "ha")
        p = overlay_payload(o, NOW)
        assert p["delta"] == 2.0 and p["source"] == "ha"
        assert p["expires_at"] == (NOW + timedelta(hours=2)).isoformat()
        # Expired overlays read as zero.
        assert overlay_payload(o, NOW + timedelta(hours=3))["delta"] == 0.0

    def test_comfort_status(self) -> None:
        b = ComfortBounds(69.0, 71.0, 73.0, 74.0)
        assert [comfort_status(v, b) for v in (68, 70, 72, 73.5, 75)] == ["cold", "cool", "ok", "warm", "hot"]

    def test_room_payload(self) -> None:
        p = room_payload(_snapshot(), "bedroom_temp", "bedroom", source_entity="sensor.x", unit="°F")
        assert p is not None
        assert p["temperature"] == 70.0 and p["feels_like"] == 69.5 and p["mrt_offset"] == 0.5
        assert p["status"] == "cool"
        assert (p["preferred_lo"], p["preferred_hi"], p["acceptable_lo"], p["acceptable_hi"]) == (71, 73, 69, 74)
        assert p["overlay"] == 2.0 and p["overlay_until"] == (NOW + timedelta(hours=2)).isoformat()
        assert p["predictions"] == {"1h": 70.6, "2h": 71.2}
        assert p["cost"] == 1.234 and p["baseline_cost"] == 4.5
        assert p["comfort_entity"] == "number.weatherstat_bedroom_comfort"
        assert p["source_entity"] == "sensor.x"

    def test_room_payload_without_reading(self) -> None:
        assert room_payload(_snapshot(temps={}), "bedroom_temp", "bedroom", source_entity=None, unit="°F") is None

    def test_status_payload(self) -> None:
        p = status_payload(_snapshot())
        assert p["mode"] == "live" and p["profile"] == "Home"
        assert p["active_count"] == 1
        split = p["effectors"][0]
        assert split["title"] == "Mini split bedroom"
        assert split["duration_minutes"] == 120 and split["delay_minutes"] == 0
        assert split["rationale"] == "bedroom 1.5° below preferred"
        assert p["overlays"] == {"bedroom": {"delta": 2.0, "expires_at": (NOW + timedelta(hours=2)).isoformat()}}
        json.dumps(p)  # serializable


# ── Bridge behaviour ──────────────────────────────────────────────────────


class TestBridge:
    def test_on_connected_publishes_discovery_and_state(self, bridge) -> None:
        b, client, _, overlays, _ = bridge
        overlays["bedroom_temp"] = ComfortOverlay("bedroom_temp", 2.0, NOW, datetime.now(UTC) + timedelta(hours=1), "cli")
        b.on_connected()
        configs = [t for t in client.topics() if t.endswith("/config")]
        assert len(configs) == 3 * len(LABELS) + 1
        assert all(retain for t, _, retain in client.published if t.endswith("/config"))
        assert ("weatherstat/status", "online", True) in client.published
        assert client.subscribed == ["weatherstat/overlay/+/set"]
        assert client.last("weatherstat/overlay/bedroom")["delta"] == 2.0
        assert client.last("weatherstat/overlay/family_room")["delta"] == 0.0

    def test_publish_overlays_skips_unchanged(self, bridge) -> None:
        b, client, _, overlays, _ = bridge
        assert b.publish_overlays(now=NOW) == len(LABELS)
        assert b.publish_overlays(now=NOW) == 0
        overlays["bedroom_temp"] = ComfortOverlay("bedroom_temp", -1.0, NOW, NOW + timedelta(hours=3), "tui")
        assert b.publish_overlays(now=NOW) == 1
        assert client.last("weatherstat/overlay/bedroom")["delta"] == -1.0
        # 'remaining' alone changing does not republish
        assert b.publish_overlays(now=NOW + timedelta(minutes=10)) == 0
        assert b.publish_overlays(force=True, now=NOW) == len(LABELS)

    def test_command_echoes_then_applies_after_debounce(self, bridge) -> None:
        b, client, applied, _, _ = bridge
        cmd = b.handle_command("weatherstat/overlay/bedroom/set", b"1")
        assert cmd == OverlayCommand("bedroom_temp", "set", 1.0)
        assert client.last("weatherstat/overlay/bedroom")["delta"] == 1.0  # optimistic echo
        assert applied == []
        ManualTimer.created[-1].fire()
        assert applied == [OverlayCommand("bedroom_temp", "set", 1.0)]

    def test_rapid_presses_apply_only_the_last(self, bridge) -> None:
        b, _, applied, _, _ = bridge
        for v in (b"1", b"2", b"3"):
            b.handle_command("weatherstat/overlay/bedroom/set", v)
        first, second, third = ManualTimer.created
        assert first.cancelled and second.cancelled and not third.cancelled
        third.fire()
        assert applied == [OverlayCommand("bedroom_temp", "set", 3.0)]

    def test_rooms_debounce_independently(self, bridge) -> None:
        b, _, applied, _, _ = bridge
        b.handle_command("weatherstat/overlay/bedroom/set", b"1")
        b.handle_command("weatherstat/overlay/family_room/set", b"0")
        for t in ManualTimer.created:
            t.fire()
        assert applied == [OverlayCommand("bedroom_temp", "set", 1.0), OverlayCommand("family_room_temp", "clear")]

    def test_bad_commands_are_logged_not_applied(self, bridge) -> None:
        b, _, applied, _, logs = bridge
        assert b.handle_command("weatherstat/overlay/garage/set", b"1") is None
        assert b.handle_command("weatherstat/overlay/bedroom/set", b"warmer") is None
        assert ManualTimer.created == [] and applied == []
        assert any("unknown topic" in line for line in logs)
        assert any("Bad command" in line for line in logs)

    def test_apply_failure_resyncs_state(self, bridge) -> None:
        b, client, _, overlays, logs = bridge

        def boom(_cmd):
            raise RuntimeError("disk full")

        b._apply = boom
        b.handle_command("weatherstat/overlay/bedroom/set", b"2")
        ManualTimer.created[-1].fire()
        assert any("disk full" in line for line in logs)
        # HA is told the truth (no overlay) rather than left showing the echo
        assert client.last("weatherstat/overlay/bedroom")["delta"] == 0.0

    def test_publish_snapshot_dedupes_ignoring_timestamp(self, bridge) -> None:
        b, client, _, _, _ = bridge
        assert b.publish_snapshot(_snapshot()) == 1 + len(LABELS)
        room = client.last("weatherstat/room/bedroom")
        assert room["feels_like"] == 69.5
        assert room["source_entity"] == "sensor.climate_bedroom_air_temperature"
        assert client.last("weatherstat/state")["mode"] == "live"
        # Same content, later timestamp: nothing new
        assert b.publish_snapshot(_snapshot(timestamp=NOW + timedelta(minutes=1))) == 0
        # One room changes
        assert b.publish_snapshot(_snapshot(temps={"bedroom_temp": 71.0, "family_room_temp": 72.0})) == 1

    def test_snapshot_before_connect_is_sent_on_connect(self, bridge) -> None:
        b, client, _, _, _ = bridge
        b._connected = False
        assert b.publish_snapshot(_snapshot()) == 0
        b._connected = True
        b.on_connected()
        assert client.last("weatherstat/room/bedroom")["feels_like"] == 69.5


class TestConfig:
    def _load(self, tmp_path, extra: str):
        from pathlib import Path

        from weatherstat.yaml_config import load_config

        example = Path(__file__).resolve().parents[1] / "weatherstat.yaml.example"
        cfg_path = tmp_path / "weatherstat.yaml"
        cfg_path.write_text(example.read_text() + "\n" + extra)
        return load_config(str(cfg_path))

    def test_absent_means_disabled(self, tmp_path) -> None:
        assert self._load(tmp_path, "").mqtt is None

    def test_block_parsed(self, tmp_path) -> None:
        cfg = self._load(tmp_path, "mqtt:\n  host: broker.local\n  port: 1884\n  debounce_seconds: 1.5\n").mqtt
        assert cfg == MqttConfig(host="broker.local", port=1884, debounce_seconds=1.5)

    def test_empty_block_enables_defaults(self, tmp_path) -> None:
        assert self._load(tmp_path, "mqtt: {}\n").mqtt == MqttConfig()

    def test_broker_host_defaults_to_ha(self) -> None:
        from weatherstat.mqtt_bridge import broker_host

        assert broker_host(MqttConfig(), "https://ha.example.com:8123") == "ha.example.com"
        assert broker_host(MqttConfig(host="mq"), "https://ha.example.com") == "mq"
        assert broker_host(MqttConfig(), "") is None
