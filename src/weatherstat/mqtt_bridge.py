"""MQTT bridge — Weatherstat as a Home Assistant device.

Publishes MQTT discovery messages so Home Assistant creates a "Weatherstat"
device with, for every constrained sensor:

- ``number.weatherstat_<label>_comfort`` — the room's comfort overlay in
  degrees (−max … +max, one ``overlay_step`` per press). Setting it from HA
  (a tile card's −/+ buttons, a voice assistant, an automation) applies an
  absolute overlay for the default duration; 0 clears it.
- ``sensor.weatherstat_<label>_comfort_until`` — when that overlay expires
  (a timestamp; unknown when no overlay is active).
- ``sensor.weatherstat_<label>`` — the room's feels-like temperature (air
  temperature minus the MRT offset), with everything else the TUI knows about
  the room as attributes: comfort band, overlay, predictions, comfort cost,
  status (cold / cool / ok / warm / hot), and the HA sensor it reads.

Plus ``sensor.weatherstat_status`` (live / dry run) carrying the effector plan
with per-device rationale, costs, advisories, warnings and data health. The
room and status sensors come from the TUI's status snapshot, so they are
published by the TUI only; the headless loop publishes the comfort controls.

Weatherstat stays the source of truth: overlays live in ``overlays.json`` and
every change still goes through ``apply_overlay_command`` (same events, same
context). The bridge republishes overlay state every ``STATE_INTERVAL``
seconds, so changes from the CLI, the TUI and the web page, and expiries,
reach HA without any extra plumbing.

Rapid presses are debounced: each command is echoed back immediately (the HA
tile updates at once) but only the final value after ``debounce_seconds`` of
quiet is applied — three taps of + are one recorded "set +3", not three
events and three control cycles.

Configuration: an ``mqtt:`` block in weatherstat.yaml enables the bridge;
``MQTT_USERNAME`` / ``MQTT_PASSWORD`` come from ``.env``. The broker host
defaults to the host of ``HA_URL`` (the Mosquitto app on the HA box).
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Protocol
from urllib.parse import urlparse

from weatherstat.overlay import ComfortOverlay, OverlayCommand, OverlayStore, format_remaining
from weatherstat.yaml_config import MqttConfig

STATE_INTERVAL = 30.0  # seconds between overlay state republishes (catches CLI changes and expiry)

LogFn = Callable[[str], None]
ApplyFn = Callable[[OverlayCommand], Any]


# ── Topics and payloads (pure) ─────────────────────────────────────────────


@dataclass(frozen=True)
class BridgeTopics:
    prefix: str = "weatherstat"
    discovery_prefix: str = "homeassistant"

    @property
    def availability(self) -> str:
        return f"{self.prefix}/status"

    @property
    def overlay_set_wildcard(self) -> str:
        return f"{self.prefix}/overlay/+/set"

    def overlay_state(self, label: str) -> str:
        return f"{self.prefix}/overlay/{label}"

    def overlay_set(self, label: str) -> str:
        return f"{self.prefix}/overlay/{label}/set"

    def number_config(self, label: str) -> str:
        return f"{self.discovery_prefix}/number/weatherstat/{label}_comfort/config"

    def until_config(self, label: str) -> str:
        return f"{self.discovery_prefix}/sensor/weatherstat/{label}_comfort_until/config"

    @property
    def status_state(self) -> str:
        return f"{self.prefix}/state"

    def room_state(self, label: str) -> str:
        return f"{self.prefix}/room/{label}"

    def room_config(self, label: str) -> str:
        return f"{self.discovery_prefix}/sensor/weatherstat/{label}/config"

    @property
    def status_config(self) -> str:
        return f"{self.discovery_prefix}/sensor/weatherstat/status/config"

    def label_from_set_topic(self, topic: str) -> str | None:
        head = f"{self.prefix}/overlay/"
        if not (topic.startswith(head) and topic.endswith("/set")):
            return None
        label = topic[len(head) : -len("/set")]
        return label if label and "/" not in label else None


def display_name(label: str) -> str:
    """'family_room' → 'Family room'."""
    return label.replace("_", " ").strip().capitalize()


def _package_version() -> str:
    try:
        return version("weatherstat")
    except PackageNotFoundError:
        return "dev"


def device_info() -> dict:
    return {
        "identifiers": ["weatherstat"],
        "name": "Weatherstat",
        "manufacturer": "Weatherstat",
        "model": "Comfort controller",
        "sw_version": _package_version(),
    }


def discovery_messages(
    topics: BridgeTopics,
    labels: dict[str, str],
    *,
    step: float,
    max_delta: float,
    unit: str,
) -> list[tuple[str, dict]]:
    """Discovery config messages for every constrained sensor (``labels``: column → label)."""
    device = device_info()
    origin = {"name": "Weatherstat", "sw_version": device["sw_version"]}
    messages: list[tuple[str, dict]] = []
    for label in labels.values():
        name = display_name(label)
        state_topic = topics.overlay_state(label)
        common = {
            "state_topic": state_topic,
            "json_attributes_topic": state_topic,
            "availability_topic": topics.availability,
            "device": device,
            "origin": origin,
        }
        messages.append(
            (
                topics.number_config(label),
                {
                    "name": f"{name} comfort",
                    "unique_id": f"weatherstat_{label}_comfort",
                    "default_entity_id": f"number.weatherstat_{label}_comfort",
                    "command_topic": topics.overlay_set(label),
                    "value_template": "{{ value_json.delta }}",
                    "min": -max_delta,
                    "max": max_delta,
                    "step": step,
                    "mode": "box",
                    "unit_of_measurement": unit,
                    "icon": "mdi:thermometer-plus",
                    **common,
                },
            )
        )
        messages.append(
            (
                topics.until_config(label),
                {
                    "name": f"{name} comfort until",
                    "unique_id": f"weatherstat_{label}_comfort_until",
                    "default_entity_id": f"sensor.weatherstat_{label}_comfort_until",
                    "value_template": "{{ value_json.expires_at if value_json.expires_at else None }}",
                    "device_class": "timestamp",
                    "icon": "mdi:timer-sand",
                    **common,
                },
            )
        )
        room_topic = topics.room_state(label)
        messages.append(
            (
                topics.room_config(label),
                {
                    "name": f"{name} feels like",
                    "unique_id": f"weatherstat_{label}",
                    "default_entity_id": f"sensor.weatherstat_{label}",
                    "state_topic": room_topic,
                    "json_attributes_topic": room_topic,
                    "value_template": "{{ value_json.feels_like }}",
                    "device_class": "temperature",
                    "state_class": "measurement",
                    "unit_of_measurement": unit,
                    "suggested_display_precision": 1,
                    "availability_topic": topics.availability,
                    "device": device,
                    "origin": origin,
                },
            )
        )
    messages.append(
        (
            topics.status_config,
            {
                "name": "Status",
                "unique_id": "weatherstat_status",
                "default_entity_id": "sensor.weatherstat_status",
                "state_topic": topics.status_state,
                "json_attributes_topic": topics.status_state,
                "value_template": "{{ value_json.mode }}",
                "icon": "mdi:home-thermometer",
                "availability_topic": topics.availability,
                "device": device,
                "origin": origin,
            },
        )
    )
    return messages


def overlay_payload(overlay: ComfortOverlay | None, now: datetime) -> dict:
    """State payload for one sensor: the active delta (0 when none) and its expiry."""
    if overlay is None or not overlay.is_active(now):
        return {"delta": 0.0, "expires_at": None, "remaining": None, "source": None}
    return {
        "delta": round(overlay.delta, 2),
        "expires_at": overlay.expires_at.isoformat(),
        "remaining": format_remaining(overlay.remaining(now)),
        "source": overlay.source,
    }


def state_messages(
    topics: BridgeTopics,
    labels: dict[str, str],
    overlays: dict[str, ComfortOverlay],
    now: datetime,
) -> dict[str, dict]:
    """topic → payload for every constrained sensor."""
    return {topics.overlay_state(label): overlay_payload(overlays.get(col), now) for col, label in labels.items()}


def comfort_status(value: float, bounds: Any) -> str:
    """Where a (feels-like) temperature sits: cold / cool / ok / warm / hot."""
    if value < bounds.acceptable_lo:
        return "cold"
    if value < bounds.preferred_lo:
        return "cool"
    if value > bounds.acceptable_hi:
        return "hot"
    if value > bounds.preferred_hi:
        return "warm"
    return "ok"


def _r(value: float | None, ndigits: int = 2) -> float | None:
    return None if value is None else round(float(value), ndigits)


def room_payload(snap: Any, col: str, label: str, *, source_entity: str | None, unit: str) -> dict | None:
    """Everything the TUI knows about one room, as a flat-ish JSON object."""
    temp = snap.temps.get(col)
    bounds = snap.comfort.get(snap.sensor_labels.get(col, label))  # snapshot comfort is keyed by display label
    if temp is None:
        return None
    offset = snap.mrt_offsets.get(col, 0.0) or 0.0
    feels = temp - offset
    overlay = snap.overlays.get(col)
    payload: dict[str, Any] = {
        "name": display_name(label),
        "label": label,
        "sensor": col,
        "unit": unit,
        "temperature": _r(temp),
        "mrt_offset": _r(offset),
        "feels_like": _r(feels),
        "status": comfort_status(feels, bounds) if bounds is not None else None,
        "preferred_lo": _r(bounds.preferred_lo) if bounds else None,
        "preferred_hi": _r(bounds.preferred_hi) if bounds else None,
        "acceptable_lo": _r(bounds.acceptable_lo) if bounds else None,
        "acceptable_hi": _r(bounds.acceptable_hi) if bounds else None,
        "overlay": _r(overlay.delta) if overlay else 0.0,
        "overlay_until": overlay.expires_at.isoformat() if overlay else None,
        "predictions": {h: _r(v) for h, v in (snap.predictions.get(col) or {}).items()},
        "cost": _r(snap.sensor_costs.get(col), 3),
        "baseline_cost": _r(snap.baseline_sensor_costs.get(col), 3),
        "source_entity": source_entity,
        "comfort_entity": f"number.weatherstat_{label}_comfort",
        "updated": snap.timestamp.isoformat(),
    }
    return payload


def opportunity_text(opp: dict) -> str:
    """'Open office window in 20m for 60m' — the TUI's opportunity line, without markup."""
    try:
        from weatherstat.yaml_config import environment_display, load_config

        dev = str(opp.get("device", "?"))
        label, kind = environment_display(dev, load_config().environment.get(dev))
    except Exception:  # noqa: BLE001 — display only
        label, kind = str(opp.get("device", "?")), ""
    mins = opp.get("in_minutes") or 0
    dur = opp.get("duration_minutes")
    timing = "now" if mins == 0 else f"in {format_remaining(timedelta(minutes=mins))}"
    dur_str = f" for {format_remaining(timedelta(minutes=dur))}" if dur else ""
    action = str(opp.get("action", "?")).capitalize()
    return " ".join(part for part in (action, label, kind) if part) + f" {timing}{dur_str}"


_MAX_LIST = 10  # cap advisory lists so attributes stay small enough for the recorder


def status_payload(snap: Any) -> dict:
    """System-level view: mode, schedule, effector plan + rationale, costs, advisories, health."""
    active = [e for e in snap.effectors if e.mode not in ("off", "", None)]
    costs = snap.costs
    return {
        "mode": "live" if snap.live else "dry run",
        "profile": snap.profile,
        "cycle_running": snap.cycle_running,
        "sysid_running": snap.sysid_running,
        "next_cycle_at": snap.next_cycle_at.isoformat() if snap.next_cycle_at else None,
        "sysid_age_minutes": _r(snap.sysid_age.total_seconds() / 60, 1) if snap.sysid_age else None,
        "collector_age_minutes": _r(snap.collector_age.total_seconds() / 60, 1) if snap.collector_age else None,
        "outdoor_temp": _r(snap.outdoor_temp, 1),
        "weather_condition": snap.weather_condition,
        "forecast": dict(snap.forecast),
        "active_count": len(active),
        "effectors": [
            {
                "name": e.name,
                "title": display_name(e.name),
                "mode": e.mode,
                "target": _r(e.target, 1),
                "delay_minutes": e.delay_steps * 5,
                "duration_minutes": e.duration_steps * 5 if e.duration_steps is not None else None,
                "override": e.override,
                "rationale": snap.rationale.get(e.name),
            }
            for e in snap.effectors
        ],
        "costs": (
            {
                "total": _r(costs.total, 3),
                "comfort": _r(costs.comfort, 3),
                "energy": _r(costs.energy, 3),
                "baseline": _r(costs.baseline, 3),
            }
            if costs
            else None
        ),
        "environment": [{"label": e.label, "kind": e.kind, "active": e.is_active} for e in snap.environment],
        "opportunities": [{**o, "text": opportunity_text(o)} for o in list(snap.opportunities)[:_MAX_LIST]],
        "warnings": list(snap.warnings)[:_MAX_LIST],
        "overlays": {snap.sensor_labels.get(col, col): {"delta": _r(o.delta), "expires_at": o.expires_at.isoformat()} for col, o in snap.overlays.items()},
        "ha_error": snap.ha_error,
        "updated": snap.timestamp.isoformat(),
    }


def snapshot_messages(
    topics: BridgeTopics,
    snap: Any,
    labels: dict[str, str],
    *,
    source_entities: dict[str, str],
    unit: str,
) -> dict[str, dict]:
    """topic → payload for the status sensor and every room with a reading."""
    out: dict[str, dict] = {topics.status_state: status_payload(snap)}
    for col, label in labels.items():
        payload = room_payload(snap, col, label, source_entity=source_entities.get(col), unit=unit)
        if payload is not None:
            out[topics.room_state(label)] = payload
    return out


def parse_set_payload(payload: bytes | str, *, max_delta: float) -> float:
    """Decode a number command payload ('2', '-1.0') into a clamped delta."""
    text = payload.decode() if isinstance(payload, bytes) else payload
    try:
        value = float(text.strip())
    except ValueError as e:
        raise ValueError(f"not a number: {text!r}") from e
    if value != value:  # NaN
        raise ValueError("not a number: nan")
    return max(-max_delta, min(max_delta, value))


def command_for(sensor: str, delta: float) -> OverlayCommand:
    """A number value is absolute: 0 clears, anything else sets."""
    if abs(delta) < 0.05:
        return OverlayCommand(sensor, "clear")
    return OverlayCommand(sensor, "set", delta)


# ── Bridge ─────────────────────────────────────────────────────────────────


class MqttClient(Protocol):
    """The subset of paho.mqtt.client.Client the bridge uses (fakeable in tests)."""

    def publish(self, topic: str, payload: str | None = None, qos: int = 0, retain: bool = False) -> Any: ...
    def subscribe(self, topic: str, qos: int = 0) -> Any: ...


class TimerLike(Protocol):
    def start(self) -> None: ...
    def cancel(self) -> None: ...


TimerFactory = Callable[[float, Callable[[], None]], TimerLike]


class MqttBridge:
    """Owns the MQTT connection, discovery, state publishing and command handling.

    Thread model: paho's network thread calls ``_on_connect`` / ``_on_message``;
    debounce timers apply commands on their own threads; a publisher thread
    republishes state every ``STATE_INTERVAL`` seconds. ``publish_overlays``
    is safe to call from anywhere (it serializes on ``self._lock``).
    """

    def __init__(
        self,
        cfg: MqttConfig,
        *,
        apply: ApplyFn,
        log: LogFn,
        labels: dict[str, str] | None = None,
        step: float | None = None,
        max_delta: float | None = None,
        unit: str | None = None,
        source_entities: dict[str, str] | None = None,
        load_overlays: Callable[[], dict[str, ComfortOverlay]] | None = None,
        timer_factory: TimerFactory | None = None,
    ) -> None:
        from weatherstat.config import OVERLAY_MAX_DELTA, OVERLAY_STEP, SENSOR_LABELS, UNIT_SYMBOL

        self.cfg = cfg
        self.topics = BridgeTopics(cfg.topic_prefix, cfg.discovery_prefix)
        self._apply = apply
        self._log = log
        self.labels = dict(labels if labels is not None else SENSOR_LABELS)
        self._by_label = {label: col for col, label in self.labels.items()}
        self.step = step if step is not None else OVERLAY_STEP
        self.max_delta = max_delta if max_delta is not None else OVERLAY_MAX_DELTA
        self.unit = unit or UNIT_SYMBOL
        self.source_entities = dict(source_entities if source_entities is not None else _temp_entities())
        self._load_overlays = load_overlays or (lambda: {o.sensor: o for o in OverlayStore.load().all()})
        self._timer_factory: TimerFactory = timer_factory or (lambda secs, fn: threading.Timer(secs, fn))

        self._lock = threading.RLock()
        self._client: MqttClient | None = None
        self._connected = False
        self._published: dict[str, str] = {}  # topic → last payload (skip unchanged republishes)
        self._pending: dict[str, TimerLike] = {}  # sensor → debounce timer
        self._stop = threading.Event()
        self._publisher: threading.Thread | None = None
        self._last_snapshot: Any = None  # re-sent on (re)connect

    # ── Lifecycle ──

    def start(self, *, host: str, username: str | None, password: str | None) -> None:
        import paho.mqtt.client as mqtt

        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"weatherstat-{os.getpid()}",
        )
        if username:
            client.username_pw_set(username, password)
        client.will_set(self.topics.availability, "offline", qos=1, retain=True)
        client.reconnect_delay_set(min_delay=1, max_delay=60)
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        self.attach(client)
        client.connect_async(host, self.cfg.port, keepalive=60)
        client.loop_start()
        self._publisher = threading.Thread(target=self._publish_loop, daemon=True, name="weatherstat-mqtt-state")
        self._publisher.start()

    def attach(self, client: MqttClient) -> None:
        """Use ``client`` for publishing (``start`` does this; tests call it directly)."""
        with self._lock:
            self._client = client

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            for timer in self._pending.values():
                timer.cancel()
            self._pending.clear()
            client = self._client
        if client is None:
            return
        try:
            client.publish(self.topics.availability, "offline", qos=1, retain=True)
            client.disconnect()  # type: ignore[attr-defined]
            client.loop_stop()  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001 — shutting down
            self._log(f"[mqtt] Error stopping: {e}")

    @property
    def connected(self) -> bool:
        return self._connected

    # ── paho callbacks ──

    def _on_connect(self, client: Any, _userdata: Any, _flags: Any, reason_code: Any, _properties: Any = None) -> None:
        if getattr(reason_code, "is_failure", False):
            self._connected = False
            self._log(f"[mqtt] [red]Connection refused: {reason_code}[/]")
            return
        self._connected = True
        self.on_connected()

    def _on_disconnect(self, _client: Any, _userdata: Any, _flags: Any, reason_code: Any, _properties: Any = None) -> None:
        self._connected = False
        if not self._stop.is_set():
            self._log(f"[mqtt] Disconnected ({reason_code}); reconnecting")

    def _on_message(self, _client: Any, _userdata: Any, msg: Any) -> None:
        self.handle_command(msg.topic, msg.payload)

    # ── Connection setup ──

    def on_connected(self) -> None:
        """Publish discovery + availability, subscribe to commands, push current state."""
        client = self._client
        if client is None:
            return
        for topic, payload in discovery_messages(
            self.topics,
            self.labels,
            step=self.step,
            max_delta=self.max_delta,
            unit=self.unit,
        ):
            client.publish(topic, json.dumps(payload), qos=1, retain=True)
        client.publish(self.topics.availability, "online", qos=1, retain=True)
        client.subscribe(self.topics.overlay_set_wildcard, qos=1)
        self.publish_overlays(force=True)
        with self._lock:
            for topic in [t for t in self._published if t.startswith(f"{self.topics.prefix}/room/")]:
                self._published.pop(topic, None)
            self._published.pop(self.topics.status_state, None)
            snap = self._last_snapshot
        if snap is not None:
            self.publish_snapshot(snap)
        self._log(f"[mqtt] Connected; publishing {len(self.labels)} room comfort control(s) to Home Assistant")

    # ── State ──

    def publish_overlays(self, *, force: bool = False, now: datetime | None = None) -> int:
        """Publish each sensor's overlay state (retained). Returns how many topics were sent."""
        now = now or datetime.now(UTC)
        try:
            overlays = self._load_overlays()
        except Exception as e:  # noqa: BLE001 — never kill the publisher thread
            self._log(f"[mqtt] [red]Could not read overlays: {e}[/]")
            return 0
        sent = 0
        with self._lock:
            client = self._client
            if client is None:
                return 0
            for topic, payload in state_messages(self.topics, self.labels, overlays, now).items():
                if not force and self._published.get(topic) == _stable_key(payload):
                    continue
                client.publish(topic, json.dumps(payload), qos=1, retain=True)
                self._published[topic] = _stable_key(payload)
                sent += 1
        return sent

    def publish_snapshot(self, snap: Any) -> int:
        """Publish the TUI's status snapshot (status sensor + one sensor per room).

        Unchanged payloads are skipped (ignoring their 'updated' stamps), so
        calling this on every snapshot refresh is cheap.
        """
        try:
            messages = snapshot_messages(
                self.topics,
                snap,
                self.labels,
                source_entities=self.source_entities,
                unit=self.unit,
            )
        except Exception as e:  # noqa: BLE001 — a bad snapshot must not break the TUI refresh
            self._log(f"[mqtt] [red]Could not serialize snapshot: {e}[/]")
            return 0
        sent = 0
        with self._lock:
            client = self._client
            if client is None or not self._connected:
                self._last_snapshot = snap
                return 0
            self._last_snapshot = snap
            for topic, payload in messages.items():
                key = _stable_key(payload)
                if self._published.get(topic) == key:
                    continue
                client.publish(topic, json.dumps(payload), qos=1, retain=True)
                self._published[topic] = key
                sent += 1
        return sent

    def _publish_loop(self) -> None:
        while not self._stop.wait(STATE_INTERVAL):
            if self._connected:
                self.publish_overlays()

    # ── Commands ──

    def handle_command(self, topic: str, payload: bytes | str) -> OverlayCommand | None:
        """Echo the requested value at once; apply it after the debounce window."""
        label = self.topics.label_from_set_topic(topic)
        sensor = self._by_label.get(label) if label else None
        if sensor is None:
            self._log(f"[mqtt] Ignoring command on unknown topic {topic}")
            return None
        try:
            delta = parse_set_payload(payload, max_delta=self.max_delta)
        except ValueError as e:
            self._log(f"[mqtt] [red]Bad command for {label}: {e}[/]")
            return None
        cmd = command_for(sensor, delta)

        from weatherstat.config import OVERLAY_DURATION_HOURS

        now = datetime.now(UTC)
        echo = (
            overlay_payload(None, now)
            if cmd.action == "clear"
            else {
                "delta": round(delta, 2),
                "expires_at": (now + timedelta(hours=OVERLAY_DURATION_HOURS)).isoformat(),
                "remaining": format_remaining(timedelta(hours=OVERLAY_DURATION_HOURS)),
                "source": "ha",
            }
        )
        with self._lock:
            client = self._client
            if client is not None:
                state_topic = self.topics.overlay_state(label)  # type: ignore[arg-type]
                client.publish(state_topic, json.dumps(echo), qos=1, retain=True)
                self._published.pop(state_topic, None)  # the next publish_overlays re-sends the applied truth
            previous = self._pending.pop(sensor, None)
            if previous is not None:
                previous.cancel()
            timer = self._timer_factory(self.cfg.debounce_seconds, lambda: self._apply_pending(sensor, cmd))
            self._pending[sensor] = timer
        timer.start()
        return cmd

    def _apply_pending(self, sensor: str, cmd: OverlayCommand) -> None:
        with self._lock:
            self._pending.pop(sensor, None)
        try:
            self._apply(cmd)
        except Exception as e:  # noqa: BLE001 — report, then resync HA with the truth
            self._log(f"[mqtt] [red]Applying {cmd.action} on {sensor} failed: {e}[/]")
        self.publish_overlays(force=True)


_VOLATILE_KEYS = frozenset({"remaining", "updated"})


def _stable_key(payload: dict) -> str:
    # 'remaining' ticks every minute and 'updated' every refresh; republishing for those alone would spam.
    return json.dumps({k: v for k, v in payload.items() if k not in _VOLATILE_KEYS}, sort_keys=True, default=str)


def _temp_entities() -> dict[str, str]:
    from weatherstat.yaml_config import load_config

    return {col: cfg.entity_id for col, cfg in load_config().temp_sensors.items()}


# ── Factory ────────────────────────────────────────────────────────────────


def broker_host(cfg: MqttConfig, ha_url: str) -> str | None:
    if cfg.host:
        return cfg.host
    return urlparse(ha_url).hostname if ha_url else None


def start_mqtt_bridge(apply: ApplyFn, log: LogFn) -> MqttBridge | None:
    """Start the bridge if ``mqtt:`` is configured; None (and a log line) otherwise."""
    from weatherstat.config import HA_URL
    from weatherstat.yaml_config import load_config

    cfg = load_config().mqtt
    if cfg is None:
        return None
    host = broker_host(cfg, HA_URL)
    if not host:
        log("[mqtt] [red]No broker host: set mqtt.host in weatherstat.yaml or HA_URL in .env[/]")
        return None
    username = os.environ.get("MQTT_USERNAME") or None
    password = os.environ.get("MQTT_PASSWORD") or None
    if username is None:
        log("[mqtt] MQTT_USERNAME not set in .env; connecting anonymously")
    bridge = MqttBridge(cfg, apply=apply, log=log)
    bridge.start(host=host, username=username, password=password)
    log(f"[mqtt] Connecting to {host}:{cfg.port}")
    return bridge
