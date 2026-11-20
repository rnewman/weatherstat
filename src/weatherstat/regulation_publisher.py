"""Publishing the virtual regulation sensors (PLAN-41 §3, §7).

Every ``PUBLISH_INTERVAL`` seconds, independent of the control cycle:

1. one HA REST ``/api/states`` call reads every constrained sensor and every
   regulating effector's climate entity;
2. frozen members are found with ``sensor_health.frozen_sensors_now`` (the
   live values say whether a sensor changed since the last snapshot);
3. V = Σ wₛ·Tₛ over the available members of each unit's adopted weight set
   (``regulation.virtual_reading``), unavailable below min coverage;
4. the values go to the MQTT bridge (``sensor.weatherstat_<effector>_input``).

**One publisher.** The TUI and the headless loop both run this loop; a
``flock`` on ``regulation_publisher.lock`` in the data dir decides which one
publishes. The other process keeps evaluating V for its own display and
retries the lock every tick, so it takes over within a minute when the
publisher exits. Only the publisher publishes, raises follow alerts and
requests control cycles.

**Membership changes** (a member frozen or unavailable, or back; a newly
adopted weight set) shift V's level, so the publisher asks for an immediate
control cycle to re-plan the target on the new scale.

**Does the unit follow V?** For effectors listed in
``defaults.regulation_live`` (switched over to read V), the unit's reported
``current_temperature`` should equal the published V within its 0.5°C steps
and whole-°F reporting. A difference over ``FOLLOW_TOLERANCE_F`` for
``FOLLOW_MINUTES`` raises a safety alert (persistent notification when running live); recovery
dismisses it. It means the remote reading timed out, the ESP reads another
entity, or the unit fell back to its internal thermistor.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import threading
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

from weatherstat import regulation

PUBLISH_INTERVAL = 60.0  # seconds between publishes; the entity expires after 180 s without one
# Unit reading vs published V (°F; converted to the configured unit). Units work in 0.5°C
# steps that HA reports as whole °F, and a unit can read a few tenths low: at 1.0°F, rounding
# alone sustained a mismatch for 10 minutes several times a day in real data; 1.5°F did not.
FOLLOW_TOLERANCE_F = 1.5
FOLLOW_MINUTES = 10.0  # sustained this long before alerting
FOLLOW_COOLDOWN_SECONDS = 6 * 3600  # re-notify a persisting mismatch at most this often
FOLLOW_ALERT_PREFIX = "regulation_follow_"  # SafetyAlert key prefix; cooldown category "regulation_follow"
LOCK_FILE_NAME = "regulation_publisher.lock"

LogFn = Callable[[str], None]


# ── Types ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FollowStatus:
    """Whether a switched-over unit reads what Weatherstat publishes."""

    state: str  # "ok" | "mismatch" (over tolerance, not yet sustained) | "alert" | "no input" | "no reading"
    diff: float | None = None  # unit reading − V
    since: datetime | None = None  # when the current mismatch started

    @property
    def healthy(self) -> bool:
        return self.state not in ("mismatch", "alert")


@dataclass(frozen=True)
class UnitInput:
    """One evaluation of a regulating effector's input."""

    effector: str
    value: float | None  # V; None = unavailable (see ``reason``)
    weights: dict[str, float]  # the full weight set (adopted, or computed when none adopted)
    used: dict[str, float]  # renormalized weights actually applied
    dropped: dict[str, str]  # member -> reason
    coverage: float  # share of total weight available
    adopted_at: str | None  # None = computed from current gains, not adopted
    members: dict[str, float] = field(default_factory=dict)  # member -> reading used
    unit_reading: float | None = None  # the unit's reported current_temperature
    unit_mode: str | None = None  # the unit's hvac mode
    follows: FollowStatus | None = None  # None unless the effector is in regulation_live
    reason: str | None = None  # why there is no value

    @property
    def membership(self) -> tuple[tuple[str, ...], str | None]:
        """What V is made of: used members + weight-set stamp. A change shifts V's level."""
        return (tuple(sorted(self.used)), self.adopted_at)


@dataclass(frozen=True)
class RegulationStatus:
    """The publisher's latest view, for display (TUI, web, MQTT status)."""

    role: str  # "publisher" | "standby"
    holder: str | None  # pid holding the lock (when standby)
    updated: datetime | None
    error: str | None
    units: dict[str, UnitInput]
    live: tuple[str, ...] = ()  # effectors with the follows-V check enabled


# ── HA readings (pure given the states) ───────────────────────────────────


def fetch_ha_states(timeout: float = 10.0) -> dict[str, dict]:
    """All entity states from one ``/api/states`` call: entity_id -> state object."""
    from weatherstat.config import HA_TOKEN, HA_URL

    if not HA_URL or not HA_TOKEN:
        raise RuntimeError("HA_URL / HA_TOKEN not set")
    resp = requests.get(f"{HA_URL}/api/states", headers={"Authorization": f"Bearer {HA_TOKEN}"}, timeout=timeout)
    resp.raise_for_status()
    return {s["entity_id"]: s for s in resp.json() if isinstance(s, dict) and "entity_id" in s}


def _sensor_sources() -> dict[str, tuple[str, str | None]]:
    """column -> (entity_id, attribute or None): temperature sensors, then climate effector temps."""
    from weatherstat.yaml_config import load_config

    cfg = load_config()
    out: dict[str, tuple[str, str | None]] = {col: (s.entity_id, None) for col, s in cfg.temp_sensors.items()}
    for name, eff in cfg.effectors.items():
        if eff.domain == "climate":
            out.setdefault(f"{name}_temp", (eff.entity_id, "current_temperature"))
    return out


def _as_float(raw: Any) -> float | None:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v if v == v and v not in (float("inf"), float("-inf")) else None


def member_values(
    states: dict[str, dict],
    columns: Iterable[str],
    sources: dict[str, tuple[str, str | None]] | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Readings for ``columns`` from HA states: (values, unavailable reasons)."""
    sources = sources if sources is not None else _sensor_sources()
    values: dict[str, float] = {}
    missing: dict[str, str] = {}
    for col in columns:
        src = sources.get(col)
        if src is None:
            missing[col] = "not configured"
            continue
        entity_id, attr = src
        st = states.get(entity_id)
        if st is None:
            missing[col] = "not in HA"
            continue
        raw = st.get("attributes", {}).get(attr) if attr else st.get("state")
        if raw in ("unavailable", "unknown", None, ""):
            missing[col] = "unavailable" if raw in ("unavailable", None, "") else "unknown"
            continue
        v = _as_float(raw)
        if v is None:
            missing[col] = f"not a number ({raw!r})"
        else:
            values[col] = v
    return values, missing


def unit_reading(states: dict[str, dict], effector: str) -> tuple[float | None, str | None]:
    """A regulating effector's own reported temperature and hvac mode."""
    from weatherstat.config import EFFECTOR_MAP

    eff = EFFECTOR_MAP.get(effector)
    st = states.get(eff.entity_id) if eff is not None else None
    if st is None:
        return None, None
    return _as_float(st.get("attributes", {}).get("current_temperature")), st.get("state")


def evaluate_unit(
    effector: str,
    weights: dict[str, float],
    adopted_at: str | None,
    values: dict[str, float],
    unavailable: dict[str, str],
    *,
    min_coverage: float | None = None,
) -> UnitInput:
    """V for one unit from current readings (``regulation.virtual_reading``) plus display context."""
    if not weights:
        return UnitInput(effector, None, {}, {}, {}, 0.0, adopted_at, reason="no weights (no positive gains yet)")
    vr = regulation.virtual_reading(
        weights,
        values,
        unavailable=unavailable,
        min_coverage=regulation.MIN_COVERAGE if min_coverage is None else min_coverage,
    )
    reason = None
    if vr.value is None:
        cov = regulation.MIN_COVERAGE if min_coverage is None else min_coverage
        reason = f"coverage {vr.coverage:.0%} < {cov:.0%}"
    return UnitInput(
        effector=effector,
        value=vr.value,
        weights=dict(weights),
        used=vr.used,
        dropped=vr.dropped,
        coverage=vr.coverage,
        adopted_at=adopted_at,
        members={s: values[s] for s in weights if s in values and s not in vr.dropped},
        reason=reason,
    )


# ── Weights ───────────────────────────────────────────────────────────────

_gains_cache: dict[str, Any] = {"mtime": None, "gains": {}}
_gains_lock = threading.Lock()


def _sim_gains() -> dict[tuple[str, str], tuple[float, float]]:
    """The simulator's filtered gains, reloaded only when thermal_params.json changes."""
    from weatherstat.config import DATA_DIR
    from weatherstat.simulator import load_sim_params

    path = DATA_DIR / "thermal_params.json"
    mtime = path.stat().st_mtime if path.exists() else None
    with _gains_lock:
        if mtime is not None and _gains_cache["mtime"] == mtime:
            return _gains_cache["gains"]
    gains = load_sim_params().gains if mtime is not None else {}
    with _gains_lock:
        _gains_cache.update(mtime=mtime, gains=gains)
    return gains


def load_weight_sets() -> dict[str, tuple[dict[str, float], str | None]]:
    """effector -> (weights, adopted_at): adopted sets, else computed from current gains (adopted_at None)."""
    adopted = regulation.load_adopted()
    out: dict[str, tuple[dict[str, float], str | None]] = {}
    gains: dict[tuple[str, str], tuple[float, float]] | None = None
    for name in regulation.regulating_effectors():
        entry = adopted.get(name)
        if entry and entry.get("weights"):
            out[name] = ({str(k): float(v) for k, v in entry["weights"].items()}, entry.get("adopted_at"))
            continue
        if gains is None:
            gains = _sim_gains()
        out[name] = (regulation.compute_weights(gains, name), None)
    return out


# ── Single-publisher lock ─────────────────────────────────────────────────


class PublisherLock:
    """Non-blocking ``flock`` on a file in the data dir; held for the process lifetime.

    flock locks belong to the open file description, so two instances contend
    even within one process (as two processes do). The holder writes its pid
    into the file for display. The kernel drops the lock when the process
    dies, so a crashed publisher never blocks the next one.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> bool:
        if self._fd is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._fd = fd
        return True

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    def holder(self) -> str | None:
        try:
            return self.path.read_text().strip() or None
        except OSError:
            return None


def default_lock() -> PublisherLock:
    from weatherstat.config import DATA_DIR

    return PublisherLock(DATA_DIR / LOCK_FILE_NAME)


# ── Does the unit follow V? ───────────────────────────────────────────────


def _alert_key(effector: str) -> str:
    return f"{FOLLOW_ALERT_PREFIX}{effector}"


def follow_alert(effector: str, value: float, reading: float, minutes: float):  # -> SafetyAlert
    from weatherstat.config import UNIT_SYMBOL
    from weatherstat.safety import SafetyAlert

    name = effector.replace("_", " ")
    return SafetyAlert(
        key=_alert_key(effector),
        title=f"{name.capitalize()} not following its input",
        message=(
            f"{name.capitalize()} reads {reading:.1f}{UNIT_SYMBOL} but Weatherstat publishes "
            f"{value:.1f}{UNIT_SYMBOL} ({reading - value:+.1f}) for {minutes:.0f} min. The remote reading may have "
            "timed out, the ESP may read another entity, or the unit is on its internal thermistor."
        ),
        severity="warning",
        category="regulation_follow",
    )


def _send_alert(alert: Any) -> bool:
    """Notify HA and record the cooldown key (as safety.process_safety_alerts does)."""
    from weatherstat.advisory import load_advisory_state, save_advisory_state, send_ha_notification
    from weatherstat.yaml_config import load_config

    ok = send_ha_notification(f"⚠ {alert.title}", alert.message, f"weatherstat_safety_{alert.key}", load_config().notification_target)
    if ok:
        state = load_advisory_state()
        state[f"safety_{alert.key}"] = time.time()
        save_advisory_state(state)
    return ok


def _dismiss_alert(effector: str) -> None:
    from weatherstat.advisory import dismiss_ha_notification, load_advisory_state, save_advisory_state
    from weatherstat.yaml_config import load_config

    key = _alert_key(effector)
    dismiss_ha_notification(f"weatherstat_safety_{key}", load_config().notification_target)
    state = load_advisory_state()
    if state.pop(f"safety_{key}", None) is not None:
        save_advisory_state(state)


def _sent_alerts() -> dict[str, float]:
    """effector -> when its follow alert was last sent (outstanding alerts survive restarts)."""
    from weatherstat.advisory import load_advisory_state

    prefix = f"safety_{FOLLOW_ALERT_PREFIX}"
    return {k.removeprefix(prefix): float(v) for k, v in load_advisory_state().items() if k.startswith(prefix)}


def _follow_cooldown() -> float:
    from weatherstat.yaml_config import load_config

    return float(load_config().safety.cooldowns.get("regulation_follow", FOLLOW_COOLDOWN_SECONDS))


class FollowChecker:
    """Tracks |unit reading − V| per live effector; alerts when sustained, clears on recovery.

    Notifications go out only when the process runs live (as every safety
    alert); in dry run the alert is logged once per episode. An alert sent
    before a restart is remembered through its advisory-state cooldown key,
    so recovery after a restart still dismisses it.
    """

    def __init__(
        self,
        *,
        log: LogFn,
        tolerance: float,
        sustain: timedelta = timedelta(minutes=FOLLOW_MINUTES),
        cooldown_seconds: float = FOLLOW_COOLDOWN_SECONDS,
        send: Callable[[Any], bool] = _send_alert,
        dismiss: Callable[[str], None] = _dismiss_alert,
        sent: dict[str, float] | None = None,
    ) -> None:
        self._log = log
        self.tolerance = tolerance
        self.sustain = sustain
        self.cooldown = cooldown_seconds
        self._send = send
        self._dismiss = dismiss
        self._since: dict[str, datetime] = {}
        self._episode: set[str] = set()  # effectors in a sustained mismatch
        self._sent: dict[str, float] = dict(sent or {})  # effector -> epoch seconds of the last notification

    def check(self, effector: str, value: float | None, reading: float | None, now: datetime, *, live: bool) -> FollowStatus:
        if value is None or reading is None:
            # No input published: the unit is meant to fall back to its own thermistor.
            self._since.pop(effector, None)
            if value is None:
                self._recover(effector, live=live, why="no input published")
            return FollowStatus("no input" if value is None else "no reading")
        diff = round(reading - value, 2)
        if abs(diff) <= self.tolerance:
            self._since.pop(effector, None)
            self._recover(effector, live=live, why=f"reading {reading:.1f} follows V {value:.1f}")
            return FollowStatus("ok", diff)
        since = self._since.setdefault(effector, now)
        if now - since < self.sustain:
            return FollowStatus("mismatch", diff, since)
        minutes = (now - since).total_seconds() / 60
        first = effector not in self._episode
        self._episode.add(effector)
        if first:
            self._log(
                f"[regulation] [bold red]{effector} not following its input[/]: reads {reading:.1f}, "
                f"V {value:.1f} ({diff:+.1f}) for {minutes:.0f} min"
            )
        last = self._sent.get(effector)
        if live and (last is None or now.timestamp() - last >= self.cooldown):
            alert = follow_alert(effector, value, reading, minutes)
            try:
                if self._send(alert):
                    self._sent[effector] = now.timestamp()
                    self._log(f"[regulation] Alert sent: {alert.title}")
            except Exception as e:  # noqa: BLE001 — a failed notification must not stop publishing
                self._log(f"[regulation] [red]Alert failed: {e}[/]")
        return FollowStatus("alert", diff, since)

    def _recover(self, effector: str, *, live: bool, why: str) -> None:
        was_episode = effector in self._episode
        self._episode.discard(effector)
        if was_episode:
            self._log(f"[regulation] [green]{effector} following again[/] ({why})")
        if effector in self._sent and live:
            try:
                self._dismiss(effector)
                self._sent.pop(effector, None)
            except Exception as e:  # noqa: BLE001
                self._log(f"[regulation] [red]Dismissing alert failed: {e}[/]")


# ── The loop ──────────────────────────────────────────────────────────────


def _default_frozen(values: dict[str, float]) -> dict[str, str]:
    from weatherstat.config import UNIT_SYMBOL
    from weatherstat.sensor_health import frozen_sensors_now

    return {col: f"frozen ({fz.describe(UNIT_SYMBOL)})" for col, fz in frozen_sensors_now(live_values=values).items()}


def _describe_change(old: UnitInput, new: UnitInput) -> str:
    parts: list[str] = []
    for s in sorted(set(old.used) - set(new.used)):
        parts.append(f"{s} dropped ({new.dropped.get(s, '?')})")
    for s in sorted(set(new.used) - set(old.used)):
        parts.append(f"{s} back")
    if old.adopted_at != new.adopted_at:
        parts.append("new weight set")
    if (old.value is None) != (new.value is None):
        parts.append("input unavailable" if new.value is None else "input available again")
    return ", ".join(parts) or "membership changed"


class RegulationPublisher:
    """Evaluates V for every regulating effector each minute and publishes it (lock holder only).

    Everything with side effects is injectable for tests: ``fetch_states``,
    ``frozen`` (column -> reason), ``weight_sets``, ``lock``, ``follow``,
    ``clock``/``wait`` (the cadence) and ``now`` (wall time).
    """

    def __init__(
        self,
        *,
        log: LogFn,
        live: Callable[[], bool],
        publish: Callable[[dict[str, UnitInput]], Any] | None = None,
        request_cycle: Callable[[str], None] | None = None,
        fetch_states: Callable[[], dict[str, dict]] = fetch_ha_states,
        frozen: Callable[[dict[str, float]], dict[str, str]] = _default_frozen,
        weight_sets: Callable[[], dict[str, tuple[dict[str, float], str | None]]] = load_weight_sets,
        lock: PublisherLock | None = None,
        follow: FollowChecker | None = None,
        live_effectors: Iterable[str] | None = None,
        sources: dict[str, tuple[str, str | None]] | None = None,
        sensors: Iterable[str] | None = None,
        interval: float = PUBLISH_INTERVAL,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        from weatherstat.config import PREDICTION_SENSORS, delta_temp
        from weatherstat.yaml_config import load_config

        self._log = log
        self._live = live
        self._publish = publish
        self._request_cycle = request_cycle
        self._fetch = fetch_states
        self._frozen = frozen
        self._weight_sets = weight_sets
        self._lock_file = lock or default_lock()
        self.live_effectors = tuple(live_effectors if live_effectors is not None else load_config().regulation_live)
        self._follow = follow
        if self._follow is None and self.live_effectors:
            self._follow = FollowChecker(
                log=log, tolerance=delta_temp(FOLLOW_TOLERANCE_F), cooldown_seconds=_follow_cooldown(), sent=_sent_alerts()
            )
        self._sources = sources
        self._sensors = list(sensors if sensors is not None else PREDICTION_SENSORS)
        self.interval = interval
        self._clock = clock
        self._now = now

        self._state_lock = threading.Lock()
        self._status = RegulationStatus("standby", None, None, None, {}, self.live_effectors)
        self._last: dict[str, UnitInput] = {}  # publisher's previous evaluation (membership baseline)
        self._error: str | None = None
        self._role: str | None = None
        self._frozen_error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ── Wiring ──

    def attach_publisher(self, publish: Callable[[dict[str, UnitInput]], Any] | None) -> None:
        self._publish = publish

    def status(self) -> RegulationStatus:
        with self._state_lock:
            return self._status

    @property
    def is_publisher(self) -> bool:
        return self._lock_file.held

    # ── One tick ──

    def tick(self) -> RegulationStatus:
        """Fetch, evaluate, and (as the lock holder) publish, check follow, request cycles."""
        now = self._now()
        holder = self._lock_file.acquire()
        self._note_role(holder)
        try:
            states = self._fetch()
        except Exception as e:  # noqa: BLE001 — HA down: publish nothing; the entity expires after 180 s
            self._note_error(f"HA fetch failed: {e}")
            with self._state_lock:
                self._status = RegulationStatus(
                    "publisher" if holder else "standby",
                    None if holder else self._lock_file.holder(),
                    self._status.updated,
                    self._error,
                    self._status.units,
                    self.live_effectors,
                )
                return self._status
        self._note_error(None)

        weight_sets = self._weight_sets()
        columns = sorted(set(self._sensors) | {s for w, _ in weight_sets.values() for s in w})
        values, unavailable = member_values(states, columns, self._sources)
        try:
            frozen = self._frozen(values)
            self._frozen_error = None
        except Exception as e:  # noqa: BLE001 — without frozen detection V still beats no V
            if str(e) != self._frozen_error:  # once per distinct failure, not every minute
                self._log(f"[regulation] [red]Frozen-sensor check failed: {e}[/]")
                self._log(traceback.format_exc())
            self._frozen_error = str(e)
            frozen = {}
        for col, why in frozen.items():
            unavailable.setdefault(col, why)
            values.pop(col, None)

        live = bool(self._live())
        units: dict[str, UnitInput] = {}
        for name, (weights, adopted_at) in weight_sets.items():
            u = evaluate_unit(name, weights, adopted_at, values, unavailable)
            reading, mode = unit_reading(states, name)
            follows = None
            if holder and self._follow is not None and name in self.live_effectors:
                follows = self._follow.check(name, u.value, reading, now, live=live)
            units[name] = replace(u, unit_reading=reading, unit_mode=mode, follows=follows)

        if holder:
            self._publish_units(units)
            self._membership(units)

        with self._state_lock:
            self._status = RegulationStatus(
                "publisher" if holder else "standby",
                None if holder else self._lock_file.holder(),
                now,
                None,
                units,
                self.live_effectors,
            )
            return self._status

    def _publish_units(self, units: dict[str, UnitInput]) -> None:
        if self._publish is None:
            return
        try:
            self._publish(units)
        except Exception as e:  # noqa: BLE001
            self._log(f"[regulation] [red]Publish failed: {e}[/]")

    def _membership(self, units: dict[str, UnitInput]) -> None:
        changes = []
        for name, u in units.items():
            old = self._last.get(name)
            if old is not None and (old.membership != u.membership or (old.value is None) != (u.value is None)):
                changes.append(f"{name}: {_describe_change(old, u)}")
        self._last = dict(units)
        if not changes:
            return
        reason = "Regulation input changed — " + "; ".join(changes)
        if self._request_cycle is None:
            self._log(f"[regulation] {reason}")
        else:
            try:
                self._request_cycle(reason)
            except Exception as e:  # noqa: BLE001
                self._log(f"[regulation] [red]Requesting a cycle failed: {e}[/]")

    def _note_role(self, holder: bool) -> None:
        role = "publisher" if holder else "standby"
        if role == self._role:
            return
        self._role = role
        if holder:
            self._log(f"[regulation] Publishing regulation inputs every {self.interval:.0f}s")
        else:
            pid = self._lock_file.holder()
            self._log(f"[regulation] Another process publishes regulation inputs (pid {pid or '?'}); standing by")

    def _note_error(self, error: str | None) -> None:
        if error == self._error:
            return
        if error is not None:
            self._log(f"[regulation] [red]{error}[/]")
        elif self._error is not None:
            self._log("[regulation] [green]HA readings back[/]")
        self._error = error

    # ── Cadence ──

    def run(self, wait: Callable[[float], bool] | None = None) -> None:
        """Tick every ``interval`` seconds on a fixed schedule until ``wait`` returns True.

        A slow tick does not shift the schedule; a tick longer than the interval
        skips the missed slots rather than bursting.
        """
        wait = wait or self._stop.wait
        next_at = self._clock()
        while True:
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001 — never let the publisher thread die
                self._log(f"[regulation] [red]Tick failed: {e}[/]")
                self._log(traceback.format_exc())
            next_at += self.interval
            now = self._clock()
            while next_at <= now:
                next_at += self.interval
            if wait(next_at - now):
                return

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run, daemon=True, name="weatherstat-regulation")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._lock_file.release()


def adopt_if_missing(log: LogFn) -> list[str]:
    """Adopt weights at startup for any regulating effector with no adopted set yet.

    So every published V comes from a recorded weight set (regulation_history.jsonl).
    """
    adopted = regulation.load_adopted()
    if all(name in adopted for name in regulation.regulating_effectors()):
        return []
    return regulation.adopt_weights(_sim_gains(), log=log)


def start_regulation_publisher(
    *,
    log: LogFn,
    live: Callable[[], bool],
    publish: Callable[[dict[str, UnitInput]], Any] | None = None,
    request_cycle: Callable[[str], None] | None = None,
) -> RegulationPublisher | None:
    """Start the 60 s publisher thread; None when no regulating effector is configured."""
    if not regulation.regulating_effectors():
        return None
    pub = RegulationPublisher(log=log, live=live, publish=publish, request_cycle=request_cycle)
    if pub._lock_file.acquire():
        try:
            adopt_if_missing(log)
        except Exception as e:  # noqa: BLE001 — fall back to computed weights
            log(f"[regulation] [red]Weight adoption failed: {e}[/]")
    pub.start()
    return pub


# ── Display helpers (TUI, web, debug) ─────────────────────────────────────


def member_label(col: str) -> str:
    """'living_room_temp' → 'living room' (the constraint's display label when there is one)."""
    from weatherstat.config import SENSOR_LABELS

    return SENSOR_LABELS.get(col, col.removesuffix("_temp")).replace("_", " ")


def effector_title(name: str) -> str:
    """'mini_split_living_room' → 'living room split'."""
    base = name.removeprefix("mini_split_")
    return f"{base.replace('_', ' ')} split" if base != name else name.replace("_", " ")


def weights_text(weights: dict[str, float]) -> str:
    """'kitchen .5 · piano .3 · office .2' (largest first)."""
    items = sorted(weights.items(), key=lambda kv: -kv[1])
    return " · ".join(f"{member_label(s)} {w:.2f}".replace(" 0.", " .") for s, w in items)


def follows_text(f: FollowStatus | None, now: datetime | None = None) -> str | None:
    """'follows V', 'off by +1.4 for 4m', 'NOT FOLLOWING (+1.4 for 12m)', or None (check not enabled)."""
    if f is None:
        return None
    if f.state == "ok":
        return "follows V"
    if f.state in ("no input", "no reading"):
        return f.state
    mins = ""
    if f.since is not None:
        mins = f" for {max(0, int(((now or datetime.now(UTC)) - f.since).total_seconds() // 60))}m"
    diff = f"{f.diff:+.1f}" if f.diff is not None else "?"
    return f"NOT FOLLOWING ({diff}{mins})" if f.state == "alert" else f"off by {diff}{mins}"
