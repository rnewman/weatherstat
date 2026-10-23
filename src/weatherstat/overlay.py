"""Comfort overlays — temporary, per-sensor comfort adjustments.

"The bedroom is too cold" is a request, not a schedule edit. An overlay
captures that request as a signed temperature delta on one sensor with an
expiry, layered on top of the configured schedules and the active comfort
profile:

    base schedules → profile offsets → **overlays** → (MRT applied to predictions)

The delta shifts all three comfort tiers (preferred band, acceptable rails,
backup hedge) uniformly for the hours the overlay is active, so the optimizer
is asked to *move* the room rather than merely tolerate a different band.
Expiry is honoured at hour granularity: an overlay that ends in two hours
shifts the current hour and the next, and the sweep sees the bounds return
to normal after that. Repeated bumps on the same sensor accumulate and
refresh the expiry; a clear removes the overlay.

Every change is recorded. The current overlays live in ``overlays.json``;
the event history (set / bump / clear / expire, with the room and outdoor
temperature at the time) goes to the ``overlay_events`` table in the
decision log, and each control decision stores the overlays that were active
when it was made. The history is the raw material for learning: a room that
is bumped warmer every evening at the same hour is a schedule asking to be
changed. Overlay events are never pruned by retention.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from weatherstat.config import (
    DATA_DIR,
    OVERLAY_DURATION_HOURS,
    OVERLAY_MAX_DELTA,
    OVERLAY_STEP,
    SENSOR_LABELS,
    TIMEZONE,
    UNIT_SYMBOL,
)
from weatherstat.types import ComfortSchedule, ComfortScheduleEntry, RoomComfort

OVERLAY_STATE_FILE = DATA_DIR / "overlays.json"

# ── Data model ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ComfortOverlay:
    """A temporary comfort shift on one sensor.

    delta is in the configured unit: positive = warmer (the room felt cold),
    negative = cooler. Timestamps are UTC.
    """

    sensor: str  # sensor column, e.g. "bedroom_temp"
    delta: float
    created_at: datetime
    expires_at: datetime
    source: str = "cli"  # "cli" | "tui" | "web" | "ha" | ...
    note: str | None = None

    @property
    def label(self) -> str:
        return SENSOR_LABELS.get(self.sensor, self.sensor.removesuffix("_temp"))

    def is_active(self, now: datetime) -> bool:
        return now < self.expires_at and abs(self.delta) >= 0.05

    def remaining(self, now: datetime) -> timedelta:
        return max(self.expires_at - now, timedelta(0))

    def to_dict(self) -> dict:
        return {
            "sensor": self.sensor,
            "delta": round(self.delta, 3),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "source": self.source,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ComfortOverlay:
        return cls(
            sensor=str(data["sensor"]),
            delta=float(data["delta"]),
            created_at=_parse_ts(data["created_at"]),
            expires_at=_parse_ts(data["expires_at"]),
            source=str(data.get("source", "cli")),
            note=data.get("note"),
        )

    def describe(self, now: datetime | None = None) -> str:
        """Human-readable summary: 'bedroom +2.0°F (2h40m left)'."""
        s = f"{self.label} {self.delta:+.1f}{UNIT_SYMBOL}"
        if now is not None:
            s += f" ({format_remaining(self.remaining(now))} left)"
        return s


def _parse_ts(value: str) -> datetime:
    ts = datetime.fromisoformat(value)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def format_remaining(td: timedelta) -> str:
    secs = max(0, int(td.total_seconds()))
    h, rem = divmod(secs, 3600)
    m = rem // 60
    if h and m:
        return f"{h}h{m:02d}m"
    if h:
        return f"{h}h"
    return f"{m}m"


# ── Sensor resolution ──────────────────────────────────────────────────────


def resolve_sensor(name: str) -> str:
    """Map a user-typed room name to a constraint sensor column.

    Accepts the sensor column ("bedroom_temp"), the constraint label
    ("bedroom"), or a case/space/hyphen-insensitive variant ("Living Room").
    Raises ValueError when nothing matches.
    """
    key = _normalize(name)
    if not key:
        raise ValueError("empty sensor name")
    candidates: dict[str, str] = {}
    for sensor_col, label in SENSOR_LABELS.items():
        candidates[_normalize(sensor_col)] = sensor_col
        candidates[_normalize(label)] = sensor_col
    if key in candidates:
        return candidates[key]
    # Unique prefix match on labels ("bed" → bedroom)
    matches = {col for k, col in candidates.items() if k.startswith(key)}
    if len(matches) == 1:
        return matches.pop()
    known = ", ".join(sorted(set(SENSOR_LABELS.values())))
    raise ValueError(f"unknown sensor '{name}' (known: {known})")


def _normalize(s: str) -> str:
    return re.sub(r"[\s\-]+", "_", s.strip().lower())


# ── Command parsing (shared by TUI prompt and CLI) ──────────────────────────


@dataclass(frozen=True)
class OverlayCommand:
    """Parsed user command: a bump, an absolute set, or a clear."""

    sensor: str
    action: str  # "bump" | "set" | "clear"
    delta: float = 0.0
    duration: timedelta | None = None  # None → default


_DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?)(h|hr|hrs|hours?|m|min|mins|minutes?)$", re.IGNORECASE)


def parse_duration(token: str) -> timedelta:
    """'3h', '90m', '1.5h' → timedelta. Raises ValueError otherwise."""
    m = _DURATION_RE.match(token.strip())
    if not m:
        raise ValueError(f"bad duration '{token}' (use e.g. 3h or 90m)")
    value = float(m.group(1))
    unit = m.group(2).lower()
    if unit.startswith("h"):
        return timedelta(hours=value)
    return timedelta(minutes=value)


def parse_overlay_command(text: str) -> OverlayCommand:
    """Parse 'bedroom +2', 'bedroom -1 90m', 'bedroom =+3 4h', 'bedroom clear'.

    - ``+N`` / ``-N`` bumps by N (accumulates with an existing overlay)
    - ``=+N`` / ``=-N`` sets the overlay to exactly N
    - ``clear`` / ``off`` / ``0`` removes the overlay
    - an optional trailing duration (``3h``, ``90m``) overrides the default
    Room names may contain spaces ("living room +1").
    """
    tokens = text.strip().split()
    if not tokens:
        raise ValueError("empty command (try 'bedroom +2')")

    # Find the first token that parses as an action; everything before is the room name.
    for i, tok in enumerate(tokens):
        if i == 0:
            continue
        action = _parse_action(tok)
        if action is not None:
            break
    else:
        raise ValueError(f"no action in '{text}' (try 'bedroom +2' or 'bedroom clear')")

    sensor = resolve_sensor(" ".join(tokens[:i]))
    kind, delta = action
    rest = tokens[i + 1 :]
    duration: timedelta | None = None
    if rest:
        if len(rest) > 1:
            raise ValueError(f"unexpected trailing input: {' '.join(rest[1:])}")
        duration = parse_duration(rest[0])
    if kind == "clear":
        return OverlayCommand(sensor=sensor, action="clear")
    return OverlayCommand(sensor=sensor, action=kind, delta=delta, duration=duration)


def _parse_action(tok: str) -> tuple[str, float] | None:
    low = tok.lower()
    if low in ("clear", "off", "reset", "none"):
        return ("clear", 0.0)
    kind = "bump"
    body = tok
    if body.startswith("="):
        kind = "set"
        body = body[1:]
    if not body or body[0] not in "+-":
        # Bare numbers are ambiguous (absolute temp or delta?) — require a sign.
        return None
    try:
        value = float(body)
    except ValueError:
        return None
    if kind == "bump" and value == 0.0:
        return ("clear", 0.0)
    return (kind, value)


# ── Store ─────────────────────────────────────────────────────────────────


class OverlayStore:
    """Current overlays, one per sensor, persisted to ``overlays.json``.

    Mutations log an event to the decision log (``overlay_events``) and save.
    """

    def __init__(self, overlays: dict[str, ComfortOverlay] | None = None, *, path: Path | None = None) -> None:
        self._overlays: dict[str, ComfortOverlay] = dict(overlays or {})
        self._path = path or OVERLAY_STATE_FILE

    # ── Persistence ──

    @classmethod
    def load(cls, path: Path | None = None) -> OverlayStore:
        p = path or OVERLAY_STATE_FILE
        if not p.exists():
            return cls(path=p)
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return cls(path=p)
        overlays: dict[str, ComfortOverlay] = {}
        for item in data.get("overlays", []):
            with contextlib.suppress(KeyError, ValueError, TypeError):
                o = ComfortOverlay.from_dict(item)
                overlays[o.sensor] = o
        return cls(overlays, path=p)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"overlays": [o.to_dict() for o in sorted(self._overlays.values(), key=lambda o: o.sensor)]}
        self._path.write_text(json.dumps(payload, indent=2))

    # ── Queries ──

    def get(self, sensor: str) -> ComfortOverlay | None:
        return self._overlays.get(sensor)

    def all(self) -> list[ComfortOverlay]:
        return sorted(self._overlays.values(), key=lambda o: o.sensor)

    def active(self, now: datetime | None = None) -> list[ComfortOverlay]:
        now = now or datetime.now(UTC)
        return [o for o in self.all() if o.is_active(now)]

    def active_deltas(self, now: datetime | None = None) -> dict[str, float]:
        """sensor column → delta, for active overlays only."""
        return {o.sensor: o.delta for o in self.active(now)}

    # ── Mutations ──

    def prune_expired(self, now: datetime | None = None, *, db_path: Path | None = None) -> list[ComfortOverlay]:
        """Drop overlays past their expiry, logging an 'expire' event for each."""
        now = now or datetime.now(UTC)
        expired = [o for o in self._overlays.values() if not o.is_active(now)]
        if not expired:
            return []
        for o in expired:
            del self._overlays[o.sensor]
            _log_event("expire", o, delta_change=-o.delta, now=now, context=None, db_path=db_path)
        self.save()
        return expired

    def bump(
        self,
        sensor: str,
        delta: float,
        *,
        duration: timedelta | None = None,
        source: str = "cli",
        note: str | None = None,
        now: datetime | None = None,
        context: dict | None = None,
        db_path: Path | None = None,
    ) -> ComfortOverlay:
        """Add ``delta`` to the sensor's overlay (creating it) and refresh the expiry."""
        now = now or datetime.now(UTC)
        existing = self._overlays.get(sensor)
        base = existing.delta if existing is not None and existing.is_active(now) else 0.0
        return self._write(
            sensor,
            base + delta,
            "bump",
            delta_change=delta,
            duration=duration,
            source=source,
            note=note,
            now=now,
            context=context,
            db_path=db_path,
        )

    def set(
        self,
        sensor: str,
        delta: float,
        *,
        duration: timedelta | None = None,
        source: str = "cli",
        note: str | None = None,
        now: datetime | None = None,
        context: dict | None = None,
        db_path: Path | None = None,
    ) -> ComfortOverlay:
        """Set the sensor's overlay to exactly ``delta``."""
        now = now or datetime.now(UTC)
        existing = self._overlays.get(sensor)
        base = existing.delta if existing is not None and existing.is_active(now) else 0.0
        return self._write(
            sensor,
            delta,
            "set",
            delta_change=delta - base,
            duration=duration,
            source=source,
            note=note,
            now=now,
            context=context,
            db_path=db_path,
        )

    def clear(
        self,
        sensor: str,
        *,
        source: str = "cli",
        now: datetime | None = None,
        context: dict | None = None,
        db_path: Path | None = None,
    ) -> ComfortOverlay | None:
        """Remove the sensor's overlay. Returns the removed overlay, or None if there was none."""
        now = now or datetime.now(UTC)
        existing = self._overlays.pop(sensor, None)
        if existing is None:
            return None
        _log_event(
            "clear",
            replace(existing, source=source),
            delta_change=-existing.delta,
            now=now,
            context=context,
            db_path=db_path,
        )
        self.save()
        return existing

    def clear_all(self, *, source: str = "cli", now: datetime | None = None, db_path: Path | None = None) -> int:
        n = 0
        for sensor in list(self._overlays):
            if self.clear(sensor, source=source, now=now, db_path=db_path) is not None:
                n += 1
        return n

    def apply_command(
        self,
        cmd: OverlayCommand,
        *,
        source: str,
        note: str | None = None,
        now: datetime | None = None,
        context: dict | None = None,
        db_path: Path | None = None,
    ) -> ComfortOverlay | None:
        """Execute a parsed command. Returns the resulting overlay, or None after a clear."""
        if cmd.action == "clear":
            return self.clear(cmd.sensor, source=source, now=now, context=context, db_path=db_path)
        fn = self.set if cmd.action == "set" else self.bump
        return fn(
            cmd.sensor,
            cmd.delta,
            duration=cmd.duration,
            source=source,
            note=note,
            now=now,
            context=context,
            db_path=db_path,
        )

    def _write(
        self,
        sensor: str,
        delta: float,
        event: str,
        *,
        delta_change: float,
        duration: timedelta | None,
        source: str,
        note: str | None,
        now: datetime,
        context: dict | None,
        db_path: Path | None,
    ) -> ComfortOverlay:
        if sensor not in SENSOR_LABELS:
            raise ValueError(f"'{sensor}' is not a constrained sensor")
        delta = max(-OVERLAY_MAX_DELTA, min(OVERLAY_MAX_DELTA, delta))
        duration = duration or timedelta(hours=OVERLAY_DURATION_HOURS)
        if duration <= timedelta(0):
            raise ValueError("overlay duration must be positive")
        if abs(delta) < 0.05:
            # Bumping back to zero is a clear.
            removed = self.clear(sensor, source=source, now=now, context=context, db_path=db_path)
            return removed if removed is not None else ComfortOverlay(sensor, 0.0, now, now, source, note)
        overlay = ComfortOverlay(
            sensor=sensor,
            delta=delta,
            created_at=now,
            expires_at=now + duration,
            source=source,
            note=note,
        )
        self._overlays[sensor] = overlay
        _log_event(event, overlay, delta_change=delta_change, now=now, context=context, db_path=db_path)
        self.save()
        return overlay


def _log_event(
    event: str,
    overlay: ComfortOverlay,
    *,
    delta_change: float,
    now: datetime,
    context: dict | None,
    db_path: Path | None,
) -> None:
    from weatherstat.decision_log import log_overlay_event

    log_overlay_event(
        timestamp=now,
        event=event,
        sensor=overlay.sensor,
        label=overlay.label,
        delta=0.0 if event in ("clear", "expire") else overlay.delta,
        delta_change=delta_change,
        expires_at=overlay.expires_at if event in ("bump", "set") else None,
        source=overlay.source,
        note=overlay.note,
        context=context,
        db_path=db_path,
    )


# ── Context capture (what the house looked like when the user asked) ───────


def capture_context(sensor: str, extra: dict | None = None) -> dict:
    """Room temp, outdoor temp and local hour at the moment of the request.

    Read from the latest collector snapshot; best-effort (missing values are
    simply omitted). The decision log carries the full picture at 5-minute
    resolution, so this only needs enough to make an event self-describing.
    """
    ctx: dict = {}
    try:
        from weatherstat.extract import latest_snapshot_values
        from weatherstat.yaml_config import load_config

        values = latest_snapshot_values()
        room = values.get(sensor)
        if room not in (None, ""):
            with contextlib.suppress(TypeError, ValueError):
                ctx["room_temp"] = round(float(room), 2)
        outdoor_col = load_config().outdoor_sensor
        for col in (outdoor_col, "met_outdoor_temp"):
            v = values.get(col) if col else None
            if v not in (None, ""):
                with contextlib.suppress(TypeError, ValueError):
                    ctx["outdoor_temp"] = round(float(v), 2)
                    break
    except Exception as e:  # noqa: BLE001 — context is best-effort
        ctx["context_error"] = str(e)
    local = datetime.now(ZoneInfo(TIMEZONE)) if TIMEZONE else datetime.now().astimezone()
    ctx["local_hour"] = local.hour
    ctx["weekday"] = local.weekday()
    if extra:
        ctx.update(extra)
    return ctx


# ── Applying overlays to schedules ────────────────────────────────────────


def apply_overlays(
    schedules: list[ComfortSchedule],
    overlays: list[ComfortOverlay],
    now: datetime | None = None,
    tz: ZoneInfo | None = None,
) -> list[ComfortSchedule]:
    """Shift each overlaid sensor's schedule by its delta for the hours it is active.

    Hours are local clock hours (the schedule's own granularity). An overlay
    expiring within the day shifts only the hours from now until expiry, so
    horizon scoring sees the bounds revert; one lasting a day or more shifts
    every hour.
    """
    now = now or datetime.now(UTC)
    active = {o.sensor: o for o in overlays if o.is_active(now)}
    if not active:
        return schedules
    tz = tz or (ZoneInfo(TIMEZONE) if TIMEZONE else UTC)
    out: list[ComfortSchedule] = []
    for schedule in schedules:
        overlay = active.get(schedule.sensor)
        if overlay is None:
            out.append(schedule)
            continue
        hours = active_hours(now, overlay.expires_at, tz)
        out.append(shift_schedule_hours(schedule, overlay.delta, hours))
    return out


def active_hours(now: datetime, expires_at: datetime, tz: ZoneInfo) -> frozenset[int]:
    """Local clock hours touched by the interval [now, expires_at), capped at 24."""
    t = now.astimezone(tz).replace(minute=0, second=0, microsecond=0)
    end = expires_at.astimezone(tz)
    hours: set[int] = set()
    while t < end and len(hours) < 24:
        hours.add(t.hour)
        t += timedelta(hours=1)
    return frozenset(hours)


def shift_comfort(c: RoomComfort, delta: float) -> RoomComfort:
    """Shift all three tiers of a RoomComfort uniformly."""
    return replace(
        c,
        preferred_lo=c.preferred_lo + delta,
        preferred_hi=c.preferred_hi + delta,
        acceptable_lo=c.acceptable_lo + delta,
        acceptable_hi=c.acceptable_hi + delta,
        backup_lo=c.backup_lo + delta,
        backup_hi=c.backup_hi + delta,
    )


def shift_schedule_hours(schedule: ComfortSchedule, delta: float, hours: frozenset[int]) -> ComfortSchedule:
    """Rebuild a schedule with ``delta`` applied during ``hours`` only.

    Expands to a 24-slot hourly table, shifts the selected slots, and
    coalesces runs of identical comfort back into entries (merging across
    midnight when hours 23 and 0 match). ``comfort_at(h)`` is preserved for
    every unshifted hour.
    """
    hourly: list[RoomComfort | None] = []
    for h in range(24):
        c = schedule.comfort_at(h)
        if c is not None and h in hours:
            c = shift_comfort(c, delta)
        hourly.append(c)
    return ComfortSchedule(sensor=schedule.sensor, label=schedule.label, entries=_coalesce(hourly))


def _coalesce(hourly: list[RoomComfort | None]) -> tuple[ComfortScheduleEntry, ...]:
    runs: list[tuple[int, int, RoomComfort]] = []  # (start, end_exclusive, comfort)
    for h, c in enumerate(hourly):
        if c is None:
            continue
        if runs and runs[-1][1] == h and runs[-1][2] == c:
            start, _, _ = runs[-1]
            runs[-1] = (start, h + 1, c)
        else:
            runs.append((h, h + 1, c))
    if not runs:
        return ()
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == 24 and runs[0][2] == runs[-1][2]:
        # Merge across midnight into a wrapping entry: start > end means wrap in comfort_at.
        first, last = runs[0], runs[-1]
        runs = [(last[0], first[1], first[2]), *runs[1:-1]]
    return tuple(ComfortScheduleEntry(start, end, c) for start, end, c in runs)


# ── Display helpers ────────────────────────────────────────────────────────


def overlay_summary(overlays: list[ComfortOverlay], now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    return ", ".join(o.describe(now) for o in overlays)


def active_overlay_labels(overlays: list[ComfortOverlay], now: datetime | None = None) -> dict[str, float]:
    """label → delta for active overlays (the decision-log serialization)."""
    now = now or datetime.now(UTC)
    return {o.label: round(o.delta, 3) for o in overlays if o.is_active(now)}


# ── CLI ───────────────────────────────────────────────────────────────────


def _print_active(store: OverlayStore, now: datetime) -> None:
    active = store.active(now)
    if not active:
        print("No active overlays.")
        return
    print("Active overlays:")
    for o in active:
        note = f"  — {o.note}" if o.note else ""
        print(f"  {o.describe(now)}  [{o.source}]{note}")


def _print_history(limit: int) -> None:
    from weatherstat.decision_log import load_overlay_events

    rows = load_overlay_events(limit=limit)
    if not rows:
        print("No overlay events recorded.")
        return
    print(f"Last {len(rows)} overlay events (newest first):")
    for r in rows:
        ctx = r.get("context") or {}
        room = f" room {ctx['room_temp']:.1f}" if "room_temp" in ctx else ""
        out = f" out {ctx['outdoor_temp']:.0f}" if "outdoor_temp" in ctx else ""
        delta = f"{r['delta']:+.1f}" if r["event"] in ("bump", "set") else "—"
        change = f" ({r['delta_change']:+.1f})" if r["event"] == "bump" else ""
        print(f"  {r['timestamp'][:16]}  {r['event']:<6} {r['label']:<14} {delta:>5}{change}  [{r['source']}]{room}{out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="weatherstat.overlay",
        description="Temporary comfort overlays: 'bedroom +2' means bedroom felt cold, make it 2° warmer for a while.",
    )
    parser.add_argument(
        "command",
        nargs="*",
        help="e.g. 'bedroom +2', 'living room -1 90m', 'bedroom =+3', 'bedroom clear' (omit to list active)",
    )
    parser.add_argument("--hours", type=float, default=None, help=f"duration (default {OVERLAY_DURATION_HOURS:g}h)")
    parser.add_argument("--note", default=None, help="free-text reason, kept with the event")
    parser.add_argument("--source", default="cli", help="who/what issued this (default cli)")
    parser.add_argument("--clear-all", action="store_true", help="remove every overlay")
    parser.add_argument("--history", type=int, nargs="?", const=20, default=None, metavar="N", help="show last N events")
    args = parser.parse_args(argv)

    now = datetime.now(UTC)
    store = OverlayStore.load()
    store.prune_expired(now)

    if args.history is not None:
        _print_history(args.history)
        return
    if args.clear_all:
        n = store.clear_all(source=args.source, now=now)
        print(f"Cleared {n} overlay(s).")
        return
    if not args.command:
        _print_active(store, now)
        print(f"\nStep {OVERLAY_STEP:g}{UNIT_SYMBOL}, max ±{OVERLAY_MAX_DELTA:g}{UNIT_SYMBOL}, default {OVERLAY_DURATION_HOURS:g}h.")
        return

    cmd = parse_overlay_command(" ".join(args.command))
    if args.hours is not None:
        cmd = replace(cmd, duration=timedelta(hours=args.hours))
    context = capture_context(cmd.sensor)
    if cmd.action == "clear":
        removed = store.clear(cmd.sensor, source=args.source, now=now, context=context)
        label = SENSOR_LABELS.get(cmd.sensor, cmd.sensor)
        print(f"Cleared {label}." if removed else f"No overlay on {label}.")
    else:
        overlay = store.apply_command(cmd, source=args.source, note=args.note, now=now, context=context)
        if overlay is not None and overlay.is_active(now):
            print(f"Overlay: {overlay.describe(now)}")
        else:
            print("Overlay cleared (net zero).")
    print()
    _print_active(store, now)


if __name__ == "__main__":
    main()
