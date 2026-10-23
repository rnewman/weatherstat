"""Tests for comfort overlays — temporary per-sensor comfort shifts.

Runs against the conftest sandbox (example YAML), with the overlay state file
and event log pointed at a per-test temp directory.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from weatherstat.control import default_comfort_schedules
from weatherstat.decision_log import load_overlay_events
from weatherstat.overlay import (
    ComfortOverlay,
    OverlayStore,
    active_hours,
    active_overlay_labels,
    apply_overlays,
    parse_duration,
    parse_overlay_command,
    resolve_sensor,
    shift_schedule_hours,
)

NOW = datetime(2026, 10, 22, 21, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path) -> OverlayStore:
    return OverlayStore(path=tmp_path / "overlays.json")


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "decision_log.db"


def _bedroom(schedules):
    return next(s for s in schedules if s.sensor == "bedroom_temp")


# ── Parsing ────────────────────────────────────────────────────────────────


class TestResolveSensor:
    def test_label_and_column(self) -> None:
        assert resolve_sensor("bedroom") == "bedroom_temp"
        assert resolve_sensor("bedroom_temp") == "bedroom_temp"

    def test_case_and_spaces(self) -> None:
        assert resolve_sensor("Family Room") == "family_room_temp"
        assert resolve_sensor("family-room") == "family_room_temp"

    def test_thermostat_label(self) -> None:
        assert resolve_sensor("upstairs") == "thermostat_upstairs_temp"

    def test_unique_prefix(self) -> None:
        assert resolve_sensor("bed") == "bedroom_temp"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown sensor"):
            resolve_sensor("attic")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_sensor("   ")


class TestParseCommand:
    def test_bump_default_duration(self) -> None:
        cmd = parse_overlay_command("bedroom +2")
        assert (cmd.sensor, cmd.action, cmd.delta, cmd.duration) == ("bedroom_temp", "bump", 2.0, None)

    def test_negative_with_minutes(self) -> None:
        cmd = parse_overlay_command("family room -1 90m")
        assert cmd.sensor == "family_room_temp"
        assert cmd.action == "bump"
        assert cmd.delta == -1.0
        assert cmd.duration == timedelta(minutes=90)

    def test_set_with_hours(self) -> None:
        cmd = parse_overlay_command("bedroom =+3 4h")
        assert cmd.action == "set"
        assert cmd.delta == 3.0
        assert cmd.duration == timedelta(hours=4)

    def test_clear_words(self) -> None:
        for word in ("clear", "off", "reset", "+0"):
            assert parse_overlay_command(f"bedroom {word}").action == "clear"

    def test_unsigned_number_is_rejected(self) -> None:
        # "bedroom 72" is ambiguous (absolute target? delta?) — require a sign.
        with pytest.raises(ValueError, match="no action"):
            parse_overlay_command("bedroom 72")

    def test_unknown_room(self) -> None:
        with pytest.raises(ValueError, match="unknown sensor"):
            parse_overlay_command("attic +1")

    def test_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_overlay_command("")

    def test_trailing_garbage(self) -> None:
        with pytest.raises(ValueError, match="trailing"):
            parse_overlay_command("bedroom +1 2h please")

    def test_duration_forms(self) -> None:
        assert parse_duration("3h") == timedelta(hours=3)
        assert parse_duration("1.5hr") == timedelta(hours=1.5)
        assert parse_duration("45min") == timedelta(minutes=45)
        with pytest.raises(ValueError):
            parse_duration("soon")


# ── Store ──────────────────────────────────────────────────────────────────


class TestStore:
    def test_bump_creates_and_accumulates(self, store: OverlayStore, db: Path) -> None:
        o1 = store.bump("bedroom_temp", 1.0, now=NOW, db_path=db)
        assert o1.delta == 1.0
        assert o1.expires_at == NOW + timedelta(hours=3)  # default duration
        o2 = store.bump("bedroom_temp", 1.0, now=NOW + timedelta(minutes=10), db_path=db)
        assert o2.delta == 2.0
        assert o2.expires_at == NOW + timedelta(minutes=10, hours=3)  # expiry refreshed
        assert store.active_deltas(NOW + timedelta(minutes=10)) == {"bedroom_temp": 2.0}

    def test_set_replaces(self, store: OverlayStore, db: Path) -> None:
        store.bump("bedroom_temp", 1.0, now=NOW, db_path=db)
        o = store.set("bedroom_temp", 3.0, duration=timedelta(hours=1), now=NOW, db_path=db)
        assert o.delta == 3.0
        assert o.expires_at == NOW + timedelta(hours=1)

    def test_clamped_to_max_delta(self, store: OverlayStore, db: Path) -> None:
        from weatherstat.config import OVERLAY_MAX_DELTA

        o = store.bump("bedroom_temp", 50.0, now=NOW, db_path=db)
        assert o.delta == OVERLAY_MAX_DELTA
        o = store.set("bedroom_temp", -50.0, now=NOW, db_path=db)
        assert o.delta == -OVERLAY_MAX_DELTA

    def test_bump_back_to_zero_clears(self, store: OverlayStore, db: Path) -> None:
        store.bump("bedroom_temp", 1.0, now=NOW, db_path=db)
        store.bump("bedroom_temp", -1.0, now=NOW, db_path=db)
        assert store.get("bedroom_temp") is None
        assert store.active(NOW) == []

    def test_clear_absent_returns_none(self, store: OverlayStore, db: Path) -> None:
        assert store.clear("bedroom_temp", now=NOW, db_path=db) is None

    def test_unknown_sensor_rejected(self, store: OverlayStore, db: Path) -> None:
        with pytest.raises(ValueError, match="not a constrained sensor"):
            store.bump("attic_temp", 1.0, now=NOW, db_path=db)

    def test_persistence_roundtrip(self, store: OverlayStore, db: Path) -> None:
        store.bump("bedroom_temp", 2.0, source="tui", note="too cold", now=NOW, db_path=db)
        store.bump("office_temp", -1.0, now=NOW, db_path=db)
        reloaded = OverlayStore.load(store._path)
        assert reloaded.active_deltas(NOW) == {"bedroom_temp": 2.0, "office_temp": -1.0}
        o = reloaded.get("bedroom_temp")
        assert o is not None
        assert (o.source, o.note, o.expires_at) == ("tui", "too cold", NOW + timedelta(hours=3))

    def test_corrupt_file_loads_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "overlays.json"
        p.write_text("{not json")
        assert OverlayStore.load(p).all() == []

    def test_prune_expired(self, store: OverlayStore, db: Path) -> None:
        store.bump("bedroom_temp", 2.0, duration=timedelta(hours=1), now=NOW, db_path=db)
        store.bump("office_temp", 1.0, duration=timedelta(hours=5), now=NOW, db_path=db)
        later = NOW + timedelta(hours=2)
        assert store.active_deltas(later) == {"office_temp": 1.0}  # inactive before prune too
        expired = store.prune_expired(later, db_path=db)
        assert [o.sensor for o in expired] == ["bedroom_temp"]
        assert OverlayStore.load(store._path).all()[0].sensor == "office_temp"
        events = load_overlay_events(db_path=db)
        assert events[0]["event"] == "expire"
        assert events[0]["sensor"] == "bedroom_temp"
        assert events[0]["delta_change"] == -2.0

    def test_events_are_logged_with_context(self, store: OverlayStore, db: Path) -> None:
        store.bump("bedroom_temp", 2.0, source="web", now=NOW, context={"room_temp": 67.2, "local_hour": 21}, db_path=db)
        store.bump("bedroom_temp", -0.5, source="web", now=NOW + timedelta(minutes=5), db_path=db)
        store.clear("bedroom_temp", source="cli", now=NOW + timedelta(minutes=10), db_path=db)
        events = load_overlay_events(db_path=db)
        assert [e["event"] for e in events] == ["clear", "bump", "bump"]  # newest first
        first = events[-1]
        assert (first["label"], first["delta"], first["delta_change"], first["source"]) == ("bedroom", 2.0, 2.0, "web")
        assert first["context"] == {"room_temp": 67.2, "local_hour": 21}
        assert first["expires_at"] is not None
        second = events[1]
        assert (second["delta"], second["delta_change"]) == (1.5, -0.5)
        cleared = events[0]
        assert (cleared["delta"], cleared["delta_change"], cleared["source"]) == (0.0, -1.5, "cli")
        assert load_overlay_events(sensor="office_temp", db_path=db) == []
        assert len(load_overlay_events(since=NOW + timedelta(minutes=7), db_path=db)) == 1

    def test_apply_command(self, store: OverlayStore, db: Path) -> None:
        cmd = parse_overlay_command("bedroom +2 1h")
        o = store.apply_command(cmd, source="tui", note="chilly", now=NOW, db_path=db)
        assert o is not None and o.delta == 2.0 and o.note == "chilly"
        assert o.expires_at == NOW + timedelta(hours=1)
        assert store.apply_command(parse_overlay_command("bedroom clear"), source="tui", now=NOW, db_path=db) is not None
        assert store.active(NOW) == []


# ── Applying to schedules ──────────────────────────────────────────────────


class TestActiveHours:
    def test_two_hours(self) -> None:
        assert active_hours(NOW, NOW + timedelta(hours=2), UTC) == frozenset({21, 22})

    def test_partial_hour_rounds_to_touched_hours(self) -> None:
        start = NOW.replace(minute=50)
        assert active_hours(start, start + timedelta(minutes=20), UTC) == frozenset({21, 22})

    def test_wraps_midnight(self) -> None:
        assert active_hours(NOW, NOW + timedelta(hours=5), UTC) == frozenset({21, 22, 23, 0, 1})

    def test_day_or_more_covers_all(self) -> None:
        assert active_hours(NOW, NOW + timedelta(hours=30), UTC) == frozenset(range(24))

    def test_local_timezone(self) -> None:
        tz = ZoneInfo("America/Los_Angeles")
        # 21:00 UTC on Oct 22 is 14:00 PDT
        assert active_hours(NOW, NOW + timedelta(hours=1), tz) == frozenset({14})


class TestShiftSchedule:
    def test_zero_shift_preserves_every_hour(self) -> None:
        # Round-trip through the hourly table must be lossless, including the
        # bedroom schedule's midnight-wrapping [21, 6) entry.
        for sched in default_comfort_schedules():
            rebuilt = shift_schedule_hours(sched, 0.0, frozenset(range(24)))
            for h in range(24):
                assert rebuilt.comfort_at(h) == sched.comfort_at(h), (sched.sensor, h)

    def test_shift_all_tiers_only_in_active_hours(self) -> None:
        sched = _bedroom(default_comfort_schedules())
        shifted = shift_schedule_hours(sched, 2.0, frozenset({21, 22}))
        base = sched.comfort_at(21)
        got = shifted.comfort_at(21)
        assert base is not None and got is not None
        assert got.preferred_lo == base.preferred_lo + 2.0
        assert got.preferred_hi == base.preferred_hi + 2.0
        assert got.acceptable_lo == base.acceptable_lo + 2.0
        assert got.acceptable_hi == base.acceptable_hi + 2.0
        assert got.backup_lo == base.backup_lo + 2.0
        assert got.backup_hi == base.backup_hi + 2.0
        assert got.cold_penalty == base.cold_penalty
        assert got.hot_penalty == base.hot_penalty
        # Hour 23 is in the same base entry [21, 6) but not in the overlay window.
        assert shifted.comfort_at(23) == sched.comfort_at(23)
        assert shifted.comfort_at(3) == sched.comfort_at(3)

    def test_entries_coalesced(self) -> None:
        sched = _bedroom(default_comfort_schedules())
        shifted = shift_schedule_hours(sched, 2.0, frozenset({21, 22}))
        # Base has 3 entries; the shift splits [21, 6) into [21,23) shifted + [23,6) unshifted.
        assert len(shifted.entries) == 4
        wrapping = [e for e in shifted.entries if e.start_hour > e.end_hour]
        assert len(wrapping) == 1
        assert (wrapping[0].start_hour, wrapping[0].end_hour) == (23, 6)


class TestApplyOverlays:
    def test_no_active_returns_same_list(self) -> None:
        schedules = default_comfort_schedules()
        expired = ComfortOverlay("bedroom_temp", 2.0, NOW - timedelta(hours=4), NOW - timedelta(hours=1))
        assert apply_overlays(schedules, [expired], NOW, UTC) is schedules

    def test_only_overlaid_sensor_changes(self) -> None:
        schedules = default_comfort_schedules()
        o = ComfortOverlay("bedroom_temp", 2.0, NOW, NOW + timedelta(hours=2))
        out = apply_overlays(schedules, [o], NOW, UTC)
        assert len(out) == len(schedules)
        for before, after in zip(schedules, out, strict=True):
            if before.sensor == "bedroom_temp":
                b, a = before.comfort_at(21), after.comfort_at(21)
                assert a is not None and b is not None and a.preferred_lo == b.preferred_lo + 2.0
                a3, b3 = after.comfort_at(23), before.comfort_at(23)
                assert a3 == b3  # beyond expiry: horizon scoring sees bounds revert
            else:
                assert after is before

    def test_negative_overlay(self) -> None:
        schedules = default_comfort_schedules()
        o = ComfortOverlay("office_temp", -1.5, NOW, NOW + timedelta(hours=1))
        out = apply_overlays(schedules, [o], NOW, UTC)
        office = next(s for s in out if s.sensor == "office_temp")
        base = next(s for s in schedules if s.sensor == "office_temp")
        a, b = office.comfort_at(21), base.comfort_at(21)
        assert a is not None and b is not None
        assert a.acceptable_hi == b.acceptable_hi - 1.5

    def test_labels_for_decision_log(self) -> None:
        o = ComfortOverlay("bedroom_temp", 2.0, NOW, NOW + timedelta(hours=2), source="web")
        stale = ComfortOverlay("office_temp", 1.0, NOW - timedelta(hours=5), NOW - timedelta(hours=2))
        assert active_overlay_labels([o, stale], NOW) == {"bedroom": 2.0}
        assert o.describe(NOW) == "bedroom +2.0°F (2h left)"
