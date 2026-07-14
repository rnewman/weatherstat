"""Tests for retention of command JSON files and decision log rows."""

from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from weatherstat.decision_log import _connect, init_db
from weatherstat.retention import prune_command_files, prune_decision_log

_NOW = datetime(2026, 7, 2, 12, 0, 0, tzinfo=UTC)


def _touch_command(directory: Path, name: str, age_days: float) -> Path:
    p = directory / name
    p.write_text("{}")
    mtime = (_NOW - timedelta(days=age_days)).timestamp()
    os.utime(p, (mtime, mtime))
    return p


def _insert_decision(db: Path, age_days: float) -> str:
    ts = (_NOW - timedelta(days=age_days)).isoformat()
    conn = _connect(db)
    conn.execute(
        "INSERT INTO decisions (timestamp, live, current_temps, predictions) VALUES (?, 0, '{}', '{}')",
        (ts,),
    )
    conn.commit()
    conn.close()
    return ts


def test_prune_command_files_by_age(tmp_path: Path) -> None:
    old = _touch_command(tmp_path, "command_20260601_000000.json", age_days=30)
    fresh = _touch_command(tmp_path, "command_20260701_000000.json", age_days=1)
    newest = _touch_command(tmp_path, "command_20260702_000000.json", age_days=0)

    deleted = prune_command_files(retention_days=14, predictions_dir=tmp_path, now=_NOW)

    assert deleted == 1
    assert not old.exists()
    assert fresh.exists() and newest.exists()


def test_prune_command_files_always_keeps_newest(tmp_path: Path) -> None:
    only = _touch_command(tmp_path, "command_20260101_000000.json", age_days=180)

    deleted = prune_command_files(retention_days=14, predictions_dir=tmp_path, now=_NOW)

    assert deleted == 0
    assert only.exists()


def test_prune_command_files_disabled(tmp_path: Path) -> None:
    old = _touch_command(tmp_path, "command_20260101_000000.json", age_days=180)
    _touch_command(tmp_path, "command_20260702_000000.json", age_days=0)

    assert prune_command_files(retention_days=0, predictions_dir=tmp_path, now=_NOW) == 0
    assert old.exists()


def test_prune_command_files_ignores_other_files(tmp_path: Path) -> None:
    keeper = tmp_path / "notes.json"
    keeper.write_text("{}")
    os.utime(keeper, ((_NOW - timedelta(days=365)).timestamp(),) * 2)
    _touch_command(tmp_path, "command_20260702_000000.json", age_days=0)

    prune_command_files(retention_days=14, predictions_dir=tmp_path, now=_NOW)

    assert keeper.exists()


def test_prune_decision_log_by_age(tmp_path: Path) -> None:
    db = tmp_path / "decision_log.db"
    init_db(db)
    old_ts = _insert_decision(db, age_days=120)
    fresh_ts = _insert_decision(db, age_days=5)

    deleted = prune_decision_log(retention_days=90, db_path=db, now=_NOW)

    assert deleted == 1
    conn = sqlite3.connect(str(db))
    remaining = [r[0] for r in conn.execute("SELECT timestamp FROM decisions").fetchall()]
    conn.close()
    assert remaining == [fresh_ts]
    assert old_ts not in remaining


def test_prune_decision_log_disabled(tmp_path: Path) -> None:
    db = tmp_path / "decision_log.db"
    init_db(db)
    _insert_decision(db, age_days=120)

    assert prune_decision_log(retention_days=0, db_path=db, now=_NOW) == 0


def test_prune_decision_log_missing_db(tmp_path: Path) -> None:
    assert prune_decision_log(retention_days=90, db_path=tmp_path / "nope.db", now=_NOW) == 0


def test_run_retention_if_due_respects_marker(tmp_path: Path) -> None:
    from weatherstat.runtime import run_retention_if_due

    marker = tmp_path / ".retention_last_run"

    first = run_retention_if_due(log=lambda _: None, marker_path=marker)
    assert first is not None
    assert marker.exists()

    # Immediately after, retention is not due
    assert run_retention_if_due(log=lambda _: None, marker_path=marker) is None

    # A day-old marker makes it due again
    old = (_NOW - timedelta(days=2)).timestamp()
    os.utime(marker, (old, old))
    assert run_retention_if_due(log=lambda _: None, marker_path=marker) is not None
