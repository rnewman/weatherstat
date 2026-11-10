"""Collector schema: the (name, timestamp) index and its in-place migration."""

from __future__ import annotations

import sqlite3

from weatherstat.collector import _CREATE_READINGS_SQL, _NAME_INDEX, _apply_schema

_SPARSE_LOOKUP = "SELECT value FROM readings WHERE name = ? AND timestamp < ? ORDER BY timestamp DESC LIMIT 1"


def _has_index(con: sqlite3.Connection) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?", (_NAME_INDEX,)).fetchone() is not None


def _old_database(tmp_path) -> sqlite3.Connection:
    """A database from before the index: table only, with history."""
    con = sqlite3.connect(tmp_path / "snapshots.db")
    con.execute(_CREATE_READINGS_SQL)
    rows = [(f"2026-03-01T{h:02d}:00:00.000Z", "office_temp", str(68 + h % 3)) for h in range(24)]
    rows += [("2026-03-01T03:00:00.000Z", "shade_lowered", "1"), ("2026-03-01T20:00:00.000Z", "shade_lowered", "0")]
    con.executemany("INSERT INTO readings VALUES (?, ?, ?)", rows)
    con.commit()
    return con


class TestSchema:
    def test_fresh_database_gets_index_quietly(self, tmp_path) -> None:
        con = sqlite3.connect(tmp_path / "snapshots.db")
        logs: list[str] = []
        _apply_schema(con, log=logs.append)
        assert _has_index(con)
        assert logs == []  # nothing to migrate

    def test_existing_database_is_migrated_once(self, tmp_path) -> None:
        con = _old_database(tmp_path)
        before = [con.execute(_SPARSE_LOOKUP, ("shade_lowered", ts)).fetchone() for ts in ("2026-03-01T10", "2026-03-02")]
        assert not _has_index(con)

        logs: list[str] = []
        _apply_schema(con, log=logs.append)
        assert _has_index(con)
        assert any("Migration" in line and "26 readings" in line for line in logs)

        after = [con.execute(_SPARSE_LOOKUP, ("shade_lowered", ts)).fetchone() for ts in ("2026-03-01T10", "2026-03-02")]
        assert after == before == [("1",), ("0",)]

        logs.clear()
        _apply_schema(con, log=logs.append)  # next collector start: no-op
        assert logs == []

    def test_sparse_lookup_uses_name_index(self, tmp_path) -> None:
        con = _old_database(tmp_path)
        _apply_schema(con, log=lambda _m: None)
        plan = " ".join(str(r) for r in con.execute("EXPLAIN QUERY PLAN " + _SPARSE_LOOKUP, ("x", "y")).fetchall())
        assert _NAME_INDEX in plan
