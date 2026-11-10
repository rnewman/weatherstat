"""Incremental snapshot load: a cached process must see exactly what a fresh full read sees."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

import weatherstat.extract as ex

# Names typed by the example config: REAL, INTEGER (binary env, ffilled → boolean), TEXT;
# plus one the config doesn't know (stays text).
_REAL, _INT, _TEXT, _UNTYPED = "office_temp", "kitchen_open", "thermostat_upstairs_action", "mystery_sensor"


def _ts(i: int) -> str:
    return f"2026-03-01T{i // 12:02d}:{i % 12 * 5:02d}:00.000Z"


@pytest.fixture
def db(tmp_path: Path):
    path = tmp_path / "snapshots.db"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE readings (timestamp TEXT NOT NULL, name TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY (timestamp, name))"
    )
    ex.clear_snapshot_cache()
    yield path, con
    con.close()
    ex.clear_snapshot_cache()


def _append(con: sqlite3.Connection, rows: list[tuple[str, str, str]]) -> None:
    con.executemany("INSERT OR IGNORE INTO readings VALUES (?, ?, ?)", rows)
    con.commit()


def _snapshot(i: int, *, temp: str | None = None) -> list[tuple[str, str, str]]:
    return [
        (_ts(i), _REAL, temp if temp is not None else f"{68 + i % 3}.{i % 10}"),
        (_ts(i), _INT, str(i // 4 % 2)),
        (_ts(i), _TEXT, "heating" if i % 3 else "idle"),
        (_ts(i), _UNTYPED, "x"),
    ]


def _fresh(path: Path) -> pd.DataFrame:
    """What a process with no cache would load."""
    saved = dict(ex._snapshot_cache)
    ex._snapshot_cache.clear()
    try:
        return ex.load_collector_snapshots(path)
    finally:
        ex._snapshot_cache.clear()
        ex._snapshot_cache.update(saved)


@pytest.fixture
def reads(monkeypatch):
    """Record each _read_rows call as 'full' or 'tail'."""
    calls: list[str] = []
    real = ex._read_rows

    def spy(conn, hi, lo=None):
        calls.append("full" if lo is None else "tail")
        return real(conn, hi, lo)

    monkeypatch.setattr(ex, "_read_rows", spy)
    return calls


def _assert_same_as_fresh(path: Path, reads: list[str] | None = None, expect: str | None = None) -> pd.DataFrame:
    """Load through the cache (checking which read it did), then compare with a cache-free load."""
    if reads is not None:
        reads.clear()
    cached = ex.load_collector_snapshots(path)
    if expect is not None:
        assert reads == ([] if expect == "none" else [expect])
    pd.testing.assert_frame_equal(cached, _fresh(path))
    return cached


class TestIncrementalLoad:
    def test_full_read_matches_reference_reader(self, db) -> None:
        path, con = db
        _append(con, [r for i in range(20) for r in _snapshot(i)])
        con2 = sqlite3.connect(path)
        reference = ex._load_from_readings(con2)
        con2.close()
        raw = ex._snapshot_cache  # load fills it
        loaded = ex.load_collector_snapshots(path)
        assert path.resolve() in raw
        pd.testing.assert_frame_equal(loaded[[_REAL, _TEXT, _UNTYPED]], reference[[_REAL, _TEXT, _UNTYPED]])

    def test_appended_snapshots_read_incrementally(self, db, reads) -> None:
        path, con = db
        _append(con, [r for i in range(20) for r in _snapshot(i)])
        _assert_same_as_fresh(path)
        _append(con, [r for i in range(20, 32) for r in _snapshot(i)])
        df = _assert_same_as_fresh(path, reads, expect="tail")  # only the new rows were read
        assert len(df) == 32
        _assert_same_as_fresh(path, reads, expect="none")  # nothing new: no rows read at all

    def test_late_names_new_columns_and_junk(self, db) -> None:
        path, con = db
        _append(con, [r for i in range(10) for r in _snapshot(i)])
        _assert_same_as_fresh(path)
        _append(con, [
            (_ts(3), "bedroom_temp", "66.5"),  # new REAL column, written late for an old timestamp
            (_ts(10), _REAL, "unavailable"),  # junk in a numeric column → NaN
            (_ts(11), "late_text", "on"),  # new untyped column at a new timestamp
        ])
        df = _assert_same_as_fresh(path)
        assert df["bedroom_temp"].notna().sum() == 1
        assert pd.isna(df.loc[df["timestamp"] == _ts(10), _REAL]).all()

    def test_integer_columns_keep_full_load_dtype(self, db) -> None:
        """to_numeric gives int64 only when every row has an integer; the merge must agree."""
        path, con = db
        con.executemany("INSERT INTO readings VALUES (?, ?, ?)", [(_ts(i), "combi_gas_usage", str(i)) for i in range(6)])
        con.commit()
        assert ex.load_collector_snapshots(path)["combi_gas_usage"].dtype == "int64"
        _append(con, [(_ts(i), "combi_gas_usage", str(i)) for i in range(6, 9)])
        assert _assert_same_as_fresh(path)["combi_gas_usage"].dtype == "int64"
        _append(con, [(_ts(9), _TEXT, "idle")])  # a timestamp without the name → NaN → float
        assert _assert_same_as_fresh(path)["combi_gas_usage"].dtype == "float64"
        _append(con, [(_ts(9), "combi_gas_usage", "2.5")])
        assert _assert_same_as_fresh(path)["combi_gas_usage"].dtype == "float64"

    def test_deleted_rows_force_full_reread(self, db, reads) -> None:
        path, con = db
        _append(con, [r for i in range(10) for r in _snapshot(i)])
        ex.load_collector_snapshots(path)
        con.execute("DELETE FROM readings WHERE timestamp = ?", (_ts(2),))
        con.commit()
        _append(con, _snapshot(10))
        df = _assert_same_as_fresh(path, reads, expect="full")
        assert _ts(2) not in set(df["timestamp"])

    def test_max_age_forces_full_reread(self, db, reads, monkeypatch) -> None:
        """Bounds how long an in-place edit (a rename migration) goes unseen."""
        path, con = db
        _append(con, [r for i in range(10) for r in _snapshot(i)])
        ex.load_collector_snapshots(path)
        con.execute("UPDATE readings SET name = 'renamed' WHERE name = ?", (_UNTYPED,))
        con.commit()
        monkeypatch.setattr(ex, "_CACHE_MAX_AGE_SECONDS", -1.0)
        df = _assert_same_as_fresh(path, reads, expect="full")
        assert "renamed" in df.columns and _UNTYPED not in df.columns

    def test_callers_cannot_modify_the_cache(self, db) -> None:
        path, con = db
        _append(con, [r for i in range(10) for r in _snapshot(i)])
        first = ex.load_collector_snapshots(path)
        first.loc[:, _REAL] = -999.0
        first[_TEXT] = "tampered"
        second = ex.load_collector_snapshots(path)
        assert (second[_REAL] != -999.0).all()
        assert "tampered" not in set(second[_TEXT])

    def test_empty_table(self, db) -> None:
        path, _ = db
        assert ex.load_collector_snapshots(path).empty
