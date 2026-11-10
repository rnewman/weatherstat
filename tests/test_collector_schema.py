"""readings schema: the (name, timestamp) key, its in-place migration, and readers on either layout."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

import weatherstat.collector as collector
import weatherstat.decision_log as dl
from weatherstat.collector import _INTERIM_NAME_INDEX, _apply_schema, _primary_key_columns
from weatherstat.extract import latest_snapshot_values, latest_timestamp, snapshot_status

_OLD_TABLE = """CREATE TABLE readings (
  timestamp TEXT NOT NULL, name TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY (timestamp, name))"""
_SPARSE_LOOKUP = "SELECT value FROM readings WHERE name = ? AND timestamp < ? ORDER BY timestamp DESC LIMIT 1"
_NAMES = ("office_temp", "bedroom_temp", "shade_lowered")


def _ts(i: int) -> str:
    return (datetime(2026, 3, 1, tzinfo=UTC) + timedelta(minutes=5 * i)).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _rows(n: int = 60) -> list[tuple[str, str, str]]:
    return [(_ts(i), name, f"{68 + i % 3}.{k}") for i in range(n) for k, name in enumerate(_NAMES)]


def _old_database(path, *, interim_index: bool) -> sqlite3.Connection:
    """A database from before the re-key: timestamp-first key, with history."""
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode = WAL")
    con.execute(_OLD_TABLE)
    if interim_index:
        con.execute(f"CREATE INDEX {_INTERIM_NAME_INDEX} ON readings (name, timestamp)")
    con.executemany("INSERT INTO readings VALUES (?, ?, ?)", _rows())
    con.execute("DELETE FROM readings WHERE rowid % 17 = 3")  # gaps, like past deletes
    con.commit()
    return con


def _index_names(con: sqlite3.Connection) -> set[str]:
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'readings'")}


def _plan(con: sqlite3.Connection, sql: str, args: tuple) -> str:
    return " ".join(r[3] for r in con.execute("EXPLAIN QUERY PLAN " + sql, args).fetchall())


class TestSchema:
    def test_fresh_database_is_name_first(self, tmp_path) -> None:
        con = sqlite3.connect(tmp_path / "snapshots.db")
        logs: list[str] = []
        _apply_schema(con, log=logs.append)
        assert _primary_key_columns(con) == ["name", "timestamp"]
        assert _INTERIM_NAME_INDEX not in _index_names(con)
        assert logs == []

    @pytest.mark.parametrize("interim_index", [False, True])
    def test_old_layout_is_rekeyed_in_place(self, tmp_path, interim_index: bool) -> None:
        con = _old_database(tmp_path / "snapshots.db", interim_index=interim_index)
        before = con.execute("SELECT rowid, timestamp, name, value FROM readings ORDER BY rowid").fetchall()

        logs: list[str] = []
        _apply_schema(con, log=logs.append)
        assert _primary_key_columns(con) == ["name", "timestamp"]
        assert _index_names(con) == {"sqlite_autoindex_readings_1"}  # the key is the only index
        # Every row, with its rowid (the incremental snapshot cache keys on rowids)
        assert con.execute("SELECT rowid, timestamp, name, value FROM readings ORDER BY rowid").fetchall() == before
        assert any("re-keying" in line for line in logs) and any("Migration done" in line for line in logs)

        logs.clear()
        _apply_schema(con, log=logs.append)  # next collector start: nothing to do
        assert logs == []

    def test_key_still_dedupes_like_the_collector_expects(self, tmp_path) -> None:
        con = _old_database(tmp_path / "snapshots.db", interim_index=False)
        _apply_schema(con, log=lambda _m: None)
        n = con.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
        present = set(con.execute("SELECT timestamp, name FROM readings").fetchall())
        missing = sum((t, name) not in present for t, name, _ in _rows(5))
        assert missing >= 1  # some of these were deleted in _old_database
        con.executemany("INSERT OR IGNORE INTO readings (timestamp, name, value) VALUES (?, ?, ?)", _rows(5))
        assert con.execute("SELECT COUNT(*) FROM readings").fetchone()[0] == n + missing  # duplicates ignored

    def test_failed_rebuild_changes_nothing(self, tmp_path) -> None:
        real = _old_database(tmp_path / "snapshots.db", interim_index=True)
        before = real.execute("SELECT rowid, timestamp, name, value FROM readings ORDER BY rowid").fetchall()

        class FailingCopy:
            """Forwards to the connection but fails the copy step, mid-transaction."""

            def __getattr__(self, attr):
                return getattr(real, attr)

            def execute(self, sql, *args):
                if sql.startswith("INSERT INTO readings_rekey"):
                    raise sqlite3.OperationalError("disk I/O error")
                return real.execute(sql, *args)

        logs: list[str] = []
        collector._rekey_name_first(FailingCopy(), logs.append)
        assert any("old layout kept" in line for line in logs)
        assert _primary_key_columns(real) == ["timestamp", "name"]
        assert "readings_rekey" not in {r[0] for r in real.execute("SELECT name FROM sqlite_master")}
        assert real.execute("SELECT rowid, timestamp, name, value FROM readings ORDER BY rowid").fetchall() == before

    def test_skips_without_disk_space(self, tmp_path, monkeypatch) -> None:
        import shutil

        con = _old_database(tmp_path / "snapshots.db", interim_index=False)
        monkeypatch.setattr(shutil, "disk_usage", lambda _p: shutil._ntuple_diskusage(1, 1, 0))
        logs: list[str] = []
        _apply_schema(con, log=logs.append)
        assert any("Migration skipped" in line for line in logs)
        assert _primary_key_columns(con) == ["timestamp", "name"]  # old layout, still working

    def test_query_plans_on_the_name_first_key(self, tmp_path) -> None:
        con = _old_database(tmp_path / "snapshots.db", interim_index=False)
        _apply_schema(con, log=lambda _m: None)  # re-key + ANALYZE
        assert "(name=? AND timestamp<?)" in _plan(con, _SPARSE_LOOKUP, ("office_temp", _ts(10)))
        # Timestamp lookups skip-scan the one index (a seek per name), not the table
        assert "ANY(name) AND timestamp=?" in _plan(con, "SELECT name, value FROM readings WHERE timestamp = ?", (_ts(5),))
        assert "ANY(name) AND timestamp>?" in _plan(con, "SELECT MAX(timestamp) FROM readings WHERE timestamp > ?", (_ts(5),))


class TestReadersOnEitherLayout:
    @pytest.fixture(params=["old", "rekeyed"])
    def db(self, request, tmp_path):
        path = tmp_path / "snapshots.db"
        con = _old_database(path, interim_index=False)
        if request.param == "rekeyed":
            _apply_schema(con, log=lambda _m: None)
        yield path, con
        con.close()

    def test_latest_timestamp(self, db) -> None:
        _, con = db
        assert latest_timestamp(con) == _ts(59)
        con.execute("INSERT INTO readings VALUES (?, 'office_temp', '1')", (_ts(-500),))  # backfill, inserted last
        con.commit()
        assert latest_timestamp(con) == _ts(59)
        con.execute("INSERT INTO readings VALUES (?, 'office_temp', '1')", (_ts(80),))  # a later one, out of order
        con.execute("INSERT INTO readings VALUES (?, 'office_temp', '1')", (_ts(-501),))
        con.commit()
        assert latest_timestamp(con) == _ts(80)

    def test_status_and_latest_values(self, db) -> None:
        path, con = db
        ts, count = snapshot_status(path)
        assert ts == _ts(59)
        assert count == con.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
        values = latest_snapshot_values(path)
        expected = dict(con.execute("SELECT name, value FROM readings WHERE timestamp = ?", (_ts(59),)).fetchall())
        assert values == expected and values

    def test_empty_table(self, tmp_path) -> None:
        con = sqlite3.connect(tmp_path / "empty.db")
        _apply_schema(con, log=lambda _m: None)
        assert latest_timestamp(con) is None
        assert snapshot_status(tmp_path / "empty.db") == ("", 0)
        assert latest_snapshot_values(tmp_path / "empty.db") == {}


class TestOutcomeBackfill:
    """The per-cycle outcome backfill reads only timestamps a pending decision could match."""

    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        snaps = tmp_path / "snapshots.db"
        con = sqlite3.connect(snaps)
        _apply_schema(con, log=lambda _m: None)
        monkeypatch.setattr(dl, "SNAPSHOTS_DB", snaps)
        log_path = tmp_path / "decision_log.db"
        dl.init_db(log_path)
        yield con, log_path
        con.close()

    @staticmethod
    def _decide(log_path, when: datetime, predicted: float) -> None:
        with sqlite3.connect(log_path) as c:
            c.execute(
                "INSERT INTO decisions (timestamp, live, current_temps, predictions, comfort_bounds) VALUES (?, 0, '{}', ?, '{}')",
                (when.isoformat(), json.dumps({"bedroom_temp": {"1h": predicted, "2h": predicted}})),
            )

    @staticmethod
    def _snapshot(con, when: datetime, bedroom: float) -> None:
        con.execute(
            "INSERT INTO readings (timestamp, name, value) VALUES (?, 'bedroom_temp', ?)",
            (when.strftime("%Y-%m-%dT%H:%M:%S.000Z"), str(bedroom)),
        )
        con.commit()

    def test_matches_nearest_snapshot_and_ignores_history(self, env) -> None:
        con, log_path = env
        now = datetime.now(UTC).replace(second=0, microsecond=0)
        decided = now - timedelta(hours=3)
        for k in range(-3000, 40):  # ten days of history before the decision, then the next 3h
            self._snapshot(con, decided + timedelta(minutes=5 * k), 60.0 + (k % 7))
        self._decide(log_path, decided, predicted=65.0)
        assert dl.backfill_outcomes(log_path) == 1
        with sqlite3.connect(log_path) as c:
            outcomes = json.loads(c.execute("SELECT outcomes FROM decisions").fetchone()[0])
        assert outcomes["bedroom_temp"]["1h"]["actual"] == 60.0 + (12 % 7)  # snapshot at +60 min
        assert outcomes["bedroom_temp"]["2h"]["actual"] == 60.0 + (24 % 7)

    def test_old_decisions_without_nearby_snapshots_are_still_closed(self, env) -> None:
        """During a collector outage there is nothing after the decision; it is closed after 7h as before."""
        con, log_path = env
        now = datetime.now(UTC)
        self._snapshot(con, now - timedelta(days=3), 66.0)  # last snapshot long before
        self._decide(log_path, now - timedelta(hours=8), predicted=65.0)
        assert dl.backfill_outcomes(log_path) == 1
        with sqlite3.connect(log_path) as c:
            assert c.execute("SELECT outcome_backfilled, outcomes FROM decisions").fetchone() == (1, None)
