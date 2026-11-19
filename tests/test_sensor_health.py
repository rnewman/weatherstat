"""Frozen sensors: detection, load-time quarantine, live check, alerts.

The failure being modelled: a battery multisensor keeps reporting its last value
for days and nothing notices. Synthetic data only (example config).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import weatherstat.extract as ex
from weatherstat import sensor_health as sh
from weatherstat.sensor_health import (
    DEFAULT_LIMIT_HOURS,
    MIN_RUNS,
    STEP_HOURS,
    FrozenSensor,
    clear_recovered_alerts,
    detect_frozen_spans,
    flat_run_limit,
    flat_runs,
    frozen_sensor_alerts,
    frozen_sensors_now,
    mask_frozen,
    monitored_sensors,
    quarantine_frozen,
)

T0 = datetime(2026, 11, 1, tzinfo=UTC)
STEPS_PER_HOUR = 12


def _ts(i: int) -> str:
    return (T0 + timedelta(minutes=5 * i)).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _room(n: int, seed: int, resolution: float = 0.1, base: float = 70.0) -> np.ndarray:
    """A room sensor: smooth diurnal swing + noise, quantized — short flat runs, thousands of them."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    walk = base + 1.5 * np.sin(2 * np.pi * t / (24 * STEPS_PER_HOUR)) + rng.normal(0, 0.08, n)
    return np.round(walk / resolution) * resolution


def _coarse(n: int, seed: int) -> np.ndarray:
    """A 0.5° thermostat that sits still for hours at a time (legitimately long runs)."""
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    v, i = 71.0, 0
    while i < n:
        hold = int(rng.integers(1, 10 * STEPS_PER_HOUR))
        out[i : i + hold] = v
        v += rng.choice([-0.5, 0.5])
        i += hold
    return out


def _frame(n: int = 30 * 24 * STEPS_PER_HOUR, **cols: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"timestamp": [_ts(i) for i in range(n)]})
    for name, values in cols.items():
        df[name] = values
    return df


# ── Flat runs and limits ──────────────────────────────────────────────────


class TestFlatRuns:
    def test_runs_skip_missing_readings(self) -> None:
        v = np.array([1.0, 1.0, np.nan, 1.0, 2.0, np.nan, np.nan, 2.0, 3.0])
        first, last, n = flat_runs(v)
        assert first.tolist() == [0, 4, 8]
        assert last.tolist() == [3, 7, 8]
        assert n.tolist() == [3, 2, 1]  # NaNs neither break nor lengthen a run

    def test_empty(self) -> None:
        first, last, n = flat_runs(np.array([np.nan, np.nan]))
        assert len(first) == len(last) == len(n) == 0

    def test_limit_relative_to_own_history(self) -> None:
        rng = np.random.default_rng(0)
        short = rng.integers(1, 6, 2000)  # ≤25 min runs: a 0.1° room sensor
        long = rng.integers(1, 120, 2000)  # ≤10h runs: a coarse thermostat
        assert flat_run_limit(short, floor_hours=3.0, multiplier=2.0) == 3.0  # floor
        coarse = flat_run_limit(long, floor_hours=3.0, multiplier=2.0)
        assert coarse == pytest.approx(2.0 * np.quantile(long * STEP_HOURS, 0.999), abs=0.01)
        assert coarse > 15.0

    def test_limit_without_history_is_default(self) -> None:
        assert flat_run_limit(np.ones(MIN_RUNS - 1, dtype=int), 3.0, 2.0) == DEFAULT_LIMIT_HOURS


# ── Detection over history ────────────────────────────────────────────────


class TestDetectFrozenSpans:
    def test_frozen_room_sensor_detected(self) -> None:
        n = 30 * 24 * STEPS_PER_HOUR
        living = _room(n, 1)
        start, stop = 10 * 24 * STEPS_PER_HOUR, 20 * 24 * STEPS_PER_HOUR  # dead for 10 days, then a new battery
        living[start:stop] = 69.83  # off the 0.1° grid: no accidental neighbours
        df = _frame(n, living_room_temp=living, bedroom_temp=_room(n, 2), thermostat_upstairs_temp=_coarse(n, 3))

        spans = detect_frozen_spans(df, ["living_room_temp", "bedroom_temp", "thermostat_upstairs_temp"])

        assert len(spans) == 1
        s = spans[0]
        assert s.sensor == "living_room_temp" and s.value == 69.83 and s.reason == "flat"
        assert s.start == _ts(start) and s.end == _ts(stop - 1)
        assert s.hours == pytest.approx(10 * 24, abs=1)

    def test_coarse_sensor_long_runs_not_flagged(self) -> None:
        n = 30 * 24 * STEPS_PER_HOUR
        df = _frame(n, thermostat_upstairs_temp=_coarse(n, 4), bedroom_temp=_room(n, 5))
        assert detect_frozen_spans(df, ["thermostat_upstairs_temp", "bedroom_temp"]) == []

    def test_house_wide_freeze_catches_coarse_sensors(self) -> None:
        """HA stops updating every device: rooms trip their own limits, the thermostat rides along."""
        n = 30 * 24 * STEPS_PER_HOUR
        a, b = 15 * 24 * STEPS_PER_HOUR, 15 * 24 * STEPS_PER_HOUR + 6 * STEPS_PER_HOUR  # 6h freeze
        rooms = {f"r{i}": _room(n, 10 + i) for i in range(3)}
        thermo = _coarse(n, 20)
        for v in [*rooms.values(), thermo]:
            v[a:b] = v[a]
        df = _frame(n, **rooms, thermo=thermo)

        spans = {s.sensor: s for s in detect_frozen_spans(df, [*rooms, "thermo"])}

        assert set(spans) == {"r0", "r1", "r2", "thermo"}
        assert spans["thermo"].reason == "house-wide"
        assert spans["r0"].reason == "flat"

    def test_one_flat_room_is_not_house_wide(self) -> None:
        """A coarse thermostat isn't dragged in by a single dead room sensor."""
        n = 30 * 24 * STEPS_PER_HOUR
        a, b = 15 * 24 * STEPS_PER_HOUR, 15 * 24 * STEPS_PER_HOUR + 6 * STEPS_PER_HOUR
        r0, r1, r2, thermo = _room(n, 30), _room(n, 31), _room(n, 32), _coarse(n, 33)
        r0[a:b] = r0[a]
        thermo[a:b] = thermo[a]
        df = _frame(n, r0=r0, r1=r1, r2=r2, thermo=thermo)
        assert {s.sensor for s in detect_frozen_spans(df, ["r0", "r1", "r2", "thermo"])} == {"r0"}

    def test_mask_and_quarantine(self) -> None:
        n = 30 * 24 * STEPS_PER_HOUR
        bedroom = _room(n, 6)
        bedroom[1000:1500] = 66.6
        office = _room(n, 7)
        df = _frame(n, bedroom_temp=bedroom, office_temp=office, other=np.arange(n, dtype=float))

        out = quarantine_frozen(df)

        spans = out.attrs[sh.ATTR_SPANS]
        assert [s.sensor for s in spans] == ["bedroom_temp"]
        assert out["bedroom_temp"].iloc[1000:1500].isna().all()
        assert out["bedroom_temp"].iloc[:999].notna().all() and out["bedroom_temp"].iloc[1500:].notna().all()
        assert out["office_temp"].notna().all() and out["other"].notna().all()
        assert set(out.attrs[sh.ATTR_LIMITS]) == {"bedroom_temp", "office_temp"}

    def test_mask_upcasts_integer_columns(self) -> None:
        df = pd.DataFrame({"timestamp": [_ts(i) for i in range(4)], "x": [1, 1, 1, 2]})
        spans = [sh.FrozenSpan("x", _ts(0), _ts(2), 1.0, 0.25, 0.1)]
        out = mask_frozen(df, spans)
        assert out["x"].isna().tolist() == [True, True, True, False]

    def test_monitored_sensors_are_constrained_plus_outdoor(self) -> None:
        from weatherstat.yaml_config import load_config

        cfg = load_config()
        cols = monitored_sensors()
        assert set(cfg.prediction_sensors) <= set(cols)
        assert cfg.outdoor_sensor in cols


# ── Load-time quarantine through the snapshot cache ───────────────────────


def _db(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE readings (timestamp TEXT NOT NULL, name TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY (name, timestamp))"
    )
    return con


def _write(con: sqlite3.Connection, name: str, values: np.ndarray, offset: int = 0) -> None:
    con.executemany(
        "INSERT OR IGNORE INTO readings VALUES (?, ?, ?)",
        [(_ts(offset + i), name, str(float(v))) for i, v in enumerate(values) if not np.isnan(v)],
    )
    con.commit()


class TestLoadQuarantine:
    def test_load_masks_frozen_span_and_keeps_raw_readings(self, tmp_path: Path) -> None:
        n = 20 * 24 * STEPS_PER_HOUR
        bedroom = _room(n, 8)
        bedroom[2000:3000] = 69.8  # ~83h frozen
        path = tmp_path / "snapshots.db"
        con = _db(path)
        _write(con, "bedroom_temp", bedroom)
        _write(con, "office_temp", _room(n, 9))
        ex.clear_snapshot_cache()
        try:
            df = ex.load_collector_snapshots(path)
            assert df["bedroom_temp"].iloc[2000:3000].isna().all()
            assert df["office_temp"].notna().all()
            raw = ex.load_collector_snapshots(path, quarantine=False)
            assert (raw["bedroom_temp"].iloc[2000:3000] == 69.8).all()
            # snapshots.db is untouched: canonical storage, never pruned.
            assert con.execute("SELECT COUNT(*) FROM readings WHERE name='bedroom_temp' AND value='69.8'").fetchone()[0] >= 1000

            # The incremental (cached) load after appends still masks — detection runs on every load.
            _write(con, "bedroom_temp", _room(STEPS_PER_HOUR, 10), offset=n)
            _write(con, "office_temp", _room(STEPS_PER_HOUR, 11), offset=n)
            again = ex.load_collector_snapshots(path)
            assert len(again) == n + STEPS_PER_HOUR
            assert again["bedroom_temp"].iloc[2000:3000].isna().all()
        finally:
            con.close()
            ex.clear_snapshot_cache()

    def test_sysid_excludes_masked_rows(self) -> None:
        """Masked readings drop out of both tau segments and the regression target."""
        from weatherstat.sysid import _enumerate_effectors, _enumerate_sensors, _preprocess, newton_residual

        n = 6 * 24 * STEPS_PER_HOUR
        bedroom = _room(n, 12)
        bedroom[500:900] = np.nan  # what quarantine leaves
        df = _frame(n, bedroom_temp=bedroom, met_outdoor_temp=np.full(n, 45.0))
        sensors = [s for s in _enumerate_sensors() if s.name == "bedroom_temp"]
        out = _preprocess(df, _enumerate_effectors(), sensors)
        resid = newton_residual(out, sensors[0], 45.0)
        assert resid is not None
        assert np.isnan(resid[505:895]).all()  # regression rows dropped (NaN target)


# ── Live detection from snapshots.db ──────────────────────────────────────


class TestFrozenSensorsNow:
    def _setup(self, tmp_path: Path, frozen_tail: int) -> Path:
        n = 20 * 24 * STEPS_PER_HOUR
        living = _room(n, 13)
        if frozen_tail:
            living[n - frozen_tail :] = 69.8
        path = tmp_path / "snapshots.db"
        con = _db(path)
        _write(con, "living_room_temp", living)
        _write(con, "bedroom_temp", _room(n, 14))
        con.close()
        sh.clear_limits_cache()
        return path

    def test_detects_current_frozen_run(self, tmp_path: Path) -> None:
        path = self._setup(tmp_path, frozen_tail=20 * STEPS_PER_HOUR)
        out = frozen_sensors_now(path, ["living_room_temp", "bedroom_temp"])
        assert set(out) == {"living_room_temp"}
        fz = out["living_room_temp"]
        assert fz.value == 69.8 and fz.hours >= 20 and fz.limit_hours == 3.0

    def test_short_flat_run_is_fine(self, tmp_path: Path) -> None:
        path = self._setup(tmp_path, frozen_tail=STEPS_PER_HOUR)  # 1h: a quiet room
        assert frozen_sensors_now(path, ["living_room_temp", "bedroom_temp"]) == {}

    def test_live_value_change_clears(self, tmp_path: Path) -> None:
        """Battery replaced: HA already reads 75.0 though the collector hasn't written it yet."""
        path = self._setup(tmp_path, frozen_tail=20 * STEPS_PER_HOUR)
        out = frozen_sensors_now(path, ["living_room_temp"], live_values={"living_room_temp": 75.0})
        assert out == {}
        assert "living_room_temp" in frozen_sensors_now(path, ["living_room_temp"], live_values={"living_room_temp": 69.8})

    def test_recovery_in_database(self, tmp_path: Path) -> None:
        path = self._setup(tmp_path, frozen_tail=20 * STEPS_PER_HOUR)
        con = sqlite3.connect(path)
        _write(con, "living_room_temp", np.array([75.0, 76.4]), offset=20 * 24 * STEPS_PER_HOUR)
        con.close()
        assert frozen_sensors_now(path, ["living_room_temp"]) == {}

    def test_missing_db(self, tmp_path: Path) -> None:
        assert frozen_sensors_now(tmp_path / "nope.db") == {}


# ── Alerts ────────────────────────────────────────────────────────────────


class TestAlerts:
    _FZ = {"bedroom_temp": FrozenSensor("bedroom_temp", 69.8, _ts(0), 23.0, 3.0)}

    def test_alert_per_frozen_sensor(self) -> None:
        alerts = frozen_sensor_alerts(self._FZ)
        assert len(alerts) == 1
        a = alerts[0]
        assert a.key == "sensor_frozen_bedroom_temp" and a.category == "sensor_frozen"
        assert "69.8" in a.message and "23h" in a.message

    def test_frozen_alert_uses_its_own_cooldown(self, tmp_path: Path) -> None:
        """Re-sent at most every 12h, not hourly; the cooldown entry marks it outstanding."""
        from weatherstat.safety import process_safety_alerts

        state_file = tmp_path / "advisory_state.json"
        with (
            patch("weatherstat.advisory.ADVISORY_STATE_FILE", state_file),
            patch("weatherstat.safety.send_ha_notification", return_value=True) as send,
        ):
            alerts = frozen_sensor_alerts(self._FZ)
            assert len(process_safety_alerts(alerts, live=True)) == 1
            # Two hours later: still within the 12h category cooldown.
            from weatherstat.advisory import load_advisory_state, save_advisory_state

            state = load_advisory_state()
            state["safety_sensor_frozen_bedroom_temp"] -= 2 * 3600
            save_advisory_state(state)
            assert process_safety_alerts(alerts, live=True) == []
            assert send.call_count == 1
            assert send.call_args.args[2] == "weatherstat_safety_sensor_frozen_bedroom_temp"

    def test_recovery_dismisses_notification(self, tmp_path: Path) -> None:
        from weatherstat.advisory import load_advisory_state, save_advisory_state

        state_file = tmp_path / "advisory_state.json"
        with (
            patch("weatherstat.advisory.ADVISORY_STATE_FILE", state_file),
            patch("weatherstat.advisory.dismiss_ha_notification", return_value=True) as dismiss,
        ):
            save_advisory_state({"safety_sensor_frozen_bedroom_temp": 1.0, "safety_combi_fault": 2.0})
            # Still frozen: nothing to clear.
            assert clear_recovered_alerts(self._FZ, live=True) == []
            # Dry run: reported, nothing changed.
            assert clear_recovered_alerts({}, live=False) == ["bedroom_temp"]
            dismiss.assert_not_called()
            assert "safety_sensor_frozen_bedroom_temp" in load_advisory_state()
            # Live: dismissed and forgotten, other safety state kept.
            assert clear_recovered_alerts({}, live=True) == ["bedroom_temp"]
            dismiss.assert_called_once()
            assert dismiss.call_args.args[0] == "weatherstat_safety_sensor_frozen_bedroom_temp"
            assert load_advisory_state() == {"safety_combi_fault": 2.0}
