"""Frozen sensors: detect readings that stopped changing, and treat them as missing.

A dead battery device often doesn't go "unavailable": Home Assistant keeps
reporting its last value (and restores it across restarts), the collector
writes it every cycle, and everything downstream believes it: control scores a
room by a reading that stopped days ago, and hourly sysid refits fit a flat line
into its tau and gains.

HA's own signals don't catch this. Sensors that report only on change have
``last_reported`` equal to ``last_changed`` in a steady room too; both reset
when HA restarts and restores the old value; a battery Z-Wave node reads
"asleep" either way. A battery-level entity is the only warning, and that is a
configurable ``health:`` check. So the detector works from the readings
themselves.

**Rule.** A *flat run* is a stretch of identical consecutive readings of one
sensor (missing readings are skipped: they neither break nor lengthen a run),
measured in observed time = readings × the 5-minute snapshot interval, so a
collector outage never counts as flat time. Sensors differ hugely in how long
they legitimately sit still — 0.1°F room sensors ≤ ~2h, a 0.5°F thermostat
with reporting hysteresis 50h on a quiet summer weekend — so the limit is
relative to each sensor's own history::

    limit = max(floor_hours, multiplier × p99.9 of that sensor's flat runs)

(defaults: 3h floor, ×2). With fewer than ``MIN_RUNS`` runs there is no
history to be relative to, and the limit is ``DEFAULT_LIMIT_HOURS``. A run
longer than its limit is frozen. One more rule catches a house-wide freeze
(HA or the Z-Wave stack stops updating every device at once): while at least
half the monitored sensors are frozen, any other monitored sensor that has
been flat for at least the floor is frozen too — a coarse thermostat's limit
is days, but its flat run in that window is evidence of the same fault.

**Uses.** History: ``quarantine_frozen`` masks frozen spans to NaN at load time
(``extract.load_collector_snapshots``), so sysid tau segments, the regression
and the residual tracker skip them while snapshots.db keeps the raw readings —
detection reruns on every load, so a span that ends (battery replaced) is
still excluded and nothing needs cleaning up. Live: ``frozen_sensors_now``
reads each monitored sensor's current run from snapshots.db; control drops a
frozen sensor from current temperatures (it is not scored and has no
predictions), the TUI/web/MQTT show it as frozen, and a safety alert goes to
HA until it changes again.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from weatherstat.safety import SafetyAlert
    from weatherstat.yaml_config import WeatherstatConfig

STEP_HOURS = 5.0 / 60.0  # one collector snapshot
QUANTILE = 0.999
MIN_RUNS = 500  # fewer runs than this: no history to be relative to
DEFAULT_LIMIT_HOURS = 12.0
HOUSE_WIDE_FRACTION = 0.5  # this share of monitored sensors frozen at once = a house-wide freeze

ATTR_SPANS = "frozen_spans"  # DataFrame.attrs key: spans masked at load (for reporting)
ATTR_LIMITS = "flat_run_limits"  # DataFrame.attrs key: per-sensor limits used for that masking

_LIMITS_TTL_SECONDS = 6 * 3600


# ── Types ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FrozenSpan:
    """A flat run long enough to count as a dead sensor."""

    sensor: str
    start: str  # timestamp of the first reading in the run
    end: str  # timestamp of the last reading in the run
    value: float
    hours: float  # observed flat time (readings × 5 min)
    limit_hours: float  # the sensor's limit; hours > limit (or house-wide)
    reason: str = "flat"  # "flat" (own limit) or "house-wide"


@dataclass(frozen=True)
class FrozenSensor:
    """A monitored sensor whose current reading is frozen."""

    sensor: str
    value: float
    since: str  # timestamp of the first reading of the current run
    hours: float
    limit_hours: float
    reason: str = "flat"

    def describe(self, unit: str = "") -> str:
        """'69.8°F for 23h (limit 18h)'."""
        why = ", house-wide" if self.reason == "house-wide" else ""
        return f"{self.value:g}{unit} for {_fmt_hours(self.hours)} (limit {_fmt_hours(self.limit_hours)}{why})"


def _fmt_hours(h: float) -> str:
    return f"{h:.1f}h" if h < 10 else f"{h:.0f}h"


# ── Config ────────────────────────────────────────────────────────────────


def monitored_sensors(cfg: WeatherstatConfig | None = None) -> list[str]:
    """Sensors checked for freezing: constrained (comfort/sysid) sensors plus the outdoor sensor."""
    if cfg is None:
        from weatherstat.yaml_config import load_config

        cfg = load_config()
    cols = list(cfg.prediction_sensors)
    if cfg.outdoor_sensor and cfg.outdoor_sensor not in cols:
        cols.append(cfg.outdoor_sensor)
    return cols


def _knobs(cfg: WeatherstatConfig | None = None) -> tuple[float, float]:
    """(floor_hours, multiplier) from ``defaults:`` (frozen_floor_hours, frozen_multiplier)."""
    if cfg is None:
        from weatherstat.yaml_config import load_config

        cfg = load_config()
    return cfg.frozen_floor_hours, cfg.frozen_multiplier


# ── Flat runs ─────────────────────────────────────────────────────────────


def flat_runs(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs of identical consecutive readings, skipping missing ones.

    Returns (first_pos, last_pos, n_readings): positions into ``values`` of
    each run's first and last reading, and how many readings it has. NaNs
    neither break a run nor count toward it.
    """
    v_all = np.asarray(values, dtype=float)
    pos = np.flatnonzero(~np.isnan(v_all))
    if len(pos) == 0:
        empty = np.array([], dtype=int)
        return empty, empty, empty
    v = v_all[pos]
    new = np.r_[True, v[1:] != v[:-1]]
    first = np.flatnonzero(new)
    last = np.r_[first[1:], len(v)] - 1
    return pos[first], pos[last], last - first + 1


def flat_run_limit(n_readings: np.ndarray, floor_hours: float, multiplier: float) -> float:
    """A sensor's frozen limit (hours) from the lengths of its flat runs."""
    if len(n_readings) < MIN_RUNS:
        return max(floor_hours, DEFAULT_LIMIT_HOURS)
    typical_long = float(np.quantile(np.asarray(n_readings, dtype=float) * STEP_HOURS, QUANTILE))
    return round(max(floor_hours, multiplier * typical_long), 2)


def flat_run_limits(df: pd.DataFrame, sensors: list[str] | None = None) -> dict[str, float]:
    """Per-sensor limits (hours) for every monitored sensor present in a wide snapshot frame."""
    floor, mult = _knobs()
    out: dict[str, float] = {}
    for col in sensors if sensors is not None else monitored_sensors():
        if col in df.columns:
            _, _, n = flat_runs(_numeric(df[col]))
            out[col] = flat_run_limit(n, floor, mult)
    return out


def _numeric(s: pd.Series) -> np.ndarray:
    import pandas as pd

    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def detect_frozen_spans(
    df: pd.DataFrame,
    sensors: list[str] | None = None,
    limits: dict[str, float] | None = None,
) -> list[FrozenSpan]:
    """Frozen spans in a wide snapshot frame (``timestamp`` column + one column per sensor)."""
    floor, _ = _knobs()
    cols = [c for c in (sensors if sensors is not None else monitored_sensors()) if c in df.columns]
    if not cols or df.empty:
        return []
    lim = limits if limits is not None else flat_run_limits(df, cols)
    ts = df["timestamp"].astype(str).to_numpy()
    n_rows = len(df)

    runs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    own = np.zeros((n_rows, len(cols)), dtype=bool)  # row is inside a sensor's own-limit frozen run
    has_data = np.zeros((n_rows, len(cols)), dtype=bool)
    for j, col in enumerate(cols):
        vals = _numeric(df[col])
        first, last, n = flat_runs(vals)
        runs[col] = (first, last, n, vals)
        # A sensor votes on house-wide freezes from its first reading to its last
        # (not before it was installed, not after it was removed).
        if len(first):
            has_data[first[0] : last[-1] + 1, j] = True
        hours = n * STEP_HOURS
        for a, b in zip(first[hours > lim[col]], last[hours > lim[col]], strict=True):
            own[a : b + 1, j] = True

    voters = has_data.sum(axis=1)
    house_wide = (own.sum(axis=1) >= np.maximum(2, np.ceil(HOUSE_WIDE_FRACTION * voters))) & (voters > 0)

    spans: list[FrozenSpan] = []
    for col in cols:
        first, last, n, vals = runs[col]
        hours = n * STEP_HOURS
        cand = hours >= min(floor, lim[col])  # only runs this long can qualify either way
        for a, b, h in zip(first[cand], last[cand], hours[cand], strict=True):
            if h > lim[col]:
                reason = "flat"
            elif h >= floor and house_wide[a : b + 1].any():
                reason = "house-wide"
            else:
                continue
            spans.append(FrozenSpan(col, str(ts[a]), str(ts[b]), float(vals[a]), round(float(h), 2), lim[col], reason))
    spans.sort(key=lambda s: (s.start, s.sensor))
    return spans


def mask_frozen(df: pd.DataFrame, spans: list[FrozenSpan]) -> pd.DataFrame:
    """Set each span's readings to NaN (in place on ``df``, which is returned)."""
    if not spans:
        return df
    ts = df["timestamp"].astype(str)
    by_col: dict[str, np.ndarray] = {}
    for s in spans:
        m = by_col.setdefault(s.sensor, np.zeros(len(df), dtype=bool))
        m |= ((ts >= s.start) & (ts <= s.end)).to_numpy()
    for col, m in by_col.items():
        df[col] = df[col].astype(float).mask(m)
    return df


def quarantine_frozen(df: pd.DataFrame) -> pd.DataFrame:
    """Mask frozen spans of monitored sensors; record spans + limits in ``df.attrs``.

    The load-time exclusion for history: raw readings stay in snapshots.db,
    every load re-detects, so a span that has since ended is still excluded.
    """
    if df.empty or "timestamp" not in df.columns:
        return df
    cols = [c for c in monitored_sensors() if c in df.columns]
    limits = flat_run_limits(df, cols)
    spans = detect_frozen_spans(df, cols, limits)
    mask_frozen(df, spans)
    df.attrs[ATTR_SPANS] = spans
    df.attrs[ATTR_LIMITS] = limits
    return df


def summarize_spans(spans: list[FrozenSpan], unit: str = "") -> list[str]:
    """One line per span, for sysid output and `just debug frozen`."""
    return [
        f"{s.sensor:<30} {s.start[:16]} → {s.end[:16]}  {s.value:g}{unit} for {_fmt_hours(s.hours)}"
        f" (limit {_fmt_hours(s.limit_hours)}{', house-wide' if s.reason == 'house-wide' else ''})"
        for s in spans
    ]


# ── Live detection (snapshots.db) ─────────────────────────────────────────

_limits_lock = threading.Lock()
_limits_cache: dict[Path, tuple[float, dict[str, float]]] = {}


def clear_limits_cache() -> None:
    with _limits_lock:
        _limits_cache.clear()


def db_flat_run_limits(db_path: Path, sensors: list[str]) -> dict[str, float]:
    """Per-sensor limits from the sensors' full history in snapshots.db (cached per process for 6h)."""
    key = db_path.resolve()
    with _limits_lock:
        hit = _limits_cache.get(key)
        if hit is not None and time.monotonic() - hit[0] < _LIMITS_TTL_SECONDS and set(sensors) <= set(hit[1]):
            return hit[1]
    floor, mult = _knobs()
    limits: dict[str, float] = {}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        for col in sensors:
            rows = conn.execute(
                "SELECT CAST(value AS REAL) FROM readings WHERE name = ? ORDER BY timestamp", (col,)
            ).fetchall()
            _, _, n = flat_runs(np.array([r[0] for r in rows], dtype=float))
            limits[col] = flat_run_limit(n, floor, mult)
    finally:
        conn.close()
    with _limits_lock:
        _limits_cache[key] = (time.monotonic(), limits)
    return limits


@dataclass(frozen=True)
class _CurrentRun:
    value: float
    since: str
    n_readings: int


def _current_run(conn: sqlite3.Connection, name: str) -> _CurrentRun | None:
    """The sensor's latest value and how many consecutive readings (to now) have had it."""
    latest = conn.execute(
        "SELECT timestamp, CAST(value AS REAL) FROM readings WHERE name = ? ORDER BY timestamp DESC LIMIT 1", (name,)
    ).fetchone()
    if latest is None:
        return None
    value = float(latest[1])
    changed = conn.execute(
        "SELECT timestamp FROM readings WHERE name = ? AND CAST(value AS REAL) != ? ORDER BY timestamp DESC LIMIT 1",
        (name, value),
    ).fetchone()
    after = changed[0] if changed else ""
    n, since = conn.execute(
        "SELECT COUNT(*), MIN(timestamp) FROM readings WHERE name = ? AND timestamp > ?", (name, after)
    ).fetchone()
    return _CurrentRun(value, str(since), int(n))


def frozen_sensors_now(
    db_path: Path | None = None,
    sensors: list[str] | None = None,
    limits: dict[str, float] | None = None,
    live_values: dict[str, float] | None = None,
) -> dict[str, FrozenSensor]:
    """Monitored sensors whose current reading is frozen, from snapshots.db.

    ``live_values`` (control's fresh HA readings): a sensor whose live value
    differs from its run in the database has changed since the last
    snapshot, so it is not frozen. Returns {} when the database is missing.
    """
    from weatherstat.config import SNAPSHOTS_DB

    path = db_path or SNAPSHOTS_DB
    if not path.exists():
        return {}
    cols = sensors if sensors is not None else monitored_sensors()
    lim = limits if limits is not None else db_flat_run_limits(path, cols)
    floor, _ = _knobs()

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        current = {c: r for c in cols if (r := _current_run(conn, c)) is not None}
    finally:
        conn.close()

    def _changed_since(col: str, run: _CurrentRun) -> bool:
        live = (live_values or {}).get(col)
        return live is not None and abs(float(live) - run.value) > 1e-9

    out: dict[str, FrozenSensor] = {}
    for col, run in current.items():
        hours = run.n_readings * STEP_HOURS
        limit = lim.get(col, DEFAULT_LIMIT_HOURS)
        if hours > limit and not _changed_since(col, run):
            out[col] = FrozenSensor(col, run.value, run.since, round(hours, 2), limit)
    if current and len(out) >= max(2, np.ceil(HOUSE_WIDE_FRACTION * len(current))):
        for col, run in current.items():
            hours = run.n_readings * STEP_HOURS
            if col not in out and hours >= floor and not _changed_since(col, run):
                out[col] = FrozenSensor(col, run.value, run.since, round(hours, 2), lim.get(col, DEFAULT_LIMIT_HOURS), "house-wide")
    return out


# ── Alerts ────────────────────────────────────────────────────────────────

ALERT_PREFIX = "sensor_frozen_"  # SafetyAlert key prefix; cooldown category "sensor_frozen"


def frozen_sensor_alerts(frozen: dict[str, FrozenSensor]) -> list[SafetyAlert]:
    """One safety alert per frozen sensor (dispatched with the other safety alerts)."""
    from weatherstat.config import SENSOR_LABELS, UNIT_SYMBOL
    from weatherstat.safety import SafetyAlert

    alerts: list[SafetyAlert] = []
    for col, fz in sorted(frozen.items()):
        label = SENSOR_LABELS.get(col, col.removesuffix("_temp")).replace("_", " ")
        alerts.append(SafetyAlert(
            key=f"{ALERT_PREFIX}{col}",
            title=f"{label.capitalize()} sensor frozen",
            message=(
                f"{label.capitalize()} has read {fz.describe(UNIT_SYMBOL)} without changing. Weatherstat is "
                "ignoring it (not scored, excluded from sysid) until it changes — check the device battery."
            ),
            severity="warning",
            category="sensor_frozen",
        ))
    return alerts


def clear_recovered_alerts(
    frozen: dict[str, FrozenSensor],
    *,
    live: bool = False,
    notification_target: str = "persistent_notification",
    now: datetime | None = None,
) -> list[str]:
    """Dismiss the HA notification of every sensor that was alerted frozen and has recovered.

    The alert's cooldown entry (``safety_sensor_frozen_<col>``, set when it
    was sent) is the record that it is outstanding; recovery dismisses the
    notification and drops the entry, so a later freeze alerts at once.
    Returns the recovered sensor columns.
    """
    from weatherstat.advisory import dismiss_ha_notification, load_advisory_state, save_advisory_state

    state = load_advisory_state()
    prefix = f"safety_{ALERT_PREFIX}"
    recovered = [k.removeprefix(prefix) for k in state if k.startswith(prefix) and k.removeprefix(prefix) not in frozen]
    if not recovered:
        return []
    stamp = (now or datetime.now(UTC)).strftime("%H:%M UTC")
    for col in recovered:
        print(f"  [safety] {col} reading again — clearing frozen-sensor alert ({stamp})")
        if live:
            dismiss_ha_notification(f"weatherstat_safety_{ALERT_PREFIX}{col}", notification_target)
            state.pop(f"{prefix}{col}", None)
    if live:
        save_advisory_state(state)
    return recovered
