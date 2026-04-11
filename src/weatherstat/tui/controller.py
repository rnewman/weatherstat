"""TUI controller — owns mutable runtime state for the Weatherstat dashboard.

The controller is the only object that mutates `live_mode`, `_overrides`,
`_latest_decision`, `_cycle_running`, etc. Both the keybinding actions on
`WeatherstatApp` and the HTTP handlers in `weatherstat.web` call into the
controller.

The controller does NOT import textual. It receives:

- `log(msg)`        — for sending text to the UI's log view
- `worker(fn)`      — for spawning a worker thread (Textual: run_worker thread=True)
- `on_snapshot(s)`  — called when a fresh `StatusSnapshot` is available
- `on_main(fn)`     — Textual `call_from_thread` (or noop in tests)

This separation lets the controller be tested without Textual installed.
"""

from __future__ import annotations

import contextlib
import io
import json
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

# ── Snapshot data structures ────────────────────────────────────────────────


@dataclass(frozen=True)
class ComfortBounds:
    acceptable_lo: float
    preferred_lo: float
    preferred_hi: float
    acceptable_hi: float


@dataclass(frozen=True)
class EnvEntry:
    label: str
    kind: str
    is_active: bool


@dataclass(frozen=True)
class EffectorView:
    name: str
    mode: str
    target: float | None
    delay_steps: int
    duration_steps: int | None
    override: str | None  # description if a manual override was detected, else None


@dataclass(frozen=True)
class CycleCosts:
    total: float
    comfort: float
    energy: float
    baseline: float | None


@dataclass(frozen=True)
class StatusSnapshot:
    """Plain-data view of the system, rebuilt by `TUIController.build_snapshot`.

    Both the TUI panels and the web Status page render from this single
    representation.
    """

    timestamp: datetime
    profile: str | None
    live: bool
    cycle_running: bool
    sysid_running: bool
    next_cycle_at: datetime | None
    sysid_age: timedelta | None
    collector_age: timedelta | None
    collector_rows: int
    local_tz: str
    outdoor_temp: float | None
    weather_condition: str
    temps: dict[str, float]                 # sensor_col → current value
    sensor_labels: dict[str, str]           # sensor_col → display label
    comfort: dict[str, ComfortBounds]       # label → bounds
    mrt_offsets: dict[str, float]           # sensor_col → offset
    environment: tuple[EnvEntry, ...]
    forecast: dict[str, float]              # "1h" → value
    effectors: tuple[EffectorView, ...]
    command_targets: dict[str, float]
    rationale: dict[str, str]
    costs: CycleCosts | None
    sensor_costs: dict[str, float]
    baseline_sensor_costs: dict[str, float]
    predictions: dict[str, dict[str, float]]
    opportunities: list[dict]
    warnings: list[dict]


@dataclass(frozen=True)
class ActionResult:
    """Outcome of a controller action.

    `status` is one of: "ok", "started", "noop", "already_running", "error".
    """

    status: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "started", "noop")


# ── Type aliases for callbacks ──────────────────────────────────────────────

LogFn = Callable[[str], None]
WorkerFn = Callable[[Callable[[], None]], None]
OnMainFn = Callable[[Callable[[], None]], None]
OnSnapshotFn = Callable[[StatusSnapshot], None]


def _noop_main(fn: Callable[[], None]) -> None:
    fn()


def _noop_snapshot(_snap: StatusSnapshot) -> None:
    return None


# ── Controller ──────────────────────────────────────────────────────────────


class TUIController:
    """Owns mutable state. Safe to call from any thread.

    Mutations of in-memory state happen under `self._lock` (a single RLock).
    The lock is **never** held across HA / executor / sysid I/O — only around
    dict/attr access. HTTP handlers, worker threads, and the Textual UI thread
    all coexist under this rule.
    """

    def __init__(
        self,
        *,
        live: bool,
        log: LogFn,
        worker: WorkerFn,
        on_snapshot: OnSnapshotFn = _noop_snapshot,
        on_main: OnMainFn = _noop_main,
    ) -> None:
        self._log = log
        self._worker = worker
        self._on_snapshot = on_snapshot
        self._on_main = on_main

        self._lock = threading.RLock()

        # Mutable state (always accessed under _lock)
        self._live = live
        self._cycle_running = False
        self._sysid_running = False
        self._latest_decision: Any = None
        self._overrides: dict[str, str] = {}
        self._next_cycle_at: datetime | None = None
        self._solar_elevation_gains: dict[str, float] = {}

        # Cached snapshot, refreshed by build_snapshot()
        self._latest_snapshot: StatusSnapshot | None = None

        self._load_solar_elevation_gains()

    # ── Read-only state access ──────────────────────────────────────────────

    @property
    def live_mode(self) -> bool:
        with self._lock:
            return self._live

    def is_cycle_running(self) -> bool:
        with self._lock:
            return self._cycle_running

    def is_sysid_running(self) -> bool:
        with self._lock:
            return self._sysid_running

    def overrides(self) -> dict[str, str]:
        with self._lock:
            return dict(self._overrides)

    def latest_snapshot(self) -> StatusSnapshot | None:
        with self._lock:
            return self._latest_snapshot

    def next_cycle_at(self) -> datetime | None:
        with self._lock:
            return self._next_cycle_at

    def set_next_cycle_at(self, when: datetime | None) -> None:
        with self._lock:
            self._next_cycle_at = when

    # ── Snapshot building ───────────────────────────────────────────────────

    def _load_solar_elevation_gains(self) -> None:
        from weatherstat.config import DATA_DIR

        params_file = DATA_DIR / "thermal_params.json"
        if not params_file.exists():
            return
        try:
            data = json.loads(params_file.read_text())
            with self._lock:
                self._solar_elevation_gains = data.get("solar_elevation_gains", {}) or {}
        except Exception as e:
            self._log(f"[config] [red]Failed to load solar gains: {e}[/]")

    def _read_sysid_age(self) -> timedelta | None:
        from weatherstat.config import DATA_DIR

        params_file = DATA_DIR / "thermal_params.json"
        if not params_file.exists():
            return None
        try:
            data = json.loads(params_file.read_text())
            ts = data.get("timestamp", "")
            if not ts:
                return None
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return datetime.now(UTC) - dt
        except Exception as e:
            self._log(f"[config] [red]Failed to read sysid age: {e}[/]")
            return None

    def _read_collector_status(self) -> tuple[timedelta | None, int]:
        try:
            from weatherstat.extract import snapshot_status

            ts, count = snapshot_status()
            if ts:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return (datetime.now(UTC) - dt, count)
            return (None, 0)
        except Exception as e:
            self._log(f"[collector] [red]Snapshot status error: {e}[/]")
            return (None, 0)

    def _read_advisory_state(self) -> tuple[list[dict], list[dict]]:
        from weatherstat.config import ADVISORY_STATE_FILE

        if not ADVISORY_STATE_FILE.exists():
            return ([], [])
        try:
            data = json.loads(ADVISORY_STATE_FILE.read_text())
            return (
                list(data.get("opportunities") or []),
                list(data.get("warnings") or []),
            )
        except Exception as e:
            self._log(f"[config] [red]Failed to load opportunities: {e}[/]")
            return ([], [])

    def build_snapshot(self) -> StatusSnapshot:
        """Read disk + memory and assemble a fresh `StatusSnapshot`.

        Safe to call from any thread. The lock is acquired only briefly to
        snapshot in-memory state — never across the disk reads or HA calls.
        """
        from weatherstat.config import (
            PREDICTION_SENSORS,
            SENSOR_LABELS,
            TIMEZONE,
            UNIT_SYMBOL,  # noqa: F401
        )
        from weatherstat.control import (
            apply_comfort_profile,
            apply_mrt_correction,
            default_comfort_schedules,
            fetch_active_comfort_profile,
        )
        from weatherstat.extract import latest_snapshot_values
        from weatherstat.yaml_config import load_config

        cfg = load_config()
        values = latest_snapshot_values()

        # Comfort schedules: base → profile offsets → MRT correction
        schedules = default_comfort_schedules()
        try:
            active_profile = fetch_active_comfort_profile()
        except Exception as e:
            self._log(f"[profile] [red]Failed to fetch active profile: {e}[/]")
            active_profile = None
        schedules = apply_comfort_profile(schedules, active_profile)
        profile_name = active_profile.name if active_profile else None

        # Outdoor temp
        outdoor_temp: float | None = None
        outdoor_col = cfg.outdoor_sensor
        for col in [outdoor_col, "met_outdoor_temp"]:
            if col and col in values:
                with contextlib.suppress(ValueError, TypeError):
                    outdoor_temp = float(values[col])
                    break

        # Sun-aware MRT correction
        mrt_per_sensor: dict[str, float] = {}
        if outdoor_temp is not None and cfg.mrt_correction:
            try:
                from weatherstat.weather import (
                    condition_to_solar_fraction as _csf,
                )
                from weatherstat.weather import (
                    solar_sin_elevation as _sse,
                )

                mrt_weights = {
                    c.sensor: c.mrt_weight
                    for c in cfg.constraints
                    if c.mrt_weight != 1.0
                }
                _now_utc = datetime.now(UTC)
                _solar_elev = _sse(cfg.location.latitude, cfg.location.longitude, _now_utc)
                _condition = values.get("weather_condition", "")
                _solar_frac = _csf(_condition) if _condition else 0.5
                with self._lock:
                    solar_gains = dict(self._solar_elevation_gains)
                schedules, _mrt_offset, mrt_per_sensor = apply_mrt_correction(
                    schedules,
                    outdoor_temp,
                    cfg.mrt_correction,
                    mrt_weights or None,
                    solar_elevation_gains=solar_gains or None,
                    current_solar_elev=_solar_elev,
                    current_solar_fraction=_solar_frac,
                )
            except Exception as e:
                self._log(f"[mrt] [red]MRT correction failed: {e}[/]")

        # Current temperatures
        temps: dict[str, float] = {}
        for sensor_col in PREDICTION_SENSORS:
            val = values.get(sensor_col)
            if val is not None:
                with contextlib.suppress(ValueError, TypeError):
                    temps[sensor_col] = float(val)

        # Comfort bounds at the current hour
        now_hour = datetime.now().hour
        comfort: dict[str, ComfortBounds] = {}
        for s in schedules:
            c = s.comfort_at(now_hour)
            if c is not None:
                comfort[s.label] = ComfortBounds(
                    acceptable_lo=c.acceptable_lo,
                    preferred_lo=c.preferred_lo,
                    preferred_hi=c.preferred_hi,
                    acceptable_hi=c.acceptable_hi,
                )

        # Environment factor states
        env_entries: list[EnvEntry] = []
        for env_cfg in cfg.environment.values():
            val = values.get(env_cfg.column)
            if val is not None:
                is_active = val in ("True", "true", "1", "1.0")
                env_entries.append(
                    EnvEntry(label=env_cfg.label, kind=env_cfg.kind, is_active=is_active)
                )

        # Forecast
        forecast: dict[str, float] = {}
        for h in [1, 2, 4, 6, 12]:
            val = values.get(f"forecast_temp_{h}h")
            if val is not None:
                with contextlib.suppress(ValueError, TypeError):
                    forecast[f"{h}h"] = float(val)
        weather_condition = values.get("weather_condition", "") or ""

        # Decision-derived data (mutable, under lock)
        with self._lock:
            decision = self._latest_decision
            cycle_running = self._cycle_running
            sysid_running = self._sysid_running
            live = self._live
            next_at = self._next_cycle_at
            override_map = dict(self._overrides)

        effectors: tuple[EffectorView, ...] = ()
        command_targets: dict[str, float] = {}
        rationale: dict[str, str] = {}
        costs: CycleCosts | None = None
        sensor_costs: dict[str, float] = {}
        baseline_sensor_costs: dict[str, float] = {}
        predictions: dict[str, dict[str, float]] = {}

        if decision is not None:
            try:
                from weatherstat.types import ControlDecision

                if isinstance(decision, ControlDecision):
                    effectors = tuple(
                        EffectorView(
                            name=ed.name,
                            mode=ed.mode,
                            target=ed.target,
                            delay_steps=ed.delay_steps,
                            duration_steps=ed.duration_steps,
                            override=override_map.get(ed.name),
                        )
                        for ed in decision.effectors
                    )
                    command_targets = dict(decision.command_targets)
                    rationale = dict(decision.rationale)
                    costs = CycleCosts(
                        total=decision.total_cost,
                        comfort=decision.comfort_cost,
                        energy=decision.energy_cost,
                        baseline=decision.baseline_cost or None,
                    )
                    sensor_costs = dict(decision.sensor_costs)
                    baseline_sensor_costs = dict(decision.baseline_sensor_costs)
                    predictions = {k: dict(v) for k, v in decision.predictions.items()}
            except Exception as e:
                self._log(f"[snapshot] [red]Decision serialization failed: {e}[/]")

        # Try to fall back to control_state.json if no in-memory decision yet
        if not effectors:
            from weatherstat.config import CONTROL_STATE_FILE

            if CONTROL_STATE_FILE.exists():
                try:
                    data = json.loads(CONTROL_STATE_FILE.read_text())
                    modes = data.get("modes", {})
                    setpoints = data.get("setpoints", {}) or {}
                    effectors = tuple(
                        EffectorView(
                            name=name,
                            mode=mode,
                            target=setpoints.get(name),
                            delay_steps=0,
                            duration_steps=None,
                            override=override_map.get(name),
                        )
                        for name, mode in modes.items()
                    )
                    command_targets = dict(setpoints)
                except Exception as e:
                    self._log(f"[config] [red]Failed to load control state: {e}[/]")

        opportunities, warnings = self._read_advisory_state()

        sysid_age = self._read_sysid_age()
        collector_age, collector_rows = self._read_collector_status()

        snap = StatusSnapshot(
            timestamp=datetime.now(UTC),
            profile=profile_name,
            live=live,
            cycle_running=cycle_running,
            sysid_running=sysid_running,
            next_cycle_at=next_at,
            sysid_age=sysid_age,
            collector_age=collector_age,
            collector_rows=collector_rows,
            local_tz=TIMEZONE or "",
            outdoor_temp=outdoor_temp,
            weather_condition=weather_condition,
            temps=temps,
            sensor_labels=dict(SENSOR_LABELS),
            comfort=comfort,
            mrt_offsets=mrt_per_sensor,
            environment=tuple(env_entries),
            forecast=forecast,
            effectors=effectors,
            command_targets=command_targets,
            rationale=rationale,
            costs=costs,
            sensor_costs=sensor_costs,
            baseline_sensor_costs=baseline_sensor_costs,
            predictions=predictions,
            opportunities=opportunities,
            warnings=warnings,
        )
        with self._lock:
            self._latest_snapshot = snap
        return snap

    def publish_snapshot(self) -> StatusSnapshot:
        """Build a snapshot and notify listeners. Returns the snapshot."""
        snap = self.build_snapshot()
        try:
            self._on_snapshot(snap)
        except Exception as e:
            self._log(f"[snapshot] [red]on_snapshot failed: {e}[/]")
        return snap

    # ── Actions ─────────────────────────────────────────────────────────────

    def toggle_live(self) -> ActionResult:
        with self._lock:
            self._live = not self._live
            new = self._live
        msg = f"Switched to {'LIVE' if new else 'DRY-RUN'} mode"
        self._log(f"[control] {msg}")
        self.publish_snapshot()
        return ActionResult("ok", msg, {"live": new})

    def set_live(self, live: bool) -> ActionResult:
        with self._lock:
            if self._live == live:
                return ActionResult("noop", f"Already in {'LIVE' if live else 'DRY-RUN'} mode")
            self._live = live
        msg = f"Switched to {'LIVE' if live else 'DRY-RUN'} mode"
        self._log(f"[control] {msg}")
        self.publish_snapshot()
        return ActionResult("ok", msg, {"live": live})

    def run_cycle(self) -> ActionResult:
        with self._lock:
            if self._cycle_running:
                return ActionResult("already_running", "Control cycle already running")
            self._cycle_running = True
        self._worker(self._cycle_worker)
        return ActionResult("started", "Control cycle started")

    def _cycle_worker(self) -> None:
        live = self.live_mode
        self._log(f"\n[control] Starting {'LIVE' if live else 'DRY-RUN'} control cycle...")
        self.publish_snapshot()

        decision = None
        buf = io.StringIO()
        try:
            from weatherstat.control import run_control_cycle

            with contextlib.redirect_stdout(buf):
                decision = run_control_cycle(live=live)
        except Exception as e:
            self._log(f"[control] [red]Error: {e}[/]")
            self._log(traceback.format_exc())

        for line in buf.getvalue().splitlines():
            self._log(line)

        if decision is None:
            self._log("[control] No decision returned (skipped or error)")
        else:
            with self._lock:
                self._latest_decision = decision
            if live:
                self._run_executor_inline(force=False)

        with self._lock:
            self._cycle_running = False
            self._next_cycle_at = datetime.now(UTC) + timedelta(
                seconds=_control_interval()
            )
        self.publish_snapshot()

    def run_sysid(self) -> ActionResult:
        with self._lock:
            if self._sysid_running:
                return ActionResult("already_running", "Sysid already running")
            self._sysid_running = True
        self._worker(self._sysid_worker)
        return ActionResult("started", "Sysid started")

    def _sysid_worker(self) -> None:
        self._log("\n[sysid] Starting system identification...")
        self.publish_snapshot()

        buf = io.StringIO()
        try:
            from weatherstat.sysid import fit_sysid, save_sysid_result

            with contextlib.redirect_stdout(buf):
                result, diagnostics = fit_sysid()

            n_taus = len(result.fitted_taus)
            n_gains = sum(1 for g in result.effector_sensor_gains if not g.negligible)
            if n_taus == 0:
                self._log("[sysid] [red]Rejected: no sensors fitted[/]")
            elif n_gains == 0:
                self._log("[sysid] [red]Rejected: no significant gains found[/]")
            else:
                with contextlib.redirect_stdout(buf):
                    save_sysid_result(result, sensor_diagnostics=diagnostics)
                self._log(
                    f"[sysid] Complete. {result.n_snapshots} snapshots,"
                    f" {n_taus} sensors, {n_gains} gains."
                )
        except Exception as e:
            self._log(f"[sysid] [red]Error: {e}[/]")
            self._log(traceback.format_exc())

        for line in buf.getvalue().splitlines():
            self._log(f"[sysid] {line}")

        self._load_solar_elevation_gains()
        with self._lock:
            self._sysid_running = False
        self.publish_snapshot()

    def force_execute(self) -> ActionResult:
        with self._lock:
            override_count = len(self._overrides)
            override_names = list(self._overrides)
        if override_count == 0:
            return ActionResult("noop", "No overrides detected")
        self._worker(lambda: self._run_executor_inline(force=True))
        return ActionResult(
            "started",
            f"Force-executing on {override_count} effector(s)",
            {"effectors": override_names},
        )

    def _run_executor_inline(self, *, force: bool) -> None:
        try:
            from weatherstat.executor import execute

            result = execute(force=force, log=self._log)
            with self._lock:
                self._overrides = dict(result.overrides)
        except Exception as e:
            self._log(f"[executor] [red]Error: {e}[/]")
            self._log(traceback.format_exc())
        self.publish_snapshot()

    def toggle_profile(self) -> ActionResult:
        return self._profile_action(target=None)

    def set_profile(self, name: str) -> ActionResult:
        return self._profile_action(target=name)

    def _profile_action(self, *, target: str | None) -> ActionResult:
        from weatherstat.yaml_config import load_config

        cfg = load_config()
        entity_id = cfg.comfort_entity
        if entity_id is None:
            msg = "comfort_entity not configured in weatherstat.yaml"
            self._log(f"[profile] [red]{msg}[/]")
            return ActionResult("error", msg)

        try:
            import os

            import requests

            ha_url = os.environ.get("HA_URL", "")
            ha_token = os.environ.get("HA_TOKEN", "")
            if not ha_url or not ha_token:
                msg = "HA_URL/HA_TOKEN not set"
                self._log(f"[profile] [red]{msg}[/]")
                return ActionResult("error", msg)

            headers = {
                "Authorization": f"Bearer {ha_token}",
                "Content-Type": "application/json",
            }

            resp = requests.get(
                f"{ha_url}/api/states/{entity_id}",
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
            payload = resp.json()
            current = payload.get("state", "")
            options = payload.get("attributes", {}).get("options") or list(
                cfg.comfort_profiles
            )

            if target is None:
                if current in options:
                    next_profile = options[(options.index(current) + 1) % len(options)]
                else:
                    next_profile = options[0] if options else current
            else:
                if target not in options:
                    msg = f"Profile '{target}' not in {options}"
                    self._log(f"[profile] [red]{msg}[/]")
                    return ActionResult("error", msg)
                next_profile = target

            requests.post(
                f"{ha_url}/api/services/input_select/select_option",
                headers=headers,
                json={"entity_id": entity_id, "option": next_profile},
                timeout=10,
            )
            self._log(f"[profile] Switched to {next_profile}")
            self.publish_snapshot()
            return ActionResult("ok", f"Switched to {next_profile}", {"profile": next_profile})
        except Exception as e:
            self._log(f"[profile] [red]Error: {e}[/]")
            self._log(traceback.format_exc())
            return ActionResult("error", str(e))


def _control_interval() -> int:
    """Imported lazily so config-reload sees the new value."""
    from weatherstat.config import CONTROL_INTERVAL

    return int(CONTROL_INTERVAL)
