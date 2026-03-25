"""Weatherstat TUI application — unified HVAC dashboard."""

from __future__ import annotations

import contextlib
import io
import json
import traceback
from datetime import UTC, datetime, timedelta

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Static, TabbedContent, TabPane

from weatherstat.tui.widgets import (
    AccuracyPanel,
    EffectorPanel,
    ForecastPanel,
    HealthPanel,
    HistoryPanel,
    LogView,
    OpportunityPanel,
    PredictionPanel,
    StatusHeader,
    TemperaturePanel,
    WindowPanel,
)

# Intervals
MONITOR_INTERVAL = 30  # seconds
SNAPSHOT_INTERVAL = 300  # 5 minutes
CONTROL_INTERVAL = 900  # 15 minutes
SYSID_TIMEOUT = 300  # 5 minutes


# ── Confirmation modal ──────────────────────────────────────────────────────


class ConfirmScreen(ModalScreen[bool]):
    """Simple yes/no confirmation dialog."""

    BINDINGS = [
        Binding("y", "confirm", "Yes"),
        Binding("n", "cancel", "No"),
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        yield Static(f"\n  {self._message}\n\n  [bold]Y[/]es / [bold]N[/]o\n", id="confirm-dialog")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


# ── Main App ────────────────────────────────────────────────────────────────


class WeatherstatApp(App):
    """Weatherstat HVAC control dashboard."""

    TITLE = "Weatherstat"
    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("1", "tab_1", "Status", show=False),
        Binding("2", "tab_2", "Predictions", show=False),
        Binding("3", "tab_3", "History", show=False),
        Binding("4", "tab_4", "Log", show=False),
        Binding("l", "toggle_live", "Live/Dry"),
        Binding("r", "run_cycle", "Run Now"),
        Binding("s", "run_sysid", "Sysid"),
        Binding("p", "toggle_profile", "Profile"),
        Binding("f", "force_execute", "Force Execute"),
        Binding("q", "quit_app", "Quit"),
        Binding("question_mark", "help", "Help"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.live_mode = False
        self._cycle_running = False
        self._sysid_running = False
        self._cycle_timer: object | None = None
        self._next_cycle_at: datetime | None = None

        # Latest decision (from control cycle)
        self._latest_decision: object | None = None
        self._baseline_cost: float | None = None
        self._overrides: dict[str, str] = {}  # effector_name -> description

        # MRT weights from sysid (loaded once at startup, refreshed after sysid runs)
        self._mrt_weights_cache: dict[str, float] = {}
        self._load_mrt_weights()

    def compose(self) -> ComposeResult:
        yield StatusHeader()
        with TabbedContent(initial="status"):
            with TabPane("Status", id="status"):  # noqa: SIM117
                with Horizontal(id="status-grid"):
                    with Vertical(id="left-col"):
                        yield TemperaturePanel()
                        yield WindowPanel()
                        yield ForecastPanel()
                    with Vertical(id="right-col"):
                        yield EffectorPanel()
                        yield OpportunityPanel()
                        yield HealthPanel()
            with TabPane("Predictions", id="predictions"):
                yield PredictionPanel()
            with TabPane("History", id="history"):  # noqa: SIM117
                with Vertical(id="history-content"):
                    yield AccuracyPanel()
                    yield HistoryPanel()
            with TabPane("Log", id="log"):
                yield LogView()
        yield Footer()

    def on_mount(self) -> None:
        self._log("Weatherstat TUI starting...")
        self._load_initial_state()
        self._collect_snapshot()  # initial snapshot
        self.set_interval(MONITOR_INTERVAL, self._monitor_tick)
        self.set_interval(SNAPSHOT_INTERVAL, self._collect_snapshot)
        self._schedule_next_cycle()

    # ── Tab switching ───────────────────────────────────────────────────────

    def action_tab_1(self) -> None:
        self.query_one(TabbedContent).active = "status"

    def action_tab_2(self) -> None:
        self.query_one(TabbedContent).active = "predictions"

    def action_tab_3(self) -> None:
        self.query_one(TabbedContent).active = "history"
        self._refresh_history()

    def action_tab_4(self) -> None:
        self.query_one(TabbedContent).active = "log"

    # ── Key actions ─────────────────────────────────────────────────────────

    def action_toggle_live(self) -> None:
        if self.live_mode:
            self.live_mode = False
            self._log("[control] Switched to DRY-RUN mode")
            self._update_header()
        else:
            self.push_screen(
                ConfirmScreen("Enable LIVE execution? Commands will be sent to Home Assistant."),
                self._on_live_confirmed,
            )

    def _on_live_confirmed(self, confirmed: bool) -> None:
        if confirmed:
            self.live_mode = True
            self._log("[control] Switched to LIVE mode")
            self._update_header()

    def action_force_execute(self) -> None:
        if not self._overrides:
            self.notify("No overrides detected", severity="information")
            return
        names = ", ".join(self._overrides)
        self.push_screen(
            ConfirmScreen(f"Force-execute past overrides on: {names}?"),
            self._on_force_confirmed,
        )

    def _on_force_confirmed(self, confirmed: bool) -> None:
        if confirmed:
            self._force_execute()

    @work(thread=True)
    def _force_execute(self) -> None:
        self._run_executor(force=True)

    def action_run_cycle(self) -> None:
        if self._cycle_running:
            self.notify("Control cycle already running", severity="warning")
            return
        self._run_control_cycle()

    def action_run_sysid(self) -> None:
        if self._sysid_running:
            self.notify("Sysid already running", severity="warning")
            return
        self._run_sysid()

    def action_toggle_profile(self) -> None:
        self._toggle_profile()

    def action_quit_app(self) -> None:
        if self.live_mode:
            self.push_screen(
                ConfirmScreen("Live mode is active. Control loop will stop. Quit?"),
                self._on_quit_confirmed,
            )
        else:
            self.exit()

    def _on_quit_confirmed(self, confirmed: bool) -> None:
        if confirmed:
            self.exit()

    def action_help(self) -> None:
        help_text = (
            "[bold]Keybindings[/]\n\n"
            "  1-4    Switch tabs (Status, Predictions, History, Log)\n"
            "  L      Toggle Live / Dry-run mode\n"
            "  R      Run control cycle now\n"
            "  F      Force-execute past overrides\n"
            "  S      Run system identification (sysid)\n"
            "  P      Toggle comfort profile (Home/Away)\n"
            "  Q      Quit\n"
            "  ?      This help\n"
        )
        self.notify(help_text, title="Help", timeout=10)

    # ── Initial state loading ───────────────────────────────────────────────

    def _load_initial_state(self) -> None:
        self._monitor_tick()
        self._load_sysid_status()
        self._load_control_state()
        self._refresh_history()

    def _load_mrt_weights(self) -> None:
        """Load sysid-derived MRT weights from thermal_params.json."""
        from weatherstat.config import DATA_DIR

        params_file = DATA_DIR / "thermal_params.json"
        if params_file.exists():
            try:
                data = json.loads(params_file.read_text())
                self._mrt_weights_cache = data.get("mrt_weights", {})
            except Exception:
                pass

    def _load_sysid_status(self) -> None:
        from weatherstat.config import DATA_DIR

        params_file = DATA_DIR / "thermal_params.json"
        if params_file.exists():
            try:
                data = json.loads(params_file.read_text())
                ts = data.get("timestamp", "")
                if ts:
                    dt = datetime.fromisoformat(ts)
                    age = datetime.now(UTC) - dt.replace(tzinfo=UTC) if dt.tzinfo is None else datetime.now(UTC) - dt
                    self.query_one(StatusHeader).set_state(sysid_age=_format_age(age))
            except Exception:
                pass

    def _load_control_state(self) -> None:
        from weatherstat.config import ADVISORY_STATE_FILE, CONTROL_STATE_FILE

        # Load effector state from control_state.json
        if CONTROL_STATE_FILE.exists():
            try:
                data = json.loads(CONTROL_STATE_FILE.read_text())
                decisions = []
                modes = data.get("modes", {})
                setpoints = data.get("setpoints", {})
                for name, mode in modes.items():
                    decisions.append({"name": name, "mode": mode})
                self.query_one(EffectorPanel).set_data(
                    decisions=decisions,
                    command_targets=setpoints,
                )
            except Exception:
                pass

        # Load opportunities from advisory_state.json
        if ADVISORY_STATE_FILE.exists():
            try:
                data = json.loads(ADVISORY_STATE_FILE.read_text())
                active = data.get("active", {})
                self.query_one(OpportunityPanel).set_data(list(active.values()))
            except Exception:
                pass

    # ── Collector (5-min) ─────────────────────────────────────────────────

    @work(thread=True)
    def _collect_snapshot(self) -> None:
        try:
            from weatherstat.collector import collect_once

            collect_once(log=self._log)
        except Exception as e:
            self._log(f"[collector] [red]Error: {e}[/]")

    # ── Monitor timer (30s) ─────────────────────────────────────────────────

    def _monitor_tick(self) -> None:
        self._refresh_snapshot_status()
        self._refresh_temps()
        self._update_header()

    def _refresh_snapshot_status(self) -> None:
        try:
            from weatherstat.extract import snapshot_status

            ts, count = snapshot_status()
            header = self.query_one(StatusHeader)
            if ts:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                age = datetime.now(UTC) - dt
                header.set_state(collector_age=_format_age(age), collector_rows=count)
            else:
                header.set_state(collector_age="no data", collector_rows=0)
        except Exception:
            pass

    def _refresh_temps(self) -> None:
        try:
            from weatherstat.config import PREDICTION_SENSORS, SENSOR_LABELS
            from weatherstat.control import (
                apply_comfort_profile,
                apply_mrt_correction,
                default_comfort_schedules,
                fetch_active_comfort_profile,
            )
            from weatherstat.extract import latest_snapshot_values
            from weatherstat.yaml_config import load_config

            values = latest_snapshot_values()
            if not values:
                return

            cfg = load_config()

            # Current temperatures for constraint sensors
            temps: dict[str, float] = {}
            for sensor_col in PREDICTION_SENSORS:
                val = values.get(sensor_col)
                if val is not None:
                    with contextlib.suppress(ValueError, TypeError):
                        temps[sensor_col] = float(val)

            # Comfort schedules: base → profile offsets → MRT correction
            schedules = default_comfort_schedules()
            active_profile = fetch_active_comfort_profile()
            schedules = apply_comfort_profile(schedules, active_profile)
            if active_profile is not None:
                self.query_one(StatusHeader).set_state(profile=active_profile.name)

            # Outdoor temp (needed for MRT and header)
            outdoor_col = cfg.outdoor_sensor
            outdoor_temp = None
            for col in [outdoor_col, "met_outdoor_temp"]:
                if col and col in values:
                    with contextlib.suppress(ValueError, TypeError):
                        outdoor_temp = float(values[col])
                        break
            if outdoor_temp is not None:
                self.query_one(StatusHeader).set_state(outdoor_temp=outdoor_temp)

            # MRT correction using outdoor temp and sysid-derived weights
            if outdoor_temp is not None and cfg.mrt_correction:
                mrt_weights: dict[str, float] = {}
                for c in cfg.constraints:
                    w = c.mrt_weight
                    if w == 1.0:
                        # Use sysid-derived weight if available
                        w = self._mrt_weights_cache.get(c.sensor, 1.0)
                    mrt_weights[c.sensor] = w
                schedules, _mrt_offset = apply_mrt_correction(
                    schedules, outdoor_temp, cfg.mrt_correction, mrt_weights
                )

            # Extract comfort bounds from adjusted schedules
            now_hour = datetime.now().hour
            comfort: dict[str, tuple[float, float, float]] = {}
            for s in schedules:
                c = s.comfort_at(now_hour)
                if c is not None:
                    comfort[s.label] = (c.min_temp, c.preferred, c.max_temp)

            self.query_one(TemperaturePanel).set_data(temps, comfort, SENSOR_LABELS)

            # Window states
            windows: dict[str, bool] = {}
            for col_name, display_name in cfg.window_display_map.items():
                val = values.get(col_name)
                if val is not None:
                    windows[display_name] = val in ("True", "true", "1", "1.0")
            self.query_one(WindowPanel).set_data(windows)

            # Forecast
            forecasts: dict[str, float] = {}
            for h in [1, 2, 4, 6, 12]:
                val = values.get(f"forecast_temp_{h}h")
                if val is not None:
                    with contextlib.suppress(ValueError, TypeError):
                        forecasts[f"{h}h"] = float(val)
            condition = values.get("weather_condition", "")
            self.query_one(ForecastPanel).set_data(forecasts, condition)

        except Exception as e:
            self._log(f"[monitor] Error refreshing temps: {e}")

    def _update_header(self) -> None:
        header = self.query_one(StatusHeader)
        header.set_state(live=self.live_mode)
        if self._next_cycle_at:
            remaining = self._next_cycle_at - datetime.now(UTC)
            secs = max(0, int(remaining.total_seconds()))
            header.set_state(next_cycle=f"{secs // 60}m")
        try:
            from weatherstat.config import TIMEZONE

            header.set_state(local_tz=TIMEZONE)
        except Exception:
            pass

    # ── Control cycle worker ────────────────────────────────────────────────

    def _schedule_next_cycle(self) -> None:
        self._next_cycle_at = datetime.now(UTC) + timedelta(seconds=CONTROL_INTERVAL)
        if self._cycle_timer is not None:
            # Cancel doesn't exist on Timer, but we track via _next_cycle_at
            pass
        self._cycle_timer = self.set_timer(CONTROL_INTERVAL, self._auto_cycle)

    def _auto_cycle(self) -> None:
        if not self._cycle_running:
            self._run_control_cycle()
        self._schedule_next_cycle()

    @work(thread=True)
    def _run_control_cycle(self) -> None:
        self._cycle_running = True
        self.call_from_thread(self._update_header_running, True)
        self._log(f"\n[control] Starting {'LIVE' if self.live_mode else 'DRY-RUN'} control cycle...")

        buf = io.StringIO()
        decision = None
        try:
            from weatherstat.control import run_control_cycle

            with contextlib.redirect_stdout(buf):
                decision = run_control_cycle(live=self.live_mode)
        except Exception as e:
            self._log(f"[control] [red]Error: {e}[/]")
            self._log(traceback.format_exc())

        # Log captured output
        output = buf.getvalue()
        if output:
            for line in output.splitlines():
                self._log(line)

        if decision:
            self._latest_decision = decision
            self.call_from_thread(self._update_from_decision, decision)

            # Execute via TS executor in live mode
            if self.live_mode:
                self._run_executor(force=False)
        else:
            self._log("[control] No decision returned (skipped or error)")

        self._cycle_running = False
        self.call_from_thread(self._update_header_running, False)
        self.call_from_thread(self._monitor_tick)

    def _update_header_running(self, running: bool) -> None:
        self.query_one(StatusHeader).set_state(cycle_running=running)

    def _update_from_decision(self, decision: object) -> None:
        from weatherstat.types import ControlDecision

        if not isinstance(decision, ControlDecision):
            return

        from weatherstat.config import SENSOR_LABELS

        # Update effector panel
        eff_dicts = [
            {"name": e.name, "mode": e.mode, "target": e.target, "delay_steps": e.delay_steps, "duration_steps": e.duration_steps}
            for e in decision.effectors
        ]
        self.query_one(EffectorPanel).set_data(
            decisions=eff_dicts,
            command_targets=decision.command_targets,
            costs=(decision.total_cost, decision.comfort_cost, decision.energy_cost),
            baseline_cost=self._baseline_cost,
        )

        # Update predictions panel
        self.query_one(PredictionPanel).set_data(decision.predictions, SENSOR_LABELS)

        # Refresh opportunities from state file
        self._load_control_state()

    # ── Executor ───────────────────────────────────────────────────────────

    def _run_executor(self, *, force: bool = False) -> None:
        """Execute latest command via HA REST API. Called from worker threads."""
        from weatherstat.executor import execute

        result = execute(force=force, log=self._log)
        self._overrides = result.overrides
        self.call_from_thread(self._update_override_display)

    def _update_override_display(self) -> None:
        self.query_one(EffectorPanel).set_overrides(self._overrides)

    # ── Sysid worker ────────────────────────────────────────────────────────

    @work(thread=True)
    def _run_sysid(self) -> None:
        self._sysid_running = True
        self.call_from_thread(self.query_one(StatusHeader).set_state, sysid_running=True)
        self._log("\n[sysid] Starting system identification...")

        buf = io.StringIO()
        try:
            from weatherstat.sysid import run_sysid

            with contextlib.redirect_stdout(buf):
                result = run_sysid()
            self._log(f"[sysid] Complete. {result.n_snapshots} snapshots, {len(result.fitted_taus)} sensors fitted.")
        except Exception as e:
            self._log(f"[sysid] [red]Error: {e}[/]")
            self._log(traceback.format_exc())

        output = buf.getvalue()
        if output:
            for line in output.splitlines():
                self._log(f"[sysid] {line}")

        self._sysid_running = False
        self._load_mrt_weights()  # refresh sysid-derived MRT weights
        self.call_from_thread(self.query_one(StatusHeader).set_state, sysid_running=False)
        self.call_from_thread(self._load_sysid_status)

    # ── Profile toggle ──────────────────────────────────────────────────────

    @work(thread=True)
    def _toggle_profile(self) -> None:
        try:
            import os

            import requests

            ha_url = os.environ.get("HA_URL", "")
            ha_token = os.environ.get("HA_TOKEN", "")
            if not ha_url or not ha_token:
                self._log("[profile] HA_URL/HA_TOKEN not set")
                return

            headers = {"Authorization": f"Bearer {ha_token}", "Content-Type": "application/json"}
            entity_id = "input_select.thermostat_mode"

            # Get current state
            resp = requests.get(f"{ha_url}/api/states/{entity_id}", headers=headers, timeout=10)
            resp.raise_for_status()
            current = resp.json().get("state", "Home")
            options = resp.json().get("attributes", {}).get("options", ["Home", "Away"])

            # Cycle to next
            idx = options.index(current) if current in options else 0
            next_profile = options[(idx + 1) % len(options)]

            # Set it
            requests.post(
                f"{ha_url}/api/services/input_select/select_option",
                headers=headers,
                json={"entity_id": entity_id, "option": next_profile},
                timeout=10,
            )
            self._log(f"[profile] Switched to {next_profile}")
            self.call_from_thread(self.query_one(StatusHeader).set_state, profile=next_profile)

        except Exception as e:
            self._log(f"[profile] [red]Error: {e}[/]")

    # ── History tab ─────────────────────────────────────────────────────────

    def _refresh_history(self) -> None:
        try:
            from weatherstat.decision_log import accuracy_summary, load_decision_log

            df = load_decision_log(limit=50)
            if not df.empty:
                rows = df.to_dict("records")
                self.query_one(HistoryPanel).set_data(rows)

            summary = accuracy_summary(hours=24)
            self.query_one(AccuracyPanel).set_data(summary)
        except Exception as e:
            self._log(f"[history] Error: {e}")

    # ── Logging ─────────────────────────────────────────────────────────────

    def _log(self, message: str) -> None:
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            log_widget = self.query_one(LogView)
            log_widget.write(f"[dim]{ts}[/] {message}")
        except Exception:
            pass


# ── Utilities ───────────────────────────────────────────────────────────────


def _format_age(td: timedelta) -> str:
    """Format a timedelta as a human-readable age string."""
    total_seconds = int(td.total_seconds())
    if total_seconds < 0:
        return "future?"
    if total_seconds < 60:
        return f"{total_seconds}s"
    minutes = total_seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    days = hours // 24
    return f"{days}d"
