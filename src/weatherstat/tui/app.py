"""Weatherstat TUI application — unified HVAC dashboard.

`WeatherstatApp` is a thin Textual frontend over `TUIController`. The
controller owns all mutable runtime state and all business logic; the app
owns widgets, key bindings, and the Textual event loop. See
`controller.py` for the data flow.
"""

from __future__ import annotations

import threading
import traceback
from datetime import UTC, datetime, timedelta

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Static, TabbedContent, TabPane

from weatherstat.config import CONTROL_INTERVAL, SYSID_INTERVAL
from weatherstat.tui.controller import ActionResult, StatusSnapshot, TUIController
from weatherstat.tui.widgets import (
    AccuracyPanel,
    EffectorPanel,
    EnvironmentPanel,
    ForecastPanel,
    HealthPanel,
    HistoryPanel,
    LogView,
    OpportunityPanel,
    PredictionPanel,
    StatusHeader,
    TemperaturePanel,
)

# Intervals
MONITOR_INTERVAL = 30  # seconds
SNAPSHOT_INTERVAL = 300  # 5 minutes
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
        Binding("c", "reload_config", "Reload Config"),
        Binding("q", "quit_app", "Quit"),
        Binding("question_mark", "help", "Help"),
    ]

    def __init__(
        self,
        live: bool = False,
        *,
        web_enabled: bool = False,
        web_host: str = "0.0.0.0",
        web_port: int = 8765,
    ) -> None:
        super().__init__()
        self._cycle_timer: object | None = None
        self._sysid_timer: object | None = None
        self._web_enabled = web_enabled
        self._web_host = web_host
        self._web_port = web_port
        self._web_server: object | None = None

        self.controller = TUIController(
            live=live,
            log=self._log,
            worker=_spawn_worker,
            on_snapshot=self._on_snapshot,
            on_main=self.call_from_thread,
        )

    def compose(self) -> ComposeResult:
        yield StatusHeader()
        with TabbedContent(initial="status"):
            with TabPane("Status", id="status"):  # noqa: SIM117
                with Horizontal(id="status-grid"):
                    with Vertical(id="left-col"):
                        yield TemperaturePanel()
                        yield EnvironmentPanel()
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
        self._log(f"  Control interval: {CONTROL_INTERVAL}s, Sysid interval: {SYSID_INTERVAL}s")

        # Initial population — synchronous from UI thread
        self._refresh_snapshot_now()
        self._collect_snapshot()
        self._refresh_history()

        # Periodic timers
        self.set_interval(MONITOR_INTERVAL, self._monitor_tick)
        self.set_interval(SNAPSHOT_INTERVAL, self._collect_snapshot)
        self._schedule_next_cycle()
        self._schedule_sysid()

        # Optional embedded web server
        if self._web_enabled:
            self._start_web_server()

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
        result = self.controller.toggle_live()
        self._notify_result(result)

    def action_force_execute(self) -> None:
        result = self.controller.force_execute()
        if result.status == "noop":
            self.notify(result.message, severity="information")
            return
        self._notify_result(result)

    def action_run_cycle(self) -> None:
        result = self.controller.run_cycle()
        if result.status == "already_running":
            self.notify(result.message, severity="warning")
            return
        self._notify_result(result)

    def action_run_sysid(self) -> None:
        result = self.controller.run_sysid()
        if result.status == "already_running":
            self.notify(result.message, severity="warning")
            return
        self._notify_result(result)

    def action_toggle_profile(self) -> None:
        _spawn_worker(lambda: self._notify_result_main(self.controller.toggle_profile()))

    def action_reload_config(self) -> None:
        """Reload weatherstat.yaml and refresh all config-derived state."""
        import importlib

        import weatherstat.config
        import weatherstat.control
        import weatherstat.extract
        import weatherstat.types
        import weatherstat.yaml_config

        try:
            importlib.reload(weatherstat.types)
            importlib.reload(weatherstat.yaml_config)
            importlib.reload(weatherstat.config)
            importlib.reload(weatherstat.extract)
            importlib.reload(weatherstat.control)
            self.controller.publish_snapshot()
            self._log("[config] Reloaded weatherstat.yaml")
            self.notify("Config reloaded", severity="information")
        except Exception as e:
            self._log(f"[config] [red]Reload failed: {e}[/]")
            self.notify(f"Config reload failed: {e}", severity="error")

    def action_quit_app(self) -> None:
        if self.controller.live_mode:
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
            "  C      Reload weatherstat.yaml\n"
            "  S      Run system identification (sysid)\n"
            "  P      Toggle comfort profile (Home/Away)\n"
            "  Q      Quit\n"
            "  ?      This help\n"
        )
        self.notify(help_text, title="Help", timeout=10)

    # ── Notification helpers ───────────────────────────────────────────────

    def _notify_result(self, result: ActionResult) -> None:
        severity = "information" if result.ok else "error"
        self.notify(result.message, severity=severity)

    def _notify_result_main(self, result: ActionResult) -> None:
        self.call_from_thread(self._notify_result, result)

    # ── Snapshot pump ───────────────────────────────────────────────────────

    def _on_snapshot(self, snap: StatusSnapshot) -> None:
        """Called by the controller (possibly from a worker thread)."""
        self.call_from_thread(self._apply_snapshot, snap)

    def _refresh_snapshot_now(self) -> None:
        """Build and apply a snapshot synchronously. Call from UI thread only."""
        try:
            snap = self.controller.build_snapshot()
            self._apply_snapshot(snap)
        except Exception as e:
            self._log(f"[snapshot] [red]Refresh error: {e}[/]")
            self._log(traceback.format_exc())

    def _apply_snapshot(self, snap: StatusSnapshot) -> None:
        """Push a snapshot into all widgets. Always runs on the UI thread."""
        try:
            self._update_header_from_snapshot(snap)
            self._update_temperatures(snap)
            self._update_environment(snap)
            self._update_forecast(snap)
            self._update_effectors(snap)
            self._update_opportunities(snap)
            self._update_predictions(snap)
        except Exception as e:
            self._log(f"[snapshot] [red]Apply failed: {e}[/]")
            self._log(traceback.format_exc())

    def _update_header_from_snapshot(self, snap: StatusSnapshot) -> None:
        header = self.query_one(StatusHeader)
        next_cycle_str = "?"
        if snap.next_cycle_at is not None:
            remaining = snap.next_cycle_at - datetime.now(UTC)
            secs = max(0, int(remaining.total_seconds()))
            next_cycle_str = f"{secs // 60}m"
        header.set_state(
            profile=snap.profile or "?",
            live=snap.live,
            collector_age=_format_age(snap.collector_age) if snap.collector_age is not None else "no data",
            collector_rows=snap.collector_rows,
            sysid_age=_format_age(snap.sysid_age) if snap.sysid_age is not None else "?",
            next_cycle=next_cycle_str,
            outdoor_temp=snap.outdoor_temp,
            cycle_running=snap.cycle_running,
            sysid_running=snap.sysid_running,
            local_tz=snap.local_tz,
        )

    def _update_temperatures(self, snap: StatusSnapshot) -> None:
        comfort_tuples = {
            label: (
                cb.acceptable_lo,
                cb.preferred_lo,
                cb.preferred_hi,
                cb.acceptable_hi,
            )
            for label, cb in snap.comfort.items()
        }
        self.query_one(TemperaturePanel).set_data(
            snap.temps,
            comfort_tuples,
            snap.sensor_labels,
            mrt_offsets=snap.mrt_offsets or None,
        )
        self.query_one(EffectorPanel).set_current_temps(snap.temps)

    def _update_environment(self, snap: StatusSnapshot) -> None:
        entries = [(e.label, e.kind, e.is_active) for e in snap.environment]
        self.query_one(EnvironmentPanel).set_data(entries)

    def _update_forecast(self, snap: StatusSnapshot) -> None:
        self.query_one(ForecastPanel).set_data(snap.forecast, snap.weather_condition)

    def _update_effectors(self, snap: StatusSnapshot) -> None:
        if not snap.effectors:
            return
        decisions = [
            {
                "name": e.name,
                "mode": e.mode,
                "target": e.target,
                "delay_steps": e.delay_steps,
                "duration_steps": e.duration_steps,
            }
            for e in snap.effectors
        ]
        costs_tuple = (0.0, 0.0, 0.0)
        baseline = None
        if snap.costs is not None:
            costs_tuple = (snap.costs.total, snap.costs.comfort, snap.costs.energy)
            baseline = snap.costs.baseline
        panel = self.query_one(EffectorPanel)
        panel.set_data(
            decisions=decisions,
            command_targets=snap.command_targets,
            costs=costs_tuple,
            baseline_cost=baseline,
            rationale=snap.rationale,
            sensor_costs=snap.sensor_costs,
            baseline_sensor_costs=snap.baseline_sensor_costs,
        )
        # Override map (mirrored from controller)
        overrides = {e.name: e.override for e in snap.effectors if e.override}
        panel.set_overrides(overrides)

    def _update_opportunities(self, snap: StatusSnapshot) -> None:
        self.query_one(OpportunityPanel).set_data(
            opportunities=snap.opportunities or None,
            warnings=snap.warnings or None,
        )

    def _update_predictions(self, snap: StatusSnapshot) -> None:
        if snap.predictions:
            self.query_one(PredictionPanel).set_data(snap.predictions, snap.sensor_labels)

    # ── Periodic ticks ──────────────────────────────────────────────────────

    def _monitor_tick(self) -> None:
        """Refresh from disk + memory and push to widgets."""
        self._refresh_snapshot_now()

    def _collect_snapshot(self) -> None:
        """Run the collector once in a worker thread (5-min cadence)."""
        def _run() -> None:
            try:
                from weatherstat.collector import collect_once

                collect_once(log=self._log)
            except Exception as e:
                self._log(f"[collector] [red]Error: {e}[/]")
                self._log(traceback.format_exc())

        _spawn_worker(_run)

    def _schedule_next_cycle(self) -> None:
        when = datetime.now(UTC) + timedelta(seconds=CONTROL_INTERVAL)
        self.controller.set_next_cycle_at(when)
        self._cycle_timer = self.set_timer(CONTROL_INTERVAL, self._auto_cycle)

    def _auto_cycle(self) -> None:
        if not self.controller.is_cycle_running():
            self.controller.run_cycle()
        self._schedule_next_cycle()

    def _schedule_sysid(self) -> None:
        if SYSID_INTERVAL <= 0:
            return
        self._sysid_timer = self.set_timer(SYSID_INTERVAL, self._auto_sysid)

    def _auto_sysid(self) -> None:
        if not self.controller.is_sysid_running():
            self._log("[sysid] Periodic sysid starting...")
            self.controller.run_sysid()
        self._schedule_sysid()

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
            self._log(f"[history] [red]Error: {e}[/]")
            self._log(traceback.format_exc())

    # ── Web server ──────────────────────────────────────────────────────────

    def _start_web_server(self) -> None:
        try:
            from weatherstat.web import start_web_server

            self._web_server = start_web_server(
                self.controller,
                host=self._web_host,
                port=self._web_port,
                log=self._log,
            )
            self._log(f"[web] Listening on http://{self._web_host}:{self._web_port}")
        except Exception as e:
            self._log(f"[web] [red]Failed to start web server: {e}[/]")
            self._log(traceback.format_exc())

    # ── Logging ─────────────────────────────────────────────────────────────

    def _log(self, message: str) -> None:
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            log_widget = self.query_one(LogView)
            log_widget.write(f"[dim]{ts}[/] {message}")
        except Exception:
            pass


# ── Module helpers ──────────────────────────────────────────────────────────


def _spawn_worker(fn) -> None:
    """Spawn a daemon thread for a worker callable."""
    threading.Thread(target=fn, daemon=True).start()


def _format_age(td: timedelta) -> str:
    """Format a timedelta as a human-readable age string."""
    total_seconds = int(td.total_seconds())
    if total_seconds < 0:
        return "<1s"
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
