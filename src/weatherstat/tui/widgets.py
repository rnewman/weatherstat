"""TUI widgets for the Weatherstat dashboard."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import DataTable, RichLog, Static

from weatherstat.config import UNIT_SYMBOL


class StatusHeader(Static):
    """Top status bar showing system state at a glance."""

    def __init__(self) -> None:
        super().__init__("", id="status-header")
        self._profile = "?"
        self._live = False
        self._collector_age = "?"
        self._collector_rows = 0
        self._sysid_age = "?"
        self._next_cycle = "?"
        self._outdoor_temp: float | None = None
        self._cycle_running = False
        self._sysid_running = False
        self._local_tz = ""

    def set_state(
        self,
        *,
        profile: str | None = None,
        live: bool | None = None,
        collector_age: str | None = None,
        collector_rows: int | None = None,
        sysid_age: str | None = None,
        next_cycle: str | None = None,
        outdoor_temp: float | None = None,
        cycle_running: bool | None = None,
        sysid_running: bool | None = None,
        local_tz: str | None = None,
    ) -> None:
        if profile is not None:
            self._profile = profile
        if live is not None:
            self._live = live
        if collector_age is not None:
            self._collector_age = collector_age
        if collector_rows is not None:
            self._collector_rows = collector_rows
        if sysid_age is not None:
            self._sysid_age = sysid_age
        if next_cycle is not None:
            self._next_cycle = next_cycle
        if outdoor_temp is not None:
            self._outdoor_temp = outdoor_temp
        if cycle_running is not None:
            self._cycle_running = cycle_running
        if sysid_running is not None:
            self._sysid_running = sysid_running
        if local_tz is not None:
            self._local_tz = local_tz
        self._refresh()

    def _refresh(self) -> None:
        from datetime import datetime

        now_str = datetime.now().strftime("%H:%M")
        tz = f" {self._local_tz}" if self._local_tz else ""

        mode_str = "[bold red]LIVE[/]" if self._live else "[bold green]DRY-RUN[/]"
        profile_str = self._profile.upper()

        out_str = f"{self._outdoor_temp:.0f}{UNIT_SYMBOL}" if self._outdoor_temp is not None else "?"

        row_k = f"{self._collector_rows / 1000:.0f}K" if self._collector_rows >= 1000 else str(self._collector_rows)

        cycle_str = self._next_cycle
        if self._cycle_running:
            cycle_str = "[bold yellow]running...[/]"

        sysid_str = self._sysid_age
        if self._sysid_running:
            sysid_str = "[bold yellow]fitting...[/]"

        line1 = f"[bold]Weatherstat HVAC Control[/]                              {profile_str} | {mode_str} | {now_str}{tz}"
        line2 = f"Collector: {self._collector_age} ago ({row_k} rows) | Sysid: {sysid_str} | Next cycle: {cycle_str} | Outdoor: {out_str}"
        self.update(f"{line1}\n{line2}")


def _comfort_bar(
    current: float, min_t: float, pref_lo: float, pref_hi: float, max_t: float, width: int = 16,
) -> Text:
    """Render a comfort-position bar showing where temp sits in the band.

    Three zones: outside [min, max] = red, within [min, max] but outside
    [pref_lo, pref_hi] = yellow/dim green, within preferred band = bright green.
    """
    margin = 2.0
    lo = min_t - margin
    hi = max_t + margin
    rng = hi - lo
    if rng <= 0:
        return Text("?" * width)

    pos = int((current - lo) / rng * width)
    pos = max(0, min(width - 1, pos))
    min_pos = int((min_t - lo) / rng * width)
    max_pos = int((max_t - lo) / rng * width)
    plo_pos = int((pref_lo - lo) / rng * width)
    phi_pos = int((pref_hi - lo) / rng * width)

    bar = Text()
    for i in range(width):
        if i == pos:
            char = "#"
        elif min_pos <= i <= max_pos:
            char = "="
        else:
            char = "-"

        if i == pos:
            if current < min_t or current > max_t:
                bar.append(char, style="bold red")
            elif current < pref_lo or current > pref_hi:
                bar.append(char, style="bold yellow")
            else:
                bar.append(char, style="bold green")
        elif plo_pos <= i <= phi_pos:
            bar.append(char, style="green")
        elif min_pos <= i <= max_pos:
            bar.append(char, style="dim green")
        else:
            bar.append(char, style="dim")

    return bar


class TemperaturePanel(Static):
    """Room temperatures with comfort bars."""

    # comfort tuple: (min, pref_lo, pref_hi, max)
    def __init__(self) -> None:
        super().__init__("", classes="panel")
        self._temps: dict[str, float] = {}
        self._comfort: dict[str, tuple[float, float, float, float]] = {}
        self._labels: dict[str, str] = {}  # sensor_col -> label
        self._mrt_offsets: dict[str, float] = {}  # sensor_col -> MRT offset

    def set_data(
        self,
        temps: dict[str, float],
        comfort: dict[str, tuple[float, float, float, float]],
        labels: dict[str, str],
        mrt_offsets: dict[str, float] | None = None,
    ) -> None:
        self._temps = temps
        self._comfort = comfort
        self._labels = labels
        self._mrt_offsets = mrt_offsets or {}
        self._refresh()

    def _refresh(self) -> None:
        lines: list[str] = ["[bold $accent]Temperatures[/]"]
        for sensor_col, temp in self._temps.items():
            label = self._labels.get(sensor_col, sensor_col.removesuffix("_temp"))
            comfort = self._comfort.get(label)
            if comfort:
                min_t, plo, phi, max_t = comfort
                bar = _comfort_bar(temp, min_t, plo, phi, max_t)
                band_str = f"[{min_t:.0f}-{max_t:.0f}]"
                mrt = self._mrt_offsets.get(sensor_col, 0.0)
                mrt_str = f" [dim]{mrt:+.1f}mrt[/]" if abs(mrt) >= 0.05 else ""
                line = f"  {label:<22} {temp:>5.1f}{UNIT_SYMBOL}  {band_str:>8} "
                text = Text.from_markup(line)
                text.append_text(bar)
                if mrt_str:
                    text.append_text(Text.from_markup(mrt_str))
                lines.append(text.markup if hasattr(text, "markup") else str(text))
            else:
                lines.append(f"  {label:<22} {temp:>5.1f}{UNIT_SYMBOL}")
        self.update("\n".join(lines))


class EnvironmentPanel(Static):
    """Current environment factor states (windows, doors, shades, etc.)."""

    def __init__(self) -> None:
        super().__init__("", classes="panel")
        self._entries: list[tuple[str, str, bool]] = []  # (label, kind, is_active)

    def set_data(self, entries: list[tuple[str, str, bool]]) -> None:
        self._entries = entries
        self._refresh()

    def _refresh(self) -> None:
        from collections import defaultdict

        from ..types import _ACTIVE_DESCRIPTIONS

        # Group active entries by their active description (e.g., "open", "lowered")
        grouped: dict[str, list[str]] = defaultdict(list)
        for label, kind, is_active in self._entries:
            if is_active:
                desc = _ACTIVE_DESCRIPTIONS.get(kind, "active")
                grouped[desc].append(f"{label} {kind}")

        if grouped:
            lines = ["[bold]Environment[/]"]
            for desc, items in grouped.items():
                lines.append(f"  [yellow]{desc.capitalize()}: {', '.join(items)}[/]")
            self.update("\n".join(lines))
        else:
            self.update("[bold]Environment[/]\n  All at default")


class ForecastPanel(Static):
    """Weather forecast and MRT info."""

    def __init__(self) -> None:
        super().__init__("", classes="panel")

    def set_data(
        self,
        forecasts: dict[str, float],
        condition: str = "",
        mrt_offset: float | None = None,
    ) -> None:
        parts = [f"{h}:{t:.0f}°" for h, t in forecasts.items()]
        line1 = "  " + " ".join(parts)
        if condition:
            line1 += f"  {condition}"
        lines = ["[bold]Forecast[/]", line1]
        if mrt_offset is not None and abs(mrt_offset) > 0.05:
            sign = "+" if mrt_offset > 0 else ""
            lines.append(f"  MRT offset: {sign}{mrt_offset:.1f}{UNIT_SYMBOL}")
        self.update("\n".join(lines))


class EffectorPanel(Static):
    """Current effector decisions."""

    def __init__(self) -> None:
        super().__init__("", classes="panel")
        self._decisions: list[dict] = []
        self._command_targets: dict[str, float] = {}
        self._current_temps: dict[str, float] = {}
        self._costs: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._baseline_cost: float | None = None
        self._overrides: dict[str, str] = {}  # effector_name -> description
        self._rationale: dict[str, str] = {}  # effector_name -> explanation
        self._sensor_costs: dict[str, float] = {}
        self._baseline_sensor_costs: dict[str, float] = {}

    def set_overrides(self, overrides: dict[str, str]) -> None:
        self._overrides = overrides
        self._refresh()

    def set_current_temps(self, temps: dict[str, float]) -> None:
        self._current_temps = temps
        self._refresh()

    def set_data(
        self,
        decisions: list[dict],
        command_targets: dict[str, float],
        costs: tuple[float, float, float] = (0.0, 0.0, 0.0),
        baseline_cost: float | None = None,
        rationale: dict[str, str] | None = None,
        sensor_costs: dict[str, float] | None = None,
        baseline_sensor_costs: dict[str, float] | None = None,
    ) -> None:
        self._decisions = decisions
        self._command_targets = command_targets
        self._costs = costs
        self._baseline_cost = baseline_cost
        self._rationale = rationale or {}
        self._sensor_costs = sensor_costs or {}
        self._baseline_sensor_costs = baseline_sensor_costs or {}
        self._refresh()

    def _refresh(self) -> None:
        lines: list[str] = ["[bold]Effector Decisions[/]"]
        if not self._decisions:
            lines.append("  (no decision yet)")
        for d in self._decisions:
            name = d.get("name", "?")
            mode = d.get("mode", "off")
            short_name = name.replace("thermostat_", "t:").replace("mini_split_", "ms:").replace("blower_", "bl:")
            target = self._command_targets.get(name)

            override = self._overrides.get(name)
            if mode == "off":
                line = f"  {short_name:<18} [dim]OFF[/]"
            else:
                # Distinguish active heating/cooling from mode-only (standby).
                # A thermostat in "heat" mode with setpoint below room temp isn't
                # actually heating — it's on standby as a safety net.
                room_temp = self._current_temps.get(f"{name}_temp")
                is_heat = mode in ("heat", "heating")
                is_cool = mode in ("cool", "cooling")
                actively_working = (
                    target is None
                    or room_temp is None
                    or not ((is_heat and target <= room_temp) or (is_cool and target >= room_temp))
                )

                if actively_working:
                    display_mode = f"[bold green]{mode.upper()}[/]"
                else:
                    display_mode = f"[dim]{mode.upper()}[/]"

                parts = [display_mode]
                if target is not None:
                    parts.append(f"({target:.0f}{UNIT_SYMBOL})")
                delay = d.get("delay_steps", 0)
                dur = d.get("duration_steps")
                if delay:
                    parts.append(f"delay:{delay * 5}m")
                if dur is not None:
                    parts.append(f"dur:{dur * 5}m")
                line = f"  {short_name:<18} {' '.join(parts)}"
            if override:
                line += f"  [bold red]OVERRIDE[/] ({override})"
            lines.append(line)

            # Inline rationale
            r = self._rationale.get(name)
            if r:
                lines.append(f"    [dim]{r}[/]")

        total, comfort, energy = self._costs
        lines.append(f"\n  Cost: {total:.1f}  (comfort {comfort:.1f} + energy {energy:.2f})")
        if self._baseline_cost is not None:
            lines.append(f"  vs all-off: {self._baseline_cost:.1f}")

        # Per-sensor cost breakdown (only sensors with non-trivial cost)
        sensors_with_cost = sorted(
            s
            for s in set(self._sensor_costs) | set(self._baseline_sensor_costs)
            if self._sensor_costs.get(s, 0) > 0.01 or self._baseline_sensor_costs.get(s, 0) > 0.01
        )
        if sensors_with_cost:
            lines.append("")
            for s in sensors_with_cost:
                dc = self._sensor_costs.get(s, 0)
                oc = self._baseline_sensor_costs.get(s, 0)
                saving = oc - dc
                label = s.removesuffix("_temp")
                if abs(saving) < 0.01:
                    lines.append(f"  [dim]{label:<16} {dc:>6.1f}[/]")
                elif saving > 0:
                    lines.append(f"  {label:<16} {dc:>6.1f}  [green]{saving:>+6.1f}[/]")
                else:
                    lines.append(f"  {label:<16} {dc:>6.1f}  [red]{saving:>+6.1f}[/]")

        self.update("\n".join(lines))


class OpportunityPanel(Static):
    """Active window opportunities and advisory recommendations."""

    def __init__(self) -> None:
        super().__init__("", classes="panel")

    def set_data(
        self,
        opportunities: list[dict],
        recommendations: list[dict] | None = None,
        warnings: list[dict] | None = None,
        proactive: list[dict] | None = None,
    ) -> None:
        from weatherstat.yaml_config import environment_display, load_config

        _env = load_config().environment
        lines: list[str] = ["[bold]Opportunities[/]"]

        # Advisory warnings (backup breaches) — most urgent first
        if warnings:
            for w in warnings:
                msg = w.get("message", "?")
                lines.append(f"  [bold red]{msg}[/]")

        # Advisory recommendations (reasonable layer)
        if recommendations:
            for rec in recommendations:
                dev = rec.get("device", "?")
                action = rec.get("action", "?")
                mins = rec.get("in_minutes", 0)
                delta = rec.get("cost_delta", 0)
                timing = "now" if mins == 0 else f"in {mins}m"
                label, kind = environment_display(dev, _env.get(dev))
                lines.append(f"  [cyan]{action} {label} {kind}[/] {timing} (saving {-delta:+.2f})")

        # Environment opportunities (existing system)
        if opportunities:
            for opp in opportunities:
                entry_name = opp.get("entry", opp.get("window", "?"))
                action = opp.get("action", "?")
                benefit = opp.get("total_benefit", 0)
                label, kind = environment_display(entry_name, _env.get(entry_name))
                lines.append(f"  [yellow]{action} {label} {kind}[/] (benefit: {benefit:.2f})")

        # Proactive advice
        if proactive:
            for p in proactive:
                dev = p.get("device", "?")
                action = p.get("action", "?")
                delta = p.get("cost_delta", 0)
                dur = p.get("duration_minutes")
                dur_str = f" for {dur}m" if dur else ""
                label, kind = environment_display(dev, _env.get(dev))
                lines.append(f"  [dim]{action} {label} {kind}{dur_str} ({delta:+.2f})[/]")

        if len(lines) == 1:
            lines.append("  (none)")
        self.update("\n".join(lines))


class HealthPanel(Static):
    """Health check status."""

    def __init__(self) -> None:
        super().__init__("", classes="panel")

    def set_data(self, alerts: list[str]) -> None:
        lines: list[str] = ["[bold]Health[/]"]
        if not alerts:
            lines.append("  [green]All checks passing[/]")
        else:
            for alert in alerts:
                lines.append(f"  [red]{alert}[/]")
        self.update("\n".join(lines))


class PredictionPanel(Static):
    """Prediction table showing decision vs all-off at each horizon."""

    def __init__(self) -> None:
        super().__init__("", id="predictions-content")

    def set_data(
        self,
        predictions: dict[str, dict[str, float]],
        labels: dict[str, str],
    ) -> None:
        if not predictions:
            self.update("[dim]No predictions yet. Press R to run a control cycle.[/]")
            return

        horizons = ["1h", "2h", "4h", "6h"]
        header = f"  {'Sensor':<22}" + "".join(f"  {h:>6}" for h in horizons)
        sep = "  " + "-" * (22 + len(horizons) * 8)
        lines = ["[bold]Predicted Temperatures[/]", header, sep]

        for sensor_col, horizon_temps in predictions.items():
            label = labels.get(sensor_col, sensor_col.removesuffix("_temp"))
            parts = [f"  {label:<22}"]
            for h in horizons:
                t = horizon_temps.get(h)
                if t is not None:
                    parts.append(f"  {t:>5.1f}°")
                else:
                    parts.append(f"  {'--':>6}")
            lines.append("".join(parts))

        self.update("\n".join(lines))


class HistoryPanel(DataTable):
    """Decision history table."""

    def __init__(self) -> None:
        super().__init__(id="history-table")
        self._initialized = False

    def set_data(self, rows: list[dict]) -> None:
        if not self._initialized:
            self.add_columns("Time", "Mode", "Total", "Comfort", "Energy", "Outdoor")
            self._initialized = True
        self.clear()
        for row in rows:
            ts = row.get("timestamp", "")
            # Show just time portion
            time_str = ts.split("T")[1][:5] if "T" in ts else ts[:16]
            mode = "LIVE" if row.get("live") else "DRY"
            total = f"{row.get('total_cost', 0):.1f}"
            comfort = f"{row.get('comfort_cost', 0):.1f}"
            energy = f"{row.get('energy_cost', 0):.2f}"
            outdoor = f"{row.get('outdoor_temp', 0):.0f}{UNIT_SYMBOL}" if row.get("outdoor_temp") else "?"
            self.add_row(time_str, mode, total, comfort, energy, outdoor)


class AccuracyPanel(Static):
    """Prediction accuracy summary."""

    def __init__(self) -> None:
        super().__init__("", id="accuracy-panel")

    def set_data(self, summary: dict[str, dict[str, float]]) -> None:
        if not summary:
            self.update("[dim]No accuracy data yet (need backfilled outcomes).[/]")
            return
        lines = ["[bold]Prediction Accuracy (24h)[/]"]
        header = f"  {'Horizon':<10} {'MAE':>8} {'Bias':>8} {'N':>6}"
        lines.append(header)
        for h in ("1h", "2h", "4h", "6h"):
            stats = summary.get(h)
            if stats:
                lines.append(f"  {h:<10} {stats['mae']:>7.2f}° {stats['bias']:>+7.2f}° {stats['n']:>6.0f}")
        self.update("\n".join(lines))


class LogView(RichLog):
    """Scrolling log of control cycle and sysid output."""

    def __init__(self) -> None:
        super().__init__(id="log-view", wrap=True, highlight=True, markup=True)
