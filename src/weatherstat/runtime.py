"""Shared runtime orchestration for the control loop.

The TUI and the headless loop (`just control --loop`) both need the same
periodic behaviors: control cycles every `control_interval`, sysid refits
every `sysid_interval` behind a quality gate, and daily retention. This
module owns that orchestration so headless operation doesn't lose the
automatic refitting that used to live only in the TUI.
"""

from __future__ import annotations

import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from weatherstat.config import CONTROL_INTERVAL, DATA_DIR, SYSID_INTERVAL

_RETENTION_MARKER = DATA_DIR / ".retention_last_run"
_RETENTION_INTERVAL_SECONDS = 24 * 3600


@dataclass(frozen=True)
class SysidOutcome:
    """Result of a quality-gated sysid refit."""

    saved: bool
    message: str
    duration_seconds: float


def sysid_refit() -> SysidOutcome:
    """Fit sysid, apply the quality gate, and save if it passes.

    The gate rejects fits with zero fitted taus or zero significant gains,
    preserving the previous thermal_params.json. Callers own logging and
    stdout capture (sysid prints its own progress).
    """
    from weatherstat.sysid import fit_sysid, save_sysid_result

    t0 = time.monotonic()
    result, diagnostics = fit_sysid()
    n_taus = len(result.fitted_taus)
    n_gains = sum(1 for g in result.effector_sensor_gains if not g.negligible)

    if n_taus == 0:
        return SysidOutcome(False, "Rejected: no sensors fitted", time.monotonic() - t0)
    if n_gains == 0:
        return SysidOutcome(False, "Rejected: no significant gains found", time.monotonic() - t0)

    save_sysid_result(result, sensor_diagnostics=diagnostics)

    # Update multi-timescale residual trackers against the fresh batch fit.
    # Tracker failure must not fail the refit — batch params are already saved.
    try:
        from weatherstat.tracker import update_trackers

        tracker_msg = update_trackers(log=lambda _msg: None)
    except Exception as e:  # noqa: BLE001 — isolate tracker faults from the refit
        tracker_msg = f"tracker: error — {e}"

    return SysidOutcome(
        True,
        f"Complete. {result.n_snapshots} snapshots, {n_taus} sensors, {n_gains} gains. {tracker_msg}",
        time.monotonic() - t0,
    )


def run_retention_if_due(
    log: Callable[[str], None] = print,
    marker_path: Path | None = None,
) -> dict[str, int] | None:
    """Run retention at most once per day, tracked via a marker file.

    The marker file (mtime) survives restarts and is shared between the TUI
    and the headless loop, so retention runs daily no matter which is up.
    Returns deletion counts if it ran, None if not due.
    """
    marker = marker_path or _RETENTION_MARKER
    now = time.time()
    if marker.exists() and (now - marker.stat().st_mtime) < _RETENTION_INTERVAL_SECONDS:
        return None

    from weatherstat.retention import run_retention

    counts = run_retention()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    if counts["command_files"] or counts["decision_rows"]:
        log(
            f"[retention] Deleted {counts['command_files']} command file(s),"
            f" {counts['decision_rows']} decision row(s)"
        )
    return counts


def run_headless_loop(live: bool = False) -> None:
    """Run the full control loop without the TUI.

    Each iteration: control cycle (+ execute when live), then a sysid refit
    when `sysid_interval` has elapsed (0 disables), then retention when due.
    Sequential by design — a sysid refit delays the next cycle by its
    duration, which is harmless at 5-minute intervals.
    """
    from weatherstat.control import run_control_cycle

    mode_str = "LIVE" if live else "DRY-RUN"
    sysid_str = f"{SYSID_INTERVAL}s" if SYSID_INTERVAL > 0 else "disabled"
    print(
        f"[runtime] Starting headless loop ({mode_str},"
        f" control interval: {CONTROL_INTERVAL}s, sysid interval: {sysid_str})"
    )
    print("  Press Ctrl+C to stop\n")

    last_sysid_at = time.monotonic()  # fresh start: params on disk are current enough

    while True:
        try:
            decision = run_control_cycle(live=live)
            if live and decision:
                from weatherstat.executor import execute

                execute()
        except Exception as e:
            print(f"[runtime] Error in control cycle: {e}", file=sys.stderr)
            traceback.print_exc()

        if SYSID_INTERVAL > 0 and (time.monotonic() - last_sysid_at) >= SYSID_INTERVAL:
            print(f"\n[runtime] Sysid refit due ({datetime.now(UTC):%H:%M UTC})")
            try:
                outcome = sysid_refit()
                print(f"[sysid] {outcome.message} ({outcome.duration_seconds:.1f}s)")
            except Exception as e:
                print(f"[runtime] Error in sysid refit: {e}", file=sys.stderr)
                traceback.print_exc()
            last_sysid_at = time.monotonic()

        try:
            run_retention_if_due()
        except Exception as e:
            print(f"[runtime] Error in retention: {e}", file=sys.stderr)
            traceback.print_exc()

        print(f"\n[runtime] Next cycle in {CONTROL_INTERVAL // 60} minutes...\n")
        time.sleep(CONTROL_INTERVAL)
