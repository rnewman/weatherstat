"""Physics-based window advisories — open/close recommendations.

Evaluates window state toggles using the physics simulator to determine
if opening or closing a window would improve comfort. Called from the
control loop after the electronic plan is committed.

Notification dispatch via HA persistent notifications with per-window
cooldown timers and quiet hour suppression.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import requests

from weatherstat.config import (
    ADVISORY_COOLDOWNS,
    ADVISORY_EFFORT_COST,
    ADVISORY_QUIET_HOURS,
    ADVISORY_STATE_FILE,
    HA_TOKEN,
    HA_URL,
)

if TYPE_CHECKING:
    from weatherstat.simulator import HouseState, SimParams
    from weatherstat.types import ComfortSchedule, TrajectoryScenario

# ── Types ─────────────────────────────────────────────────────────────────


class AdvisoryType(StrEnum):
    FREE_COOLING = "free_cooling"
    CLOSE_WINDOWS = "close_windows"


@dataclass(frozen=True)
class Advisory:
    advisory_type: AdvisoryType
    title: str
    message: str
    window: str = ""  # window name for per-window cooldown tracking
    improvement: float = 0.0  # comfort cost improvement


# ── Cooldown state ────────────────────────────────────────────────────────


def _cooldown_key(advisory: Advisory) -> str:
    """Composite cooldown key: type_window for per-window tracking."""
    if advisory.window:
        return f"{advisory.advisory_type.value}_{advisory.window}"
    return advisory.advisory_type.value


def load_advisory_state() -> dict[str, float]:
    """Load advisory cooldown state: key → last_sent_unix_timestamp."""
    if not ADVISORY_STATE_FILE.exists():
        return {}
    try:
        return json.loads(ADVISORY_STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_advisory_state(state: dict[str, float]) -> None:
    """Persist advisory cooldown state."""
    ADVISORY_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ADVISORY_STATE_FILE.write_text(json.dumps(state, indent=2))


def _is_on_cooldown(state: dict[str, float], key: str, advisory_type: AdvisoryType) -> bool:
    """Return True if the advisory was sent recently enough to suppress."""
    last_sent = state.get(key)
    if last_sent is None:
        return False
    cooldown = ADVISORY_COOLDOWNS.get(advisory_type.value, 3600)
    return (time.time() - last_sent) < cooldown


# ── Physics-based evaluation ──────────────────────────────────────────────


def evaluate_window_advisories(
    state: HouseState,
    winning_scenario: TrajectoryScenario,
    sim_params: SimParams,
    schedules: list[ComfortSchedule],
    base_hour: int,
) -> list[Advisory]:
    """Evaluate whether toggling any window would improve comfort.

    For each window, simulates the winning electronic plan with the window
    toggled and compares comfort cost against the current state. The
    simulator determines which sensors each window affects (via the tau
    model); no configured window→room mapping is needed.

    Uses original (non-window-adjusted) comfort schedules for both baseline
    and toggled evaluation to measure true comfort impact.

    Args:
        state: Current house state (including window_states).
        winning_scenario: The committed electronic trajectory.
        sim_params: Thermal model parameters from sysid.
        schedules: Original comfort schedules (NOT window-adjusted).
        base_hour: Current hour of day (0-23) for schedule lookup.

    Returns:
        List of advisories sorted by comfort improvement (best first).
    """
    from weatherstat.control import CONTROL_HORIZONS, compute_comfort_cost
    from weatherstat.simulator import HouseState as _HS
    from weatherstat.simulator import predict
    from weatherstat.yaml_config import load_config

    cfg = load_config()

    # Baseline: predict with current window states, score against original schedules
    target_names, baseline_preds = predict(state, [winning_scenario], sim_params, CONTROL_HORIZONS)
    baseline_dict = {t: float(baseline_preds[0, j]) for j, t in enumerate(target_names)}
    baseline_cost = compute_comfort_cost(baseline_dict, schedules, base_hour)

    advisories: list[Advisory] = []

    for window_name in cfg.windows:
        is_open = state.window_states.get(window_name, False)
        toggled_windows = dict(state.window_states)
        toggled_windows[window_name] = not is_open

        toggled_state = _HS(
            current_temps=state.current_temps,
            outdoor_temp=state.outdoor_temp,
            forecast_temps=state.forecast_temps,
            window_states=toggled_windows,
            hour_of_day=state.hour_of_day,
            recent_history=state.recent_history,
        )

        _, toggled_preds = predict(toggled_state, [winning_scenario], sim_params, CONTROL_HORIZONS)
        toggled_dict = {t: float(toggled_preds[0, j]) for j, t in enumerate(target_names)}
        toggled_cost = compute_comfort_cost(toggled_dict, schedules, base_hour)

        improvement = baseline_cost - toggled_cost
        if improvement <= ADVISORY_EFFORT_COST:
            continue

        action = "Open" if not is_open else "Close"
        adv_type = AdvisoryType.FREE_COOLING if not is_open else AdvisoryType.CLOSE_WINDOWS

        # Find the most-affected constrained sensor for the message
        best_label = window_name
        best_delta = 0.0
        best_baseline_temp: float | None = None
        best_toggled_temp: float | None = None
        for c in cfg.constraints:
            key_2h = f"{c.sensor}_t+24"  # 2h horizon (24 × 5min steps)
            bt = baseline_dict.get(key_2h)
            tt = toggled_dict.get(key_2h)
            if bt is not None and tt is not None:
                delta = abs(tt - bt)
                if delta > best_delta:
                    best_delta = delta
                    best_label = c.label
                    best_baseline_temp = bt
                    best_toggled_temp = tt

        if best_baseline_temp is not None and best_toggled_temp is not None:
            if not is_open:
                message = (
                    f"{action} {window_name} window — {state.outdoor_temp:.0f}°F outside, "
                    f"{best_label} predicted {best_toggled_temp:.0f}°F at 2h "
                    f"(vs {best_baseline_temp:.0f}°F with window closed)"
                )
            else:
                message = (
                    f"{action} {window_name} window — {best_label} predicted "
                    f"{best_toggled_temp:.0f}°F at 2h "
                    f"(vs {best_baseline_temp:.0f}°F with window open)"
                )
        else:
            message = f"{action} {window_name} window (comfort improvement: {improvement:.1f})"

        advisories.append(Advisory(
            advisory_type=adv_type,
            title=f"{action} {window_name} window",
            message=message,
            window=window_name,
            improvement=round(improvement, 2),
        ))

    advisories.sort(key=lambda a: a.improvement, reverse=True)
    return advisories


# ── HA notification dispatch ─────────────────────────────────────────────


def send_ha_notification(advisory: Advisory, target: str = "persistent_notification") -> bool:
    """Send a notification to Home Assistant.

    Dispatches to the appropriate HA service based on the target:
    - "persistent_notification" → persistent_notification/create (sidebar)
    - "notify.mobile_app_*" → notify/mobile_app_* (mobile push)
    - "notify.*" → notify/* (notification group)

    Uses notification_id/tag so new notifications replace old ones (no stacking).
    Returns True on success, False on failure (logged but not raised).
    """
    tag = f"weatherstat_{_cooldown_key(advisory)}"

    if target == "persistent_notification":
        service = "persistent_notification/create"
        payload: dict[str, object] = {
            "title": advisory.title,
            "message": advisory.message,
            "notification_id": tag,
        }
    else:
        # notify.mobile_app_foo → notify/mobile_app_foo
        service = target.replace(".", "/", 1)
        payload = {
            "title": advisory.title,
            "message": advisory.message,
            "data": {"tag": tag},
        }

    url = f"{HA_URL}/api/services/{service}"
    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"  WARNING: Failed to send HA notification: {e}")
        return False


# ── Main dispatch ─────────────────────────────────────────────────────────


def process_advisories(
    advisories: list[Advisory],
    live: bool = False,
    notification_target: str = "persistent_notification",
    current_hour: int | None = None,
) -> list[Advisory]:
    """Apply cooldowns, check quiet hours, and dispatch notifications.

    Args:
        advisories: Pre-evaluated advisories from evaluate_window_advisories.
        live: If True, send HA notifications and persist cooldown state.
        notification_target: HA service target.
        current_hour: Current hour for quiet hours check.

    Returns:
        List of advisories that passed cooldown/quiet filtering.
    """
    state = load_advisory_state()
    dispatched: list[Advisory] = []

    quiet = False
    if current_hour is not None:
        start, end = ADVISORY_QUIET_HOURS
        quiet = start <= current_hour < end if start <= end else current_hour >= start or current_hour < end

    print("\n[advisory] Evaluating advisories...")

    if not advisories:
        print("  No advisories triggered")
        return []

    for advisory in advisories:
        key = _cooldown_key(advisory)

        if _is_on_cooldown(state, key, advisory.advisory_type):
            print(f"  {advisory.title} (on cooldown)")
            continue

        if quiet:
            print(f"  {advisory.title} (quiet hours)")
            continue

        print(f"  {advisory.title}: {advisory.message}")
        dispatched.append(advisory)

        if live and send_ha_notification(advisory, target=notification_target):
            state[key] = time.time()
            print(f"  → Sent to HA ({notification_target})")

    if live:
        save_advisory_state(state)

    return dispatched
