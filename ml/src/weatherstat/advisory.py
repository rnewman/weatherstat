"""Human advisory notifications — window open/close recommendations.

Evaluates rules based on current sensor state and sends HA persistent notifications.
Cooldown timers prevent notification fatigue. State persisted to JSON.

Called from the control loop after each decision cycle.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import StrEnum
from statistics import median

import requests

from weatherstat.config import ADVISORY_STATE_FILE, HA_TOKEN, HA_URL

# ── Types ─────────────────────────────────────────────────────────────────


class AdvisoryType(StrEnum):
    FREE_COOLING = "free_cooling"
    CLOSE_WINDOWS = "close_windows"


@dataclass(frozen=True)
class Advisory:
    advisory_type: AdvisoryType
    title: str
    message: str


# ── Cooldown configuration (seconds) ─────────────────────────────────────

COOLDOWNS: dict[AdvisoryType, int] = {
    AdvisoryType.FREE_COOLING: 4 * 3600,  # 4 hours
    AdvisoryType.CLOSE_WINDOWS: 1 * 3600,  # 1 hour
}


# ── State persistence ────────────────────────────────────────────────────


def load_advisory_state() -> dict[str, float]:
    """Load advisory cooldown state: advisory_type → last_sent_unix_timestamp."""
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


def is_on_cooldown(state: dict[str, float], advisory_type: AdvisoryType) -> bool:
    """Return True if the advisory was sent recently enough to suppress."""
    last_sent = state.get(advisory_type.value)
    if last_sent is None:
        return False
    cooldown = COOLDOWNS.get(advisory_type, 3600)
    return (time.time() - last_sent) < cooldown


# ── Rule evaluation ──────────────────────────────────────────────────────


def evaluate_free_cooling(
    outdoor_temp: float | None,
    indoor_temps: dict[str, float],
    window_states: dict[str, bool],
    heating_active: bool,
) -> Advisory | None:
    """Suggest opening windows when outdoor air is cooler than indoor.

    Triggers when:
    - outdoor_temp is known and >= 50°F (not too cold)
    - outdoor_temp is >= 3°F below median indoor temp
    - no windows are already open
    - heating is off (no point opening windows while heating)
    """
    if outdoor_temp is None or len(indoor_temps) == 0:
        return None
    if outdoor_temp < 50.0:
        return None
    if any(window_states.values()):
        return None
    if heating_active:
        return None

    median_indoor = median(indoor_temps.values())
    delta = median_indoor - outdoor_temp
    if delta < 3.0:
        return None

    return Advisory(
        advisory_type=AdvisoryType.FREE_COOLING,
        title="Free cooling available",
        message=(
            f"It's {outdoor_temp:.0f}°F outside, {median_indoor:.0f}°F inside"
            f" ({delta:.0f}°F cooler). Consider opening windows."
        ),
    )


def evaluate_close_windows(
    window_states: dict[str, bool],
    heating_active: bool,
    outdoor_temp: float | None,
) -> Advisory | None:
    """Suggest closing windows when heating is running with windows open.

    Triggers when:
    - any window is open
    - either thermostat zone is heating

    Names the specific open windows in the message.
    """
    open_names = [name for name, is_open in window_states.items() if is_open]
    if not open_names:
        return None
    if not heating_active:
        return None

    window_list = " and ".join(", ".join(open_names).rsplit(", ", 1))
    temp_note = f" It's {outdoor_temp:.0f}°F outside." if outdoor_temp is not None else ""
    return Advisory(
        advisory_type=AdvisoryType.CLOSE_WINDOWS,
        title="Close windows",
        message=(
            f"The {window_list} window{'s are' if len(open_names) > 1 else ' is'} open"
            f" but heating is running.{temp_note} Close to avoid wasting energy."
        ),
    )


# ── HA notification dispatch ─────────────────────────────────────────────


def send_ha_notification(advisory: Advisory) -> bool:
    """Send a persistent notification to Home Assistant.

    Uses notification_id so new notifications replace old ones (no stacking).
    Returns True on success, False on failure (logged but not raised).
    """
    url = f"{HA_URL}/api/services/persistent_notification/create"
    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "title": advisory.title,
        "message": advisory.message,
        "notification_id": f"weatherstat_{advisory.advisory_type.value}",
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"  WARNING: Failed to send HA notification: {e}")
        return False


# ── Main entry point ─────────────────────────────────────────────────────


def process_advisories(
    outdoor_temp: float | None,
    indoor_temps: dict[str, float],
    window_states: dict[str, bool],
    heating_active: bool,
    live: bool = False,
) -> list[Advisory]:
    """Evaluate all advisory rules, apply cooldowns, and dispatch notifications.

    Args:
        outdoor_temp: Current outdoor temperature (°F), or None if unavailable.
        indoor_temps: Room name → current temperature for all rooms.
        window_states: Window name → open boolean for each window sensor.
        heating_active: Whether either thermostat zone is heating.
        live: If True, send HA notifications and persist cooldown state.

    Returns:
        List of advisories that were triggered (regardless of cooldown/send status).
    """
    state = load_advisory_state()
    triggered: list[Advisory] = []

    # Evaluate all rules
    candidates: list[Advisory | None] = [
        evaluate_free_cooling(outdoor_temp, indoor_temps, window_states, heating_active),
        evaluate_close_windows(window_states, heating_active, outdoor_temp),
    ]

    print("\n[advisory] Evaluating advisories...")
    for advisory in candidates:
        if advisory is None:
            continue
        triggered.append(advisory)

        if is_on_cooldown(state, advisory.advisory_type):
            print(f"  {advisory.advisory_type}: {advisory.title} (on cooldown)")
            continue

        print(f"  {advisory.advisory_type}: {advisory.message}")
        if live and send_ha_notification(advisory):
            state[advisory.advisory_type.value] = time.time()
            print("  → Sent to HA")

    if not triggered:
        print("  No advisories triggered")

    if live:
        save_advisory_state(state)

    return triggered
