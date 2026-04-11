"""Physics-based environment opportunities — persistent advisory recommendations.

Reads the per-device marginals from the trajectory sweep's ``AdvisoryPlan``,
filters to beneficial changes (``cost_delta < 0``), persists them for TUI
display, and dispatches a single rolled-up Home Assistant notification per
cycle for opportunities whose magnitude exceeds the notification threshold —
subject to cooldown and quiet-hour gates.

The cooldown dict in the state file is shared with ``safety.py``, which keys
its own entries under a ``safety_*`` namespace.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import requests

from weatherstat.config import (
    ADVISORY_COOLDOWNS,
    ADVISORY_NOTIFICATION_THRESHOLD,
    ADVISORY_QUIET_HOURS,
    ADVISORY_STATE_FILE,
    HA_TOKEN,
    HA_URL,
)

if TYPE_CHECKING:
    from weatherstat.types import AdvisoryPlan, DeviceOpportunity

# ── Persistent state ──────────────────────────────────────────────────────


@dataclass
class AdvisoryState:
    """Persistent state for advisory opportunities and notification cooldowns.

    cooldowns are shared with ``safety.py`` (which prefixes its own keys with
    ``safety_``). opportunities and warnings are written by the advisory
    pipeline for TUI display, refreshed every control cycle.
    """

    cooldowns: dict[str, float] = field(default_factory=dict)
    opportunities: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls) -> AdvisoryState:
        """Load from disk. Returns an empty state when the file is missing or unreadable."""
        if not ADVISORY_STATE_FILE.exists():
            return cls()
        try:
            data = json.loads(ADVISORY_STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return cls()
        return cls(
            cooldowns=data.get("cooldowns", {}),
            opportunities=data.get("opportunities", []),
            warnings=data.get("warnings", []),
        )

    def save(self) -> None:
        """Persist to disk."""
        ADVISORY_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        ADVISORY_STATE_FILE.write_text(json.dumps(
            {
                "cooldowns": self.cooldowns,
                "opportunities": self.opportunities,
                "warnings": self.warnings,
            },
            indent=2,
        ))


# ── Cooldown helpers (used by safety.py) ──────────────────────────────────


def load_advisory_state() -> dict[str, float]:
    """Load cooldown state: key → last_sent_unix_timestamp."""
    return AdvisoryState.load().cooldowns


def save_advisory_state(cooldowns: dict[str, float]) -> None:
    """Persist cooldown state, preserving opportunity/warning data."""
    state = AdvisoryState.load()
    state.cooldowns = cooldowns
    state.save()


# ── Opportunity serialization ─────────────────────────────────────────────


def _serialize_opportunity(opp: DeviceOpportunity) -> dict[str, object]:
    """Serialize a per-device opportunity for the TUI/JSON state file."""
    entry: dict[str, object] = {
        "device": opp.device,
        "current_state": opp.current_state,
        "action": opp.advisory.action,
        "in_minutes": opp.advisory.transition_step * 5,
        "cost_delta": round(opp.cost_delta, 4),
    }
    if opp.advisory.return_step is not None:
        entry["duration_minutes"] = (
            opp.advisory.return_step - opp.advisory.transition_step
        ) * 5
    return entry


# ── HA notification dispatch ─────────────────────────────────────────────


def send_ha_notification(
    title: str,
    message: str,
    tag: str,
    target: str = "persistent_notification",
) -> bool:
    """Send a notification to Home Assistant.

    Dispatches to the appropriate HA service based on the target:
    - "persistent_notification" → persistent_notification/create (sidebar)
    - "notify.mobile_app_*" → notify/mobile_app_* (mobile push)
    - "notify.*" → notify/* (notification group)

    Uses notification_id/tag so new notifications replace old ones (no stacking).
    Returns True on success, False on failure (logged but not raised).
    """
    if target == "persistent_notification":
        service = "persistent_notification/create"
        payload: dict[str, object] = {
            "title": title,
            "message": message,
            "notification_id": tag,
        }
    else:
        # notify.mobile_app_foo → notify/mobile_app_foo
        service = target.replace(".", "/", 1)
        payload = {
            "title": title,
            "message": message,
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


def dismiss_ha_notification(
    tag: str,
    target: str = "persistent_notification",
) -> bool:
    """Dismiss a persistent notification in Home Assistant."""
    if target == "persistent_notification":
        service = "persistent_notification/dismiss"
        payload: dict[str, object] = {"notification_id": tag}
    else:
        service = target.replace(".", "/", 1)
        payload = {"message": "clear_notification", "data": {"tag": tag}}

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
        print(f"  WARNING: Failed to dismiss HA notification: {e}")
        return False


# ── Unified advisory pipeline ────────────────────────────────────────────


_ROLLUP_TAG = "weatherstat_opportunities"


def process_advisory_plan(
    advisory_plan: AdvisoryPlan | None,
    *,
    live: bool = False,
    notification_target: str = "persistent_notification",
    current_hour: int | None = None,
) -> None:
    """Persist advisory recommendations and dispatch notifications.

    Filters per-device opportunities to beneficial ones (``cost_delta < 0``),
    sorts them by magnitude, persists them to ``ADVISORY_STATE_FILE`` alongside
    backup-breach warnings, and pushes a single rolled-up Home Assistant
    notification for opportunities whose ``|cost_delta|`` exceeds
    ``ADVISORY_NOTIFICATION_THRESHOLD`` — subject to cooldown and quiet-hour
    gates. The rollup is dismissed when the previously-active set drains.
    """
    from weatherstat.yaml_config import environment_display, load_config

    state = AdvisoryState.load()
    _env = load_config().environment
    had_opportunities = bool(state.opportunities)

    if advisory_plan is None:
        beneficial: list[DeviceOpportunity] = []
        warnings: list[dict] = []
    else:
        beneficial = sorted(
            (o for o in advisory_plan.opportunities if o.cost_delta < 0),
            key=lambda o: o.cost_delta,
        )
        warnings = [{"message": b} for b in advisory_plan.backup_breaches]

    state.opportunities = [_serialize_opportunity(o) for o in beneficial]
    state.warnings = warnings

    # ── Notification gating ──
    quiet = False
    if current_hour is not None:
        start, end = ADVISORY_QUIET_HOURS
        quiet = (
            start <= current_hour < end
            if start <= end
            else current_hour >= start or current_hour < end
        )

    cooldown_secs = ADVISORY_COOLDOWNS.get("free_cooling", 14400)
    now_ts = time.time()
    newly_notify: list[DeviceOpportunity] = []
    if not quiet:
        for opp in beneficial:
            magnitude = -opp.cost_delta
            if magnitude < ADVISORY_NOTIFICATION_THRESHOLD:
                continue
            cooldown_key = f"opportunity_{opp.device}"
            last = state.cooldowns.get(cooldown_key)
            if last is not None and (now_ts - last) < cooldown_secs:
                continue
            newly_notify.append(opp)
            state.cooldowns[cooldown_key] = now_ts

    # ── Console output ──
    print("\n[advisory] Evaluating environment opportunities...")
    if not beneficial:
        print("  No beneficial opportunities")
    else:
        notify_set = {id(o) for o in newly_notify}
        for opp in beneficial:
            label, kind = environment_display(opp.device, _env.get(opp.device))
            tag = " (notify)" if id(opp) in notify_set else ""
            timing = "now" if opp.advisory.transition_step == 0 else f"in {opp.advisory.transition_step * 5}m"
            print(
                f"  {opp.advisory.action.title()} {label} {kind}: "
                f"cost_delta={opp.cost_delta:+.2f} {timing}{tag}"
            )

    # ── Persistence + notification dispatch ──
    if not live:
        return

    state.save()

    if newly_notify:
        lines: list[str] = []
        for opp in newly_notify:
            label, kind = environment_display(opp.device, _env.get(opp.device))
            magnitude = -opp.cost_delta
            lines.append(f"{opp.advisory.action.title()} {label} {kind} (benefit {magnitude:.2f})")
        send_ha_notification("Free cooling", "\n".join(lines), _ROLLUP_TAG, notification_target)
        print(f"  → Notified ({len(newly_notify)} opportunities)")
    elif had_opportunities and not beneficial:
        # The previously-active rollup no longer applies — clear it.
        dismiss_ha_notification(_ROLLUP_TAG, notification_target)
