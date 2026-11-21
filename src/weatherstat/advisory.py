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
from datetime import timedelta
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
    from weatherstat.types import AdvisoryPlan, BackupBreach, DeviceOpportunity

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
    # Devices listed in the suggestion notification currently on the phone.
    notified: list[str] = field(default_factory=list)

    @classmethod
    def load(cls) -> AdvisoryState:
        """Load from disk. Returns an empty state when the file is missing or unreadable."""
        if not ADVISORY_STATE_FILE.exists():
            return cls()
        try:
            data = json.loads(ADVISORY_STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return cls()
        opportunities = data.get("opportunities", [])
        return cls(
            cooldowns=data.get("cooldowns", {}),
            opportunities=opportunities,
            warnings=data.get("warnings", []),
            # Files written before ``notified`` existed: any tracked opportunity
            # may be on the phone.
            notified=data.get("notified", [str(o.get("device")) for o in opportunities]),
        )

    def save(self) -> None:
        """Persist to disk."""
        ADVISORY_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        ADVISORY_STATE_FILE.write_text(json.dumps(
            {
                "cooldowns": self.cooldowns,
                "opportunities": self.opportunities,
                "warnings": self.warnings,
                "notified": self.notified,
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


def opportunity_text(opp: dict) -> str:
    """'Open office window in 20m for 1h' — one serialized opportunity as a sentence.

    Shared by the phone notification and the MQTT status sensor. The TUI builds
    the same line with markup.
    """
    from weatherstat.overlay import format_remaining

    try:
        from weatherstat.yaml_config import environment_display, load_config

        dev = str(opp.get("device", "?"))
        label, kind = environment_display(dev, load_config().environment.get(dev))
    except Exception:  # noqa: BLE001 — display only
        label, kind = str(opp.get("device", "?")), ""
    mins = opp.get("in_minutes") or 0
    dur = opp.get("duration_minutes")
    timing = "now" if mins == 0 else f"in {format_remaining(timedelta(minutes=mins))}"
    dur_str = f" for {format_remaining(timedelta(minutes=dur))}" if dur else ""
    action = str(opp.get("action", "?")).replace("_", " ").capitalize()
    return " ".join(part for part in (action, label, kind) if part) + f" {timing}{dur_str}"


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
    backup-breach warnings, and keeps one rolled-up Home Assistant notification
    for opportunities whose ``|cost_delta|`` reaches
    ``ADVISORY_NOTIFICATION_THRESHOLD``.

    A suggestion past its cooldown (and outside quiet hours) sends the rollup.
    The rollup lists every suggestion currently worth notifying, not only the
    new one: it replaces the previous notification (same tag), which would
    otherwise drop suggestions that still apply. Each listed suggestion's
    cooldown restarts. The notification is dismissed once none of the
    suggestions it lists is still worth notifying.

    Suggestions can warm or cool (close a window on a cold day, open one on a
    warm evening), so the title says neither.
    """
    state = AdvisoryState.load()

    if advisory_plan is None:
        beneficial: list[DeviceOpportunity] = []
        warnings: list[dict] = []
    else:
        beneficial = sorted(
            (o for o in advisory_plan.opportunities if o.cost_delta < 0),
            key=lambda o: o.cost_delta,
        )
        warnings = [{"sensor": b.sensor, "message": b.message} for b in advisory_plan.backup_breaches]

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

    # The opportunity cooldown key predates suggestions that close or warm.
    cooldown_secs = ADVISORY_COOLDOWNS.get("free_cooling", 14400)
    now_ts = time.time()
    worth = [o for o in beneficial if -o.cost_delta >= ADVISORY_NOTIFICATION_THRESHOLD]
    due: list[DeviceOpportunity] = []
    if not quiet:
        for opp in worth:
            last = state.cooldowns.get(f"opportunity_{opp.device}")
            if last is None or (now_ts - last) >= cooldown_secs:
                due.append(opp)
    texts = {id(o): opportunity_text(_serialize_opportunity(o)) for o in beneficial}

    # ── Console output ──
    print("\n[advisory] Evaluating environment opportunities...")
    if not beneficial:
        print("  No beneficial opportunities")
    else:
        due_set = {id(o) for o in due}
        for opp in beneficial:
            tag = " (notify)" if id(opp) in due_set else ""
            print(f"  {texts[id(opp)]}: cost_delta={opp.cost_delta:+.2f}{tag}")

    # ── Persistence + notification dispatch ──
    if not live:
        return

    if due:
        lines = [f"{texts[id(o)]} (benefit {-o.cost_delta:.2f})" for o in worth]
        title = "Weatherstat suggestion" if len(worth) == 1 else "Weatherstat suggestions"
        if send_ha_notification(title, "\n".join(lines), _ROLLUP_TAG, notification_target):
            for opp in worth:
                state.cooldowns[f"opportunity_{opp.device}"] = now_ts
            state.notified = [o.device for o in worth]
            print(f"  → Notified ({len(worth)} suggestion{'s' if len(worth) != 1 else ''})")
    elif state.notified and not {o.device for o in worth} & set(state.notified):
        # Nothing the notification lists is still worth doing — clear it.
        if dismiss_ha_notification(_ROLLUP_TAG, notification_target):
            state.notified = []

    state.save()


# ── Comfort breach notifications ──────────────────────────────────────────

# Default: re-warn at most once per hour per sensor. Tunable via
# weatherstat.yaml ``advisory.cooldowns.comfort_breach`` (seconds).
_COMFORT_BREACH_DEFAULT_COOLDOWN = 3600
_COMFORT_BREACH_TAG = "weatherstat_advisory_warning"


def process_comfort_breaches(
    breaches: list[BackupBreach],
    *,
    live: bool = False,
    notification_target: str = "persistent_notification",
) -> list[BackupBreach]:
    """Dispatch comfort-breach warnings with per-sensor cooldown.

    Cooldown is per (sensor, direction) so a low-side breach doesn't suppress a
    later high-side breach on the same sensor. State is shared with the rest of
    the advisory cooldowns dict, keyed under ``comfort_breach_<sensor>_<dir>``.

    The whole batch is sent as a single notification (replacing any prior one
    via the shared tag) — but we only send when at least one breach has cleared
    its cooldown, and we only mark the breaches we actually announce.
    """
    if not breaches:
        return []

    state = AdvisoryState.load()
    cooldown_secs = ADVISORY_COOLDOWNS.get("comfort_breach", _COMFORT_BREACH_DEFAULT_COOLDOWN)
    now_ts = time.time()

    fresh: list[BackupBreach] = []
    for b in breaches:
        key = f"comfort_breach_{b.sensor}_{b.direction}"
        last = state.cooldowns.get(key)
        if last is not None and (now_ts - last) < cooldown_secs:
            continue
        fresh.append(b)

    if not fresh:
        print(f"  [advisory] {len(breaches)} comfort breach(es) on cooldown")
        return []

    print(f"  [advisory] {len(fresh)} comfort breach(es) past cooldown — notifying")

    if not live:
        return fresh

    body = "\n".join(b.message for b in fresh)
    sent = send_ha_notification("Comfort warning", body, _COMFORT_BREACH_TAG, notification_target)
    if sent:
        for b in fresh:
            state.cooldowns[f"comfort_breach_{b.sensor}_{b.direction}"] = now_ts
        state.save()
        print(f"    → Sent to HA ({notification_target})")
    return fresh
