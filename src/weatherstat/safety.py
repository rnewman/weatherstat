"""Infrastructure safety checks — detect hardware/config problems.

Checks for conditions that prevent the control loop from working:
- Thermostat turned off (hvac_mode=off) while control wants to heat
- Device health thresholds (configured per-effector in weatherstat.yaml)

Called from the control loop after each decision cycle. Safety alerts
use push notifications (no quiet hours — these are urgent).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from weatherstat.yaml_config import HealthCheck

from weatherstat.advisory import (
    load_advisory_state,
    save_advisory_state,
    send_ha_notification,
)
from weatherstat.config import HA_TOKEN, HA_URL

# ── Types ─────────────────────────────────────────────────────────────────

# Safety alerts reuse Advisory infrastructure but with distinct types/keys.

_DEFAULT_COOLDOWNS: dict[str, int] = {
    "thermostat_off": 3600,   # 1 hour
    "device_fault": 1800,     # 30 min — more urgent
}


def _get_cooldowns() -> dict[str, int]:
    """Merge YAML-configured cooldowns with defaults."""
    from weatherstat.yaml_config import load_config
    cfg = load_config()
    merged = dict(_DEFAULT_COOLDOWNS)
    merged.update(cfg.safety.cooldowns)
    return merged


@dataclass(frozen=True)
class SafetyAlert:
    """A detected infrastructure problem."""

    key: str        # "thermostat_off_upstairs", "navien_fault"
    title: str
    message: str
    severity: str = "warning"  # "warning" or "critical"


# ── Thermostat mode check ────────────────────────────────────────────────


def check_thermostat_modes(latest: object, decision: object) -> list[SafetyAlert]:
    """Check if thermostats are turned off while control wants to heat.

    The thermostat has two independent properties: target temp and hvac_mode.
    If hvac_mode is "off", setting the target has no effect. The control loop
    only sets the target, so an off thermostat silently defeats heating.

    Args:
        latest: Latest snapshot row (pandas Series or dict-like).
        decision: ControlDecision with effectors tuple.
    """
    from weatherstat.config import EFFECTORS

    alerts: list[SafetyAlert] = []

    for eff in EFFECTORS:
        if eff.mode_control != "manual":
            continue
        # Find this effector's decision
        eff_dec = next((e for e in decision.effectors if e.name == eff.name), None)
        heating = eff_dec is not None and eff_dec.mode != "off"
        action = str(getattr(latest, f"{eff.name}_action", "")
                      if hasattr(latest, f"{eff.name}_action")
                      else latest.get(f"{eff.name}_action", ""))

        if heating and action == "off":
            # Strip "thermostat_" prefix for display
            display_name = eff.name.removeprefix("thermostat_")
            alerts.append(SafetyAlert(
                key=f"{eff.name}_off",
                title=f"{display_name.replace('_', ' ').title()} thermostat is off",
                message=(
                    f"Control wants {display_name} heating but the thermostat's hvac_mode is off. "
                    f"Setpoint changes will have no effect. Turn the thermostat on."
                ),
                severity="critical",
            ))

    return alerts


# ── Generic device health checks ────────────────────────────────────────


def check_device_health() -> list[SafetyAlert]:
    """Run health checks for all devices with configured thresholds.

    Reads health check definitions from the health section in
    weatherstat.yaml. For each check, fetches the
    entity's current state from HA and compares against min/max thresholds.
    """
    from weatherstat.yaml_config import load_config

    cfg = load_config()
    alerts: list[SafetyAlert] = []

    for check in cfg.device_health_checks:
        try:
            alert = _check_health_threshold(check)
            if alert is not None:
                alerts.append(alert)
        except Exception:
            pass  # Don't fail the control cycle if HA fetch fails

    return alerts


def _check_when_condition(check: HealthCheck) -> bool:
    """Return True if the check's ``when`` condition is satisfied (or absent).

    Fetches the conditioning entity from HA.  The condition passes when:
    1. The entity's current state matches ``when_state``, AND
    2. It has been in that state for at least ``when_for_minutes`` (compared
       against the ``last_changed`` timestamp in the HA state response).
    """
    if check.when_entity is None:
        return True  # no condition — always evaluate

    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }
    resp = requests.get(
        f"{HA_URL}/api/states/{check.when_entity}",
        headers=headers,
        timeout=5,
    )
    if resp.status_code != 200:
        return False  # can't verify condition — skip the check

    when_data = resp.json()
    if when_data.get("state", "") != check.when_state:
        return False

    if check.when_for_minutes > 0:
        last_changed = when_data.get("last_changed", "")
        if not last_changed:
            return False
        changed_at = datetime.fromisoformat(last_changed)
        elapsed_min = (datetime.now(UTC) - changed_at).total_seconds() / 60
        if elapsed_min < check.when_for_minutes:
            return False

    return True


def _check_sustained(check: HealthCheck) -> bool:
    """Return True if enough 1-minute samples in the sustain window violate the threshold.

    Fetches HA history for the entity over the last ``sustain_minutes``,
    resamples at 1-minute intervals by forward-filling the last known state,
    and counts how many samples violate.  This handles both high-frequency
    oscillations (many state changes, few violations) and sustained flat
    violations (one state change held for many minutes).
    """
    min_required = check.sustain_samples
    if min_required <= 0:
        return True  # misconfigured — fall through to alert

    now = datetime.now(UTC)
    window_start = now - timedelta(minutes=check.sustain_minutes)
    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }
    resp = requests.get(
        f"{HA_URL}/api/history/period/{window_start.isoformat()}",
        headers=headers,
        params={
            "filter_entity_id": check.entity_id,
            "minimal_response": "",
            "no_attributes": "",
            "significant_changes_only": "",
            "end_time": now.isoformat(),
        },
        timeout=10,
    )
    if resp.status_code != 200:
        return False  # can't verify — don't alert

    history = resp.json()
    if not history or not history[0]:
        return False

    # Build sorted (timestamp, value) pairs from state changes.
    transitions: list[tuple[datetime, float]] = []
    for entry in history[0]:
        state = entry.get("state", "")
        if state in ("unavailable", "unknown"):
            continue
        ts_str = entry.get("last_changed", "")
        if not ts_str:
            continue
        try:
            transitions.append((datetime.fromisoformat(ts_str), float(state)))
        except (ValueError, TypeError):
            continue

    if not transitions:
        return False

    # Resample at 1-minute intervals, forward-filling the last known value.
    violating = 0
    n_samples = int(check.sustain_minutes)
    for i in range(n_samples):
        sample_time = window_start + timedelta(minutes=i + 0.5)  # sample at :30s marks
        # Find last transition at or before sample_time.
        val: float | None = None
        for ts, v in transitions:
            if ts <= sample_time:
                val = v
            else:
                break
        if val is None:
            continue  # no data before this sample point
        if check.min_value is not None and val <= check.min_value or check.max_value is not None and val >= check.max_value:
            violating += 1

    return violating >= min_required


def _check_health_threshold(check: HealthCheck) -> SafetyAlert | None:
    """Fetch an entity from HA and compare against configured thresholds."""
    # Evaluate optional precondition first.
    if not _check_when_condition(check):
        return None

    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }

    resp = requests.get(
        f"{HA_URL}/api/states/{check.entity_id}",
        headers=headers,
        timeout=5,
    )
    if resp.status_code != 200:
        return None

    state_data = resp.json()
    state_val = state_data.get("state", "")

    if state_val in ("unavailable", "unknown"):
        return SafetyAlert(
            key=f"{check.name}_unavailable",
            title=f"{check.name} sensor unavailable",
            message=f"Diagnostic sensor {check.entity_id} is '{state_val}'.",
            severity="warning",
        )

    # String state check (for binary/enum entities like connection status)
    if check.expected_state is not None:
        if state_val != check.expected_state:
            return SafetyAlert(
                key=f"{check.name}_fault",
                title=f"{check.name} health check failed",
                message=check.message or f"{check.entity_id} is '{state_val}' (expected '{check.expected_state}')",
                severity=check.severity,
            )
        return None

    # Numeric threshold checks
    try:
        value = float(state_val)
    except (ValueError, TypeError):
        return None

    violated = False
    if check.min_value is not None and value <= check.min_value or check.max_value is not None and value >= check.max_value:
        violated = True

    if not violated:
        return None

    # Sustained threshold: require all readings over the window to violate.
    if check.sustain_minutes > 0 and not _check_sustained(check):
        return None

    return SafetyAlert(
        key=f"{check.name}_fault",
        title=f"{check.name} health check failed",
        message=check.message or f"{check.entity_id} = {value}",
        severity=check.severity,
    )


# ── Dispatch ──────────────────────────────────────────────────────────────


def process_safety_alerts(
    alerts: list[SafetyAlert],
    live: bool = False,
    notification_target: str = "persistent_notification",
) -> list[SafetyAlert]:
    """Dispatch safety alerts with cooldown tracking.

    Safety alerts bypass quiet hours — these are urgent.
    Cooldown state is shared with the advisory system.
    """
    if not alerts:
        return []

    state = load_advisory_state()
    dispatched: list[SafetyAlert] = []

    print("\n[safety] Infrastructure checks:")

    for alert in alerts:
        cooldown_key = f"safety_{alert.key}"
        cooldowns = _get_cooldowns()
        cooldown_seconds = cooldowns.get(
            alert.key.rsplit("_", 1)[0],  # "thermostat_off_upstairs" -> "thermostat_off"
            cooldowns.get(alert.key, 3600),
        )

        last_sent = state.get(cooldown_key)
        if last_sent is not None and (time.time() - last_sent) < cooldown_seconds:
            print(f"  {alert.severity.upper()}: {alert.title} (on cooldown)")
            continue

        print(f"  {alert.severity.upper()}: {alert.title}")
        print(f"    {alert.message}")
        dispatched.append(alert)

        if live:
            tag = f"weatherstat_safety_{alert.key}"
            if send_ha_notification(f"⚠ {alert.title}", alert.message, tag, notification_target):
                state[cooldown_key] = time.time()
                print(f"    → Sent to HA ({notification_target})")

    if live:
        save_advisory_state(state)

    return dispatched
