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
        decision: ControlDecision with upstairs_heating/downstairs_heating.
    """
    alerts: list[SafetyAlert] = []

    for zone in ("upstairs", "downstairs"):
        heating = getattr(decision, f"{zone}_heating", False)
        action = str(getattr(latest, f"thermostat_{zone}_action", "")
                      if hasattr(latest, f"thermostat_{zone}_action")
                      else latest.get(f"thermostat_{zone}_action", ""))

        if heating and action == "off":
            alerts.append(SafetyAlert(
                key=f"thermostat_off_{zone}",
                title=f"{zone.title()} thermostat is off",
                message=(
                    f"Control wants {zone} heating but the thermostat's hvac_mode is off. "
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

    for device_name, checks in cfg.device_health_checks.items():
        for check in checks:
            try:
                alert = _check_health_threshold(device_name, check)
                if alert is not None:
                    alerts.append(alert)
            except Exception:
                pass  # Don't fail the control cycle if HA fetch fails

    return alerts


def _check_health_threshold(device_name: str, check: HealthCheck) -> SafetyAlert | None:
    """Fetch an entity from HA and compare against configured thresholds."""
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
            key=f"{device_name}_unavailable",
            title=f"{device_name} sensor unavailable",
            message=f"Diagnostic sensor {check.entity_id} is '{state_val}'.",
            severity="warning",
        )

    # String state check (for binary/enum entities like connection status)
    if check.expected_state is not None:
        if state_val != check.expected_state:
            return SafetyAlert(
                key=f"{device_name}_fault",
                title=f"{device_name} health check failed",
                message=check.message or f"{check.entity_id} is '{state_val}' (expected '{check.expected_state}')",
                severity=check.severity,
            )
        return None

    # Numeric threshold checks
    try:
        value = float(state_val)
    except (ValueError, TypeError):
        return None

    if check.min_value is not None and value <= check.min_value:
        return SafetyAlert(
            key=f"{device_name}_fault",
            title=f"{device_name} health check failed",
            message=check.message or f"{check.entity_id} = {value}",
            severity=check.severity,
        )

    if check.max_value is not None and value >= check.max_value:
        return SafetyAlert(
            key=f"{device_name}_fault",
            title=f"{device_name} health check failed",
            message=check.message or f"{check.entity_id} = {value}",
            severity=check.severity,
        )

    return None


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
