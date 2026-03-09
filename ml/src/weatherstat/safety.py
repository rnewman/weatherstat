"""Infrastructure safety checks — detect hardware/config problems.

Checks for conditions that prevent the control loop from working:
- Thermostat turned off (hvac_mode=off) while control wants to heat
- Navien boiler not responding to heat calls (wrong mode, disconnected)

Called from the control loop after each decision cycle. Safety alerts
use push notifications (no quiet hours — these are urgent).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from weatherstat.advisory import (
    Advisory,
    AdvisoryType,
    load_advisory_state,
    save_advisory_state,
    send_ha_notification,
)
from weatherstat.config import HA_TOKEN, HA_URL

# ── Types ─────────────────────────────────────────────────────────────────

# Safety alerts reuse Advisory infrastructure but with distinct types/keys.
# We use AdvisoryType.CLOSE_WINDOWS as a carrier (the actual key is the
# safety alert's key, not the advisory type).

_DEFAULT_COOLDOWNS: dict[str, int] = {
    "thermostat_off": 3600,   # 1 hour
    "navien_fault": 1800,     # 30 min — more urgent
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

    key: str        # "thermostat_off_upstairs", "navien_disconnected"
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


# ── Navien health check ──────────────────────────────────────────────────


def check_navien_health(latest: object) -> list[SafetyAlert]:
    """Check for Navien boiler problems from snapshot data and HA API.

    Two-tier check:
    1. Snapshot data: thermostat calling for heat but Navien not in Space Heating
    2. HA REST API: Navien ESPHome sensors showing disconnected (return_temp ≤ 33°F)
    """
    alerts: list[SafetyAlert] = []

    # Helper to read from pandas Series or dict
    def _get(key: str, default: object = "") -> object:
        if hasattr(latest, key):
            return getattr(latest, key)
        return latest.get(key, default) if hasattr(latest, "get") else default

    # Check 1: Heat call without boiler response
    up_action = str(_get("thermostat_upstairs_action"))
    dn_action = str(_get("thermostat_downstairs_action"))
    navien_mode = str(_get("navien_heating_mode"))
    any_calling = up_action == "heating" or dn_action == "heating"

    # Log when thermostat is calling but Navien isn't in Space Heating.
    # This is normal — Navien cycles between SH, DHW, and Idle. Only the
    # ESPHome disconnection check (return_temp ≤ 33°F) is a real fault signal.
    if any_calling and navien_mode not in ("Space Heating", ""):
        print(f"  [safety] Note: thermostat calling for heat, Navien mode is '{navien_mode}'")

    # Check 2: Navien ESPHome sensors (live HA fetch)
    from weatherstat.yaml_config import load_config
    cfg = load_config()
    navien_cfg = cfg.safety_navien

    if navien_cfg is not None:
        try:
            esphome_alerts = _check_navien_esphome(navien_cfg)
            alerts.extend(esphome_alerts)
        except Exception:
            pass  # Don't fail the control cycle if HA fetch fails

    return alerts


def _check_navien_esphome(navien_cfg: object) -> list[SafetyAlert]:
    """Fetch Navien ESPHome sensors from HA and check for disconnection.

    When the Navien is in the wrong mode (Zone Pump instead of Zone Valve),
    the ESPHome device shows all zeros: return_temp=32°F, outlet_temp=32°F.
    """
    alerts: list[SafetyAlert] = []
    threshold = navien_cfg.disconnected_threshold

    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }

    # Check return temp — most reliable indicator
    if navien_cfg.return_temp_entity:
        try:
            resp = requests.get(
                f"{HA_URL}/api/states/{navien_cfg.return_temp_entity}",
                headers=headers,
                timeout=5,
            )
            if resp.status_code == 200:
                state_data = resp.json()
                state_val = state_data.get("state", "")
                try:
                    return_temp = float(state_val)
                    if return_temp <= threshold:
                        alerts.append(SafetyAlert(
                            key="navien_disconnected",
                            title="Navien appears disconnected",
                            message=(
                                f"Navien return temp is {return_temp:.0f}°F "
                                f"(threshold: {threshold}°F). "
                                f"The boiler may be in Zone Pump mode instead of Zone Valve. "
                                f"Check the Navien unit."
                            ),
                            severity="critical",
                        ))
                except (ValueError, TypeError):
                    if state_val in ("unavailable", "unknown"):
                        alerts.append(SafetyAlert(
                            key="navien_unavailable",
                            title="Navien sensor unavailable",
                            message=(
                                f"Navien return temp sensor is '{state_val}'. "
                                f"The ESPHome device may be offline."
                            ),
                            severity="warning",
                        ))
        except requests.RequestException:
            pass  # HA unreachable — don't fail the cycle

    return alerts


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
            # Send as Advisory for notification compatibility
            adv = Advisory(
                advisory_type=AdvisoryType.CLOSE_WINDOWS,  # carrier type
                title=f"⚠ {alert.title}",
                message=alert.message,
                window=f"safety_{alert.key}",
            )
            if send_ha_notification(adv, target=notification_target):
                state[cooldown_key] = time.time()
                print(f"    → Sent to HA ({notification_target})")

    if live:
        save_advisory_state(state)

    return dispatched
