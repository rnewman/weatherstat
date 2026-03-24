"""HVAC command executor — applies control decisions to Home Assistant.

Reads the latest command JSON from predictions/, compares desired state against
current HA entity state, and issues service calls for any changes.

Features:
- Lazy execution: skips commands when HA is already in the desired state
- Override detection: skips devices whose state has been manually changed
  since the last execution (respects user overrides until cleared)
- Force mode: bypasses override detection
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime

import requests

from weatherstat.config import DATA_DIR, EFFECTORS, PREDICTIONS_DIR

TARGET_TOLERANCE = 0.5
OVERRIDE_STALE_SECONDS = 30 * 60  # 30 minutes

_EXECUTOR_STATE_FILE = DATA_DIR / "executor_state.json"


# ── Types ───────────────────────────────────────────────────────────────────


@dataclass
class DeviceAction:
    """Result of attempting to execute a command for one device."""

    name: str
    action: str  # "applied", "already_correct", "override", "ineligible"
    detail: str = ""


@dataclass
class ExecutorResult:
    """Summary of an execution run."""

    actions: list[DeviceAction] = field(default_factory=list)

    @property
    def applied(self) -> int:
        return sum(1 for a in self.actions if a.action == "applied")

    @property
    def already_correct(self) -> int:
        return sum(1 for a in self.actions if a.action == "already_correct")

    @property
    def overrides(self) -> dict[str, str]:
        """effector_name -> detail for devices with detected overrides."""
        return {a.name: a.detail for a in self.actions if a.action == "override"}


# ── HA helpers ──────────────────────────────────────────────────────────────


def _ha_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {os.environ.get('HA_TOKEN', '')}",
        "Content-Type": "application/json",
    }


def _ha_url() -> str:
    return os.environ.get("HA_URL", "")


def _get_entity_state(entity_id: str) -> dict | None:
    """Fetch a single entity's state from HA REST API."""
    url = _ha_url()
    if not url:
        return None
    try:
        resp = requests.get(f"{url}/api/states/{entity_id}", headers=_ha_headers(), timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        pass
    return None


def _call_service(domain: str, service: str, entity_id: str, data: dict | None = None) -> bool:
    """Call an HA service. Returns True on success."""
    url = _ha_url()
    if not url:
        return False
    payload: dict = {"entity_id": entity_id}
    if data:
        payload.update(data)
    try:
        resp = requests.post(
            f"{url}/api/services/{domain}/{service}",
            headers=_ha_headers(),
            json=payload,
            timeout=10,
        )
        return resp.status_code == 200
    except requests.RequestException:
        return False


# ── HA state readers ────────────────────────────────────────────────────────


def _read_climate_target(entity_id: str) -> float | None:
    """Read climate entity's target temperature."""
    state = _get_entity_state(entity_id)
    if not state:
        return None
    target = state.get("attributes", {}).get("temperature")
    return float(target) if isinstance(target, (int, float)) else None


def _read_climate_mode(entity_id: str) -> tuple[str | None, float | None]:
    """Read climate entity's mode and target. Returns (mode, target)."""
    state = _get_entity_state(entity_id)
    if not state:
        return None, None
    mode = state.get("state")
    target = state.get("attributes", {}).get("temperature")
    return mode, float(target) if isinstance(target, (int, float)) else None


def _read_fan_mode(entity_id: str) -> str | None:
    """Read fan entity's effective mode (off / preset_mode)."""
    state = _get_entity_state(entity_id)
    if not state:
        return None
    if state.get("state") == "off":
        return "off"
    preset = state.get("attributes", {}).get("preset_mode")
    return str(preset) if preset else "low"


# ── Executor state persistence ──────────────────────────────────────────────


def _load_executor_state() -> dict:
    if _EXECUTOR_STATE_FILE.exists():
        try:
            return json.loads(_EXECUTOR_STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_executor_state(devices: dict[str, dict]) -> None:
    state = {"timestamp": datetime.now(UTC).isoformat(), "devices": devices}
    _EXECUTOR_STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Command loader ──────────────────────────────────────────────────────────


def _load_latest_command() -> dict | None:
    """Load the most recent command_*.json from predictions/."""
    if not PREDICTIONS_DIR.exists():
        return None
    files = sorted(PREDICTIONS_DIR.glob("command_*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _snake_to_camel(s: str) -> str:
    parts = s.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


# ── Main executor ───────────────────────────────────────────────────────────


def execute(*, force: bool = False, log: print = print) -> ExecutorResult:
    """Execute the latest command JSON against Home Assistant.

    Args:
        force: If True, ignore detected overrides.
        log: Callable for status messages (default: print).

    Returns:
        ExecutorResult with per-device actions.
    """
    result = ExecutorResult()

    command = _load_latest_command()
    if not command:
        log("[executor] No command found")
        return result

    ts = command.get("timestamp", "?")
    confidence = command.get("confidence", 0)
    log(f"[executor] Executing command from {ts} (confidence: {confidence:.2f})")
    if force:
        log("[executor] Force mode — overrides will be ignored")

    # Load last-applied state
    exec_state = _load_executor_state()
    last_devices: dict[str, dict] = exec_state.get("devices", {})

    # Check staleness
    state_ts = exec_state.get("timestamp", "")
    if state_ts:
        try:
            dt = datetime.fromisoformat(state_ts.replace("Z", "+00:00"))
            age = (datetime.now(UTC) - dt).total_seconds()
            if age > OVERRIDE_STALE_SECONDS:
                log(f"[executor] Executor state is {int(age / 60)}m old — clearing overrides")
                last_devices = {}
        except (ValueError, TypeError):
            pass

    updated_devices = dict(last_devices)

    for eff in EFFECTORS:
        if eff.control_type in ("trajectory",) and eff.mode_control == "manual":
            # Manual-mode climate (thermostat): target only
            target_key = _snake_to_camel(f"{eff.name}_target")
            desired_target = command.get(target_key)
            if desired_target is None:
                result.actions.append(DeviceAction(eff.name, "ineligible"))
                continue

            desired_target = float(desired_target)
            current_target = _read_climate_target(eff.entity_id)

            if current_target is not None and abs(current_target - desired_target) < TARGET_TOLERANCE:
                log(f"[executor] {eff.name}: already at {current_target}°F")
                result.actions.append(DeviceAction(eff.name, "already_correct", f"{current_target}°F"))
                continue

            last = last_devices.get(eff.name, {})
            last_target = last.get("target")
            if (
                not force
                and last_target is not None
                and current_target is not None
                and abs(current_target - last_target) > TARGET_TOLERANCE
            ):
                detail = f"target {current_target}°F, we set {last_target}°F"
                log(f"[executor] {eff.name}: override detected ({detail})")
                result.actions.append(DeviceAction(eff.name, "override", detail))
                continue

            _call_service("climate", "set_temperature", eff.entity_id, {"temperature": desired_target})
            log(f"[executor] {eff.name}: set to {desired_target}°F")
            updated_devices[eff.name] = {"target": desired_target}
            result.actions.append(DeviceAction(eff.name, "applied", f"{desired_target}°F"))

        elif eff.control_type == "regulating":
            # Automatic-mode climate (mini-split): mode + target
            mode_key = _snake_to_camel(f"{eff.name}_mode")
            target_key = _snake_to_camel(f"{eff.name}_target")
            desired_mode = command.get(mode_key)
            desired_target = command.get(target_key)

            if desired_mode is None:
                result.actions.append(DeviceAction(eff.name, "ineligible"))
                continue

            current_mode, current_target = _read_climate_mode(eff.entity_id)

            mode_match = current_mode == desired_mode
            target_match = (
                desired_mode == "off"
                or (desired_target is not None and current_target is not None and abs(current_target - float(desired_target)) < TARGET_TOLERANCE)
            )
            if mode_match and target_match:
                desc = "off" if desired_mode == "off" else f"{desired_mode} @ {current_target}°F"
                log(f"[executor] {eff.name}: already {desc}")
                result.actions.append(DeviceAction(eff.name, "already_correct", desc))
                continue

            last = last_devices.get(eff.name, {})
            if not force and last.get("mode") is not None and current_mode != last["mode"]:
                detail = f"mode {current_mode}, we set {last['mode']}"
                log(f"[executor] {eff.name}: override detected ({detail})")
                result.actions.append(DeviceAction(eff.name, "override", detail))
                continue

            _call_service("climate", "set_hvac_mode", eff.entity_id, {"hvac_mode": desired_mode})
            if desired_mode != "off" and desired_target is not None:
                _call_service("climate", "set_temperature", eff.entity_id, {"temperature": float(desired_target)})
            desc = "off" if desired_mode == "off" else f"{desired_mode} @ {desired_target}°F"
            log(f"[executor] {eff.name}: set to {desc}")
            updated_devices[eff.name] = {"mode": desired_mode, "target": desired_target}
            result.actions.append(DeviceAction(eff.name, "applied", desc))

        elif eff.control_type == "binary":
            # Fan entity (blower): mode only
            mode_key = _snake_to_camel(f"{eff.name}_mode")
            desired_mode = command.get(mode_key)

            if desired_mode is None:
                result.actions.append(DeviceAction(eff.name, "ineligible"))
                continue

            current_mode = _read_fan_mode(eff.entity_id)

            if current_mode == desired_mode:
                log(f"[executor] {eff.name}: already {desired_mode}")
                result.actions.append(DeviceAction(eff.name, "already_correct", desired_mode))
                continue

            last = last_devices.get(eff.name, {})
            if not force and last.get("mode") is not None and current_mode != last["mode"]:
                detail = f"mode {current_mode}, we set {last['mode']}"
                log(f"[executor] {eff.name}: override detected ({detail})")
                result.actions.append(DeviceAction(eff.name, "override", detail))
                continue

            if desired_mode == "off":
                _call_service("fan", "turn_off", eff.entity_id)
            else:
                _call_service("fan", "turn_on", eff.entity_id)
                _call_service("fan", "set_preset_mode", eff.entity_id, {"preset_mode": desired_mode})
            log(f"[executor] {eff.name}: set to {desired_mode}")
            updated_devices[eff.name] = {"mode": desired_mode}
            result.actions.append(DeviceAction(eff.name, "applied", desired_mode))

    _save_executor_state(updated_devices)

    log(f"[executor] Done: {result.applied} applied, {result.already_correct} correct, {len(result.overrides)} overrides")
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Execute latest HVAC command via Home Assistant")
    parser.add_argument("--force", action="store_true", help="Ignore manual overrides")
    args = parser.parse_args()
    execute(force=args.force)


if __name__ == "__main__":
    main()
