"""Physics-based environment opportunities — persistent advisory recommendations.

Evaluates environment entry toggles using the physics simulator to determine if
changing state (opening/closing windows, raising/lowering shades, etc.) would
improve comfort and/or save energy. Called from the control loop after the
electronic plan is committed.

Two-threshold model:
- Opportunity threshold: minimum benefit to track (visible in control output).
- Notification threshold: minimum benefit to push a notification.

Opportunities persist across control cycles: active until conditions change,
then automatically dismissed.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import requests

from weatherstat.config import (
    ADVISORY_COOLDOWNS,
    ADVISORY_NOTIFICATION_THRESHOLD,
    ADVISORY_OPPORTUNITY_THRESHOLD,
    ADVISORY_QUIET_HOURS,
    ADVISORY_STATE_FILE,
    HA_TOKEN,
    HA_URL,
    UNIT_SYMBOL,
)

if TYPE_CHECKING:
    from weatherstat.control import ControlState
    from weatherstat.simulator import HouseState, SimParams
    from weatherstat.types import ComfortSchedule, EnvironmentOpportunity, Scenario

# ── Opportunity state ────────────────────────────────────────────────────


@dataclass
class OpportunityState:
    """Persistent state for environment opportunities and advisory recommendations."""

    active: dict[str, dict]  # entry_name -> opportunity data (serializable)
    cooldowns: dict[str, float]  # cooldown_key -> unix_timestamp
    recommendations: list[dict] = field(default_factory=list)  # advisory layer recommendations
    warnings: list[dict] = field(default_factory=list)  # backup breach warnings
    proactive: list[dict] = field(default_factory=list)  # proactive advice

    @classmethod
    def load(cls) -> OpportunityState:
        """Load from disk. Handles old flat-dict format (cooldowns only)."""
        if not ADVISORY_STATE_FILE.exists():
            return cls(active={}, cooldowns={})
        try:
            data = json.loads(ADVISORY_STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return cls(active={}, cooldowns={})

        if "active" in data and "cooldowns" in data:
            return cls(
                active=data["active"],
                cooldowns=data["cooldowns"],
                recommendations=data.get("recommendations", []),
                warnings=data.get("warnings", []),
                proactive=data.get("proactive", []),
            )
        return cls(active={}, cooldowns={})

    def save(self) -> None:
        """Persist to disk."""
        ADVISORY_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        ADVISORY_STATE_FILE.write_text(json.dumps(
            {
                "active": self.active,
                "cooldowns": self.cooldowns,
                "recommendations": self.recommendations,
                "warnings": self.warnings,
                "proactive": self.proactive,
            },
            indent=2,
        ))


# ── Cooldown helpers (used by safety.py) ──────────────────────────────────


def load_advisory_state() -> dict[str, float]:
    """Load cooldown state: key → last_sent_unix_timestamp."""
    state = OpportunityState.load()
    return state.cooldowns


def save_advisory_state(cooldowns: dict[str, float]) -> None:
    """Persist cooldown state, preserving active opportunity data."""
    state = OpportunityState.load()
    state.cooldowns = cooldowns
    state.save()


def save_advisory_recommendations(advisory_layers: dict | None, live: bool = False) -> None:
    """Persist advisory layer recommendations to state file for TUI display.

    Called from the control loop after sweep_scenarios_physics returns advisory_layers.
    Converts AdvisoryDecision objects to serializable dicts.
    """
    if not live:
        return
    state = OpportunityState.load()

    if not advisory_layers:
        state.recommendations = []
        state.warnings = []
        state.proactive = []
        state.save()
        return

    # Reasonable layer recommendations
    recs: list[dict] = []
    r_advisories = advisory_layers.get("reasonable_advisories", {})
    for dev, adv in sorted(r_advisories.items()):
        if adv.action != "hold":
            recs.append({
                "device": dev,
                "action": adv.action,
                "in_minutes": adv.transition_step * 5,
                "cost_delta": round(
                    advisory_layers.get("reasonable_cost", 0) - advisory_layers.get("baseline_cost", 0), 4,
                ),
            })
    state.recommendations = recs

    # Backup breach warnings
    breaches = advisory_layers.get("backup_breaches", [])
    state.warnings = [{"message": b} for b in breaches]

    # Proactive advice
    proactive_list: list[dict] = []
    proactive_info = advisory_layers.get("proactive", {})
    proactive_advs = advisory_layers.get("proactive_advisories", {})
    for dev, info in sorted(proactive_info.items()):
        adv = proactive_advs.get(dev)
        if adv is not None:
            entry: dict[str, object] = {
                "device": dev,
                "action": adv.action,
                "cost_delta": round(info["cost_delta"], 4),
            }
            if adv.return_step is not None:
                entry["duration_minutes"] = (adv.return_step - adv.transition_step) * 5
            proactive_list.append(entry)
    state.proactive = proactive_list

    state.save()


# ── Energy-aware opportunity evaluation ──────────────────────────────────


def evaluate_environment_opportunities(
    state: HouseState,
    winning_scenario: Scenario,
    winning_comfort_cost: float,
    winning_energy_cost: float,
    sim_params: SimParams,
    schedules: list[ComfortSchedule],
    base_hour: int,
    prev_state: ControlState | None = None,
) -> list[EnvironmentOpportunity]:
    """Evaluate window toggles for comfort and energy savings.

    Two-tier evaluation:
    1. Quick check: simulate winning scenario with window toggled → comfort delta.
    2. Re-sweep (if promising): full scenario sweep with toggled window to find
       best HVAC plan. Captures "open window + turn off mini split" savings.

    Returns list of EnvironmentOpportunity sorted by total benefit (best first).
    """
    from weatherstat.control import (
        CONTROL_HORIZONS,
        HORIZON_WEIGHTS,
        compute_comfort_cost,
        sweep_scenarios_physics,
    )
    from weatherstat.simulator import HouseState as _HS
    from weatherstat.simulator import predict
    from weatherstat.types import EnvironmentOpportunity
    from weatherstat.yaml_config import load_config

    cfg = load_config()
    threshold = ADVISORY_OPPORTUNITY_THRESHOLD

    # Normalization factor: total horizon weight × number of schedules.
    # Converts raw cost difference to per-room per-weighted-horizon average,
    # so thresholds (0.3 = track, 1.5 = notify) are stable regardless of
    # system size (number of rooms/horizons).
    total_hw = sum(HORIZON_WEIGHTS.get(h, 0.5) for h in CONTROL_HORIZONS)
    cost_norm = len(schedules) * total_hw if schedules else 1.0

    # Baseline comfort cost with current window states
    target_names, baseline_preds = predict(state, [winning_scenario], sim_params, CONTROL_HORIZONS)
    baseline_dict = {t: float(baseline_preds[0, j]) for j, t in enumerate(target_names)}
    baseline_comfort = compute_comfort_cost(baseline_dict, schedules, base_hour)

    opportunities: list[EnvironmentOpportunity] = []

    for env_name in cfg.advisory_environment:
        is_open = state.environment_states.get(env_name, False)
        toggled_states = dict(state.environment_states)
        toggled_states[env_name] = not is_open

        toggled_state = _HS(
            current_temps=state.current_temps,
            outdoor_temp=state.outdoor_temp,
            forecast_temps=state.forecast_temps,
            environment_states=toggled_states,
            hour_of_day=state.hour_of_day,
            recent_history=state.recent_history,
            solar_fractions=state.solar_fractions,
            solar_elevations=state.solar_elevations,
        )

        # Quick check: comfort improvement with same HVAC plan
        _, toggled_preds = predict(toggled_state, [winning_scenario], sim_params, CONTROL_HORIZONS)
        toggled_dict = {t: float(toggled_preds[0, j]) for j, t in enumerate(target_names)}
        quick_comfort = compute_comfort_cost(toggled_dict, schedules, base_hour)
        quick_benefit = (baseline_comfort - quick_comfort) / cost_norm

        if quick_benefit <= threshold:
            continue

        # Re-sweep: find best HVAC plan with toggled environment factor
        resweep_decision, _resweep_scenario, _, _ = sweep_scenarios_physics(
            current_temps=state.current_temps,
            outdoor_temp=state.outdoor_temp,
            forecast_temps=state.forecast_temps,
            environment_states=toggled_states,
            sim_params=sim_params,
            hour_of_day=state.hour_of_day,
            recent_history=state.recent_history,
            schedules=schedules,
            base_hour=base_hour,
            prev_state=prev_state,
            solar_fractions=state.solar_fractions,
            solar_elevations=state.solar_elevations,
        )

        comfort_improvement = (winning_comfort_cost - resweep_decision.comfort_cost) / cost_norm
        energy_saving = winning_energy_cost - resweep_decision.energy_cost
        total_benefit = comfort_improvement + energy_saving

        if total_benefit <= threshold:
            continue

        env_cfg = cfg.environment[env_name]
        action = env_cfg.open_action.capitalize() if not is_open else env_cfg.close_action.capitalize()

        parts = [f"{action} {env_cfg.label} {env_cfg.kind}"]
        if not is_open:
            parts.append(f"({state.outdoor_temp:.0f}{UNIT_SYMBOL} outside)")
        if comfort_improvement > 0.01:
            parts.append(f"comfort +{comfort_improvement:.2f}")
        if energy_saving > 0.01:
            parts.append(f"energy saving +{energy_saving:.3f}")
        message = " — ".join(parts)

        opportunities.append(EnvironmentOpportunity(
            entry=env_name,
            action=action.lower(),
            comfort_improvement=round(comfort_improvement, 2),
            energy_saving=round(energy_saving, 3),
            total_benefit=round(total_benefit, 2),
            message=message,
        ))

    opportunities.sort(key=lambda o: o.total_benefit, reverse=True)
    return opportunities


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


# ── Opportunity lifecycle ────────────────────────────────────────────────


_ROLLUP_TAG = "weatherstat_opportunities"


def _legacy_notification_tag(window: str) -> str:
    """Per-window tag from pre-rollup era — used only for dismissing old notifications."""
    return f"weatherstat_opportunity_{window}"


def process_opportunities(
    new_opportunities: list[EnvironmentOpportunity],
    live: bool = False,
    notification_target: str = "persistent_notification",
    current_hour: int | None = None,
) -> tuple[list[EnvironmentOpportunity], list[str]]:
    """Manage opportunity lifecycle: add/keep/remove, dispatch notifications.

    Notifications are rolled up into a single push per control cycle listing
    all new opportunities.  Dismissed entries clear any legacy per-entry
    notification tags (transition from pre-rollup format).

    Returns:
        (active_opportunities, dismissed_entries).
    """
    from weatherstat.types import EnvironmentOpportunity
    from weatherstat.yaml_config import environment_display, load_config

    _env = load_config().environment
    opp_state = OpportunityState.load()
    now_iso = datetime.now(UTC).isoformat()
    now_ts = time.time()

    quiet = False
    if current_hour is not None:
        start, end = ADVISORY_QUIET_HOURS
        quiet = start <= current_hour < end if start <= end else current_hour >= start or current_hour < end

    # Build candidate set from new opportunities
    candidates = {opp.entry: opp for opp in new_opportunities}
    prev_active = set(opp_state.active.keys())
    curr_active: dict[str, dict] = {}
    active_list: list[EnvironmentOpportunity] = []
    dismissed: list[str] = []
    newly_notified: list[EnvironmentOpportunity] = []

    print("\n[opportunities] Evaluating environment opportunities...")

    if not candidates and not prev_active:
        print("  No opportunities")
        return [], []

    for entry_name, opp in candidates.items():
        was_active = entry_name in prev_active
        prev_data = opp_state.active.get(entry_name, {})
        first_seen = prev_data.get("first_seen", now_iso) if was_active else now_iso
        was_notified = prev_data.get("notified", False) if was_active else False

        # Check notification threshold + cooldown
        should_notify = (
            opp.total_benefit >= ADVISORY_NOTIFICATION_THRESHOLD
            and not was_notified
            and not quiet
        )
        cooldown_key = f"opportunity_{entry_name}"
        if should_notify and cooldown_key in opp_state.cooldowns:
            cooldown_secs = ADVISORY_COOLDOWNS.get("free_cooling", 14400)
            if (now_ts - opp_state.cooldowns[cooldown_key]) < cooldown_secs:
                should_notify = False

        notified = was_notified or should_notify

        active_opp = EnvironmentOpportunity(
            entry=opp.entry,
            action=opp.action,
            comfort_improvement=opp.comfort_improvement,
            energy_saving=opp.energy_saving,
            total_benefit=opp.total_benefit,
            message=opp.message,
            first_seen=first_seen,
            notified=notified,
        )
        active_list.append(active_opp)
        curr_active[entry_name] = asdict(active_opp)

        if should_notify:
            opp_state.cooldowns[cooldown_key] = now_ts
            newly_notified.append(active_opp)

        status = "active" if was_active else "new"
        label, kind = environment_display(entry_name, _env.get(entry_name))
        print(f"  {opp.action.title()} {label} {kind}: benefit={opp.total_benefit:.2f} [{status}]")

    # Dismiss expired opportunities (+ legacy per-entry tags from pre-rollup)
    for entry_name in prev_active - set(candidates.keys()):
        prev_data = opp_state.active[entry_name]
        if prev_data.get("notified", False) and live:
            dismiss_ha_notification(_legacy_notification_tag(entry_name), notification_target)
        label, kind = environment_display(entry_name, _env.get(entry_name))
        print(f"  Dismissed: {label} {kind}")
        dismissed.append(entry_name)

    # Send single rolled-up notification for all newly notified opportunities
    if newly_notified and live:
        lines: list[str] = []
        for opp in newly_notified:
            label, kind = environment_display(opp.entry, _env.get(opp.entry))
            line = f"{opp.action.title()} {label} {kind}"
            parts: list[str] = []
            if opp.comfort_improvement > 0.01:
                parts.append(f"comfort +{opp.comfort_improvement:.2f}")
            if opp.energy_saving > 0.01:
                parts.append(f"energy +{opp.energy_saving:.3f}")
            if parts:
                line += f" ({', '.join(parts)})"
            lines.append(line)

        title = "Free cooling"
        body = "\n".join(lines)
        send_ha_notification(title, body, _ROLLUP_TAG, notification_target)
        print(f"  → Notified ({len(newly_notified)} opportunities)")

    # If nothing is active and something was previously notified, dismiss the rollup
    if not curr_active and prev_active and live:
        dismiss_ha_notification(_ROLLUP_TAG, notification_target)

    opp_state.active = curr_active
    if live:
        opp_state.save()

    if not active_list and not dismissed:
        print("  No opportunities")

    return active_list, dismissed
