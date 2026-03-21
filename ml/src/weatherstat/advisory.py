"""Physics-based window opportunities — persistent open/close recommendations.

Evaluates window state toggles using the physics simulator to determine if
opening or closing a window would improve comfort and/or save energy. Called
from the control loop after the electronic plan is committed.

Two-threshold model:
- Opportunity threshold: minimum benefit to track (visible in control output).
- Notification threshold: minimum benefit to push a notification.

Opportunities persist across control cycles: active until conditions change,
then automatically dismissed.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

import requests

from weatherstat.config import (
    ADVISORY_COOLDOWNS,
    ADVISORY_EFFORT_COST,
    ADVISORY_NOTIFICATION_THRESHOLD,
    ADVISORY_OPPORTUNITY_THRESHOLD,
    ADVISORY_QUIET_HOURS,
    ADVISORY_STATE_FILE,
    HA_TOKEN,
    HA_URL,
)

if TYPE_CHECKING:
    from weatherstat.control import ControlState
    from weatherstat.simulator import HouseState, SimParams
    from weatherstat.types import ComfortSchedule, TrajectoryScenario, WindowOpportunity

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


# ── Opportunity state ────────────────────────────────────────────────────


@dataclass
class OpportunityState:
    """Persistent state for window opportunities across control cycles."""

    active: dict[str, dict]  # window_name -> opportunity data (serializable)
    cooldowns: dict[str, float]  # cooldown_key -> unix_timestamp

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
            return cls(active=data["active"], cooldowns=data["cooldowns"])
        # Old format: flat dict of cooldown timestamps
        return cls(active={}, cooldowns=data)

    def save(self) -> None:
        """Persist to disk."""
        ADVISORY_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        ADVISORY_STATE_FILE.write_text(json.dumps(
            {"active": self.active, "cooldowns": self.cooldowns}, indent=2,
        ))


# ── Legacy cooldown helpers (used by process_advisories) ─────────────────


def _cooldown_key(advisory: Advisory) -> str:
    """Composite cooldown key: type_window for per-window tracking."""
    if advisory.window:
        return f"{advisory.advisory_type.value}_{advisory.window}"
    return advisory.advisory_type.value


def load_advisory_state() -> dict[str, float]:
    """Load advisory cooldown state: key → last_sent_unix_timestamp."""
    state = OpportunityState.load()
    return state.cooldowns


def save_advisory_state(state: dict[str, float]) -> None:
    """Persist advisory cooldown state (legacy format)."""
    ADVISORY_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ADVISORY_STATE_FILE.write_text(json.dumps(state, indent=2))


def _is_on_cooldown(state: dict[str, float], key: str, advisory_type: AdvisoryType) -> bool:
    """Return True if the advisory was sent recently enough to suppress."""
    last_sent = state.get(key)
    if last_sent is None:
        return False
    cooldown = ADVISORY_COOLDOWNS.get(advisory_type.value, 3600)
    return (time.time() - last_sent) < cooldown


# ── Physics-based evaluation (legacy) ────────────────────────────────────


def evaluate_window_advisories(
    state: HouseState,
    winning_scenario: TrajectoryScenario,
    sim_params: SimParams,
    schedules: list[ComfortSchedule],
    base_hour: int,
) -> list[Advisory]:
    """Evaluate whether toggling any window would improve comfort.

    Legacy function — use evaluate_window_opportunities for energy-aware
    persistent opportunities.
    """
    from weatherstat.control import CONTROL_HORIZONS, compute_comfort_cost
    from weatherstat.simulator import HouseState as _HS
    from weatherstat.simulator import predict
    from weatherstat.yaml_config import load_config

    cfg = load_config()

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
            solar_fractions=state.solar_fractions,
        )

        _, toggled_preds = predict(toggled_state, [winning_scenario], sim_params, CONTROL_HORIZONS)
        toggled_dict = {t: float(toggled_preds[0, j]) for j, t in enumerate(target_names)}
        toggled_cost = compute_comfort_cost(toggled_dict, schedules, base_hour)

        improvement = baseline_cost - toggled_cost
        if improvement <= ADVISORY_EFFORT_COST:
            continue

        action = "Open" if not is_open else "Close"
        adv_type = AdvisoryType.FREE_COOLING if not is_open else AdvisoryType.CLOSE_WINDOWS

        best_label = window_name
        best_delta = 0.0
        best_baseline_temp: float | None = None
        best_toggled_temp: float | None = None
        for c in cfg.constraints:
            key_2h = f"{c.sensor}_t+24"
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


# ── Energy-aware opportunity evaluation ──────────────────────────────────


def evaluate_window_opportunities(
    state: HouseState,
    winning_scenario: TrajectoryScenario,
    winning_comfort_cost: float,
    winning_energy_cost: float,
    sim_params: SimParams,
    schedules: list[ComfortSchedule],
    base_hour: int,
    prev_state: ControlState | None = None,
    current_temps: dict[str, float] | None = None,
) -> list[WindowOpportunity]:
    """Evaluate window toggles for comfort and energy savings.

    Two-tier evaluation:
    1. Quick check: simulate winning scenario with window toggled → comfort delta.
    2. Re-sweep (if promising): full scenario sweep with toggled window to find
       best HVAC plan. Captures "open window + turn off mini split" savings.

    Returns list of WindowOpportunity sorted by total benefit (best first).
    """
    from weatherstat.control import (
        CONTROL_HORIZONS,
        HORIZON_WEIGHTS,
        compute_comfort_cost,
        sweep_scenarios_physics,
    )
    from weatherstat.simulator import HouseState as _HS
    from weatherstat.simulator import predict
    from weatherstat.types import WindowOpportunity
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

    opportunities: list[WindowOpportunity] = []

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
            solar_fractions=state.solar_fractions,
        )

        # Quick check: comfort improvement with same HVAC plan
        _, toggled_preds = predict(toggled_state, [winning_scenario], sim_params, CONTROL_HORIZONS)
        toggled_dict = {t: float(toggled_preds[0, j]) for j, t in enumerate(target_names)}
        quick_comfort = compute_comfort_cost(toggled_dict, schedules, base_hour)
        quick_benefit = (baseline_comfort - quick_comfort) / cost_norm

        if quick_benefit <= threshold:
            continue

        # Re-sweep: find best HVAC plan with toggled window
        # Uses the thermostat temps from current state for physical constraints
        up_temp = current_temps.get("upstairs", 71.0) if current_temps else 71.0
        dn_temp = current_temps.get("downstairs", 71.0) if current_temps else 71.0
        split_temps: dict[str, float] = {}
        if current_temps:
            for ms in cfg.mini_splits:
                for c in cfg.constraints:
                    if c.label == ms:
                        split_temps[ms] = current_temps.get(c.label, 71.0)
                        break

        resweep_decision, _resweep_scenario = sweep_scenarios_physics(
            current_temps=state.current_temps,
            outdoor_temp=state.outdoor_temp,
            forecast_temps=state.forecast_temps,
            window_states=toggled_windows,
            sim_params=sim_params,
            hour_of_day=state.hour_of_day,
            recent_history=state.recent_history,
            up_current=up_temp,
            dn_current=dn_temp,
            current_split_temps=split_temps,
            schedules=schedules,
            base_hour=base_hour,
            prev_state=prev_state,
            solar_fractions=state.solar_fractions,
        )

        comfort_improvement = (winning_comfort_cost - resweep_decision.comfort_cost) / cost_norm
        energy_saving = winning_energy_cost - resweep_decision.energy_cost
        total_benefit = comfort_improvement + energy_saving

        if total_benefit <= threshold:
            continue

        action = "Open" if not is_open else "Close"

        parts = [f"{action} {window_name} window"]
        if not is_open:
            parts.append(f"({state.outdoor_temp:.0f}°F outside)")
        if comfort_improvement > 0.01:
            parts.append(f"comfort +{comfort_improvement:.2f}")
        if energy_saving > 0.01:
            parts.append(f"energy saving +{energy_saving:.3f}")
        message = " — ".join(parts)

        opportunities.append(WindowOpportunity(
            window=window_name,
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


def _notification_tag(window: str) -> str:
    return f"weatherstat_opportunity_{window}"


def process_opportunities(
    new_opportunities: list[WindowOpportunity],
    live: bool = False,
    notification_target: str = "persistent_notification",
    current_hour: int | None = None,
) -> tuple[list[WindowOpportunity], list[str]]:
    """Manage opportunity lifecycle: add/keep/remove, dispatch notifications.

    Returns:
        (active_opportunities, dismissed_windows).
    """
    from weatherstat.types import WindowOpportunity

    opp_state = OpportunityState.load()
    now_iso = datetime.now(UTC).isoformat()
    now_ts = time.time()

    quiet = False
    if current_hour is not None:
        start, end = ADVISORY_QUIET_HOURS
        quiet = start <= current_hour < end if start <= end else current_hour >= start or current_hour < end

    # Build candidate set from new opportunities
    candidates = {opp.window: opp for opp in new_opportunities}
    prev_active = set(opp_state.active.keys())
    curr_active: dict[str, dict] = {}
    active_list: list[WindowOpportunity] = []
    dismissed: list[str] = []

    print("\n[opportunities] Evaluating window opportunities...")

    if not candidates and not prev_active:
        print("  No opportunities")
        return [], []

    for window, opp in candidates.items():
        was_active = window in prev_active
        prev_data = opp_state.active.get(window, {})
        first_seen = prev_data.get("first_seen", now_iso) if was_active else now_iso
        was_notified = prev_data.get("notified", False) if was_active else False

        # Check notification threshold + cooldown
        should_notify = (
            opp.total_benefit >= ADVISORY_NOTIFICATION_THRESHOLD
            and not was_notified
            and not quiet
        )
        cooldown_key = f"opportunity_{window}"
        if should_notify and cooldown_key in opp_state.cooldowns:
            cooldown_secs = ADVISORY_COOLDOWNS.get("free_cooling", 14400)
            if (now_ts - opp_state.cooldowns[cooldown_key]) < cooldown_secs:
                should_notify = False

        notified = was_notified or should_notify

        if should_notify and live:
            tag = _notification_tag(window)
            title = f"{opp.action.title()} {window} window"
            send_ha_notification(title, opp.message, tag, notification_target)
            opp_state.cooldowns[cooldown_key] = now_ts
            print(f"  → Notified: {title}")

        active_opp = WindowOpportunity(
            window=opp.window,
            action=opp.action,
            comfort_improvement=opp.comfort_improvement,
            energy_saving=opp.energy_saving,
            total_benefit=opp.total_benefit,
            message=opp.message,
            first_seen=first_seen,
            notified=notified,
        )
        active_list.append(active_opp)
        curr_active[window] = asdict(active_opp)

        status = "active" if was_active else "new"
        print(f"  {opp.action.title()} {window}: benefit={opp.total_benefit:.2f} [{status}]")

    # Dismiss expired opportunities
    for window in prev_active - set(candidates.keys()):
        prev_data = opp_state.active[window]
        if prev_data.get("notified", False) and live:
            tag = _notification_tag(window)
            dismiss_ha_notification(tag, notification_target)
            print(f"  Dismissed: {window}")
        dismissed.append(window)

    opp_state.active = curr_active
    if live:
        opp_state.save()

    if not active_list and not dismissed:
        print("  No opportunities")

    return active_list, dismissed


# ── Legacy dispatch (kept for backward compat) ──────────────────────────


def process_advisories(
    advisories: list[Advisory],
    live: bool = False,
    notification_target: str = "persistent_notification",
    current_hour: int | None = None,
) -> list[Advisory]:
    """Apply cooldowns, check quiet hours, and dispatch notifications.

    Legacy function — use process_opportunities for persistent lifecycle.
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

    # Send a single combined notification for all dispatched advisories
    if live and dispatched:
        now = time.time()
        lines = [a.message for a in dispatched]
        combined_message = "\n".join(lines)
        title = f"Window advisory ({len(dispatched)})" if len(dispatched) > 1 else dispatched[0].title
        if send_ha_notification(title, combined_message, "weatherstat_advisory", notification_target):
            for advisory in dispatched:
                state[_cooldown_key(advisory)] = now
            print(f"  → Sent to HA ({notification_target})")

        save_advisory_state(state)

    return dispatched
