"""Domain types: frozen dataclasses and StrEnum."""

from dataclasses import dataclass, field
from enum import StrEnum


class HVACMode(StrEnum):
    OFF = "off"
    HEAT = "heat"
    COOL = "cool"
    HEAT_COOL = "heat_cool"
    AUTO = "auto"
    FAN_ONLY = "fan_only"
    DRY = "dry"


class HVACAction(StrEnum):
    HEATING = "heating"
    COOLING = "cooling"
    IDLE = "idle"
    OFF = "off"
    DRYING = "drying"
    FAN = "fan"


class BlowerMode(StrEnum):
    OFF = "off"
    LOW = "low"
    HIGH = "high"



class WeatherCondition(StrEnum):
    """All met.no weather conditions that HA can report."""

    CLEAR_NIGHT = "clear-night"
    CLOUDY = "cloudy"
    EXCEPTIONAL = "exceptional"
    FOG = "fog"
    HAIL = "hail"
    LIGHTNING = "lightning"
    LIGHTNING_RAINY = "lightning-rainy"
    PARTLY_CLOUDY = "partlycloudy"
    POURING = "pouring"
    RAINY = "rainy"
    SNOWY = "snowy"
    SNOWY_RAINY = "snowy-rainy"
    SUNNY = "sunny"
    WINDY = "windy"
    WINDY_VARIANT = "windy-variant"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ThermostatState:
    entity_id: str
    current_temp: float
    target_temp: float
    hvac_mode: HVACMode
    hvac_action: HVACAction


@dataclass(frozen=True)
class MiniSplitState:
    entity_id: str
    current_temp: float
    target_temp: float
    hvac_mode: HVACMode
    fan_mode: str


@dataclass(frozen=True)
class WeatherState:
    temperature: float
    humidity: float
    wind_speed: float
    wind_bearing: float
    condition: WeatherCondition


# Snapshot schema is defined by weatherstat.yaml and generated dynamically.
# See WeatherstatConfig.snapshot_column_defs().

# ── Control types ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RoomComfort:
    """Comfort profile for a single constraint label (sensor).

    Three-tier comfort bounds:
    - preferred_lo/preferred_hi: dead band — zero cost within this range.
      When equal, behaves as a point target (backward compatible).
    - acceptable_lo/acceptable_hi: normal HVAC action range with steep penalty (configurable, default 3×)
      outside these bounds. Maps to current min_temp/max_temp semantics.
    - backup_lo/backup_hi: worst-case hedge — defensive HVAC action when breached.
      Defaults to acceptable ± backup_margin when not specified.
    cold_penalty/hot_penalty: asymmetric weights for deviation from preferred band.
    """

    label: str  # "bedroom", "office", "upstairs", "downstairs"
    preferred_lo: float  # lower edge of preferred band
    preferred_hi: float  # upper edge (= preferred_lo for point target)
    acceptable_lo: float  # lower acceptable bound (steep penalty below)
    acceptable_hi: float  # upper acceptable bound (steep penalty above)
    backup_lo: float  # worst-case lower bound (defensive action below)
    backup_hi: float  # worst-case upper bound (defensive action above)
    cold_penalty: float = 2.0  # weight for being below preferred band
    hot_penalty: float = 1.0  # weight for being above preferred band


@dataclass(frozen=True)
class ComfortScheduleEntry:
    """A time-of-day range with its comfort profile."""

    start_hour: int  # 0-23 inclusive
    end_hour: int  # 0-23 inclusive (wraps at midnight if start > end)
    comfort: RoomComfort


@dataclass(frozen=True)
class ComfortSchedule:
    """Time-of-day comfort schedule for a sensor."""

    sensor: str  # sensor column name: "bedroom_temp"
    label: str  # display label: "bedroom"
    entries: tuple[ComfortScheduleEntry, ...] = field(default_factory=tuple)

    def comfort_at(self, hour: int) -> RoomComfort | None:
        """Return the active RoomComfort for a given hour, or None if no entry matches."""
        for entry in self.entries:
            if entry.start_hour <= entry.end_hour:
                if entry.start_hour <= hour < entry.end_hour:
                    return entry.comfort
            else:
                # Wraps midnight (e.g. 22-6)
                if hour >= entry.start_hour or hour < entry.end_hour:
                    return entry.comfort
        return None


@dataclass(frozen=True)
class AdvisoryDecision:
    """Decision for an advisory effector in a scenario.

    Advisory effectors are user-operated devices (windows, space heaters, blinds)
    that the system can observe and advise on but not directly control.

    action: "hold" keeps current state. Other values ("close", "open", "turn_off")
    indicate the advisory transition.
    transition_step: the 5-min step at which the transition occurs (0 = now).
    return_step: optional step at which the device returns to its previous state.
        Used for proactive advice: "open window now, close in 2h" →
        action="open", transition_step=0, return_step=24.
    """

    name: str  # "piano_window", "office_heater"
    action: str = "hold"  # "hold", "close", "open", "turn_off", "turn_on"
    transition_step: int = 0  # 5-min step at which transition occurs
    return_step: int | None = None  # 5-min step at which device returns to prior state


@dataclass(frozen=True)
class DeviceOpportunity:
    """Per-device marginal benefit from the advisory sweep.

    For each environment device, holds the delta between the best scenario where
    the device changes state and the best scenario where it holds — with the rest
    of the system (HVAC + other advisories) free to optimize around each choice.
    Negative cost_delta means the change is beneficial.
    """

    device: str  # environment entry name
    current_state: bool  # True if currently non-default (active)
    advisory: AdvisoryDecision  # the change action + timing from the best-change scenario
    idx: int  # scenario index of the best-change scenario
    cost_delta: float  # cost_change - cost_hold (< 0 = beneficial)


@dataclass(frozen=True)
class AdvisoryPlan:
    """Output of the advisory sweep: baseline hold cost + per-device alternatives + hedge.

    baseline_idx/baseline_cost refer to the cheapest scenario where every
    environment device's advisory action is "hold" — i.e. the plan the system
    should commit to if the user takes no action.

    opportunities are the per-device marginals (DeviceOpportunity list), already
    filtered to devices where both a best-change and best-hold scenario exist.

    backup_breaches are worst-case predicted bound violations derived from the
    baseline scenario's predictions — used both for warnings and for the
    defensive-HVAC override in the control loop.
    """

    baseline_idx: int | None
    baseline_cost: float  # inf when no all-hold scenario exists
    opportunities: tuple[DeviceOpportunity, ...] = ()
    backup_breaches: tuple[str, ...] = ()


_ACTION_VERBS: dict[str, tuple[str, str]] = {
    # (close_action, open_action) — close = return to default state, open = leave default.
    # For shades default_state is "open", so the verbs are flipped relative to windows.
    "window": ("close", "open"),
    "door": ("close", "open"),
    "shade": ("raise", "lower"),
    "vent": ("close", "open"),
    "heater": ("turn_off", "turn_on"),
}

_ACTIVE_DESCRIPTIONS: dict[str, str] = {
    "window": "open",
    "door": "open",
    "shade": "lowered",
    "vent": "open",
    "heater": "on",
}


@dataclass(frozen=True)
class EnvironmentEntryConfig:
    """An observable factor that affects physics (tau, solar, gain) but isn't system-controlled.

    Declared in the `environment` YAML section. Covers windows, doors, shades, vents,
    space heaters — anything the system observes for physics and optionally advises on.
    """

    name: str            # config key: "basement", "moss_garden_shade"
    entity_id: str       # HA entity: "binary_sensor.window_basement_intrusion"
    column: str          # EAV readings column: "basement_open"
    kind: str            # "window", "door", "shade" — drives display + action verbs
    default_state: str   # "closed" or "open" — what the system considers normal
    active_state: str    # HA state string when non-default (e.g., "on", "closed")
    advisory: bool = False  # True = included in advisory sweep
    value_type: str = "binary"  # "binary" (0/1) or "continuous" (0.0–1.0)
    storage: str = "sampled"  # "sampled" = every 5-min tick; "sparse" = only on change

    @property
    def label(self) -> str:
        """Human-readable label: 'moss_garden_shade' → 'moss garden shade'."""
        return self.name.replace("_", " ")

    @property
    def active_description(self) -> str:
        """State description when non-default: 'open', 'lowered', 'on'."""
        return _ACTIVE_DESCRIPTIONS.get(self.kind, "active")

    @property
    def close_action(self) -> str:
        """Action verb for returning to default state."""
        return _ACTION_VERBS.get(self.kind, ("close", "open"))[0]

    @property
    def open_action(self) -> str:
        """Action verb for leaving default state."""
        return _ACTION_VERBS.get(self.kind, ("close", "open"))[1]


@dataclass(frozen=True)
class EffectorDecision:
    """Decision for a single effector in a scenario.

    Unified type for all effectors — thermostats (trajectory), mini-splits
    (regulating), blowers (binary). Fields are used per control_type:
    - trajectory: mode="heating"/"off", delay_steps, duration_steps
    - regulating: mode="heat"/"cool"/"off", target (comfort setpoint)
    - binary: mode="off"/"low"/"high"/etc.
    """

    name: str  # "thermostat_upstairs", "blower_office", "mini_split_bedroom"
    mode: str = "off"  # "off", "heating", "heat", "cool", "low", "high"
    target: float | None = None  # regulating effectors: comfort setpoint (°F)
    delay_steps: int = 0  # trajectory effectors: 5-min steps before activation
    duration_steps: int | None = None  # trajectory effectors: steps of activation (None = full horizon)


@dataclass(frozen=True)
class Scenario:
    """An HVAC plan to evaluate: one decision per effector.

    Activity timelines depend on control_type (from sysid params):
    - trajectory: [OFF × delay] → [ON × duration] → [OFF × remainder]
    - regulating: proportional activity based on (target - room_temp) / band
    - binary: constant encoded mode value over full horizon

    Dependent effectors (e.g., blowers depending on a thermostat) have their
    activity multiplied by the dependency's active mask in the simulator.

    advisories: advisory effector decisions. Empty dict = hold all at current state
    (backward compatible with pre-advisory code).
    """

    effectors: dict[str, EffectorDecision]  # effector_name -> decision
    advisories: dict[str, AdvisoryDecision] = field(default_factory=dict)


@dataclass(frozen=True)
class ControlDecision:
    """A single control decision: all effectors and derived command targets."""

    timestamp: str
    effectors: tuple[EffectorDecision, ...]
    command_targets: dict[str, float] = field(default_factory=dict)  # effector_name -> setpoint for command JSON
    total_cost: float = 0.0
    comfort_cost: float = 0.0
    energy_cost: float = 0.0
    predictions: dict[str, dict[str, float]] = field(default_factory=dict)  # label -> {horizon -> temp}
    trajectory_info: dict[str, dict[str, int | None]] = field(default_factory=dict)
    rationale: dict[str, str] = field(default_factory=dict)  # effector_name -> explanation text
    sensor_costs: dict[str, float] = field(default_factory=dict)  # sensor -> decision comfort cost
    baseline_sensor_costs: dict[str, float] = field(default_factory=dict)  # sensor -> all-off comfort cost
    baseline_cost: float = 0.0  # all-off total comfort cost
    dry_run: bool = True


@dataclass(frozen=True)
class ControlState:
    """Persisted state to prevent rapid cycling."""

    last_decision_time: str  # ISO 8601
    setpoints: dict[str, float] = field(default_factory=dict)  # effector_name -> setpoint
    modes: dict[str, str] = field(default_factory=dict)  # effector_name -> mode
    mode_times: dict[str, str] = field(default_factory=dict)  # effector_name -> ISO timestamp
