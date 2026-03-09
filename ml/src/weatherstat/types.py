"""Domain types mirroring the TypeScript side, using frozen dataclasses and StrEnum."""

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


class NavienHeatingMode(StrEnum):
    SPACE_HEATING = "Space Heating"
    IDLE = "Idle"


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
    """Comfort bounds and penalty weights for a single room."""

    room: str  # "bedroom", "office", "upstairs", "downstairs"
    min_temp: float  # lower comfort bound (°F)
    max_temp: float  # upper comfort bound (°F)
    cold_penalty: float = 2.0  # weight for being below min
    hot_penalty: float = 1.0  # weight for being above max


@dataclass(frozen=True)
class ComfortScheduleEntry:
    """A time-of-day range with its comfort profile."""

    start_hour: int  # 0-23 inclusive
    end_hour: int  # 0-23 inclusive (wraps at midnight if start > end)
    comfort: RoomComfort


@dataclass(frozen=True)
class ComfortSchedule:
    """Time-of-day comfort profile for a room."""

    room: str
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
class BlowerDecision:
    """Control decision for a single blower fan."""

    name: str  # "family_room", "office"
    mode: str  # "off", "low", "high"


@dataclass(frozen=True)
class MiniSplitDecision:
    """Control decision for a single mini-split heat pump."""

    name: str  # "bedroom", "living_room"
    mode: str  # "off", "heat", "cool"
    target: float  # command target (derived from comfort schedule after sweep)


@dataclass(frozen=True)
class ThermostatTrajectory:
    """Trajectory for a slow effector: [OFF × delay] → [ON × duration] → [OFF × remainder].

    Used in the physics sweep where constant-action evaluation systematically
    mis-evaluates slow effectors (hydronic floor heat: 45-75 min lag).
    """

    heating: bool
    delay_steps: int = 0  # 5-min steps before activation (0 = start now)
    duration_steps: int | None = None  # 5-min steps of activation (None = full horizon)


@dataclass(frozen=True)
class TrajectoryScenario:
    """HVAC scenario with trajectory parameters for slow effectors (thermostats).

    Fast effectors (mini-splits, blowers) use constant activity over the full horizon.
    Blower timelines follow their zone thermostat's on/off pattern.
    Boiler timeline is derived as the OR of both thermostat timelines.
    """

    upstairs: ThermostatTrajectory
    downstairs: ThermostatTrajectory
    blowers: tuple[BlowerDecision, ...]
    mini_splits: tuple[MiniSplitDecision, ...]


@dataclass(frozen=True)
class ControlDecision:
    """A single control decision: all HVAC devices and derived setpoints."""

    timestamp: str
    upstairs_heating: bool
    downstairs_heating: bool
    upstairs_setpoint: float  # derived: current ± CAUTIOUS_OFFSET
    downstairs_setpoint: float
    blowers: tuple[BlowerDecision, ...] = ()
    mini_splits: tuple[MiniSplitDecision, ...] = ()
    total_cost: float = 0.0
    comfort_cost: float = 0.0
    energy_cost: float = 0.0
    room_predictions: dict[str, dict[str, float]] = field(default_factory=dict)  # room -> {horizon -> temp}
    # zone -> {delay_steps, duration_steps}
    trajectory_info: dict[str, dict[str, int | None]] = field(default_factory=dict)
    dry_run: bool = True


@dataclass(frozen=True)
class ControlState:
    """Persisted state to prevent rapid cycling."""

    last_decision_time: str  # ISO 8601
    upstairs_setpoint: float
    downstairs_setpoint: float
    blower_modes: dict[str, str] = field(default_factory=dict)  # name -> mode
    mini_split_modes: dict[str, str] = field(default_factory=dict)  # name -> mode
    mini_split_targets: dict[str, float] = field(default_factory=dict)  # name -> target


# ── Unified action types ─────────────────────────────────────────────────


class ExecutionType(StrEnum):
    ELECTRONIC = "electronic"
    ADVISORY = "advisory"


@dataclass(frozen=True)
class ActionOption:
    """One possible state of a controllable action."""

    name: str  # "open", "closed", "on", "off", etc.
    feature_overrides: dict[str, float]  # feature_col -> value
    energy_cost: float = 0.0


@dataclass(frozen=True)
class Action:
    """A controllable action — electronic or advisory."""

    name: str  # "window_bedroom", "thermostat_upstairs"
    options: tuple[ActionOption, ...]
    current: str  # current option name
    execution: ExecutionType
    effort_cost: float = 0.0  # penalty for recommending change


@dataclass(frozen=True)
class ActionRecommendation:
    """An advisory recommendation from the optimizer."""

    action_name: str  # "window_bedroom"
    recommended_state: str  # "open" or "closed"
    current_state: str
    comfort_improvement: float  # comfort_cost reduction
    message: str  # human-readable
