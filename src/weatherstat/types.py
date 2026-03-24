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

    preferred: ideal temperature — continuous cost in both directions.
    min_temp/max_temp: hard rails with steep additional penalty (10×).
    cold_penalty/hot_penalty: asymmetric weights for deviation from preferred.
    """

    label: str  # "bedroom", "office", "upstairs", "downstairs"
    preferred: float  # ideal temperature (°F)
    min_temp: float  # hard lower bound (°F)
    max_temp: float  # hard upper bound (°F)
    cold_penalty: float = 2.0  # weight for being below preferred
    hot_penalty: float = 1.0  # weight for being above preferred


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
class WindowOpportunity:
    """A persistent opportunity to improve comfort/energy by toggling a window."""

    window: str  # window name
    action: str  # "open" or "close"
    comfort_improvement: float  # comfort cost reduction
    energy_saving: float  # energy cost reduction from HVAC changes
    total_benefit: float  # comfort_improvement + energy_saving
    message: str  # human-readable description
    first_seen: str = ""  # ISO timestamp when first detected
    notified: bool = False  # whether push notification was sent


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
    """

    effectors: dict[str, EffectorDecision]  # effector_name -> decision


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
    dry_run: bool = True


@dataclass(frozen=True)
class ControlState:
    """Persisted state to prevent rapid cycling."""

    last_decision_time: str  # ISO 8601
    setpoints: dict[str, float] = field(default_factory=dict)  # effector_name -> setpoint
    modes: dict[str, str] = field(default_factory=dict)  # effector_name -> mode
    mode_times: dict[str, str] = field(default_factory=dict)  # effector_name -> ISO timestamp
