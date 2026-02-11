/**
 * Shared domain types for the weatherstat HA client.
 *
 * The HAClient interface abstracts Home Assistant access so business logic
 * (collector, executor) doesn't depend on a specific transport (WebSocket, add-on API, etc.).
 */

// ---- HA Client abstraction ----

export interface HAEntityState {
  entity_id: string;
  state: string;
  attributes: Record<string, unknown>;
  last_changed: string;
  last_updated: string;
}

export interface HAServiceCall {
  domain: string;
  service: string;
  target?: { entity_id: string | string[] };
  serviceData?: Record<string, unknown>;
}

export interface HAClient {
  /** Get current states of all (or specific) entities. */
  getStates(entityIds?: string[]): Promise<HAEntityState[]>;

  /** Call a Home Assistant service. */
  callService(call: HAServiceCall): Promise<void>;

  /** Subscribe to entity state changes. Returns an unsubscribe function. */
  subscribeEntities(
    entityIds: string[],
    callback: (state: HAEntityState) => void,
  ): Promise<() => void>;

  /** Disconnect / clean up resources. */
  disconnect(): Promise<void>;
}

// ---- HVAC domain types ----

export type HVACMode = "off" | "heat" | "cool" | "heat_cool" | "auto";
export type FanMode = "auto" | "low" | "medium" | "high";

export interface ThermostatState {
  entityId: string;
  currentTemp: number;
  targetTemp: number;
  hvacMode: HVACMode;
  hvacAction: string; // "heating", "cooling", "idle", "off"
}

export interface MiniSplitState {
  entityId: string;
  currentTemp: number;
  targetTemp: number;
  hvacMode: HVACMode;
  fanMode: FanMode;
}

export interface BlowerState {
  entityId: string;
  isOn: boolean;
  speed: number | null; // null if not controllable
}

export interface WindowSensorState {
  entityId: string;
  isOpen: boolean;
}

export interface TempSensorState {
  entityId: string;
  temperature: number;
  location: string;
}

// ---- Weather (from HA weather entity) ----

export interface WeatherState {
  temperature: number;
  humidity: number;
  windSpeed: number;
  windBearing: number;
  condition: string; // "sunny", "cloudy", "rainy", etc.
  forecast: WeatherForecastEntry[];
}

export interface WeatherForecastEntry {
  datetime: string;
  temperature: number;
  templow: number | null;
  condition: string;
  precipitation: number | null;
  windSpeed: number | null;
}

// ---- Snapshot (written to Parquet) ----

export interface SnapshotRow {
  timestamp: string; // ISO 8601
  // Thermostats
  thermostatUpstairsTemp: number;
  thermostatUpstairsTarget: number;
  thermostatUpstairsAction: string;
  thermostatDownstairsTemp: number;
  thermostatDownstairsTarget: number;
  thermostatDownstairsAction: string;
  // Mini splits
  miniSplit1Temp: number;
  miniSplit1Target: number;
  miniSplit1Mode: string;
  miniSplit2Temp: number;
  miniSplit2Target: number;
  miniSplit2Mode: string;
  // Floor heat
  floorHeatOn: boolean;
  // Blowers (controllable ones)
  blower1On: boolean;
  blower2On: boolean;
  // Environment
  outdoorTemp: number;
  outdoorHumidity: number;
  windSpeed: number;
  weatherCondition: string;
  // Navien heater
  navienHeaterActive: boolean;
  // Window sensors (any open?)
  anyWindowOpen: boolean;
  // Indoor temps (various sensors)
  indoorTemps: Record<string, number>;
}

// ---- Prediction (read from JSON) ----

export interface Prediction {
  timestamp: string;
  thermostatUpstairsTarget: number;
  thermostatDownstairsTarget: number;
  miniSplit1Target: number;
  miniSplit1Mode: HVACMode;
  miniSplit2Target: number;
  miniSplit2Mode: HVACMode;
  floorHeatOn: boolean;
  blower1On: boolean;
  blower2On: boolean;
  confidence: number;
}
