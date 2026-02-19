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

  /** Call a Home Assistant service and return the response data. */
  callServiceWithResponse(call: HAServiceCall): Promise<Record<string, unknown>>;

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

export type BlowerMode = "off" | "low" | "high";

export interface BlowerState {
  entityId: string;
  mode: BlowerMode;
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
  cloud_coverage: number | null;
}

// ---- Snapshot (written to SQLite) ----

/** Dynamic snapshot row — columns are driven entirely by weatherstat.yaml.
 *
 * The only guaranteed field is `timestamp`. All other columns (temperatures,
 * HVAC states, window sensors, weather) are generated at runtime by the
 * YAML config loader. Adding a sensor to the YAML is sufficient; no TS
 * interface changes needed.
 */
export interface SnapshotRow {
  timestamp: string; // ISO 8601
  [column: string]: string | number | boolean;
}

// ---- Prediction (read from JSON) ----

export interface Prediction {
  timestamp: string;
  thermostatUpstairsTarget: number;
  thermostatDownstairsTarget: number;
  miniSplitBedroomTarget: number;
  miniSplitBedroomMode: HVACMode;
  miniSplitLivingRoomTarget: number;
  miniSplitLivingRoomMode: HVACMode;
  blowerFamilyRoomMode: string;
  blowerOfficeMode: string;
  confidence: number;
}
