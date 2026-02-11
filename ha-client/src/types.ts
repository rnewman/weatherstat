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
}

// ---- Snapshot (written to Parquet) ----

export interface SnapshotRow {
  timestamp: string; // ISO 8601
  // Thermostat zones
  thermostatUpstairsTemp: number;
  thermostatUpstairsTarget: number;
  thermostatUpstairsAction: string; // "heating" | "idle"
  thermostatDownstairsTemp: number;
  thermostatDownstairsTarget: number;
  thermostatDownstairsAction: string;
  // Mini splits (named by location)
  miniSplitBedroomTemp: number;
  miniSplitBedroomTarget: number;
  miniSplitBedroomMode: string; // "off" | "heat" | "cool" | ...
  miniSplitLivingRoomTemp: number;
  miniSplitLivingRoomTarget: number;
  miniSplitLivingRoomMode: string;
  // Blowers (mode captures speed level)
  blowerFamilyRoomMode: string; // "off" | "low" | "high"
  blowerOfficeMode: string;
  // Navien
  navienHeatingMode: string; // "Space Heating" | "Idle"
  navienHeatCapacity: number; // 0-100%
  // Environment
  outdoorTemp: number;
  outdoorHumidity: number;
  windSpeed: number;
  weatherCondition: string;
  indoorHumidity: number;
  anyWindowOpen: boolean;
  // Per-room temperatures
  upstairsAggregateTemp: number;
  downstairsAggregateTemp: number;
  familyRoomTemp: number;
  officeTemp: number;
  bedroomTemp: number;
  kitchenTemp: number;
  livingRoomTemp: number;
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
