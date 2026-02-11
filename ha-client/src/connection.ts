/**
 * WebSocket-based HAClient implementation using home-assistant-js-websocket.
 *
 * This is the concrete implementation for standalone/development use.
 * An add-on or custom integration would provide a different HAClient implementation
 * that uses HA's internal API instead.
 */

import {
  createConnection,
  createLongLivedTokenAuth,
  subscribeEntities,
  type Connection,
  type HassEntities,
} from "home-assistant-js-websocket";

import type {
  HAClient,
  HAEntityState,
  HAServiceCall,
} from "./types.ts";

function hassEntityToState(entityId: string, entities: HassEntities): HAEntityState | undefined {
  const entity = entities[entityId];
  if (!entity) return undefined;
  return {
    entity_id: entity.entity_id,
    state: entity.state,
    attributes: entity.attributes as Record<string, unknown>,
    last_changed: entity.last_changed,
    last_updated: entity.last_updated,
  };
}

export class WebSocketHAClient implements HAClient {
  private connection: Connection | null = null;
  private cachedEntities: HassEntities = {};
  private entityUnsubscribe: (() => void) | null = null;

  constructor(
    private readonly url: string,
    private readonly token: string,
  ) {}

  private async ensureConnection(): Promise<Connection> {
    if (this.connection) return this.connection;

    const auth = createLongLivedTokenAuth(this.url, this.token);
    this.connection = await createConnection({ auth });

    // Start global entity subscription to keep cache fresh
    this.entityUnsubscribe = subscribeEntities(this.connection, (entities) => {
      this.cachedEntities = entities;
    });

    return this.connection;
  }

  async getStates(entityIds?: string[]): Promise<HAEntityState[]> {
    await this.ensureConnection();

    const ids = entityIds ?? Object.keys(this.cachedEntities);
    const states: HAEntityState[] = [];

    for (const id of ids) {
      const state = hassEntityToState(id, this.cachedEntities);
      if (state) {
        states.push(state);
      }
    }

    return states;
  }

  async callService(call: HAServiceCall): Promise<void> {
    const conn = await this.ensureConnection();

    await conn.sendMessagePromise({
      type: "call_service",
      domain: call.domain,
      service: call.service,
      target: call.target,
      service_data: call.serviceData,
    });
  }

  async subscribeEntities(
    entityIds: string[],
    callback: (state: HAEntityState) => void,
  ): Promise<() => void> {
    const conn = await this.ensureConnection();

    const entitySet = new Set(entityIds);
    const unsub = subscribeEntities(conn, (entities) => {
      for (const entityId of entitySet) {
        const state = hassEntityToState(entityId, entities);
        if (state) {
          callback(state);
        }
      }
    });

    return unsub;
  }

  async disconnect(): Promise<void> {
    if (this.entityUnsubscribe) {
      this.entityUnsubscribe();
      this.entityUnsubscribe = null;
    }
    if (this.connection) {
      this.connection.close();
      this.connection = null;
    }
    this.cachedEntities = {};
  }
}
