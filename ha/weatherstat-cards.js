/*
 * Weatherstat cards for Home Assistant.
 *
 * Reads the entities Weatherstat publishes over MQTT discovery
 * (src/weatherstat/mqtt_bridge.py):
 *
 *   sensor.weatherstat_<room>                feels-like temp; comfort band, overlay,
 *                                            predictions, cost, source sensor as attributes
 *   number.weatherstat_<room>_comfort        the room's comfort overlay (−/+)
 *   sensor.weatherstat_<room>_comfort_until  when that overlay expires
 *   sensor.weatherstat_status                live / dry run; plan, rationale, advisories, health
 *
 * Cards:
 *
 *   type: custom:weatherstat-room-card      one room (room: bedroom)
 *   type: custom:weatherstat-rooms-card     every room as a compact list (rooms: [...] optional)
 *   type: custom:weatherstat-status-card    the controller: plan, why, advisories, health
 *
 * Self-contained (no imports) so it can be registered as an inline module resource.
 */

const WS_VERSION = "1.0.0";

const STATUS_COLOR = {
  cold: "var(--weatherstat-cold-color, #2f7bf5)",
  cool: "var(--weatherstat-cool-color, #5ea8f2)",
  ok: "var(--weatherstat-ok-color, var(--success-color, #3a9d4d))",
  warm: "var(--weatherstat-warm-color, #f0a020)",
  hot: "var(--weatherstat-hot-color, var(--error-color, #db4437))",
};

const MODE_COLOR = {
  heat: "var(--state-climate-heat-color, #ff8100)",
  cool: "var(--state-climate-cool-color, #2b9af9)",
  low: "var(--state-fan-on-color, #00a99d)",
  high: "var(--state-fan-on-color, #00a99d)",
  on: "var(--state-active-color, #fdd835)",
};

const PENDING_MS = 4000; // trust a local −/+ value this long while the bridge echoes it back

// ── Helpers ────────────────────────────────────────────────────────────────

const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const num = (v) => {
  if (v === null || v === undefined || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

const fmt = (v, digits = 1) => (v === null || v === undefined ? "—" : Number(v).toFixed(digits));

const fmtDelta = (v) => {
  const n = num(v) ?? 0;
  if (Math.abs(n) < 0.05) return "±0°";
  const s = Math.abs(n - Math.round(n)) < 0.05 ? Math.round(n).toString() : n.toFixed(1);
  return `${n > 0 ? "+" : ""}${s}°`;
};

const fmtDuration = (minutes) => {
  if (minutes === null || minutes === undefined) return "";
  const m = Math.max(0, Math.round(minutes));
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  const r = m % 60;
  return r ? `${h}h ${r}m` : `${h}h`;
};

const minutesUntil = (iso) => {
  if (!iso) return null;
  const t = Date.parse(iso);
  return Number.isFinite(t) ? (t - Date.now()) / 60000 : null;
};

const titleCase = (s) => String(s ?? "").replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());

const statusFor = (raw, a) => {
  if (raw === null || a.acceptable_lo === null) return null;
  const v = Math.round(raw * 10) / 10; // classify what is displayed: 70.95 shows as 71.0, so it is in band at 71
  if (v < a.acceptable_lo) return "cold";
  if (v < a.preferred_lo) return "cool";
  if (v > a.acceptable_hi) return "hot";
  if (v > a.preferred_hi) return "warm";
  return "ok";
};

const CONDITION = {
  partlycloudy: "partly cloudy", clear: "clear", "clear-night": "clear", sunny: "sunny", cloudy: "cloudy", rainy: "rain",
  pouring: "heavy rain", fog: "fog", snowy: "snow", "snowy-rainy": "sleet", windy: "windy", "windy-variant": "windy",
  lightning: "lightning", "lightning-rainy": "storms", hail: "hail", exceptional: "",
};

const STATUS_WORD = { cold: "cold", cool: "a bit cool", ok: "comfortable", warm: "a bit warm", hot: "hot" };

const effectorIcon = (name) => {
  if (name.includes("mini_split") || name.includes("split")) return "mdi:air-conditioner";
  if (name.includes("blower") || name.includes("fan")) return "mdi:fan";
  if (name.includes("thermostat")) return "mdi:heating-coil";
  return "mdi:thermostat";
};

function roomIds(room) {
  return {
    room: `sensor.weatherstat_${room}`,
    comfort: `number.weatherstat_${room}_comfort`,
    until: `sensor.weatherstat_${room}_comfort_until`,
  };
}

function fireMoreInfo(node, entityId) {
  const ev = new Event("hass-more-info", { bubbles: true, composed: true });
  ev.detail = { entityId };
  node.dispatchEvent(ev);
}

// ── Room model ─────────────────────────────────────────────────────────────

// HA drops an entity's attributes while it is unavailable (Weatherstat stopped). Remember the last
// ones per entity so an offline room still shows its live HA temperature and last-known band.
const LAST_ATTRS = new Map();

function roomModel(hass, cfg, pending) {
  const room = cfg.room || (cfg.entity || "").replace(/^sensor\.weatherstat_/, "");
  const ids = roomIds(room);
  if (cfg.entity) ids.room = cfg.entity;
  const st = hass.states[ids.room];
  if (!st) return { missing: true, room, ids };
  const available = st.state !== "unavailable" && st.state !== "unknown";
  if (available && st.attributes && st.attributes.label) LAST_ATTRS.set(ids.room, st.attributes);
  const a = available ? st.attributes || {} : { ...(LAST_ATTRS.get(ids.room) || {}), ...(st.attributes || {}) };
  if (cfg.source_entity) a.source_entity = cfg.source_entity;
  if (a.comfort_entity) ids.comfort = a.comfort_entity;

  const src = a.source_entity ? hass.states[a.source_entity] : null;
  const live = src ? num(src.state) : null;
  const temp = live ?? num(a.temperature);
  const offset = num(a.mrt_offset) ?? 0;
  const feels = temp === null ? null : temp - offset;

  const comfortSt = hass.states[ids.comfort];
  const ca = comfortSt ? comfortSt.attributes || {} : {};
  let overlay = comfortSt ? num(comfortSt.state) : num(a.overlay);
  const p = pending[ids.comfort];
  if (p && Date.now() - p.at < PENDING_MS) overlay = p.value;
  const untilSt = hass.states[ids.until];
  const untilIso = untilSt && untilSt.state && !["unknown", "unavailable"].includes(untilSt.state) ? untilSt.state : a.overlay_until;

  // Bands in the payload include the overlay that was active when Weatherstat built them.
  // Re-base them on the current overlay so a fresh −/+ moves the band at once.
  const shift = (overlay ?? 0) - (num(a.overlay) ?? 0);
  const band = (k) => (num(a[k]) === null ? null : num(a[k]) + shift);
  const bands = {
    acceptable_lo: band("acceptable_lo"),
    preferred_lo: band("preferred_lo"),
    preferred_hi: band("preferred_hi"),
    acceptable_hi: band("acceptable_hi"),
  };
  const status = statusFor(feels, bands);

  return {
    room,
    ids,
    name: cfg.name || a.name || titleCase(room),
    unit: a.unit || st.attributes.unit_of_measurement || "°",
    available,
    temp,
    feels,
    offset,
    status,
    bands,
    overlay: overlay ?? 0,
    untilMinutes: overlay && Math.abs(overlay) >= 0.05 ? minutesUntil(untilIso) : null,
    predictions: a.predictions || {},
    sourceEntity: a.source_entity || ids.room,
    step: num(ca.step) ?? 1,
    min: num(ca.min) ?? -5,
    max: num(ca.max) ?? 5,
    controllable: !!comfortSt && comfortSt.state !== "unavailable",
  };
}

// ── Rendering ──────────────────────────────────────────────────────────────

function barHtml(m, horizon) {
  const b = m.bands;
  if (b.acceptable_lo === null || m.feels === null) return `<div class="bar empty"></div>`;
  const pred = num(m.predictions[horizon]);
  const values = [b.acceptable_lo, b.acceptable_hi, m.feels, m.temp];
  if (pred !== null) values.push(pred);
  const lo = Math.min(...values) - 1.2;
  const hi = Math.max(...values) + 1.2;
  const pct = (v) => ((v - lo) / (hi - lo)) * 100;
  const span = (a, z, cls, extra = "") =>
    `<div class="${cls}" style="left:${pct(a).toFixed(2)}%;width:${Math.max(0, pct(z) - pct(a)).toFixed(2)}%;${extra}"></div>`;

  let html = `<div class="bar"><div class="track"></div>`;
  html += span(b.acceptable_lo, b.acceptable_hi, "acc");
  if (Math.abs(m.overlay) >= 0.05) html += span(b.preferred_lo - m.overlay, b.preferred_hi - m.overlay, "base");
  html += span(b.preferred_lo, b.preferred_hi, "pref");
  if (pred !== null && Math.abs(pred - m.feels) >= 0.1) {
    const a = Math.min(pred, m.feels);
    const z = Math.max(pred, m.feels);
    html += span(a, z, `trend ${pred > m.feels ? "up" : "down"}`);
    html += `<div class="pred" style="left:${pct(pred).toFixed(2)}%" title="${esc(horizon)}: ${fmt(pred)}°"></div>`;
  }
  if (Math.abs(m.offset) >= 0.2 && m.temp !== null) {
    html += `<div class="air" style="left:${pct(m.temp).toFixed(2)}%" title="Air ${fmt(m.temp)}°"></div>`;
  }
  html += `<div class="now" style="left:${pct(m.feels).toFixed(2)}%;background:${STATUS_COLOR[m.status] || "var(--primary-color)"}"></div>`;
  html += `<div class="tick" style="left:${pct(b.preferred_lo).toFixed(2)}%">${fmt(b.preferred_lo, 0)}</div>`;
  html += `<div class="tick" style="left:${pct(b.preferred_hi).toFixed(2)}%">${fmt(b.preferred_hi, 0)}</div>`;
  html += `</div>`;
  return html;
}

function controlsHtml(m) {
  const active = Math.abs(m.overlay) >= 0.05;
  const disabled = m.controllable ? "" : "disabled";
  const label = active
    ? `<span class="delta ${m.overlay > 0 ? "warmer" : "cooler"}">${fmtDelta(m.overlay)}</span>` +
      (m.untilMinutes !== null ? `<span class="left">${fmtDuration(m.untilMinutes)}</span>` : "")
    : `<span class="idle">comfort</span>`;
  return `<div class="ctl">
      <button class="btn" data-action="down" ${disabled} title="Cooler" aria-label="Cooler"><ha-icon icon="mdi:minus"></ha-icon></button>
      <button class="mid" data-action="${active ? "clear" : "noop"}" ${active ? "" : "tabindex=-1"} title="${active ? "Clear" : ""}">${label}${
        active ? `<ha-icon class="x" icon="mdi:close-circle-outline"></ha-icon>` : ""
      }</button>
      <button class="btn" data-action="up" ${disabled} title="Warmer" aria-label="Warmer"><ha-icon icon="mdi:plus"></ha-icon></button>
    </div>`;
}

function roomHtml(m, cfg, compact) {
  if (m.missing) {
    return `<div class="room missing" data-room="${esc(m.room)}">No Weatherstat room <b>${esc(m.room)}</b> — is the MQTT bridge running?</div>`;
  }
  const horizon = cfg.horizon || "2h";
  const pred = num(m.predictions[horizon]);
  const subParts = [];
  if (!m.available) subParts.push("Weatherstat offline");
  else {
    if (m.status) subParts.push(STATUS_WORD[m.status]);
    if (Math.abs(m.offset) >= 0.2) subParts.push(`feels ${fmt(m.feels)}°`);
    if (pred !== null && !compact) subParts.push(`${fmt(pred)}° in ${esc(horizon)}`);
  }
  const color = m.available ? STATUS_COLOR[m.status] || "var(--secondary-text-color)" : "var(--disabled-text-color)";
  return `<div class="room ${compact ? "compact" : ""} ${m.available ? "" : "offline"}" data-room="${esc(m.room)}">
    <div class="head" data-action="more-info">
      <ha-icon class="icon" icon="mdi:home-thermometer-outline" style="color:${color}"></ha-icon>
      <div class="names"><div class="name">${esc(m.name)}</div><div class="sub">${esc(subParts.join(" · "))}</div></div>
      <div class="temp">${fmt(m.temp)}<span class="unit">${esc(m.unit)}</span></div>
    </div>
    <div class="row2">${barHtml(m, horizon)}${controlsHtml(m)}</div>
  </div>`;
}

const ROOM_CSS = `
  :host { display: block; }
  ha-card { padding: 12px 14px; }
  ha-card.list { padding: 6px 14px; }
  .room + .room { border-top: 1px solid var(--divider-color); }
  .list .room { padding: 9px 0; }
  .head { display: flex; align-items: center; gap: 10px; cursor: pointer; min-width: 0; }
  .icon { --mdc-icon-size: 22px; flex: none; }
  .names { flex: 1; min-width: 0; }
  .name { font-weight: 500; font-size: 15px; line-height: 20px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .sub { color: var(--secondary-text-color); font-size: 12px; line-height: 16px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .temp { font-size: 24px; font-weight: 400; font-variant-numeric: tabular-nums; flex: none; }
  .compact .temp { font-size: 20px; }
  .unit { font-size: 13px; color: var(--secondary-text-color); margin-left: 1px; }
  .row2 { display: flex; align-items: center; gap: 12px; margin-top: 8px; }
  .bar { position: relative; flex: 1; height: 30px; min-width: 80px; }
  .bar .track, .bar .acc, .bar .pref, .bar .base, .bar .trend { position: absolute; top: 8px; height: 8px; border-radius: 4px; }
  .bar .track { left: 0; right: 0; background: var(--divider-color); opacity: .55; }
  .bar .acc { background: var(--weatherstat-ok-color, var(--success-color, #3a9d4d)); opacity: .22; }
  .bar .pref { background: var(--weatherstat-ok-color, var(--success-color, #3a9d4d)); opacity: .6; }
  .bar .base { top: 6px; height: 12px; border: 1px dashed var(--secondary-text-color); background: transparent; opacity: .7; box-sizing: border-box; }
  .bar .trend { top: 11px; height: 2px; border-radius: 1px; background: var(--primary-text-color); opacity: .45; }
  .bar .pred { position: absolute; top: 7px; width: 0; height: 0; margin-left: -5px; border: 5px solid transparent; border-top: 6px solid var(--primary-text-color); opacity: .6; }
  .bar .air { position: absolute; top: 5px; width: 2px; height: 14px; margin-left: -1px; background: var(--secondary-text-color); opacity: .7; border-radius: 1px; }
  .bar .now { position: absolute; top: 5px; width: 14px; height: 14px; margin-left: -7px; border-radius: 50%; box-shadow: 0 0 0 2px var(--card-background-color, #fff); }
  .bar .tick { position: absolute; top: 19px; transform: translateX(-50%); font-size: 10px; line-height: 11px; color: var(--secondary-text-color); font-variant-numeric: tabular-nums; }
  .bar.empty { height: 8px; }
  .ctl { display: flex; align-items: center; flex: none; gap: 2px; }
  .btn { width: 36px; height: 36px; border-radius: 18px; border: none; cursor: pointer; display: inline-flex; align-items: center; justify-content: center;
         background: var(--secondary-background-color, rgba(127,127,127,.12)); color: var(--primary-text-color); padding: 0; }
  .btn:hover { filter: brightness(.95); }
  .btn:active { transform: scale(.94); }
  .btn[disabled] { opacity: .35; cursor: default; }
  .btn ha-icon { --mdc-icon-size: 20px; }
  .mid { min-width: 64px; height: 36px; border: none; background: none; color: inherit; padding: 0 4px; cursor: default; display: inline-flex; flex-direction: column;
         align-items: center; justify-content: center; line-height: 14px; position: relative; font: inherit; }
  .mid[data-action="clear"] { cursor: pointer; }
  .mid .x { position: absolute; top: -3px; right: -4px; --mdc-icon-size: 13px; color: var(--secondary-text-color); }
  .delta { font-weight: 600; font-size: 15px; }
  .delta.warmer { color: var(--state-climate-heat-color, #ff8100); }
  .delta.cooler { color: var(--state-climate-cool-color, #2b9af9); }
  .left { font-size: 11px; color: var(--secondary-text-color); }
  .idle { font-size: 11px; color: var(--secondary-text-color); text-transform: lowercase; }
  .offline .bar, .offline .ctl { opacity: .4; }
  .missing { color: var(--secondary-text-color); font-size: 13px; padding: 6px 0; }
  .title { font-size: 16px; font-weight: 500; padding: 6px 0 2px; display: flex; align-items: center; gap: 8px; }
  .legend { font-size: 11px; color: var(--secondary-text-color); padding: 2px 0 6px; }
`;

// ── Shared base ────────────────────────────────────────────────────────────

class WeatherstatBase extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._pending = {};
    this._sig = "";
  }

  connectedCallback() {
    // Remaining-time labels age without any state change; re-render every 30s.
    this._timer = setInterval(() => this._render(true), 30000);
  }

  disconnectedCallback() {
    clearInterval(this._timer);
  }

  set hass(hass) {
    this._hass = hass;
    this._render(false);
  }

  _watched() {
    return [];
  }

  _render(force) {
    if (!this._hass || !this._config) return;
    const ids = this._watched();
    const sig = ids.map((id) => {
      const s = this._hass.states[id];
      return s ? `${s.state}|${s.last_updated}` : "-";
    }).join(";");
    if (!force && sig === this._sig) return;
    this._sig = sig;
    this.shadowRoot.innerHTML = this._html();
    this._bind();
  }

  _bind() {}
}

// ── Room card(s) ───────────────────────────────────────────────────────────

class WeatherstatRoomsBase extends WeatherstatBase {
  _rooms() {
    return [];
  }

  _watched() {
    const out = [];
    for (const cfg of this._rooms()) {
      const ids = roomIds(cfg.room);
      const st = this._hass.states[cfg.entity || ids.room];
      out.push(cfg.entity || ids.room, ids.comfort, ids.until);
      if (cfg.source_entity) out.push(cfg.source_entity);
      else if (st && st.attributes.source_entity) out.push(st.attributes.source_entity);
    }
    return out;
  }

  _bind() {
    this.shadowRoot.querySelectorAll(".room").forEach((el) => {
      const room = el.dataset.room;
      const cfg = this._rooms().find((r) => r.room === room) || { room };
      el.addEventListener("click", (ev) => {
        const target = ev.target.closest("[data-action]");
        if (!target || target.disabled) return;
        const m = roomModel(this._hass, cfg, this._pending);
        if (m.missing) return;
        const action = target.dataset.action;
        if (action === "more-info") fireMoreInfo(this, m.sourceEntity);
        else if (action === "up") this._setOverlay(m, m.overlay + m.step);
        else if (action === "down") this._setOverlay(m, m.overlay - m.step);
        else if (action === "clear") this._setOverlay(m, 0);
      });
    });
  }

  _setOverlay(m, value) {
    const v = Math.max(m.min, Math.min(m.max, Math.round(value / m.step) * m.step));
    this._pending[m.ids.comfort] = { value: v, at: Date.now() };
    this._hass.callService("number", "set_value", { entity_id: m.ids.comfort, value: v });
    this._render(true);
  }
}

class WeatherstatRoomCard extends WeatherstatRoomsBase {
  setConfig(config) {
    if (!config.room && !config.entity) throw new Error("Set `room:` (e.g. bedroom) or `entity:` (sensor.weatherstat_bedroom)");
    this._config = { ...config, room: config.room || config.entity.replace(/^sensor\.weatherstat_/, "") };
  }

  _rooms() {
    return [this._config];
  }

  _html() {
    const m = roomModel(this._hass, this._config, this._pending);
    return `<style>${ROOM_CSS}</style><ha-card>${roomHtml(m, this._config, false)}</ha-card>`;
  }

  getCardSize() {
    return 2;
  }

  getGridOptions() {
    return { columns: 6, min_columns: 6 };
  }

  static getStubConfig(hass) {
    const first = Object.keys(hass.states).find((id) => /^sensor\.weatherstat_(?!status)/.test(id) && !id.endsWith("_until"));
    return { room: first ? first.replace(/^sensor\.weatherstat_/, "") : "bedroom" };
  }
}

class WeatherstatRoomsCard extends WeatherstatRoomsBase {
  setConfig(config) {
    this._config = { ...config };
    const rooms = config.rooms || [];
    this._explicit = rooms.map((r) => (typeof r === "string" ? { room: r } : { ...r }));
  }

  _rooms() {
    if (this._explicit.length) return this._explicit;
    // Auto: every Weatherstat room sensor, alphabetically.
    return Object.keys(this._hass.states)
      .filter((id) => id.startsWith("sensor.weatherstat_") && this._hass.states[id].attributes.comfort_entity)
      .map((id) => ({ room: id.replace(/^sensor\.weatherstat_/, "") }))
      .sort((a, b) => a.room.localeCompare(b.room));
  }

  _html() {
    const rows = this._rooms()
      .map((cfg) => roomHtml(roomModel(this._hass, { ...this._config, ...cfg }, this._pending), { ...this._config, ...cfg }, true))
      .join("");
    const title = this._config.title ? `<div class="title">${esc(this._config.title)}</div>` : "";
    return `<style>${ROOM_CSS}</style><ha-card class="list">${title}${rows || '<div class="missing">No Weatherstat rooms found.</div>'}</ha-card>`;
  }

  getCardSize() {
    return 1 + this._rooms().length * 2;
  }

  getGridOptions() {
    return { columns: 12, min_columns: 6 };
  }

  static getStubConfig() {
    return {};
  }
}

// ── Status card ────────────────────────────────────────────────────────────

const STATUS_CSS = `
  :host { display: block; }
  ha-card { padding: 12px 14px 10px; }
  .top { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .top .title { font-size: 16px; font-weight: 500; margin-right: auto; display: flex; align-items: center; gap: 8px; }
  .chip { font-size: 12px; line-height: 22px; padding: 0 9px; border-radius: 11px; background: var(--secondary-background-color, rgba(127,127,127,.12));
          color: var(--primary-text-color); white-space: nowrap; }
  .chip.live { background: var(--success-color, #3a9d4d); color: #fff; }
  .chip.dry { background: var(--warning-color, #f0a020); color: #fff; }
  .chip.off { background: var(--disabled-color, #bdbdbd); color: #fff; }
  .chip.busy { background: var(--info-color, #039be5); color: #fff; }
  h4 { margin: 14px 0 6px; font-size: 12px; font-weight: 500; letter-spacing: .04em; text-transform: uppercase; color: var(--secondary-text-color); }
  .eff { display: grid; grid-template-columns: 26px 1fr auto; column-gap: 8px; row-gap: 1px; align-items: center; padding: 5px 0; }
  .eff + .eff { border-top: 1px solid var(--divider-color); }
  .eff ha-icon { --mdc-icon-size: 20px; grid-row: span 2; align-self: start; margin-top: 1px; }
  .eff .n { font-size: 14px; }
  .eff .m { font-size: 12px; font-weight: 600; text-transform: uppercase; }
  .eff .why { grid-column: 2 / span 2; font-size: 12px; color: var(--secondary-text-color); }
  .eff .ovr { grid-column: 2 / span 2; font-size: 12px; color: var(--warning-color, #f0a020); }
  .quiet { font-size: 13px; color: var(--secondary-text-color); }
  .line { font-size: 13px; padding: 2px 0; display: flex; gap: 6px; align-items: baseline; }
  .line.warn { color: var(--error-color, #db4437); }
  .line ha-icon { --mdc-icon-size: 16px; position: relative; top: 2px; }
  .foot { margin-top: 12px; font-size: 12px; color: var(--secondary-text-color); display: flex; flex-wrap: wrap; gap: 4px 12px; }
  .foot .bad { color: var(--error-color, #db4437); }
  .offline { color: var(--secondary-text-color); font-size: 13px; padding: 8px 0 2px; }
`;

class WeatherstatStatusCard extends WeatherstatBase {
  setConfig(config) {
    this._config = { entity: "sensor.weatherstat_status", show_off: true, ...config };
  }

  _watched() {
    return [this._config.entity];
  }

  _html() {
    const st = this._hass.states[this._config.entity];
    const head = (chips) =>
      `<div class="top"><div class="title"><ha-icon icon="mdi:home-thermometer"></ha-icon>${esc(this._config.title || "Weatherstat")}</div>${chips}</div>`;
    if (!st) {
      return `<style>${STATUS_CSS}</style><ha-card>${head("")}<div class="offline">${esc(this._config.entity)} not found — is the MQTT bridge running?</div></ha-card>`;
    }
    if (st.state === "unavailable" || st.state === "unknown") {
      return `<style>${STATUS_CSS}</style><ha-card>${head('<span class="chip off">offline</span>')}<div class="offline">Weatherstat is not running. Rooms keep their last settings.</div></ha-card>`;
    }
    const a = st.attributes || {};
    const chips = [];
    chips.push(a.mode === "live" ? '<span class="chip live">live</span>' : '<span class="chip dry">dry run</span>');
    if (a.profile) chips.push(`<span class="chip">${esc(a.profile)}</span>`);
    if (a.outdoor_temp !== null && a.outdoor_temp !== undefined)
      chips.push(`<span class="chip">${fmt(a.outdoor_temp, 0)}° ${esc(CONDITION[a.weather_condition] ?? String(a.weather_condition || "").replace(/[_-]/g, " "))}</span>`);
    if (a.cycle_running) chips.push('<span class="chip busy">planning…</span>');
    else {
      const next = minutesUntil(a.next_cycle_at);
      if (next !== null) chips.push(`<span class="chip">next ${next <= 0.5 ? "now" : fmtDuration(next)}</span>`);
    }

    const effectors = a.effectors || [];
    const on = effectors.filter((e) => e.mode && e.mode !== "off");
    const off = effectors.filter((e) => !e.mode || e.mode === "off");
    let plan = "";
    for (const e of on) {
      const timing = [];
      timing.push(e.delay_minutes ? `in ${fmtDuration(e.delay_minutes)}` : "now");
      if (e.duration_minutes) timing.push(`for ${fmtDuration(e.duration_minutes)}`);
      const target = e.target !== null && e.target !== undefined ? ` → ${fmt(e.target)}°` : "";
      plan += `<div class="eff"><ha-icon icon="${effectorIcon(e.name)}" style="color:${MODE_COLOR[e.mode] || "var(--state-active-color)"}"></ha-icon>
        <div class="n">${esc(e.title || titleCase(e.name))}<span class="quiet"> ${esc(timing.join(" "))}</span></div>
        <div class="m" style="color:${MODE_COLOR[e.mode] || "inherit"}">${esc(e.mode)}${esc(target)}</div>
        ${e.rationale ? `<div class="why">${esc(e.rationale)}</div>` : ""}
        ${e.override ? `<div class="ovr">Manual override: ${esc(e.override)}</div>` : ""}
      </div>`;
    }
    if (!on.length) plan = `<div class="quiet">Everything is off — nothing needs doing.</div>`;
    const overridden = off.filter((e) => e.override);
    for (const e of overridden) plan += `<div class="line warn"><ha-icon icon="mdi:hand-back-right"></ha-icon>${esc(e.title || titleCase(e.name))}: manual override (${esc(e.override)})</div>`;
    if (this._config.show_off && off.length)
      plan += `<div class="quiet" style="margin-top:6px">Off: ${esc(off.map((e) => (e.title || titleCase(e.name)).toLowerCase()).join(", "))}</div>`;

    const overlays = Object.entries(a.overlays || {});
    let requests = "";
    for (const [label, o] of overlays) {
      const left = minutesUntil(o.expires_at);
      requests += `<div class="line"><ha-icon icon="mdi:thermometer-plus"></ha-icon>${esc(titleCase(label))} ${esc(fmtDelta(o.delta))}${
        left !== null ? ` <span class="quiet">for ${esc(fmtDuration(left))}</span>` : ""
      }</div>`;
    }

    let advice = "";
    for (const w of a.warnings || []) advice += `<div class="line warn"><ha-icon icon="mdi:alert"></ha-icon>${esc(w.message || JSON.stringify(w))}</div>`;
    for (const o of a.opportunities || []) {
      const delta = num(o.cost_delta);
      advice += `<div class="line"><ha-icon icon="mdi:window-open-variant"></ha-icon>${esc(o.text || `${o.action} ${o.device}`)}${
        delta !== null ? ` <span class="quiet">(${delta > 0 ? "+" : ""}${delta.toFixed(2)})</span>` : ""
      }</div>`;
    }

    const foot = [];
    const c = a.costs;
    if (c && c.total !== null) {
      const vs = c.baseline !== null && c.baseline !== undefined ? ` vs ${fmt(c.baseline, 2)} all-off` : "";
      foot.push(`<span title="comfort ${fmt(c.comfort, 2)} + energy ${fmt(c.energy, 2)}">cost ${fmt(c.total, 2)}${vs}</span>`);
    }
    const age = (label, mins, staleAfter) =>
      mins === null || mins === undefined ? "" : `<span class="${mins > staleAfter ? "bad" : ""}">${label} ${fmtDuration(mins)} old</span>`;
    foot.push(age("data", a.collector_age_minutes, 15));
    foot.push(age("model", a.sysid_age_minutes, 180));
    if (a.ha_error) foot.push(`<span class="bad">HA: ${esc(a.ha_error)}</span>`);

    return `<style>${STATUS_CSS}</style><ha-card>
      ${head(chips.join(""))}
      <h4>Plan</h4>${plan}
      ${requests ? `<h4>Comfort requests</h4>${requests}` : ""}
      ${advice ? `<h4>Advisories</h4>${advice}` : ""}
      <div class="foot">${foot.filter(Boolean).join("")}</div>
    </ha-card>`;
  }

  getCardSize() {
    return 5;
  }

  getGridOptions() {
    return { columns: 12, min_columns: 6 };
  }

  static getStubConfig() {
    return {};
  }
}

// ── Registration ───────────────────────────────────────────────────────────

const defs = [
  ["weatherstat-room-card", WeatherstatRoomCard, "Weatherstat room", "One room: temperature, comfort band, forecast and −/+ comfort."],
  ["weatherstat-rooms-card", WeatherstatRoomsCard, "Weatherstat rooms", "Every room as a compact list with −/+ comfort."],
  ["weatherstat-status-card", WeatherstatStatusCard, "Weatherstat status", "What the controller is doing and why: plan, requests, advisories, health."],
];
window.customCards = window.customCards || [];
for (const [tag, cls, name, description] of defs) {
  if (!customElements.get(tag)) customElements.define(tag, cls);
  if (!window.customCards.some((c) => c.type === tag)) window.customCards.push({ type: tag, name, description, preview: false });
}
console.info(`%c WEATHERSTAT-CARDS %c ${WS_VERSION} `, "color:#fff;background:#3a9d4d;font-weight:600", "color:#3a9d4d");
