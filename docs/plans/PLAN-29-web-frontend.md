# Embedded Web Frontend for the TUI

## Context

The TUI (`src/weatherstat/tui/app.py`) is the primary control surface: it displays
status, runs control cycles, runs sysid, executes commands, toggles profiles, and shows
the decision history. It's the only place that knows the live in-memory `_latest_decision`,
the in-memory `_overrides` map, and the running cycle/sysid timers.

We want a phone-friendly remote: a small HTTP server that surfaces a "Status" page and
four buttons (Run, Force, Toggle Profile, Toggle Live) — without spawning a second
process and without duplicating any control logic. The server runs **inside** the TUI
process so it shares the same in-memory state, the same scheduler, and the same HA
session.

The user pushed back hard on a separate-process architecture: "I would much rather have
as much combined as possible, perhaps even to the point of them running in the same
process." This plan honours that.

### Pre-existing bugs to fix in passing

1. `_toggle_profile()` (`tui/app.py:632-671`) hardcodes `entity_id =
   "input_select.thermostat_mode"`. The configured `_CFG.comfort_entity` is what the
   rest of the system uses (`control.py:170`). Profile toggle is broken if the user
   configured a different entity. The web work is the right time to fix this because
   the refactor moves the profile-toggle code into a controller method that both the
   keybinding and the HTTP route call.

2. `_load_opportunities()` (`tui/app.py:313-325`) reads `advisory_state.json` on every
   decision update — fine for the TUI, but the web Status page needs the same data and
   shouldn't re-read the file. Plumb the data through the controller instead.

## Architecture

### One process, two front-ends

```
                ┌─────────────────────────────────────────┐
                │            WeatherstatApp                │
                │           (Textual App)                  │
                │                                          │
                │  ┌───────────────────────────────────┐  │
                │  │       TUIController               │  │
                │  │  (the only thing that mutates      │  │
                │  │   _latest_decision, _overrides,    │  │
                │  │   live_mode, schedules cycles)     │  │
                │  └───────────────────────────────────┘  │
                │     ▲                       ▲           │
                │     │ Textual actions       │ HTTP      │
                │     │ (key bindings)        │ handlers  │
                │     │                       │           │
                └─────┼───────────────────────┼───────────┘
                      │                       │
                  Keyboard               daemon thread
                                          (http.server)
```

`TUIController` is the only object that owns mutable state. Both the keybinding actions
in `WeatherstatApp` and the HTTP handlers in `web.py` call into it. The controller is
*not* a Textual widget — it's a plain class that holds references to the widgets it
needs to refresh, and uses `app.call_from_thread()` to marshal updates to the UI thread
when called from a worker thread or the web thread.

### Why a controller class

Today, action methods like `action_toggle_profile`, `action_force_execute`, and
`_run_control_cycle` live on `WeatherstatApp` and freely mix three concerns:

1. **Business logic** — talk to HA, run sysid, run control cycle, dispatch the executor.
2. **UI updates** — `query_one(...).set_data(...)`, `_update_header()`, `_log()`.
3. **State machine** — `_cycle_running`, `_sysid_running`, `_overrides`, `_next_cycle_at`.

The web frontend needs (1) and (3) but not (2). Splitting them out gives:

- A thin `WeatherstatApp` whose actions delegate everything to the controller.
- A `TUIController` callable from the HTTP thread without any Textual coupling.
- The same set of "things you can do" available to both UIs.

## Source Code Changes

### New: `src/weatherstat/tui/controller.py`

```python
@dataclass
class StatusSnapshot:
    """Plain-data snapshot rendered by both the TUI panels and the web Status page."""
    timestamp: datetime
    profile: str | None
    live: bool
    cycle_running: bool
    sysid_running: bool
    next_cycle_at: datetime | None
    sysid_age: timedelta | None
    collector_age: timedelta | None
    collector_rows: int
    outdoor_temp: float | None
    temps: dict[str, float]                 # sensor_col → current value
    comfort: dict[str, ComfortBounds]       # label → (acc_lo, pref_lo, pref_hi, acc_hi)
    mrt_offsets: dict[str, float]           # sensor → offset
    environment: list[EnvEntry]             # (label, kind, active)
    forecast: dict[str, float]              # "1h" → value
    weather_condition: str
    effectors: list[EffectorView]           # name, mode, target, override?
    opportunities: list[dict]               # from advisory_state.json
    warnings: list[dict]
    sensor_costs: dict[str, float] | None
    baseline_sensor_costs: dict[str, float] | None


class TUIController:
    """Owns mutable runtime state. Called from both Textual actions and HTTP handlers.

    Threading: every public method is safe to call from any thread. Methods that
    update widgets do so via `app.call_from_thread()`. A single `RLock` serialises
    state mutations (live_mode, _overrides, _latest_decision, timers). The lock is
    NOT held across HTTP / HA / executor I/O — only around state reads/writes.
    """

    def __init__(self, app: WeatherstatApp, *, live: bool) -> None: ...

    # ── State access (read-only, thread-safe) ───────────────────────────────
    def snapshot(self) -> StatusSnapshot: ...
    def overrides(self) -> dict[str, str]: ...
    def is_cycle_running(self) -> bool: ...
    def is_sysid_running(self) -> bool: ...

    # ── Actions (thread-safe; idempotent for already-in-progress) ───────────
    def run_cycle(self) -> ActionResult: ...     # returns "started"|"already_running"
    def run_sysid(self) -> ActionResult: ...
    def toggle_live(self) -> ActionResult: ...   # returns new mode; no confirm
    def set_live(self, live: bool) -> ActionResult: ...
    def force_execute(self) -> ActionResult: ...
    def toggle_profile(self) -> ActionResult: ...      # cycles HA input_select
    def set_profile(self, name: str) -> ActionResult: ... # explicit set

    # ── Snapshot helpers (called from worker threads) ───────────────────────
    def refresh_snapshot(self) -> None: ...      # rebuilds StatusSnapshot from disk + memory
```

`ActionResult` is a small frozen dataclass: `status: str`, `message: str`,
`detail: dict[str, object] = {}`. Both the keyboard and the HTTP layer use it for
notifications / JSON responses.

`StatusSnapshot` is built from the same code paths `_refresh_temps()` already uses, just
factored into pure functions that return data instead of poking widgets.

### Refactor: `src/weatherstat/tui/app.py`

**Goal**: `WeatherstatApp` becomes a thin Textual frontend over `TUIController`. No
business logic, no HA calls, no `requests.post(...)` lines.

Move into `TUIController`:

- All of `_run_control_cycle`, `_run_sysid`, `_run_executor`, `_force_execute`,
  `_toggle_profile`. These keep `@work(thread=True)` decoration *on the controller
  side* — the controller owns its own thread workers via `app.run_worker(...)`.
- The state-loading helpers `_load_solar_elevation_gains`, `_load_sysid_status`,
  `_load_control_state`, `_load_opportunities`, `_collect_snapshot`,
  `_refresh_snapshot_status`, `_refresh_temps` (renamed `build_snapshot`).
- The schedulers `_schedule_next_cycle`, `_auto_cycle`, `_schedule_sysid`,
  `_auto_sysid`. These are now controller methods that the controller arms via
  `app.set_interval(...)` from `on_mount`.
- The flags `_cycle_running`, `_sysid_running`, `_latest_decision`, `_overrides`,
  `_next_cycle_at`, `_baseline_cost`, `_solar_elevation_gains_cache`. The controller
  owns them.

Keep on `WeatherstatApp`:

- `compose()` and `on_mount()` (now constructs the controller and starts the web
  server if requested).
- `action_*` methods, each one a 1–3 line delegation to `controller.*`. They translate
  the `ActionResult` into `self.notify(...)` and `self._log(...)`.
- `_log()` (the controller calls back into this via a callback passed at construction).
- `_apply_snapshot(snapshot: StatusSnapshot)` — single method that pushes a snapshot
  into every widget. The controller calls it via `call_from_thread`.

**Profile toggle bug fix**: the new `controller.toggle_profile()` reads
`_CFG.comfort_entity` instead of hardcoding `input_select.thermostat_mode`. If
`comfort_entity` is None, returns `ActionResult(status="error",
message="comfort_entity not configured")`. The HTTP handler turns this into a 400; the
keybinding turns it into a `notify` with severity error.

**ConfirmScreen**: keep it for `q` (quit while live) but drop it from `l` (toggle live)
per user request. The web has no confirmation either.

### New: `src/weatherstat/web.py`

```python
def start_web_server(
    controller: TUIController,
    host: str,
    port: int,
    *,
    log: Callable[[str], None],
) -> ThreadingHTTPServer:
    """Start an HTTP server in a daemon thread. Returns the server handle.

    Routes:
        GET  /              → HTML status page (auto-refresh, mobile-friendly)
        GET  /status.json   → JSON snapshot (used by /, also useful for debugging)
        POST /action/run    → controller.run_cycle()
        POST /action/sysid  → controller.run_sysid()
        POST /action/force  → controller.force_execute()
        POST /action/profile → controller.toggle_profile() OR set_profile(form["name"])
        POST /action/live   → controller.toggle_live() OR set_live(form["mode"])
        GET  /style.css     → static CSS (inline string in this module)
    """
```

Single file. No Flask, no Jinja, no aiohttp — just `http.server.ThreadingHTTPServer`
and `http.server.BaseHTTPRequestHandler`. Handlers translate POSTs into controller
calls and respond `303 See Other` to `/`.

#### HTML page (mobile-first)

Single page, no JavaScript. `<meta http-equiv="refresh" content="30">` for auto-refresh.
Viewport meta for mobile. Sections, in order:

1. **Header bar** — sticky. Profile pill, mode pill (LIVE red / DRY-RUN green), outdoor
   temp, time. Cycle/sysid status spinners (just text, no animation).
2. **Action buttons** — four big touch targets in a 2×2 grid: `Run`, `Force`,
   `Profile →`, `Live ↔`. Each is a `<form method=post>` with one button. Disabled
   state for buttons that can't run right now (cycle already running, no overrides for
   force, etc.) — render as disabled HTML.
3. **Temperatures** — table of (sensor, current, comfort range). Color-code the cell
   green/yellow/red the same way the TUI does. Show MRT offset if non-zero.
4. **Effectors** — table of (name, mode, target, status pill). Override badge if
   present.
5. **Forecast** — small inline strip of `1h 2h 4h 6h 12h` with deltas vs current.
6. **Environment** — list of (label, state, kind icon).
7. **Opportunities** — bullet list. Empty → "No opportunities."
8. **Health alerts** — bullet list. Empty → omitted.

CSS: ~80 lines, system font stack, dark mode via `prefers-color-scheme`, large touch
targets (44px min), no fancy layout. The whole page should be < 6 KB so phone load is
instant even on cellular.

#### JSON endpoint

`/status.json` returns the same `StatusSnapshot` serialised as JSON. This is what the
HTML page is built from server-side (one path, two formats). Useful for debugging from
the browser, and a potential future hook for a richer client.

### Modified: `src/weatherstat/tui/__main__.py`

```python
import argparse

parser = argparse.ArgumentParser(prog="weatherstat-tui")
parser.add_argument("--live", action="store_true", help="Start in live mode")
parser.add_argument("--web", action="store_true",
                    help="Enable embedded web server")
parser.add_argument("--web-host", default="0.0.0.0",
                    help="Web server bind host (default: 0.0.0.0)")
parser.add_argument("--web-port", type=int, default=8765,
                    help="Web server bind port (default: 8765)")
args = parser.parse_args()

app = WeatherstatApp(
    live=args.live,
    web_enabled=args.web,
    web_host=args.web_host,
    web_port=args.web_port,
)
app.run()
```

`--web` is **opt-in**. Without it, no socket is opened. The justfile gets a new task:

```just
tui-web *args:
    uv run python -m weatherstat.tui --web {{args}}
```

### Modified: `src/weatherstat/control.py`

No changes. The controller calls `run_control_cycle(live=...)` exactly the way
`_run_control_cycle` does today.

### Modified: `tests/test_tui_controller.py` (new)

Test the controller in isolation, without spinning up Textual:

- `test_run_cycle_sets_running_flag` — start cycle, observe `is_cycle_running()`, wait
  for completion, observe `False`. Patch `run_control_cycle` to return a fake decision.
- `test_run_cycle_idempotent` — second concurrent call returns `already_running`.
- `test_toggle_live_no_confirmation` — flips boolean, no modal.
- `test_toggle_profile_uses_comfort_entity` — patch `_CFG.comfort_entity =
  "input_select.foo"`, mock `requests`, assert the right URL is hit. Regression for the
  hardcode bug.
- `test_toggle_profile_no_entity_returns_error` — `_CFG.comfort_entity = None` →
  `ActionResult(status="error")`.
- `test_force_execute_no_overrides` — returns `ActionResult(status="noop")`.
- `test_snapshot_shape` — call `snapshot()`, assert all expected fields present.

The controller takes a fake `app` object in tests — a minimal stub that captures
`call_from_thread` calls into a list and ignores `query_one`. This avoids dragging
Textual into the test runner.

### Modified: `tests/test_web.py` (new)

Test the HTTP layer with a real `ThreadingHTTPServer`:

- `test_status_html_renders` — GET /, assert 200 + `<title>Weatherstat</title>` +
  expected sections.
- `test_status_json_shape` — GET /status.json, parse, assert keys.
- `test_post_action_run_calls_controller` — POST /action/run, assert controller's
  `run_cycle` was called once. 303 redirect to `/`.
- `test_post_action_profile_with_name` — POST /action/profile with `name=Away`, assert
  `set_profile("Away")` called.
- `test_unknown_route_404` — GET /nope → 404.
- `test_method_not_allowed` — GET /action/run → 405.

A fake controller (`FakeController`) records calls, returns fixed snapshots. Server
spun up on `127.0.0.1:0` (port 0 = ephemeral) for each test.

## Threading model

- **Textual UI thread** — runs the event loop, all `query_one(...).set_*` happen here.
- **Worker threads** — `@work(thread=True)` for cycle / sysid / executor / profile
  toggle. Today these are methods on `WeatherstatApp`; after the refactor they're
  methods on `TUIController` invoked via `app.run_worker(...)`.
- **HTTP threads** — `ThreadingHTTPServer` spawns one thread per request. They call
  controller methods directly.

**Locking rules** (kept simple):

1. The controller has one `threading.RLock` (`self._lock`).
2. The lock is held only around mutations of: `live_mode`, `_overrides`,
   `_latest_decision`, `_cycle_running`, `_sysid_running`, `_next_cycle_at`,
   `_solar_elevation_gains_cache`.
3. The lock is **never** held across HA REST calls, executor calls, sysid calls, or
   `call_from_thread`. HTTP I/O and HA I/O happen outside the lock.
4. A "started" return from `run_cycle()`/`run_sysid()` means the worker has been
   scheduled, not that it's complete. The HTTP handler responds `202 Accepted`-like
   semantics (303 to `/` with a flash message in the next snapshot).

This is safe because:
- The only contended state is small (booleans + dicts).
- Cycle/sysid use existing `@work(thread=True)` semantics, which are already
  thread-safe via Textual's worker pool.
- Snapshot reads are atomic dict copies under the lock.

## Implementation order

1. **Controller skeleton + tests** — write `controller.py` with empty methods, write
   `test_tui_controller.py` against the fake-app stub. Define `StatusSnapshot`,
   `ActionResult` dataclasses. No behavior moved yet.
2. **Move worker methods** — move `_run_control_cycle`, `_run_sysid`,
   `_run_executor`, `_force_execute`, `_toggle_profile` into the controller. Update
   `WeatherstatApp.action_*` methods to delegate. Fix the comfort_entity bug.
3. **Move state loading + snapshots** — move `_refresh_temps` body into
   `controller.build_snapshot()`. Add `app._apply_snapshot(snapshot)` that pushes the
   snapshot into widgets. The controller calls `app._apply_snapshot` via
   `call_from_thread`.
4. **Verify TUI still works** — `just tui` smoke test. Keyboard actions still hit the
   controller. No web server yet.
5. **Web server** — write `web.py` with HTML template + handlers + tests. Add
   `--web*` args to `__main__.py`.
6. **End-to-end smoke test** — `just tui --web`, hit `http://localhost:8765/` from a
   phone on the LAN, click each button, verify the corresponding TUI panel updates.
7. **Just task + docs** — `tui-web` task in justfile, CLAUDE.md commands section
   updated.

## Verification

1. `just test` — all existing tests pass (no behavior change for the TUI).
2. `just lint` — clean.
3. `just tui` — keyboard-only TUI works exactly as before. Profile toggle now uses
   `comfort_entity` (test by switching to a different entity in the YAML and toggling).
4. `just tui --web` then from a phone:
   - GET `http://<host>:8765/` renders all panels
   - Tap **Run** → status flips to "running...", new decision appears in TUI within
     seconds
   - Tap **Force** → executor runs (when overrides present)
   - Tap **Profile →** → profile cycles in HA, TUI header updates, web page updates on
     next refresh
   - Tap **Live ↔** → mode flips, TUI header updates immediately, no confirmation
5. Kill the TUI; the web socket closes (daemon thread exits with the process).
6. `curl -X POST http://localhost:8765/action/run` from another machine works.

## Decision criteria

**Done if:**
- All four buttons work from the phone
- TUI behavior is unchanged with `--web` absent
- The hardcoded `input_select.thermostat_mode` is gone
- No business logic in `WeatherstatApp.action_*` methods
- Controller has its own test suite that doesn't import Textual

**Defer if:**
- Authentication — out of scope. The user controls who can reach the LAN port. Note in
  the help text: "Bind to 127.0.0.1 if you don't trust your network."
- HTTPS — out of scope; reverse-proxy via the user's existing HA setup if needed.
- WebSockets / live updates — meta-refresh is enough. A future enhancement could swap
  to SSE if 30s feels stale.
- A second client (Apple Watch, widget, ...) — `/status.json` is the hook; not built
  now.

## Risks

- **Textual + plain HTTP server cohabitation**: untested in this codebase. Mitigation:
  the daemon thread is fully isolated; if the HTTP server crashes it doesn't take the
  TUI with it. Wrap the server start in `try/except` and `_log` failures.
- **Lock-around-I/O footgun**: easy to accidentally hold the lock across a `requests`
  call and deadlock the TUI. Mitigation: review every `with self._lock:` block; none
  should contain function calls beyond dict/attr access. Add a helper
  `_with_lock(setter)` that takes a callable and runs it under lock to make this
  visually obvious.
- **HA token in env**: web handlers don't need the token directly — the controller
  calls `requests` the same way the TUI already does. But the web port could be a
  vector for "trigger profile change" abuse. Acceptance: LAN-only deployment, user
  accepts the risk.
- **Snapshot freshness**: the web page shows the controller's most recent snapshot,
  which is updated on the same cadence as the TUI's `_monitor_tick`. Stale-by-up-to-30s
  is fine for this use case.

## Out of scope (for this plan)

- Pretty graphs / sparklines on the web page
- Editing comfort schedules from the web
- Decision history table on the web
- Authentication
- HTTPS
- Mobile app or PWA installability
