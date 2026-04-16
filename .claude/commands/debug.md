---
description: Inspect weatherstat system state — temperatures, gains, taus, decisions, opportunities, comfort schedules
user-invocable: true
---

# Weatherstat Debug Inspector

When the user asks to debug, inspect, or understand the current state of the weatherstat system, use the `scripts/debug_state.py` script and present the results.

## Available subcommands

Run these via `just debug <subcommand>` or `uv run python scripts/debug_state.py <subcommand>`.
Add `--bundle <dir>` to inspect a point-in-time bundle instead of the live system:

| Subcommand | What it shows |
|-----------|---------------|
| (none) | Full summary: snapshots, sysid, control state, temps + comfort, opportunities |
| `temps` | Temperatures from snapshot DB + comfort bounds (⚠ stale if collector is down) |
| `live` | Live temperatures from HA REST API + comfort bounds (always current) |
| `gains [SENSOR]` | Sysid effector→sensor gains (filter by sensor name) |
| `taus` | Tau models, window betas with effective tau, interaction betas, MRT weights |
| `decisions [N]` | Last N decisions with costs, effectors, current temps, blocked info |
| `decision-why [N]` | Explain decision N: live temps + comfort + gains + predictions (consistent view) |
| `opportunities` | Active window opportunities and cooldown timers |
| `snapshots` | Snapshot DB row counts and time range |
| `comfort [SENSOR]` | Comfort schedules from config (filter by sensor label) |
| `why [EFFECTOR]` | Explain why active effectors are on (⚠ uses snapshot DB temps) |
| `advisory` | Advisory states + compound tau effects |

## CRITICAL: Two data sources for temperatures

The system has two sources of temperature data — mixing them up wastes debugging time:

| Source | Commands | Where it comes from | Freshness |
|--------|----------|---------------------|-----------|
| **Snapshot DB** | `temps`, `why`, summary | SQLite EAV table written by collector | Only as fresh as the collector — if collector is down, temps are stale |
| **Decision log** | `decisions`, `decision-why` | HA REST API, fetched live by control loop at decision time | Always current (at decision time) |
| **HA REST API** | `live` | Direct fetch from Home Assistant | Always current (right now) |

**When the collector is down**, `temps` and `why` show old data but `decisions` and `live` show real data. The staleness warning will fire automatically when snapshot data is >10 minutes old.

## Instructions

1. **Always start with `just health`** to check if the collector is running. If stale, warn that `temps`/`why` output will be wrong.
2. For "why is [effector] doing [X]?" questions, start with `just debug decision-why` (live temps) rather than `just debug why` (snapshot temps).
3. If the user argument `$ARGUMENTS` is provided, pass it as the subcommand: `just debug $ARGUMENTS`
4. Present the output and interpret it in context of the user's question.

## Bundles

To inspect a point-in-time bundle:
```
just debug --bundle ~/.weatherstat/bundles/bundle_20260330T103104p0000 temps
just debug --bundle ~/.weatherstat/bundles/bundle_20260330T103104p0000 decisions 5
```

Create bundles with `just bundle <timestamp>`, list with `just bundle --list`, replay with `just bundle --replay <dir>`.

## Common investigation patterns

**"Why is [effector] doing [X]?"** (most common question)
1. `just health` — is the collector running?
2. `just debug decision-why` — shows live temps at decision time + effector gains + predicted trajectories. This is the authoritative view of why the optimizer chose what it chose.
3. `just debug gains <sensor>` — deeper dive into specific gain concerns (confounded? implausible magnitude?).
4. `just debug taus` — check if environment state (open windows/doors) is affecting thermal dynamics.

**"Why is this room cold/hot?"**
1. `just debug live` — confirm ACTUAL current temps (not stale snapshot)
2. `just debug decision-why` — what is the system doing about it? What does it predict?
3. `just debug gains <sensor>` — which effectors affect this room, and how strongly?
4. `just debug comfort <sensor>` — what are the configured targets?

**"Is the system working?"**
1. `just health` — collector running?
2. `just debug decisions 5` — recent decision history (are decisions being made? costs reasonable?)
3. `just debug live` — current state
4. `just comfort` — visual dashboard
