---
description: Inspect weatherstat system state — temperatures, gains, taus, decisions, opportunities, comfort schedules
user-invocable: true
---

# Weatherstat Debug Inspector

When the user asks to debug, inspect, or understand the current state of the weatherstat system, use the `scripts/debug_state.py` script and present the results.

## Available subcommands

Run these via `just debug <subcommand>` or `uv run python scripts/debug_state.py <subcommand>`:

| Subcommand | What it shows |
|-----------|---------------|
| (none) | Full summary: snapshots, sysid, control state, temps + comfort, opportunities |
| `temps` | Current temperatures with comfort bounds from latest decision |
| `gains [SENSOR]` | Sysid effector→sensor gains (filter by sensor name) |
| `taus` | Tau models, window betas with effective tau, interaction betas, MRT weights |
| `decisions [N]` | Last N decisions with costs, effectors, temps, blocked info |
| `opportunities` | Active window opportunities and cooldown timers |
| `snapshots` | Snapshot DB row counts and time range |
| `comfort [SENSOR]` | Comfort schedules from config (filter by sensor label) |

## Instructions

1. If the user's question is about a specific topic, run the relevant subcommand. For broad questions, run the full summary (no args).
2. If the user argument `$ARGUMENTS` is provided, pass it as the subcommand: `just debug $ARGUMENTS`
3. Present the output and interpret it in context of the user's question.
4. For deeper investigation (e.g., "why did this opportunity fire?"), combine multiple subcommands.

## Common investigation patterns

**"Why did this opportunity fire?"**
1. `gains <sensor>` — check if the window has plausible betas for affected sensors
2. `taus` — check window_betas and effective tau changes
3. `temps` — is the room actually too warm/cold?
4. `decisions 3` — what was the system doing?

**"Why is this room cold?"**
1. `temps` — confirm current state
2. `gains <sensor>` — which effectors heat this room, and how strongly?
3. `decisions 3` — is the system trying to heat it?
4. `comfort <sensor>` — what are the targets?

**"Is the system working?"**
1. Full summary (no args) — overview
2. `decisions 10` — recent history
3. `just comfort` — visual dashboard
