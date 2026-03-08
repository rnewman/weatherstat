# Plan: Archive Orphaned ML Pipeline

## Context

The control loop now uses the physics simulator exclusively. The LightGBM
training/inference/experiment pipeline is orphaned — no live code path
calls it. It should be archived to reduce confusion, not deleted, because
the retrospective in ARCHITECTURE.md references it and the experiment
results have historical value.

## What to Archive

Move to `archive/ml/`:

| File | Reason |
|------|--------|
| `ml/src/weatherstat/train.py` | LightGBM training pipeline |
| `ml/src/weatherstat/inference.py` | ML inference + `fetch_recent_history` (needs extraction first) |
| `ml/src/weatherstat/evaluate.py` | Model evaluation on validation set |
| `ml/src/weatherstat/experiment.py` | Experiment comparison framework |
| `ml/src/weatherstat/metrics.py` | Training metrics viewer |
| `ml/src/weatherstat/visualize.py` | Data visualization utility |
| `ml/scripts/fit_tau.py` | Standalone tau fitting (superseded by sysid) |
| `ml/scripts/backtest_overnight.py` | ML model backtesting |

## What to Keep (but modify)

### `inference.py` → extract `fetch_recent_history`

`control.py` imports `fetch_recent_history()` from `inference.py`. This
function fetches live state from HA — it's data plumbing, not ML inference.

**Action:** Move `fetch_recent_history()` to `extract.py` (where the other
HA data functions live). Update the import in `control.py`. Then archive
the rest of `inference.py`.

### `features.py` → extract `ROOM_TEMP_COLUMNS`

`control.py` imports `ROOM_TEMP_COLUMNS` from `features.py`. This is just
`_CFG.room_temp_columns` — a one-liner.

**Action:** Replace the import in `control.py` with a direct call to
`load_config().room_temp_columns` or a module-level constant in `config.py`.
The rest of `features.py` (feature engineering for ML training) is only
used by archived modules, but it's also used by test_features.py.

**Decision:** Keep `features.py` in place for now — it's harmless and the
tests validate the feature engineering logic which could be useful if we
ever revisit ML. The key is removing the calling code, not the library.

## Justfile Cleanup

Remove these targets:

| Target | Reason |
|--------|--------|
| `train` | ML training |
| `retrain` | Alias for train |
| `evaluate` | ML evaluation |
| `metrics` | ML metrics |
| `predict` | ML inference |
| `predict-snapshot` | ML inference from snapshot |
| `counterfactual` | ML counterfactual |
| `train-experiment` | ML experiment training |
| `experiment-compare` | ML experiment comparison |
| `experiments` | List ML experiments |
| `backtest` | ML backtesting |
| `backtest-experiment` | ML experiment backtesting |
| `backtest-all` | ML all-experiment backtesting |
| `retrain-manual` | Scheduled ML retrain |
| `retrain-install` | Cron for ML retrain |
| `retrain-uninstall` | Cron removal |

Keep: `extract` (still useful for populating collector DB from HA history).

## CLAUDE.md / operations.md

Already updated. Just verify no references to removed Justfile targets
remain.

## Implementation Order

1. Move `fetch_recent_history` from `inference.py` to `extract.py`.
   Update import in `control.py`.
2. Replace `ROOM_TEMP_COLUMNS` import in `control.py` with direct
   config access.
3. Create `archive/ml/` directory.
4. `git mv` the archived files.
5. Remove orphaned Justfile targets.
6. Run `just lint-py && just test-py` to verify nothing breaks.
7. Update CLAUDE.md commands section if needed.

## Risk

Low. All archived code is unreachable from the live control path.
The only risk is breaking an import chain — step 6 catches this.
