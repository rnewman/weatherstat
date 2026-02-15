"""LightGBM training pipeline with multi-horizon temperature prediction.

Two training modes:
- baseline: Train on hourly data (7+ months of statistics + HVAC features)
- full: Train on 5-min full-feature data (~10 days of raw history)

Both modes use HVAC features (setpoints, actions, modes) when available.
The baseline merges HVAC data from collector/extraction into the hourly
statistics. LightGBM handles NaN natively, so pre-collector rows (where
HVAC features are missing) still contribute temperature learning.

Run:
  uv run python -m weatherstat.train --mode baseline
  uv run python -m weatherstat.train --mode full
"""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from weatherstat.config import (
    HORIZONS_5MIN,
    HORIZONS_HOURLY,
    LGBM_PARAMS,
    LGBM_PARAMS_SMALL,
    METRICS_DIR,
    MODELS_DIR,
    PREDICTION_ROOMS,
    SNAPSHOTS_DB,
    SNAPSHOTS_DIR,
    experiment_models_dir,
)
from weatherstat.extract import load_collector_snapshots
from weatherstat.features import (
    ROOM_TEMP_COLUMNS,
    add_future_targets,
    build_features,
)
from weatherstat.yaml_config import load_config

_CFG = load_config()

# Columns to exclude from features (from YAML config: identifiers, raw categoricals, targets)
EXCLUDE_COLUMNS_BASE = _CFG.exclude_columns

# HVAC columns to merge from full-feature sources into baseline hourly data (from YAML config).
# Temperature columns are NOT included — those come from hourly statistics
# (actual hourly means computed by HA, more accurate than resampled snapshots).
HVAC_MERGE_COLUMNS = _CFG.hvac_merge_columns


def _resample_to_hourly(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Resample 5-min data to hourly, keeping only specified columns.

    Numeric columns use mean; categorical/bool use last value per hour.
    """
    available = [c for c in columns if c in df.columns]
    if not available:
        return pd.DataFrame()

    subset = df[["timestamp"] + available].copy()
    subset["_ts"] = pd.to_datetime(subset["timestamp"], format="ISO8601", utc=True)

    numeric = subset[available].select_dtypes(include=["number", "bool"]).columns.tolist()
    other = [c for c in available if c not in numeric]

    grouped = subset.set_index("_ts")
    parts: list[pd.DataFrame] = []
    if numeric:
        parts.append(grouped[numeric].resample("1h").mean())
    if other:
        parts.append(grouped[other].resample("1h").last())

    if not parts:
        return pd.DataFrame()

    result = pd.concat(parts, axis=1).reset_index()
    result = result.rename(columns={"_ts": "timestamp"})
    result["timestamp"] = result["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return result


def load_baseline_data() -> pd.DataFrame:
    """Load hourly statistics + HVAC features for baseline training.

    Temperatures come from HA long-term statistics (7+ months of hourly means).
    HVAC features (setpoints, actions, modes) come from full-feature sources
    (historical extraction + collector), resampled to hourly and merged.
    Pre-collector rows have NaN for HVAC features — LightGBM handles this.
    """
    path = SNAPSHOTS_DIR / "historical_hourly.parquet"
    if not path.exists():
        print(f"Error: {path} not found. Run `just extract` first.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} hourly rows from {path}")

    # Merge HVAC features from full-feature sources
    hvac_frames: list[pd.DataFrame] = []

    parquet_path = SNAPSHOTS_DIR / "historical_full.parquet"
    if parquet_path.exists():
        full = pd.read_parquet(parquet_path)
        print(f"  HVAC source: {len(full)} rows from {parquet_path.name}")
        hvac_frames.append(full)

    if SNAPSHOTS_DB.exists():
        collector = load_collector_snapshots(SNAPSHOTS_DB)
        print(f"  HVAC source: {len(collector)} rows from {SNAPSHOTS_DB.name}")
        hvac_frames.append(collector)

    if hvac_frames:
        hvac_all = pd.concat(hvac_frames, ignore_index=True)
        hvac_all = hvac_all.drop_duplicates(subset="timestamp", keep="last")
        hvac_hourly = _resample_to_hourly(hvac_all, HVAC_MERGE_COLUMNS)

        if not hvac_hourly.empty:
            df = df.merge(hvac_hourly, on="timestamp", how="left")
            hvac_cols = [c for c in hvac_hourly.columns if c != "timestamp"]
            n_with_hvac = df[hvac_cols].notna().any(axis=1).sum()
            print(f"  Merged HVAC features: {n_with_hvac}/{len(df)} rows have data")
    else:
        print("  No HVAC sources available (run `just extract` or `just collect`)")

    return df


def load_full_data(collector_only: bool = False) -> pd.DataFrame:
    """Load 5-min full-feature data for full training.

    Merges two optional sources:
    - historical_full.parquet (bootstrap extraction from HA)
    - snapshots.db (ongoing collector data)

    At least one source must exist. On timestamp collision the collector
    row wins (it's the live truth).

    If collector_only=True, skip historical_full.parquet entirely — train
    only on collector data where every row has full HVAC features.
    """
    frames: list[pd.DataFrame] = []

    # Source 1: historical extraction (optional, skipped in collector-only mode)
    parquet_path = SNAPSHOTS_DIR / "historical_full.parquet"
    if collector_only:
        print("Collector-only mode: skipping historical_full.parquet")
    elif parquet_path.exists():
        hist = pd.read_parquet(parquet_path)
        print(f"Loaded {len(hist)} rows from {parquet_path.name}")
        frames.append(hist)

    # Source 2: collector SQLite (optional)
    if SNAPSHOTS_DB.exists():
        collector = load_collector_snapshots(SNAPSHOTS_DB)
        print(f"Loaded {len(collector)} rows from {SNAPSHOTS_DB.name}")
        frames.append(collector)

    if not frames:
        print(
            "Error: no training data found. Need historical_full.parquet or snapshots.db.\n"
            "Run `just extract` or `just collect` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)

    # Dedup on timestamp — keep last (collector appended second, so it wins)
    pre_dedup = len(df)
    df = df.drop_duplicates(subset="timestamp", keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    if pre_dedup > len(df):
        print(f"Deduped: {pre_dedup} → {len(df)} rows ({pre_dedup - len(df)} overlapping timestamps removed)")

    # Normalize timestamp format (SQLite has .000Z milliseconds, Parquet doesn't)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601").dt.strftime("%Y-%m-%dT%H:%M:%S%z")

    # Normalize window columns to bool (INTEGER in SQLite, bool in Parquet)
    for col in _CFG.window_bool_columns:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    print(f"Full dataset: {len(df)} rows")
    return df


def get_target_columns(zones: list[str], horizons: list[int]) -> list[str]:
    """Build target column names for zone/horizon combinations."""
    return [f"{zone}_temp_t+{h}" for zone in zones for h in horizons]


def _git_short_hash() -> str | None:
    """Return the short git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _save_metrics(
    mode: str,
    experiment: str | None,
    results: list[dict[str, object]],
    n_features: int,
    n_rows_raw: int,
    data_start: str,
    data_end: str,
) -> None:
    """Write a timestamped metrics JSON to data/metrics/."""
    now = datetime.now(UTC)
    ts_label = now.strftime("%Y-%m-%dT%H%M%S")

    metrics = {
        "timestamp": now.isoformat(),
        "mode": mode,
        "experiment": experiment,
        "git_hash": _git_short_hash(),
        "data": {
            "rows_raw": n_rows_raw,
            "rows_train": results[0]["n_train"] if results else 0,
            "rows_val": results[0]["n_val"] if results else 0,
            "n_features": n_features,
            "date_range": [data_start, data_end],
        },
        "targets": results,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / f"{mode}_{ts_label}.json"
    path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {path}")


def train_mode(mode: str, output_dir: Path | None = None, collector_only: bool = False) -> None:
    """Run training for the specified mode.

    Args:
        mode: "baseline" or "full".
        output_dir: Override model output directory (default: MODELS_DIR).
            Used for experiment workflows to write to data/models/{experiment}/.
        collector_only: If True, skip historical_full.parquet and train only on
            collector data. Only applies to full mode.
    """
    models_dir = output_dir or MODELS_DIR
    if mode == "baseline":
        df = load_baseline_data()
        horizons = HORIZONS_HOURLY
        params = LGBM_PARAMS
        model_prefix = "baseline"
    elif mode == "full":
        df = load_full_data(collector_only=collector_only)
        horizons = HORIZONS_5MIN
        # Auto-select params: ~17 days of 5-min data ≈ 5000 rows
        if len(df) >= 5000:
            params = LGBM_PARAMS
            print(f"Using full LGBM params ({len(df)} rows >= 5000)")
        else:
            params = LGBM_PARAMS_SMALL
            print(f"Using small LGBM params ({len(df)} rows < 5000)")
        model_prefix = "full"
    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)

    # Capture data date range before any row drops
    data_start = str(df["timestamp"].min())
    data_end = str(df["timestamp"].max())
    n_rows_raw = len(df)

    # Feature engineering
    print(f"\nBuilding features (mode={mode})...")
    df = build_features(df, mode=mode)

    # Add future temperature targets
    df = add_future_targets(df, ROOM_TEMP_COLUMNS, horizons)

    all_target_cols = get_target_columns(PREDICTION_ROOMS, horizons)

    # Filter to rooms with sufficient data (>50% non-NaN target values)
    target_cols: list[str] = []
    skipped_rooms: set[str] = set()
    for col in all_target_cols:
        if col not in df.columns:
            room = col.rsplit("_temp_t+", 1)[0]
            skipped_rooms.add(room)
            continue
        non_null_frac = df[col].notna().mean()
        if non_null_frac < 0.5:
            room = col.rsplit("_temp_t+", 1)[0]
            skipped_rooms.add(room)
        else:
            target_cols.append(col)

    if skipped_rooms:
        print(f"Skipping rooms with insufficient data: {sorted(skipped_rooms)}")

    if not target_cols:
        print("Error: no rooms have sufficient data for training.", file=sys.stderr)
        sys.exit(1)

    # Build exclude set: base excludes + all possible target columns
    exclude = EXCLUDE_COLUMNS_BASE | set(all_target_cols)

    # Determine feature columns
    feature_cols = [c for c in df.columns if c not in exclude]
    numeric_df = df[feature_cols].select_dtypes(include=["number", "bool"])
    feature_cols = list(numeric_df.columns)

    # Only drop rows where TARGET is NaN (from future shift at end of series).
    # Feature NaN is fine — LightGBM handles missing values natively.
    # This is important: HVAC features are only available for recent data, so
    # most baseline rows have NaN for setpoints/actions. LightGBM learns the
    # HVAC relationship from the rows that have data and uses temperature-only
    # splits for the rest.
    pre_drop = len(df)
    df = df.dropna(subset=target_cols)
    print(f"Dropped {pre_drop - len(df)} rows with NaN targets, {len(df)} remaining")

    if len(df) < 50:
        print("Error: too few rows after dropping NaN.", file=sys.stderr)
        sys.exit(1)

    X = df[feature_cols]
    print(f"Features: {len(feature_cols)} columns")
    n_rooms = len(target_cols) // len(horizons)
    print(f"Targets: {len(target_cols)} ({n_rooms} rooms x {len(horizons)} horizons)")

    # Time-based train/test split (no shuffle — preserves temporal order)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]

    print(f"Train: {len(X_train)} rows, Val: {len(X_val)} rows")

    models_dir.mkdir(parents=True, exist_ok=True)
    if models_dir != MODELS_DIR:
        print(f"Experiment output: {models_dir}")

    results: list[dict[str, object]] = []

    for target in target_cols:
        y_train = df[target].iloc[:split_idx]
        y_val = df[target].iloc[split_idx:]

        print(f"\n{'─' * 50}")
        print(f"Training: {target}")

        model = lgb.LGBMRegressor(**params)  # type: ignore[arg-type]
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
        )

        # Evaluate
        y_pred = model.predict(X_val)
        rmse = ((y_val - y_pred) ** 2).mean() ** 0.5
        mae = (y_val - y_pred).abs().mean()
        print(f"  RMSE: {rmse:.3f} F")
        print(f"  MAE:  {mae:.3f} F")

        # Feature importance (top 10)
        importance = pd.Series(
            model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)
        top_features = {str(feat): int(imp) for feat, imp in importance.head(10).items()}

        results.append({
            "target": target,
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "top_features": top_features,
        })

        # Save model
        model_path = models_dir / f"{model_prefix}_{target}_lgbm.txt"
        model.booster_.save_model(str(model_path))
        print(f"  Saved: {model_path.name}")

        print("  Top 10 features:")
        for feat, imp in top_features.items():
            print(f"    {feat}: {imp}")

    # Save feature columns for inference
    feature_path = models_dir / f"{model_prefix}_feature_columns.txt"
    feature_path.write_text("\n".join(feature_cols))

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"TRAINING SUMMARY ({mode} mode)")
    print(f"{'=' * 60}")
    results_df = pd.DataFrame(results)
    print(results_df[["target", "rmse", "mae", "n_train", "n_val"]].to_string(index=False))
    print(f"\nFeature columns saved to {feature_path}")

    # ── Save metrics JSON ──
    _save_metrics(
        mode=mode,
        experiment=output_dir.name if output_dir and output_dir != MODELS_DIR else None,
        results=results,
        n_features=len(feature_cols),
        n_rows_raw=n_rows_raw,
        data_start=data_start,
        data_end=data_end,
    )

    print("Training complete!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train weatherstat LightGBM models")
    parser.add_argument(
        "--mode",
        choices=["baseline", "full"],
        required=True,
        help="Training mode: baseline (hourly temp-only) or full (5-min all features)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Write models to data/models/{name}/ instead of production",
    )
    parser.add_argument(
        "--collector-only",
        action="store_true",
        help="Train only on collector data (skip historical_full.parquet). Only applies to --mode full.",
    )
    args = parser.parse_args()
    output_dir = experiment_models_dir(args.experiment) if args.experiment else None
    train_mode(args.mode, output_dir=output_dir, collector_only=args.collector_only)


if __name__ == "__main__":
    main()
