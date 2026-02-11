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
import sys

import lightgbm as lgb
import pandas as pd

from weatherstat.config import (
    HORIZONS_5MIN,
    HORIZONS_HOURLY,
    LGBM_PARAMS,
    LGBM_PARAMS_SMALL,
    MODELS_DIR,
    PREDICTION_ROOMS,
    SNAPSHOTS_DB,
    SNAPSHOTS_DIR,
)
from weatherstat.extract import load_collector_snapshots
from weatherstat.features import (
    ROOM_TEMP_COLUMNS,
    add_future_targets,
    build_features,
)

# Columns to exclude from features (identifiers, raw categoricals, targets)
EXCLUDE_COLUMNS_BASE = {
    "timestamp",
    # Raw categorical strings (encoded versions are used instead)
    "thermostat_upstairs_action",
    "thermostat_downstairs_action",
    "mini_split_bedroom_mode",
    "mini_split_living_room_mode",
    "blower_family_room_mode",
    "blower_office_mode",
    "navien_heating_mode",
    "weather_condition",
}


# HVAC columns to merge from full-feature sources into baseline hourly data.
# Temperature columns are NOT included — those come from hourly statistics
# (actual hourly means computed by HA, more accurate than resampled snapshots).
HVAC_MERGE_COLUMNS = [
    "thermostat_upstairs_target",
    "thermostat_downstairs_target",
    "thermostat_upstairs_action",
    "thermostat_downstairs_action",
    "mini_split_bedroom_temp",
    "mini_split_bedroom_target",
    "mini_split_bedroom_mode",
    "mini_split_living_room_temp",
    "mini_split_living_room_target",
    "mini_split_living_room_mode",
    "blower_family_room_mode",
    "blower_office_mode",
    "navien_heating_mode",
    "navien_heat_capacity",
    "weather_condition",
    "wind_speed",
    "outdoor_humidity",
    "any_window_open",
]


def _resample_to_hourly(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Resample 5-min data to hourly, keeping only specified columns.

    Numeric columns use mean; categorical/bool use last value per hour.
    """
    available = [c for c in columns if c in df.columns]
    if not available:
        return pd.DataFrame()

    subset = df[["timestamp"] + available].copy()
    subset["_ts"] = pd.to_datetime(subset["timestamp"], format="ISO8601", utc=True)

    numeric = subset[available].select_dtypes(include=["number"]).columns.tolist()
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


def load_full_data() -> pd.DataFrame:
    """Load 5-min full-feature data for full training.

    Merges two optional sources:
    - historical_full.parquet (bootstrap extraction from HA)
    - snapshots.db (ongoing collector data)

    At least one source must exist. On timestamp collision the collector
    row wins (it's the live truth).
    """
    frames: list[pd.DataFrame] = []

    # Source 1: historical extraction (optional)
    parquet_path = SNAPSHOTS_DIR / "historical_full.parquet"
    if parquet_path.exists():
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

    # Normalize any_window_open to bool (INTEGER in SQLite, bool in Parquet)
    if "any_window_open" in df.columns:
        df["any_window_open"] = df["any_window_open"].astype(bool)

    print(f"Full dataset: {len(df)} rows")
    return df


def get_target_columns(zones: list[str], horizons: list[int]) -> list[str]:
    """Build target column names for zone/horizon combinations."""
    return [f"{zone}_temp_t+{h}" for zone in zones for h in horizons]


def train_mode(mode: str) -> None:
    """Run training for the specified mode."""
    if mode == "baseline":
        df = load_baseline_data()
        horizons = HORIZONS_HOURLY
        params = LGBM_PARAMS
        model_prefix = "baseline"
    elif mode == "full":
        df = load_full_data()
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
    numeric_df = df[feature_cols].select_dtypes(include=["number"])
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

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

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

        results.append({
            "target": target,
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "n_train": len(X_train),
            "n_val": len(X_val),
        })

        # Save model
        model_path = MODELS_DIR / f"{model_prefix}_{target}_lgbm.txt"
        model.booster_.save_model(str(model_path))
        print(f"  Saved: {model_path.name}")

        # Feature importance (top 10)
        importance = pd.Series(
            model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)
        print("  Top 10 features:")
        for feat, imp in importance.head(10).items():
            print(f"    {feat}: {imp}")

    # Save feature columns for inference
    feature_path = MODELS_DIR / f"{model_prefix}_feature_columns.txt"
    feature_path.write_text("\n".join(feature_cols))

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"TRAINING SUMMARY ({mode} mode)")
    print(f"{'=' * 60}")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print(f"\nFeature columns saved to {feature_path}")
    print("Training complete!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train weatherstat LightGBM models")
    parser.add_argument(
        "--mode",
        choices=["baseline", "full"],
        required=True,
        help="Training mode: baseline (hourly temp-only) or full (5-min all features)",
    )
    args = parser.parse_args()
    train_mode(args.mode)


if __name__ == "__main__":
    main()
