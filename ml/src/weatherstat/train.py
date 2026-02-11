"""LightGBM training pipeline with multi-horizon temperature prediction.

Two training modes:
- baseline: Train on hourly temp-only data (5+ months of statistics)
- full: Train on 5-min full-feature data (~10 days of raw history)

Predicts future temperature at T+1h, T+2h, T+4h per zone.

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
    PREDICTION_ZONES,
    SNAPSHOTS_DIR,
)
from weatherstat.features import (
    ZONE_TEMP_COLUMNS,
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


def load_baseline_data() -> pd.DataFrame:
    """Load hourly statistics data for baseline training."""
    path = SNAPSHOTS_DIR / "historical_hourly.parquet"
    if not path.exists():
        print(f"Error: {path} not found. Run `just extract` first.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} hourly rows from {path}")
    return df


def load_full_data() -> pd.DataFrame:
    """Load 5-min full-feature data for full training."""
    path = SNAPSHOTS_DIR / "historical_full.parquet"
    if not path.exists():
        print(f"Error: {path} not found. Run `just extract` first.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} rows from {path}")
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
        params = LGBM_PARAMS_SMALL
        model_prefix = "full"
    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)

    # Feature engineering
    print(f"\nBuilding features (mode={mode})...")
    df = build_features(df, mode=mode)

    # Add future temperature targets
    df = add_future_targets(df, ZONE_TEMP_COLUMNS, horizons)

    target_cols = get_target_columns(PREDICTION_ZONES, horizons)

    # Build exclude set: base excludes + all target columns
    exclude = EXCLUDE_COLUMNS_BASE | set(target_cols)

    # Drop rows with NaN (from lag/rolling/future shift)
    pre_drop = len(df)
    df = df.dropna(subset=target_cols)
    df = df.dropna()
    print(f"Dropped {pre_drop - len(df)} rows with NaN, {len(df)} remaining")

    if len(df) < 50:
        print("Error: too few rows after dropping NaN.", file=sys.stderr)
        sys.exit(1)

    feature_cols = [c for c in df.columns if c not in exclude]

    # Filter to only numeric columns
    numeric_df = df[feature_cols].select_dtypes(include=["number"])
    feature_cols = list(numeric_df.columns)

    X = df[feature_cols]
    print(f"Features: {len(feature_cols)} columns")
    print(f"Targets: {target_cols}")

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
