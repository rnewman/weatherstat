"""LightGBM training pipeline.

Load Parquet snapshots → build features → train model → save.

Run: uv run python -m weatherstat.train
"""

import sys

import lightgbm as lgb
import pandas as pd

from weatherstat.config import LGBM_PARAMS, MODELS_DIR, SNAPSHOTS_DIR
from weatherstat.features import build_features

# Target columns the model predicts
TARGET_COLUMNS = [
    "thermostat_upstairs_target",
    "thermostat_downstairs_target",
    "mini_split_1_target",
    "mini_split_2_target",
]

# Columns to exclude from features (targets, identifiers, raw strings)
EXCLUDE_COLUMNS = {
    "timestamp",
    "thermostat_upstairs_action",
    "thermostat_downstairs_action",
    "mini_split_1_mode",
    "mini_split_2_mode",
    "weather_condition",
    "indoor_temps_json",
    *TARGET_COLUMNS,
}


def load_snapshots() -> pd.DataFrame:
    """Load all Parquet snapshot files into a single DataFrame."""
    parquet_files = sorted(SNAPSHOTS_DIR.glob("*.parquet"))
    if not parquet_files:
        print(f"No snapshot files found in {SNAPSHOTS_DIR}", file=sys.stderr)
        sys.exit(1)

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(parquet_files)} files")
    return df


def train() -> None:
    """Run the full training pipeline."""
    df = load_snapshots()
    df = build_features(df)

    # Drop rows with NaN from lag/rolling features
    df = df.dropna()
    print(f"Training on {len(df)} rows after dropping NaN")

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    X = df[feature_cols]
    y = df[TARGET_COLUMNS]

    print(f"Features: {len(feature_cols)} columns")
    print(f"Targets: {TARGET_COLUMNS}")

    # Train one model per target (multi-output via separate models)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for target in TARGET_COLUMNS:
        print(f"\nTraining model for {target}...")

        model = lgb.LGBMRegressor(**LGBM_PARAMS)  # type: ignore[arg-type]

        # Simple train/val split (last 20% as validation)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y[target].iloc[:split_idx], y[target].iloc[split_idx:]

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
        )

        model_path = MODELS_DIR / f"{target}_lgbm.txt"
        model.booster_.save_model(str(model_path))
        print(f"Saved model to {model_path}")

    # Also save feature column names for inference
    feature_path = MODELS_DIR / "feature_columns.txt"
    feature_path.write_text("\n".join(feature_cols))
    print(f"\nSaved feature columns to {feature_path}")
    print("Training complete!")


if __name__ == "__main__":
    train()
