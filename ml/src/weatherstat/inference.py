"""Inference pipeline.

Load trained models + latest snapshot → build features → predict → write JSON.

Run: uv run python -m weatherstat.inference
"""

import json
import sys
from datetime import UTC, datetime

import lightgbm as lgb
import pandas as pd

from weatherstat.config import MODELS_DIR, PREDICTIONS_DIR, SNAPSHOTS_DIR
from weatherstat.features import build_features
from weatherstat.train import TARGET_COLUMNS
from weatherstat.types import HVACMode


def load_models() -> dict[str, lgb.Booster]:
    """Load all trained LightGBM models."""
    models: dict[str, lgb.Booster] = {}
    for target in TARGET_COLUMNS:
        model_path = MODELS_DIR / f"{target}_lgbm.txt"
        if not model_path.exists():
            print(f"Model not found: {model_path}", file=sys.stderr)
            sys.exit(1)
        models[target] = lgb.Booster(model_file=str(model_path))
    return models


def load_feature_columns() -> list[str]:
    """Load the feature column names used during training."""
    feature_path = MODELS_DIR / "feature_columns.txt"
    if not feature_path.exists():
        print(f"Feature columns file not found: {feature_path}", file=sys.stderr)
        sys.exit(1)
    return feature_path.read_text().strip().split("\n")


def load_latest_snapshots(n_rows: int = 24) -> pd.DataFrame:
    """Load the most recent snapshot rows (enough for lag/rolling features)."""
    parquet_files = sorted(SNAPSHOTS_DIR.glob("*.parquet"))
    if not parquet_files:
        print(f"No snapshot files found in {SNAPSHOTS_DIR}", file=sys.stderr)
        sys.exit(1)

    # Load the last file(s) and take the last n_rows
    df = pd.read_parquet(parquet_files[-1])
    if len(parquet_files) > 1 and len(df) < n_rows:
        prev_df = pd.read_parquet(parquet_files[-2])
        df = pd.concat([prev_df, df], ignore_index=True)

    return df.tail(n_rows).reset_index(drop=True)


def infer() -> None:
    """Run the full inference pipeline."""
    models = load_models()
    feature_columns = load_feature_columns()

    df = load_latest_snapshots()
    df = build_features(df)

    # Use the last row (most recent) for prediction
    latest = df.iloc[[-1]]
    X = latest[feature_columns]

    predictions: dict[str, float] = {}
    for target, model in models.items():
        pred = model.predict(X)
        predictions[target] = float(pred[0])

    # Build prediction output
    now = datetime.now(UTC).isoformat()
    up_target = round(predictions["thermostat_upstairs_target"], 1)
    down_target = round(
        predictions["thermostat_downstairs_target"], 1
    )
    output = {
        "timestamp": now,
        "thermostat_upstairs_target": up_target,
        "thermostat_downstairs_target": down_target,
        "mini_split_1_target": round(predictions["mini_split_1_target"], 1),
        "mini_split_1_mode": HVACMode.HEAT,  # TODO: predict mode
        "mini_split_2_target": round(predictions["mini_split_2_target"], 1),
        "mini_split_2_mode": HVACMode.HEAT,  # TODO: predict mode
        "floor_heat_on": True,  # TODO: predict
        "blower_1_on": False,  # TODO: predict
        "blower_2_on": False,  # TODO: predict
        "confidence": 0.0,  # TODO: compute from model
    }

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = PREDICTIONS_DIR / f"prediction_{date_str}.json"
    output_path.write_text(json.dumps(output, indent=2))

    print(f"Wrote prediction to {output_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    infer()
