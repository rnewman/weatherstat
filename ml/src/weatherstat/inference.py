"""Inference pipeline.

Load trained models + latest snapshot → build features → predict → write JSON.

Run: uv run python -m weatherstat.inference
"""

import json
import sys
from datetime import UTC, datetime

import lightgbm as lgb
import pandas as pd

from weatherstat.config import (
    HORIZONS_5MIN,
    MODELS_DIR,
    PREDICTION_ZONES,
    PREDICTIONS_DIR,
    SNAPSHOTS_DIR,
)
from weatherstat.features import build_features


def _target_columns() -> list[str]:
    return [
        f"{zone}_temp_t+{h}"
        for zone in PREDICTION_ZONES
        for h in HORIZONS_5MIN
    ]


def load_models(prefix: str = "full") -> dict[str, lgb.Booster]:
    """Load trained LightGBM models for all targets."""
    models: dict[str, lgb.Booster] = {}
    for target in _target_columns():
        model_path = MODELS_DIR / f"{prefix}_{target}_lgbm.txt"
        if not model_path.exists():
            print(f"Model not found: {model_path}", file=sys.stderr)
            sys.exit(1)
        models[target] = lgb.Booster(model_file=str(model_path))
    return models


def load_feature_columns(prefix: str = "full") -> list[str]:
    """Load the feature column names used during training."""
    feature_path = MODELS_DIR / f"{prefix}_feature_columns.txt"
    if not feature_path.exists():
        print(f"Feature columns file not found: {feature_path}", file=sys.stderr)
        sys.exit(1)
    return feature_path.read_text().strip().split("\n")


def load_latest_snapshots(n_rows: int = 48) -> pd.DataFrame:
    """Load the most recent snapshot rows (enough for lag/rolling features)."""
    parquet_files = sorted(SNAPSHOTS_DIR.glob("snapshot_*.parquet"))
    if not parquet_files:
        print(f"No snapshot files found in {SNAPSHOTS_DIR}", file=sys.stderr)
        sys.exit(1)

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
    df = build_features(df, mode="full")

    # Use the last row (most recent) for prediction
    latest = df.iloc[[-1]]
    available = [c for c in feature_columns if c in latest.columns]
    X = latest[available]

    predictions: dict[str, float] = {}
    for target, model in models.items():
        pred = model.predict(X)
        predictions[target] = float(pred[0])

    # Build prediction output with per-zone per-horizon temperatures
    now = datetime.now(UTC).isoformat()
    output: dict[str, object] = {
        "timestamp": now,
        "predictions": {
            target: round(val, 2) for target, val in predictions.items()
        },
        "confidence": 0.0,  # TODO: compute from model uncertainty
    }

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = PREDICTIONS_DIR / f"prediction_{date_str}.json"
    output_path.write_text(json.dumps(output, indent=2))

    print(f"Wrote prediction to {output_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    infer()
