"""Model evaluation and comparison.

Compares baseline vs full model on the validation window.
Reports per-zone per-horizon RMSE, MAE, and feature importance.

Run: uv run python -m weatherstat.evaluate
"""


import lightgbm as lgb
import pandas as pd

from weatherstat.config import (
    HORIZONS_5MIN,
    HORIZONS_HOURLY,
    MODELS_DIR,
    PREDICTION_ROOMS,
    SNAPSHOTS_DIR,
)
from weatherstat.features import (
    ROOM_TEMP_COLUMNS,
    add_future_targets,
    build_features,
)


def _load_model(path_str: str) -> lgb.Booster | None:
    from pathlib import Path
    path = Path(path_str)
    if not path.exists():
        return None
    return lgb.Booster(model_file=str(path))


def _target_cols(zones: list[str], horizons: list[int]) -> list[str]:
    return [f"{zone}_temp_t+{h}" for zone in zones for h in horizons]


def evaluate_model(
    mode: str,
    df: pd.DataFrame,
    horizons: list[int],
    model_prefix: str,
) -> pd.DataFrame | None:
    """Evaluate a model set on the validation portion of data."""
    feature_path = MODELS_DIR / f"{model_prefix}_feature_columns.txt"
    if not feature_path.exists():
        print(f"  No {model_prefix} model found (missing {feature_path.name})")
        return None

    feature_cols = feature_path.read_text().strip().split("\n")
    all_target_cols = _target_cols(PREDICTION_ROOMS, horizons)

    # Build features and targets
    df = build_features(df, mode=mode)
    df = add_future_targets(df, ROOM_TEMP_COLUMNS, horizons)

    # Filter to targets that exist and have models
    target_cols = [
        t for t in all_target_cols
        if t in df.columns
        and (MODELS_DIR / f"{model_prefix}_{t}_lgbm.txt").exists()
        and df[t].notna().mean() >= 0.5
    ]
    if not target_cols:
        print("  No targets with sufficient data and trained models")
        return None

    # Drop NaN for available targets only
    df = df.dropna(subset=target_cols)
    df = df.dropna()

    if len(df) < 10:
        print(f"  Too few rows ({len(df)}) for evaluation")
        return None

    # Validation split (last 20%)
    split_idx = int(len(df) * 0.8)
    val_df = df.iloc[split_idx:]

    # Check which feature columns exist in the data
    available_features = [c for c in feature_cols if c in val_df.columns]
    if len(available_features) < len(feature_cols):
        missing = set(feature_cols) - set(available_features)
        print(f"  Warning: {len(missing)} feature columns missing from data")

    X_val = val_df[available_features]

    results: list[dict[str, object]] = []

    for target in target_cols:
        model_path = MODELS_DIR / f"{model_prefix}_{target}_lgbm.txt"
        model = _load_model(str(model_path))
        if model is None:
            print(f"  Missing model: {model_path.name}")
            continue

        y_val = val_df[target]
        y_pred = model.predict(X_val)

        rmse = ((y_val - y_pred) ** 2).mean() ** 0.5
        mae = (y_val - y_pred).abs().mean()

        # Parse room and horizon from target name
        parts = target.rsplit("_t+", 1)
        room = parts[0].replace("_temp", "")
        horizon = parts[1] if len(parts) > 1 else "?"

        results.append({
            "room": room,
            "horizon": horizon,
            "rmse": round(float(rmse), 4),
            "mae": round(float(mae), 4),
            "n_val": len(X_val),
        })

    if not results:
        return None

    return pd.DataFrame(results)


def print_feature_importance(model_prefix: str, top_n: int = 15) -> None:
    """Print aggregate feature importance across all models of a given prefix."""
    feature_path = MODELS_DIR / f"{model_prefix}_feature_columns.txt"
    if not feature_path.exists():
        return

    feature_cols = feature_path.read_text().strip().split("\n")
    all_importance = pd.Series(dtype=float)

    model_files = sorted(MODELS_DIR.glob(f"{model_prefix}_*_lgbm.txt"))
    for mf in model_files:
        model = lgb.Booster(model_file=str(mf))
        importance = pd.Series(
            model.feature_importance(importance_type="gain"),
            index=feature_cols[:len(model.feature_importance())],
        )
        all_importance = all_importance.add(importance, fill_value=0)

    if all_importance.empty:
        return

    # Normalize to mean
    all_importance = all_importance / len(model_files)
    top = all_importance.sort_values(ascending=False).head(top_n)

    print(f"\nAggregate feature importance ({model_prefix}, top {top_n}):")
    for feat, imp in top.items():
        bar = "=" * int(imp / top.max() * 30)
        print(f"  {feat:45s} {imp:8.1f}  {bar}")


def main() -> None:
    print("=" * 60)
    print("WEATHERSTAT MODEL EVALUATION")
    print("=" * 60)

    # Evaluate baseline model
    hourly_path = SNAPSHOTS_DIR / "historical_hourly.parquet"
    baseline_results = None
    if hourly_path.exists():
        print("\n--- Baseline model (hourly temp-only, 5+ months) ---")
        hourly_df = pd.read_parquet(hourly_path)
        print(f"  Data: {len(hourly_df)} hourly rows")
        baseline_results = evaluate_model(
            "baseline", hourly_df, HORIZONS_HOURLY, "baseline"
        )
        if baseline_results is not None:
            print(baseline_results.to_string(index=False))
            print_feature_importance("baseline")
    else:
        print("\nNo hourly data found — skipping baseline evaluation")

    # Evaluate full model
    full_path = SNAPSHOTS_DIR / "historical_full.parquet"
    full_results = None
    if full_path.exists():
        print("\n--- Full model (5-min all features, ~10 days) ---")
        full_df = pd.read_parquet(full_path)
        print(f"  Data: {len(full_df)} rows")
        full_results = evaluate_model(
            "full", full_df, HORIZONS_5MIN, "full"
        )
        if full_results is not None:
            print(full_results.to_string(index=False))
            print_feature_importance("full")
    else:
        print("\nNo full-feature data found — skipping full evaluation")

    # Comparison
    if baseline_results is not None and full_results is not None:
        print(f"\n{'=' * 60}")
        print("COMPARISON: Baseline vs Full")
        print(f"{'=' * 60}")

        # Compare at 1h horizon (common between both)
        bl_1h = baseline_results[baseline_results["horizon"] == "1"]
        # Full model's 1h horizon is 12 steps at 5-min intervals
        fu_1h = full_results[full_results["horizon"] == "12"]

        if not bl_1h.empty and not fu_1h.empty:
            for room in PREDICTION_ROOMS:
                bl_row = bl_1h[bl_1h["room"] == room]
                fu_row = fu_1h[fu_1h["room"] == room]
                if bl_row.empty or fu_row.empty:
                    continue
                bl_rmse = float(bl_row["rmse"].iloc[0])
                fu_rmse = float(fu_row["rmse"].iloc[0])
                diff = fu_rmse - bl_rmse
                better = "BETTER" if diff < 0 else "WORSE" if diff > 0 else "SAME"
                print(f"  {room} (1h): baseline={bl_rmse:.3f}, full={fu_rmse:.3f}, "
                      f"delta={diff:+.3f} ({better})")
        else:
            print("  Cannot compare — different horizon formats or missing data")

    # Assessment
    print(f"\n{'=' * 60}")
    print("ASSESSMENT")
    print(f"{'=' * 60}")

    if baseline_results is not None:
        avg_rmse = baseline_results["rmse"].mean()
        if avg_rmse < 2.0:
            print(f"  Baseline avg RMSE {avg_rmse:.3f}F < 2.0F — thermal model captures dynamics")
        else:
            print(f"  Baseline avg RMSE {avg_rmse:.3f}F >= 2.0F — model needs more data or tuning")

    if full_results is not None:
        avg_rmse = full_results["rmse"].mean()
        print(f"  Full model avg RMSE: {avg_rmse:.3f}F")

    if baseline_results is not None and full_results is not None:
        bl_avg = baseline_results["rmse"].mean()
        fu_avg = full_results["rmse"].mean()
        if fu_avg < bl_avg:
            print("  Full model outperforms baseline — HVAC features add signal")
        else:
            print("  Full model does not beat baseline — need more data for HVAC features")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
