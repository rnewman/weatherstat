"""Experiment comparison — evaluate an experiment's models against production.

Trains (or loads) both production and experiment models, evaluates on the same
validation split, and prints a side-by-side RMSE/MAE comparison.

Run:
  uv run python -m weatherstat.experiment compare physics_v1
  uv run python -m weatherstat.experiment list
"""

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from weatherstat.config import (
    HORIZONS_5MIN,
    MODELS_DIR,
    PREDICTION_ROOMS,
    experiment_models_dir,
)
from weatherstat.features import (
    ROOM_TEMP_COLUMNS,
    add_future_targets,
    build_features,
)


def _target_cols(rooms: list[str], horizons: list[int]) -> list[str]:
    return [f"{room}_temp_t+{h}" for room in rooms for h in horizons]


def _evaluate_dir(
    models_dir: Path,
    model_prefix: str,
    mode: str,
    df: pd.DataFrame,
    horizons: list[int],
) -> dict[str, dict[str, float]] | None:
    """Evaluate models from a directory. Returns {target: {rmse, mae}} or None."""
    feature_path = models_dir / f"{model_prefix}_feature_columns.txt"
    if not feature_path.exists():
        return None

    feature_cols = feature_path.read_text().strip().split("\n")
    all_targets = _target_cols(PREDICTION_ROOMS, horizons)

    df_eval = build_features(df.copy(), mode=mode)
    df_eval = add_future_targets(df_eval, ROOM_TEMP_COLUMNS, horizons)

    target_cols = [
        t for t in all_targets
        if t in df_eval.columns
        and (models_dir / f"{model_prefix}_{t}_lgbm.txt").exists()
        and df_eval[t].notna().mean() >= 0.5
    ]
    if not target_cols:
        return None

    df_eval = df_eval.dropna(subset=target_cols)

    # Feature matrix
    X = pd.DataFrame(index=df_eval.index)
    for col in feature_cols:
        X[col] = df_eval[col] if col in df_eval.columns else np.nan

    # 80/20 split (same as training)
    split_idx = int(len(X) * 0.8)
    X_val = X.iloc[split_idx:]

    results: dict[str, dict[str, float]] = {}
    for target in target_cols:
        model_path = models_dir / f"{model_prefix}_{target}_lgbm.txt"
        model = lgb.Booster(model_file=str(model_path))

        y_val = df_eval[target].iloc[split_idx:]
        y_pred = model.predict(X_val)

        rmse = float(((y_val - y_pred) ** 2).mean() ** 0.5)
        mae = float((y_val - y_pred).abs().mean())
        results[target] = {"rmse": rmse, "mae": mae}

    return results


def _load_data() -> pd.DataFrame:
    """Load evaluation data from collector."""
    from weatherstat.train import load_training_data
    return load_training_data()


def compare(experiment_name: str) -> None:
    """Compare an experiment's models against production."""
    exp_dir = experiment_models_dir(experiment_name)
    if not exp_dir.exists():
        print(f"Experiment '{experiment_name}' not found at {exp_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Comparing experiment '{experiment_name}' vs production")
    print(f"  Production: {MODELS_DIR}")
    print(f"  Experiment: {exp_dir}\n")

    for mode, horizons, prefix in [
        ("full", HORIZONS_5MIN, "full"),
    ]:
        # Check if experiment has this model type
        if not (exp_dir / f"{prefix}_feature_columns.txt").exists():
            continue

        print(f"{'=' * 72}")
        print("  MODEL COMPARISON")
        print(f"{'=' * 72}")

        df = _load_data()
        prod_results = _evaluate_dir(MODELS_DIR, prefix, mode, df, horizons)
        exp_results = _evaluate_dir(exp_dir, prefix, mode, df, horizons)

        if prod_results is None:
            print("  No production models found — skipping comparison")
            if exp_results:
                print(f"  Experiment has {len(exp_results)} targets")
            continue
        if exp_results is None:
            print("  No experiment models found — skipping")
            continue

        # Side-by-side table
        all_targets = sorted(set(prod_results) | set(exp_results))
        horizon_map = {12: "1h", 24: "2h", 48: "4h", 72: "6h", 144: "12h",
                       1: "1h", 2: "2h", 4: "4h", 6: "6h"}

        print(f"\n  {'Target':<30} {'Prod RMSE':>10} {'Exp RMSE':>10} {'Delta':>8} {'Better':>7}")
        print(f"  {'-' * 65}")

        wins = 0
        losses = 0
        for target in all_targets:
            prod = prod_results.get(target)
            exp = exp_results.get(target)
            if prod is None or exp is None:
                continue

            delta = exp["rmse"] - prod["rmse"]
            better = "exp" if delta < -0.005 else ("prod" if delta > 0.005 else "tie")
            if better == "exp":
                wins += 1
            elif better == "prod":
                losses += 1

            # Parse room + horizon for display
            parts = target.rsplit("_temp_t+", 1)
            room = parts[0]
            h = int(parts[1]) if len(parts) > 1 else 0
            label = f"{room} {horizon_map.get(h, f't+{h}')}"

            print(
                f"  {label:<30} {prod['rmse']:>10.4f} {exp['rmse']:>10.4f}"
                f" {delta:>+8.4f} {better:>7}"
            )

        total = wins + losses + (len(all_targets) - wins - losses)
        print(f"\n  Summary: {wins} wins, {losses} losses, {total - wins - losses} ties")
        print()

    # Save comparison
    output = {
        "experiment": experiment_name,
        "production_dir": str(MODELS_DIR),
        "experiment_dir": str(exp_dir),
    }
    report_path = exp_dir / "comparison.json"
    report_path.write_text(json.dumps(output, indent=2))
    print(f"Report saved to {report_path}")


def list_experiments() -> None:
    """List all experiment directories."""
    if not MODELS_DIR.exists():
        print("No models directory found")
        return

    experiments = [d for d in MODELS_DIR.iterdir() if d.is_dir()]
    if not experiments:
        print("No experiments found in data/models/")
        return

    print("Experiments:")
    for exp_dir in sorted(experiments):
        has_full = (exp_dir / "full_feature_columns.txt").exists()
        model_str = "full" if has_full else "empty"
        print(f"  {exp_dir.name:<20} [{model_str}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment management")
    sub = parser.add_subparsers(dest="command")

    compare_p = sub.add_parser("compare", help="Compare experiment vs production")
    compare_p.add_argument("name", help="Experiment name")

    sub.add_parser("list", help="List experiments")

    args = parser.parse_args()
    if args.command == "compare":
        compare(args.name)
    elif args.command == "list":
        list_experiments()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
