"""Backtest models against the overnight cooling test case (Feb 14-15 2026).

The house cooled from ~71°F to ~63°F with all HVAC off and outdoor temp 39°F.
This is a test of whether models understand passive cooling — any model that
predicts stable ~70°F temperatures is learning spurious correlations.

Usage:
  uv run python scripts/backtest_overnight.py                    # production full model
  uv run python scripts/backtest_overnight.py --experiment NAME  # experiment model
  uv run python scripts/backtest_overnight.py --all              # production + all experiments
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
from weatherstat.features import ROOM_TEMP_COLUMNS, build_features

FIXTURE_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "overnight_cooling_20260214.parquet"

# Overnight window (PST): 10pm Feb 14 → 6am Feb 15
# In UTC: 06:00 → 14:00 Feb 15
OVERNIGHT_START_UTC = pd.Timestamp("2026-02-15 06:00:00", tz="UTC")
OVERNIGHT_END_UTC = pd.Timestamp("2026-02-15 14:00:00", tz="UTC")

HORIZON_LABELS = {12: "1h", 24: "2h", 48: "4h", 72: "6h", 144: "12h"}


def load_models(models_dir: Path) -> tuple[dict[str, lgb.Booster], list[str]]:
    """Load full models and feature columns from a directory."""
    feature_path = models_dir / "full_feature_columns.txt"
    if not feature_path.exists():
        return {}, []
    feature_cols = feature_path.read_text().strip().split("\n")
    models: dict[str, lgb.Booster] = {}
    for room in PREDICTION_ROOMS:
        for h in HORIZONS_5MIN:
            target = f"{room}_temp_t+{h}"
            model_path = models_dir / f"full_{target}_lgbm.txt"
            if model_path.exists():
                models[target] = lgb.Booster(model_file=str(model_path))
    return models, feature_cols


def run_backtest(
    df_features: pd.DataFrame,
    models: dict[str, lgb.Booster],
    feature_cols: list[str],
    label: str,
) -> pd.DataFrame:
    """Run predictions at 15-min intervals and compare to actuals."""
    df = df_features.copy()
    df["_ts"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)

    overnight = df[(df["_ts"] >= OVERNIGHT_START_UTC) & (df["_ts"] <= OVERNIGHT_END_UTC)]
    overnight = overnight.iloc[::3]  # 15-min intervals

    df_lookup = df.set_index("_ts")
    rows: list[dict] = []

    for _, row in overnight.iterrows():
        t = row["_ts"]
        X = pd.DataFrame([{col: row[col] if col in row.index else np.nan for col in feature_cols}])

        for room in PREDICTION_ROOMS:
            temp_col = ROOM_TEMP_COLUMNS.get(room)
            if not temp_col:
                continue

            for h in HORIZONS_5MIN:
                target = f"{room}_temp_t+{h}"
                model = models.get(target)
                if model is None:
                    continue

                future_t = t + pd.Timedelta(minutes=5 * h)
                actual = np.nan
                if future_t in df_lookup.index and temp_col in df_lookup.columns:
                    actual = df_lookup.loc[future_t, temp_col]
                else:
                    nearby = df_lookup.index[abs(df_lookup.index - future_t) < pd.Timedelta(minutes=3)]
                    if len(nearby) > 0 and temp_col in df_lookup.columns:
                        actual = df_lookup.loc[nearby[0], temp_col]

                pred = float(model.predict(X)[0])
                rows.append({
                    "model": label,
                    "time_pst": t.tz_convert("US/Pacific").strftime("%H:%M"),
                    "room": room,
                    "horizon": HORIZON_LABELS[h],
                    "horizon_steps": h,
                    "actual": actual,
                    "pred": pred,
                    "error": pred - actual if not np.isnan(actual) else np.nan,
                })

    return pd.DataFrame(rows)


def print_report(results: pd.DataFrame) -> None:
    """Print per-model, per-horizon summary."""
    valid = results.dropna(subset=["actual"])
    if valid.empty:
        print("No actuals available")
        return

    model_labels = valid["model"].unique()

    # Overall summary
    print(f"\n{'=' * 78}")
    header = f"  {'Horizon':<8}"
    for _label in model_labels:
        header += f" {'RMSE':>8} {'Bias':>8} |"
    header = header.rstrip(" |")
    label_header = f"  {'':8}"
    for lbl in model_labels:
        label_header += f"  {lbl:>16} |"
    label_header = label_header.rstrip(" |")
    print(label_header)
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for h in HORIZONS_5MIN:
        hlabel = HORIZON_LABELS[h]
        line = f"  {hlabel:<8}"
        for label in model_labels:
            hdata = valid[(valid["model"] == label) & (valid["horizon_steps"] == h)]
            if hdata.empty:
                line += f" {'—':>8} {'—':>8} |"
                continue
            rmse = (hdata["error"] ** 2).mean() ** 0.5
            bias = hdata["error"].mean()
            line += f" {rmse:>8.2f} {bias:>+8.2f} |"
        print(line.rstrip(" |"))

    # Per-room breakdown
    for room in PREDICTION_ROOMS:
        room_data = valid[valid["room"] == room]
        if room_data.empty:
            continue

        print(f"\n{'─' * 78}")
        print(f"  {room.upper()}")
        header = f"  {'Horizon':<8}"
        for _label in model_labels:
            header += f" {'RMSE':>8} {'Bias':>8} |"
        header = header.rstrip(" |")
        print(header)

        for h in HORIZONS_5MIN:
            hlabel = HORIZON_LABELS[h]
            line = f"  {hlabel:<8}"
            for label in model_labels:
                hdata = room_data[(room_data["model"] == label) & (room_data["horizon_steps"] == h)]
                if hdata.empty:
                    line += f" {'—':>8} {'—':>8} |"
                    continue
                rmse = (hdata["error"] ** 2).mean() ** 0.5
                bias = hdata["error"].mean()
                line += f" {rmse:>8.2f} {bias:>+8.2f} |"
            print(line.rstrip(" |"))

    # Timeline for key rooms at 1h horizon
    for room in ["upstairs", "bedroom", "downstairs"]:
        room_1h = valid[(valid["room"] == room) & (valid["horizon_steps"] == 12)]
        if room_1h.empty:
            continue
        # Pivot: one row per timestamp
        times = room_1h["time_pst"].unique()
        print(f"\n{'─' * 78}")
        cols = f"  {'Time':<7} {'Actual':>7}"
        for label in model_labels:
            cols += f" {label[:12]:>12} {'err':>6}"
        print(f"  {room.upper()} — 1h ahead")
        print(cols)
        for t in times:
            t_data = room_1h[room_1h["time_pst"] == t]
            actual = t_data["actual"].iloc[0]
            line = f"  {t:<7} {actual:>7.1f}"
            for label in model_labels:
                m_data = t_data[t_data["model"] == label]
                if m_data.empty:
                    line += f" {'—':>12} {'—':>6}"
                else:
                    line += f" {m_data['pred'].iloc[0]:>12.1f} {m_data['error'].iloc[0]:>+6.1f}"
            print(line)


def save_metrics(results: pd.DataFrame, output_path: Path) -> None:
    """Save per-model, per-room, per-horizon RMSE/bias to JSON."""
    valid = results.dropna(subset=["actual"])
    metrics: dict = {}

    for label in valid["model"].unique():
        model_metrics: dict = {"overall": {}, "rooms": {}}
        m_data = valid[valid["model"] == label]

        for h in HORIZONS_5MIN:
            hdata = m_data[m_data["horizon_steps"] == h]
            if hdata.empty:
                continue
            model_metrics["overall"][HORIZON_LABELS[h]] = {
                "rmse": round(float((hdata["error"] ** 2).mean() ** 0.5), 4),
                "bias": round(float(hdata["error"].mean()), 4),
                "n": len(hdata),
            }

        for room in PREDICTION_ROOMS:
            room_data = m_data[m_data["room"] == room]
            if room_data.empty:
                continue
            room_metrics: dict = {}
            for h in HORIZONS_5MIN:
                hdata = room_data[room_data["horizon_steps"] == h]
                if hdata.empty:
                    continue
                room_metrics[HORIZON_LABELS[h]] = {
                    "rmse": round(float((hdata["error"] ** 2).mean() ** 0.5), 4),
                    "bias": round(float(hdata["error"].mean()), 4),
                    "n": len(hdata),
                }
            model_metrics["rooms"][room] = room_metrics

        metrics[label] = model_metrics

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nMetrics saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest against overnight cooling fixture")
    parser.add_argument("--experiment", type=str, help="Compare experiment against production")
    parser.add_argument("--all", action="store_true", help="Compare production + all experiments")
    args = parser.parse_args()

    if not FIXTURE_PATH.exists():
        print(f"Fixture not found: {FIXTURE_PATH}", file=sys.stderr)
        print("Run: uv run python scripts/save_overnight_fixture.py", file=sys.stderr)
        sys.exit(1)

    # Load and prepare fixture
    print("Loading overnight cooling fixture...")
    raw = pd.read_parquet(FIXTURE_PATH)
    print(f"  {len(raw)} rows ({raw['timestamp'].iloc[0]} to {raw['timestamp'].iloc[-1]})")

    print("Building features...")
    df = build_features(raw.copy(), mode="full")

    # Determine which model sets to evaluate
    model_sets: list[tuple[str, Path]] = [("production", MODELS_DIR)]

    if args.all:
        for d in sorted(MODELS_DIR.iterdir()):
            if d.is_dir() and (d / "full_feature_columns.txt").exists():
                model_sets.append((d.name, d))
    elif args.experiment:
        exp_dir = experiment_models_dir(args.experiment)
        if not exp_dir.exists():
            print(f"Experiment '{args.experiment}' not found at {exp_dir}", file=sys.stderr)
            sys.exit(1)
        model_sets.append((args.experiment, exp_dir))

    # Run backtest for each model set
    all_results: list[pd.DataFrame] = []
    for label, model_dir in model_sets:
        models, features = load_models(model_dir)
        if not models:
            print(f"  No full models in {model_dir}, skipping")
            continue
        print(f"  Running {label} ({len(models)} targets)...")
        results = run_backtest(df, models, features, label)
        all_results.append(results)

    if not all_results:
        print("No models to evaluate")
        sys.exit(1)

    combined = pd.concat(all_results, ignore_index=True)

    print_report(combined)
    save_metrics(combined, Path("data/metrics/backtest_overnight_cooling.json"))


if __name__ == "__main__":
    main()
