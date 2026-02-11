"""Inference pipeline — fetch live state from HA and predict temperatures.

Two modes:
- Live: Fetch recent history from HA, build features, predict with both models
- Snapshot: Use collector snapshot files (for when collector is running)

Run:
  uv run python -m weatherstat.inference          # live from HA
  uv run python -m weatherstat.inference --snapshot  # from snapshot files
"""

import argparse
import json
import sqlite3
import sys
from datetime import UTC, datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

from weatherstat.config import (
    HORIZONS_5MIN,
    HORIZONS_HOURLY,
    MODELS_DIR,
    PREDICTION_ZONES,
    PREDICTIONS_DIR,
    SNAPSHOTS_DB,
)
from weatherstat.features import build_features


def _target_columns(horizons: list[int]) -> list[str]:
    return [f"{zone}_temp_t+{h}" for zone in PREDICTION_ZONES for h in horizons]


def load_models(prefix: str, horizons: list[int]) -> dict[str, lgb.Booster]:
    """Load trained LightGBM models for all targets."""
    models: dict[str, lgb.Booster] = {}
    for target in _target_columns(horizons):
        model_path = MODELS_DIR / f"{prefix}_{target}_lgbm.txt"
        if not model_path.exists():
            return {}  # model set not available
        models[target] = lgb.Booster(model_file=str(model_path))
    return models


def load_feature_columns(prefix: str) -> list[str]:
    """Load the feature column names used during training."""
    feature_path = MODELS_DIR / f"{prefix}_feature_columns.txt"
    if not feature_path.exists():
        return []
    return feature_path.read_text().strip().split("\n")


def _prepare_feature_row(
    df_features: pd.DataFrame,
    expected_columns: list[str],
) -> pd.DataFrame:
    """Extract the last row and ensure all expected feature columns are present.

    Missing columns are filled with NaN (LightGBM handles NaN natively).
    """
    latest = df_features.iloc[-1]
    data = {col: [latest[col] if col in latest.index else np.nan] for col in expected_columns}
    return pd.DataFrame(data)


# ── Live prediction from HA ────────────────────────────────────────────────


def fetch_recent_history(hours_back: int = 14) -> pd.DataFrame:
    """Fetch recent entity history from HA for prediction context.

    Returns a DataFrame in the same schema as extract_history() but for
    a short recent window — enough for lag/rolling feature computation.
    """
    from datetime import timedelta

    from weatherstat.extract import (
        CLIMATE_ENTITIES,
        FAN_ENTITIES,
        SENSOR_ENTITIES,
        STATISTICS_ENTITIES,
        WEATHER_ENTITY,
        WINDOW_SENSORS,
        _climate_to_series,
        _fan_to_series,
        _history_to_series,
        _weather_to_series,
        fetch_history,
        fetch_history_with_attributes,
    )

    end = datetime.now(UTC)
    start = end - timedelta(hours=hours_back)

    # Fetch sensor entities (no attributes needed)
    sensor_ids = list(STATISTICS_ENTITIES.values()) + list(SENSOR_ENTITIES.values())
    sensor_history = fetch_history(sensor_ids, start, end)

    # Fetch climate + fan + weather entities (with attributes)
    attr_ids = list(CLIMATE_ENTITIES.values()) + list(FAN_ENTITIES.values()) + [WEATHER_ENTITY]
    attr_history = fetch_history_with_attributes(attr_ids, start, end)

    # Fetch window sensors
    window_history = fetch_history(WINDOW_SENSORS, start, end)

    # Build 5-minute time index
    time_index = pd.date_range(start=start, end=end, freq="5min", tz=UTC)
    result = pd.DataFrame(index=time_index)
    result.index.name = "timestamp"

    # Process temperature/numeric sensors
    for col_name, entity_id in {**STATISTICS_ENTITIES, **SENSOR_ENTITIES}.items():
        records = sensor_history.get(entity_id, [])
        if not records:
            result[col_name] = np.nan
            continue
        if col_name == "navien_heating_mode":
            series = _history_to_series(records, value_fn=lambda s: s)
        else:
            series = _history_to_series(records)
        series = series[~series.index.duplicated(keep="last")]
        result[col_name] = series.reindex(time_index, method="ffill")

    # Process climate entities
    for name, entity_id in CLIMATE_ENTITIES.items():
        records = attr_history.get(entity_id, [])
        if not records:
            continue
        series_dict = _climate_to_series(records)
        if name.startswith("thermostat_"):
            for suffix, series in [("target", series_dict["target"]), ("action", series_dict["action"])]:
                s = series[~series.index.duplicated(keep="last")]
                result[f"{name}_{suffix}"] = s.reindex(time_index, method="ffill")
        else:
            for suffix, series in [
                ("temp", series_dict["temp"]),
                ("target", series_dict["target"]),
                ("mode", series_dict["mode"]),
            ]:
                s = series[~series.index.duplicated(keep="last")]
                result[f"{name}_{suffix}"] = s.reindex(time_index, method="ffill")

    # Process fan entities
    for name, entity_id in FAN_ENTITIES.items():
        records = attr_history.get(entity_id, [])
        if not records:
            result[f"{name}_mode"] = "off"
            continue
        series = _fan_to_series(records)
        series = series[~series.index.duplicated(keep="last")]
        result[f"{name}_mode"] = series.reindex(time_index, method="ffill")

    # Process weather entity
    weather_records = attr_history.get(WEATHER_ENTITY, [])
    if weather_records:
        weather_dict = _weather_to_series(weather_records)
        for suffix, series in weather_dict.items():
            col = f"outdoor_{suffix}" if suffix != "condition" else "weather_condition"
            s = series[~series.index.duplicated(keep="last")]
            result[col] = s.reindex(time_index, method="ffill")

    # Process window sensors
    window_series_list: list[pd.Series] = []
    for entity_id in WINDOW_SENSORS:
        records = window_history.get(entity_id, [])
        if records:
            s = _history_to_series(records, value_fn=lambda s: s == "on")
            s = s[~s.index.duplicated(keep="last")]
            window_series_list.append(s.reindex(time_index, method="ffill"))
    if window_series_list:
        window_df = pd.concat(window_series_list, axis=1)
        result["any_window_open"] = window_df.any(axis=1).astype(bool)
    else:
        result["any_window_open"] = False

    # Finalize
    result = result.reset_index()
    result["timestamp"] = result["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    # Ensure numeric columns
    numeric_cols = [
        "thermostat_upstairs_temp",
        "thermostat_downstairs_temp",
        "upstairs_aggregate_temp",
        "downstairs_aggregate_temp",
        "family_room_temp",
        "office_temp",
        "kitchen_temp",
        "bedroom_temp",
        "living_room_temp",
        "outdoor_temp",
        "indoor_humidity",
        "navien_heat_capacity",
        "outdoor_humidity",
        "outdoor_wind_speed",
        "thermostat_upstairs_target",
        "thermostat_downstairs_target",
        "mini_split_bedroom_temp",
        "mini_split_bedroom_target",
        "mini_split_living_room_temp",
        "mini_split_living_room_target",
    ]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    if "outdoor_wind_speed" in result.columns:
        result = result.rename(columns={"outdoor_wind_speed": "wind_speed"})

    return result


def _resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 5-min snapshot data to hourly means for baseline model."""
    df = df.copy()
    df["_ts"] = pd.to_datetime(df["timestamp"])
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    hourly = df[["_ts"] + numeric_cols].set_index("_ts").resample("1h").mean()
    hourly = hourly.reset_index().rename(columns={"_ts": "timestamp"})
    hourly["timestamp"] = hourly["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return hourly


def _predict_with_model(
    df: pd.DataFrame,
    mode: str,
    prefix: str,
    horizons: list[int],
) -> dict[str, float] | None:
    """Build features and predict with a model set. Returns None if models unavailable."""
    feature_columns = load_feature_columns(prefix)
    models = load_models(prefix, horizons)
    if not models or not feature_columns:
        return None

    df_feat = build_features(df.copy(), mode=mode)
    X = _prepare_feature_row(df_feat, feature_columns)

    predictions: dict[str, float] = {}
    for target, model in models.items():
        pred = model.predict(X)
        predictions[target] = float(pred[0])
    return predictions


def predict_live() -> None:
    """Fetch current entity states from HA and predict at all horizons."""
    from weatherstat.extract import _check_config

    _check_config()

    print("Fetching recent history from Home Assistant...")
    df_raw = fetch_recent_history(hours_back=14)
    n_rows = len(df_raw)
    print(f"  Retrieved {n_rows} rows (5-min intervals, ~{n_rows * 5 / 60:.0f} hours)")

    if n_rows < 24:
        print(f"Error: only {n_rows} rows, need at least 24 for features.", file=sys.stderr)
        sys.exit(1)

    now_str = df_raw["timestamp"].iloc[-1]

    # Current state
    latest_row = df_raw.iloc[-1]
    up_now = latest_row.get("thermostat_upstairs_temp")
    dn_now = latest_row.get("thermostat_downstairs_temp")
    out_now = latest_row.get("outdoor_temp")
    up_action = latest_row.get("thermostat_upstairs_action", "?")
    dn_action = latest_row.get("thermostat_downstairs_action", "?")
    weather = latest_row.get("weather_condition", "?")

    print(f"\nCurrent state ({now_str}):")
    print(f"  Upstairs:    {_fmt_temp(up_now)} (action: {up_action})")
    print(f"  Downstairs:  {_fmt_temp(dn_now)} (action: {dn_action})")
    print(f"  Outdoor:     {_fmt_temp(out_now)} ({weather})")

    # Full model predictions (5-min data, all features)
    print("\nRunning full model (5-min, all features)...")
    full_preds = _predict_with_model(df_raw, "full", "full", HORIZONS_5MIN)

    # Baseline model predictions (resample to hourly, temp-only)
    print("Running baseline model (hourly, temp-only)...")
    df_hourly = _resample_to_hourly(df_raw)
    baseline_preds = _predict_with_model(df_hourly, "baseline", "baseline", HORIZONS_HOURLY)

    # Print results table
    print(f"\n{'=' * 58}")
    print("TEMPERATURE PREDICTIONS")
    print(f"{'=' * 58}")
    print(f"  {'Zone':<14} {'Horizon':<10} {'Baseline':>10} {'Full':>10}")
    print(f"  {'-' * 50}")

    horizon_labels = {1: "1h", 2: "2h", 4: "4h", 6: "6h", 12: "12h"}

    for zone in PREDICTION_ZONES:
        for bh, fh in zip(HORIZONS_HOURLY, HORIZONS_5MIN, strict=True):
            b_key = f"{zone}_temp_t+{bh}"
            f_key = f"{zone}_temp_t+{fh}"
            b_val = baseline_preds.get(b_key, float("nan")) if baseline_preds else float("nan")
            f_val = full_preds.get(f_key, float("nan")) if full_preds else float("nan")
            label = horizon_labels[bh]
            print(f"  {zone:<14} {label:<10} {_fmt_temp(b_val):>10} {_fmt_temp(f_val):>10}")

    # Save prediction output
    output: dict[str, object] = {
        "timestamp": now_str,
        "current": {
            "upstairs_temp": _round_or_none(up_now),
            "downstairs_temp": _round_or_none(dn_now),
            "outdoor_temp": _round_or_none(out_now),
            "upstairs_action": str(up_action),
            "downstairs_action": str(dn_action),
            "weather": str(weather),
        },
        "baseline": {k: round(v, 2) for k, v in baseline_preds.items()} if baseline_preds else {},
        "full": {k: round(v, 2) for k, v in full_preds.items()} if full_preds else {},
    }

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = PREDICTIONS_DIR / f"prediction_{date_str}.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {output_path}")


def _fmt_temp(val: object) -> str:
    """Format a temperature value for display."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "  --"
    return f"{float(val):.1f}\u00b0F"


def _round_or_none(val: object) -> float | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), 2)


# ── Counterfactual predictions ──────────────────────────────────────────────

# Default setpoint range to sweep (°F)
DEFAULT_SETPOINTS = [68, 70, 72, 74, 76]


def _build_setpoint_overrides(
    up_temp: float,
    dn_temp: float,
    up_setpoint: float,
    dn_setpoint: float,
) -> dict[str, float]:
    """Build feature overrides for a setpoint scenario.

    The setpoint is the real control variable. Everything else follows:
    - If setpoint > current temp → thermostat heats, Navien fires, blowers run
    - If setpoint ≤ current temp → thermostat idles
    - target_delta = setpoint - current_temp (the gap the HVAC is closing)
    """
    overrides: dict[str, float] = {}

    up_heating = up_setpoint > up_temp
    dn_heating = dn_setpoint > dn_temp

    # Upstairs
    overrides["thermostat_upstairs_target"] = up_setpoint
    overrides["upstairs_target_delta"] = up_setpoint - up_temp
    overrides["thermostat_upstairs_action_enc"] = 1.0 if up_heating else 0.0

    # Downstairs
    overrides["thermostat_downstairs_target"] = dn_setpoint
    overrides["downstairs_target_delta"] = dn_setpoint - dn_temp
    overrides["thermostat_downstairs_action_enc"] = 1.0 if dn_heating else 0.0

    # Navien fires when either zone is heating
    overrides["navien_heating_mode_enc"] = 1.0 if (up_heating or dn_heating) else 0.0

    # Blowers follow downstairs thermostat
    blower = 1.0 if dn_heating else 0.0
    overrides["blower_family_room_mode_enc"] = blower
    overrides["blower_office_mode_enc"] = blower

    return overrides


def predict_counterfactual(setpoints: list[int] | None = None) -> None:
    """Predict temperatures under different thermostat setpoint scenarios.

    The setpoint is the actual control input — "what if I set both thermostats
    to X°F?" The model sees the target, the target-current delta, and the
    resulting HVAC action. No faked time series, no autoregressive propagation.

    Args:
        setpoints: List of setpoint temperatures to try (°F). Both zones set
                   to the same value for each scenario. Defaults to [68..76].
    """
    from weatherstat.extract import _check_config

    _check_config()

    if setpoints is None:
        setpoints = DEFAULT_SETPOINTS

    print("Fetching recent history from Home Assistant...")
    df_raw = fetch_recent_history(hours_back=14)
    n_rows = len(df_raw)
    print(f"  Retrieved {n_rows} rows")

    if n_rows < 24:
        print(f"Error: only {n_rows} rows, need >= 24.", file=sys.stderr)
        sys.exit(1)

    # Current state
    latest_row = df_raw.iloc[-1]
    now_str = df_raw["timestamp"].iloc[-1]
    up_now = float(latest_row.get("thermostat_upstairs_temp", 0))
    dn_now = float(latest_row.get("thermostat_downstairs_temp", 0))
    out_now = latest_row.get("outdoor_temp")
    up_target = latest_row.get("thermostat_upstairs_target", "?")
    dn_target = latest_row.get("thermostat_downstairs_target", "?")

    print(f"\nCurrent state ({now_str}):")
    print(f"  Upstairs:    {_fmt_temp(up_now)}  (setpoint: {_fmt_temp(up_target)})")
    print(f"  Downstairs:  {_fmt_temp(dn_now)}  (setpoint: {_fmt_temp(dn_target)})")
    print(f"  Outdoor:     {_fmt_temp(out_now)}")

    # Build features once
    feature_columns = load_feature_columns("full")
    models = load_models("full", HORIZONS_5MIN)
    if not models or not feature_columns:
        print("Error: full models not found. Run `just train-full` first.", file=sys.stderr)
        sys.exit(1)

    df_feat = build_features(df_raw.copy(), mode="full")
    base_row = _prepare_feature_row(df_feat, feature_columns)

    # Run each setpoint scenario (both zones set to same value)
    # results[setpoint][target_name] = predicted_temp
    results: dict[int, dict[str, float]] = {}

    for sp in setpoints:
        overrides = _build_setpoint_overrides(up_now, dn_now, float(sp), float(sp))
        X = base_row.copy()
        for col, val in overrides.items():
            if col in X.columns:
                X[col] = val
        preds: dict[str, float] = {}
        for target, model in models.items():
            preds[target] = float(model.predict(X)[0])
        results[sp] = preds

    # Print comparison table
    col_w = 10
    horizon_labels = {12: "1h", 24: "2h", 48: "4h", 72: "6h", 144: "12h"}

    print(f"\n{'=' * 72}")
    print("SETPOINT COUNTERFACTUALS (full model)")
    print(f"{'=' * 72}")
    print('  "What temperature will we reach if both thermostats are set to X?"')
    print(f"  Current temps: up={up_now:.1f}\u00b0F, dn={dn_now:.1f}\u00b0F, out={_fmt_temp(out_now)}\n")

    sp_labels = [f"{sp}\u00b0F" for sp in setpoints]
    header = f"  {'Zone':<14} {'':>4}" + "".join(f"{lbl:>{col_w}}" for lbl in sp_labels)
    print(header)
    print(f"  {'-' * (18 + col_w * len(setpoints))}")

    for zone in PREDICTION_ZONES:
        for h in HORIZONS_5MIN:
            target = f"{zone}_temp_t+{h}"
            row = f"  {zone:<14} {horizon_labels[h]:>4}"
            for sp in setpoints:
                v = results[sp].get(target, float("nan"))
                row += f"{_fmt_temp(v):>{col_w}}"
            print(row)
        print()

    # Delta table: difference from lowest setpoint
    lo = setpoints[0]
    print(f"  Delta vs setpoint={lo}\u00b0F:")
    print(header)
    print(f"  {'-' * (18 + col_w * len(setpoints))}")

    for zone in PREDICTION_ZONES:
        for h in HORIZONS_5MIN:
            target = f"{zone}_temp_t+{h}"
            base_val = results[lo].get(target, float("nan"))
            row = f"  {zone:<14} {horizon_labels[h]:>4}"
            for sp in setpoints:
                v = results[sp].get(target, float("nan"))
                delta = v - base_val
                if sp == lo:
                    row += f"{'---':>{col_w}}"
                else:
                    row += f"{delta:>+{col_w}.2f}"
            print(row)
        print()

    # Save
    output: dict[str, object] = {
        "timestamp": now_str,
        "current": {
            "upstairs_temp": up_now,
            "downstairs_temp": dn_now,
            "outdoor_temp": _round_or_none(out_now),
            "upstairs_target": _round_or_none(up_target),
            "downstairs_target": _round_or_none(dn_target),
        },
        "setpoints": {str(sp): {t: round(v, 2) for t, v in preds.items()} for sp, preds in results.items()},
    }

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = PREDICTIONS_DIR / f"counterfactual_{date_str}.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Saved to {output_path}")


# ── Snapshot-based inference (for collector mode) ───────────────────────────


def load_latest_snapshots(n_rows: int = 48) -> pd.DataFrame:
    """Load the most recent snapshot rows from SQLite (enough for lag/rolling features)."""
    if not SNAPSHOTS_DB.exists():
        print(f"No snapshot database found at {SNAPSHOTS_DB}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(SNAPSHOTS_DB))
    query = f"SELECT * FROM snapshots ORDER BY timestamp DESC LIMIT {n_rows}"
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        print("No snapshots in database", file=sys.stderr)
        sys.exit(1)

    # Reverse to chronological order
    df = df.iloc[::-1].reset_index(drop=True)

    # Convert any_window_open from INTEGER back to bool
    if "any_window_open" in df.columns:
        df["any_window_open"] = df["any_window_open"].astype(bool)

    return df


def infer_snapshot() -> None:
    """Run inference from collector snapshot files."""
    feature_columns = load_feature_columns("full")
    models = load_models("full", HORIZONS_5MIN)
    if not models:
        print("Error: full models not found. Run `just train-full` first.", file=sys.stderr)
        sys.exit(1)

    df = load_latest_snapshots()
    df = build_features(df, mode="full")

    X = _prepare_feature_row(df, feature_columns)

    predictions: dict[str, float] = {}
    for target, model in models.items():
        pred = model.predict(X)
        predictions[target] = float(pred[0])

    now = datetime.now(UTC).isoformat()
    output: dict[str, object] = {
        "timestamp": now,
        "predictions": {t: round(v, 2) for t, v in predictions.items()},
    }

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = PREDICTIONS_DIR / f"prediction_{date_str}.json"
    output_path.write_text(json.dumps(output, indent=2))

    print(f"Wrote prediction to {output_path}")
    print(json.dumps(output, indent=2))


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Weatherstat inference pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--snapshot",
        action="store_true",
        help="Use collector snapshot files instead of fetching from HA",
    )
    group.add_argument(
        "--counterfactual",
        "--cf",
        action="store_true",
        help="Predict under different thermostat setpoint scenarios",
    )
    parser.add_argument(
        "--setpoints",
        type=str,
        default=None,
        help="Comma-separated setpoints to test, e.g. '68,70,72,74,76'",
    )
    args = parser.parse_args()

    if args.snapshot:
        infer_snapshot()
    elif args.counterfactual:
        sp = [int(s) for s in args.setpoints.split(",")] if args.setpoints else None
        predict_counterfactual(setpoints=sp)
    else:
        predict_live()


if __name__ == "__main__":
    main()
