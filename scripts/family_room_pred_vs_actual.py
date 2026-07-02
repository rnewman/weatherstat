"""Compare predicted vs actual family_room_temp at 1h/2h/4h/6h horizons.

For each decision row in a chosen window, look up the actual temp at
timestamp+horizon in the decision log itself (current_temps of the future
decision closest to that target time, within tolerance).
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta

DB = os.path.expanduser("~/.weatherstat/decision_log.db")
SENSOR = "family_room_temp"
HORIZONS = [("1h", 1), ("2h", 2), ("4h", 4), ("6h", 6)]
WINDOW_DAYS = 14


def main() -> None:
    con = sqlite3.connect(DB)
    cur = con.cursor()

    cutoff = (datetime.now().astimezone() - timedelta(days=WINDOW_DAYS)).isoformat()
    cur.execute(
        "SELECT timestamp, current_temps, predictions, weather_condition, outdoor_temp "
        "FROM decisions WHERE timestamp >= ? ORDER BY timestamp",
        (cutoff,),
    )
    rows = cur.fetchall()
    print(f"Loaded {len(rows)} decisions since {cutoff}")

    # Build a sorted list of (ts, current_temp[SENSOR]) for actual lookup
    actuals: list[tuple[datetime, float]] = []
    for ts, ct_json, _, _, _ in rows:
        ct = json.loads(ct_json)
        v = ct.get(SENSOR)
        if v is None:
            continue
        try:
            tdt = datetime.fromisoformat(ts)
        except ValueError:
            continue
        actuals.append((tdt, float(v)))

    def actual_at(target: datetime, tol_min: int = 8) -> float | None:
        # Binary search would be better, but list is small; linear is fine.
        best = None
        best_diff = timedelta(minutes=tol_min + 1)
        for t, v in actuals:
            d = abs(t - target)
            if d < best_diff:
                best_diff = d
                best = v
            if t > target + timedelta(minutes=tol_min):
                break
        return best

    # Bucket errors by sun condition + horizon
    by_cond_horizon: dict[tuple[str, str], list[float]] = {}
    counts: dict[tuple[str, str], int] = {}

    for ts, _ct_json, pred_json, weather, outdoor in rows:
        try:
            tdt = datetime.fromisoformat(ts)
        except ValueError:
            continue
        preds = json.loads(pred_json).get(SENSOR)
        if not preds:
            continue
        cond = weather or "unknown"
        for label, hours in HORIZONS:
            pred = preds.get(label)
            if pred is None:
                continue
            actual = actual_at(tdt + timedelta(hours=hours))
            if actual is None:
                continue
            err = pred - actual  # positive = over-prediction
            by_cond_horizon.setdefault((cond, label), []).append(err)
            counts[(cond, label)] = counts.get((cond, label), 0) + 1

    # Summarize
    print(f"\n=== {SENSOR} prediction error (pred - actual), last {WINDOW_DAYS}d ===\n")
    print(f"{'condition':<16} {'horizon':<6} {'n':>5}  {'mean':>7}  {'p50':>6}  {'p90':>6}  {'max':>6}")
    print("-" * 70)
    for (cond, h), errs in sorted(by_cond_horizon.items()):
        if len(errs) < 5:
            continue
        errs_sorted = sorted(errs)
        n = len(errs)
        mean = sum(errs) / n
        p50 = errs_sorted[n // 2]
        p90 = errs_sorted[int(n * 0.9)]
        mx = max(errs)
        print(f"{cond:<16} {h:<6} {n:>5}  {mean:>+7.2f}  {p50:>+6.2f}  {p90:>+6.2f}  {mx:>+6.2f}")

    # All conditions combined
    print("\n=== combined (all weather) ===")
    print(f"{'horizon':<6} {'n':>5}  {'mean':>7}  {'p50':>6}  {'p90':>6}  {'max':>6}")
    print("-" * 50)
    for h_label in [h[0] for h in HORIZONS]:
        all_err = [e for (_c, h), errs in by_cond_horizon.items() if h == h_label for e in errs]
        if not all_err:
            continue
        all_err_sorted = sorted(all_err)
        n = len(all_err)
        mean = sum(all_err) / n
        p50 = all_err_sorted[n // 2]
        p90 = all_err_sorted[int(n * 0.9)]
        mx = max(all_err)
        print(f"{h_label:<6} {n:>5}  {mean:>+7.2f}  {p50:>+6.2f}  {p90:>+6.2f}  {mx:>+6.2f}")


if __name__ == "__main__":
    main()
