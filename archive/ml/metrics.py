"""Metrics comparison CLI — view, compare, and track training run metrics.

Three subcommands:
  latest  — Most recent run summary per mode (default)
  history — RMSE trend across all runs
  compare — Side-by-side diff of two specific runs

Run:
  uv run python -m weatherstat.metrics
  uv run python -m weatherstat.metrics history
  uv run python -m weatherstat.metrics compare baseline_2026-02-12T041644 full_2026-02-12T041713
"""

import argparse
import json
import re
import sys
from pathlib import Path

from weatherstat.config import METRICS_DIR


def _load_all_metrics() -> list[dict]:
    """Load all metrics JSON files, sorted by timestamp ascending."""
    if not METRICS_DIR.exists():
        return []
    files = sorted(METRICS_DIR.glob("*.json"))
    results: list[dict] = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            data["_file"] = f.name
            results.append(data)
        except (json.JSONDecodeError, KeyError):
            print(f"  Warning: skipping malformed {f.name}", file=sys.stderr)
    results.sort(key=lambda d: d.get("timestamp", ""))
    return results


def _find_file(prefix: str, all_metrics: list[dict]) -> dict | None:
    """Find a metrics entry by prefix match on filename (without .json)."""
    for m in all_metrics:
        stem = Path(m["_file"]).stem
        if stem == prefix or stem.startswith(prefix):
            return m
    return None


def _parse_target(target: str) -> tuple[str, str]:
    """Parse 'room_temp_t+N' into (room, horizon label)."""
    match = re.match(r"(.+)_temp_t\+(\d+)", target)
    if not match:
        return target, ""
    room = match.group(1)
    h = int(match.group(2))
    # Map step count to human-readable horizon
    hour_map = {1: "1h", 2: "2h", 3: "3h", 4: "4h", 6: "6h", 12: "12h",
                24: "2h", 48: "4h", 72: "6h", 144: "12h"}
    return room, hour_map.get(h, f"t+{h}")


def cmd_latest(all_metrics: list[dict]) -> None:
    """Show latest run summary per mode."""
    if not all_metrics:
        print("No metrics files found in data/metrics/")
        return

    # Group by (mode, experiment) and take the latest
    latest: dict[tuple[str, str | None], dict] = {}
    for m in all_metrics:
        key = (m["mode"], m.get("experiment"))
        latest[key] = m  # sorted ascending, so last wins

    for (mode, experiment), m in sorted(latest.items()):
        label = f"{mode}" + (f" (experiment: {experiment})" if experiment else "")
        print(f"\n{'=' * 72}")
        print(f"  {label.upper()} — Latest Run")
        print(f"{'=' * 72}")
        print(f"  File:      {m['_file']}")
        print(f"  Date:      {m['timestamp'][:19]}")
        print(f"  Git hash:  {m.get('git_hash') or 'unknown'}")

        data = m.get("data", {})
        print(f"  Data:      {data.get('rows_raw', '?')} raw rows, "
              f"{data.get('rows_train', '?')} train, {data.get('rows_val', '?')} val")
        print(f"  Features:  {data.get('n_features', '?')}")

        date_range = data.get("date_range", [])
        if len(date_range) == 2:
            print(f"  Range:     {date_range[0][:10]} to {date_range[1][:10]}")

        targets = m.get("targets", [])
        if not targets:
            print("  No target results.")
            continue

        # Group by room, show table
        print(f"\n  {'Room':<18} {'Horizon':>8} {'RMSE':>8} {'MAE':>8}")
        print(f"  {'-' * 42}")

        current_room = ""
        for t in targets:
            room, horizon = _parse_target(t["target"])
            display_room = room if room != current_room else ""
            current_room = room
            print(f"  {display_room:<18} {horizon:>8} {t['rmse']:>8.4f} {t['mae']:>8.4f}")

        # Mean RMSE across all targets
        mean_rmse = sum(t["rmse"] for t in targets) / len(targets)
        mean_mae = sum(t["mae"] for t in targets) / len(targets)
        print(f"  {'-' * 42}")
        print(f"  {'MEAN':<18} {'':>8} {mean_rmse:>8.4f} {mean_mae:>8.4f}")


def cmd_history(all_metrics: list[dict]) -> None:
    """Show RMSE trend across all runs."""
    if not all_metrics:
        print("No metrics files found in data/metrics/")
        return

    # Group by mode
    by_mode: dict[str, list[dict]] = {}
    for m in all_metrics:
        mode = m["mode"]
        exp = m.get("experiment")
        label = f"{mode}" + (f"/{exp}" if exp else "")
        by_mode.setdefault(label, []).append(m)

    for mode_label, runs in sorted(by_mode.items()):
        print(f"\n{'=' * 78}")
        print(f"  {mode_label.upper()} — Training History ({len(runs)} runs)")
        print(f"{'=' * 78}")
        print(f"  {'Date':<20} {'Git':>8} {'Rows':>6} {'Features':>9} {'Mean RMSE':>10}")
        print(f"  {'-' * 53}")

        for m in runs:
            ts = m["timestamp"][:16].replace("T", " ")
            git_hash = (m.get("git_hash") or "?")[:7]
            data = m.get("data", {})
            rows = data.get("rows_raw", 0)
            n_feat = data.get("n_features", 0)
            targets = m.get("targets", [])
            mean_rmse = sum(t["rmse"] for t in targets) / len(targets) if targets else 0
            print(f"  {ts:<20} {git_hash:>8} {rows:>6} {n_feat:>9} {mean_rmse:>10.4f}")


def cmd_compare(file1: str, file2: str, all_metrics: list[dict]) -> None:
    """Side-by-side diff of two metrics files."""
    m1 = _find_file(file1, all_metrics)
    m2 = _find_file(file2, all_metrics)

    if m1 is None:
        print(f"No metrics file matching '{file1}'", file=sys.stderr)
        print(f"Available: {', '.join(Path(m['_file']).stem for m in all_metrics)}", file=sys.stderr)
        sys.exit(1)
    if m2 is None:
        print(f"No metrics file matching '{file2}'", file=sys.stderr)
        print(f"Available: {', '.join(Path(m['_file']).stem for m in all_metrics)}", file=sys.stderr)
        sys.exit(1)

    print("Comparing metrics:")
    print(f"  OLD: {m1['_file']}  ({m1['timestamp'][:19]})")
    print(f"  NEW: {m2['_file']}  ({m2['timestamp'][:19]})")

    d1, d2 = m1.get("data", {}), m2.get("data", {})
    print(f"\n  {'':>30} {'OLD':>10} {'NEW':>10} {'Delta':>10}")
    print(f"  {'-' * 60}")
    for label, key in [("Raw rows", "rows_raw"), ("Train rows", "rows_train"),
                       ("Val rows", "rows_val"), ("Features", "n_features")]:
        v1, v2 = d1.get(key, 0), d2.get(key, 0)
        delta = v2 - v1
        sign = "+" if delta > 0 else ""
        print(f"  {label:>30} {v1:>10} {v2:>10} {sign}{delta:>9}")

    # Per-target comparison
    targets1 = {t["target"]: t for t in m1.get("targets", [])}
    targets2 = {t["target"]: t for t in m2.get("targets", [])}
    all_targets = sorted(set(targets1) | set(targets2))

    if not all_targets:
        print("\n  No targets to compare.")
        return

    print(f"\n  {'Target':<30} {'Old RMSE':>10} {'New RMSE':>10} {'Delta':>9} {'Result':>8}")
    print(f"  {'-' * 67}")

    wins = 0
    losses = 0
    ties = 0
    total_delta = 0.0
    compared = 0

    for target in all_targets:
        t1 = targets1.get(target)
        t2 = targets2.get(target)

        room, horizon = _parse_target(target)
        label = f"{room} {horizon}"

        if t1 is None:
            print(f"  {label:<30} {'---':>10} {t2['rmse']:>10.4f} {'new':>9} {'':>8}")
            continue
        if t2 is None:
            print(f"  {label:<30} {t1['rmse']:>10.4f} {'---':>10} {'removed':>9} {'':>8}")
            continue

        delta = t2["rmse"] - t1["rmse"]
        total_delta += delta
        compared += 1

        if delta < -0.005:
            result = "better"
            wins += 1
        elif delta > 0.005:
            result = "worse"
            losses += 1
        else:
            result = "tie"
            ties += 1

        print(f"  {label:<30} {t1['rmse']:>10.4f} {t2['rmse']:>10.4f} {delta:>+9.4f} {result:>8}")

    print(f"  {'-' * 67}")
    mean_delta = total_delta / compared if compared else 0
    print(f"  Summary: {wins} better, {losses} worse, {ties} ties")
    print(f"  Mean RMSE change: {mean_delta:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="View and compare training metrics",
        prog="python -m weatherstat.metrics",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("latest", help="Latest run summary per mode (default)")
    sub.add_parser("history", help="RMSE trend across all runs")

    compare_p = sub.add_parser("compare", help="Diff two metrics files")
    compare_p.add_argument("file1", help="First metrics file (prefix match)")
    compare_p.add_argument("file2", help="Second metrics file (prefix match)")

    args = parser.parse_args()
    all_metrics = _load_all_metrics()

    if args.command == "history":
        cmd_history(all_metrics)
    elif args.command == "compare":
        cmd_compare(args.file1, args.file2, all_metrics)
    else:
        cmd_latest(all_metrics)


if __name__ == "__main__":
    main()
