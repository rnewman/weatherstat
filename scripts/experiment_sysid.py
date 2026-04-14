"""Sysid experiment harness: run fit_sysid, extract metrics, compare variants.

Usage:
    # Save baseline metrics
    uv run python scripts/experiment_sysid.py --save-baseline

    # Run variant and compare against baseline
    uv run python scripts/experiment_sysid.py --compare experiments/baseline.json

    # Just print metrics without saving
    uv run python scripts/experiment_sysid.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from weatherstat.sysid import fit_sysid
from weatherstat.validate import RegressionDiagnostics, compute_sensor_health


# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class GainEntry:
    """One effector→sensor gain with metadata."""

    effector: str
    sensor: str
    gain: float
    t_stat: float
    lag_min: float
    negligible: bool
    classification: str  # "direct", "cross", "negligible"


@dataclass
class SensorQuality:
    """Per-sensor regression quality metrics."""

    sensor: str
    r_squared: float
    durbin_watson: float
    holdout_degradation: float | None
    n_rows: int
    n_features: int
    grade: str


@dataclass
class EffectorSummary:
    """Per-effector gain analysis."""

    effector: str
    direct_sensor: str
    direct_gain: float
    direct_t: float
    n_cross: int
    mean_cross_gain: float
    max_cross_gain: float
    cross_direct_ratio: float


@dataclass
class ExperimentReport:
    """Full experiment metrics report."""

    variant: str
    data_start: str
    data_end: str
    n_snapshots: int
    sensors: list[SensorQuality]
    gains: list[GainEntry]
    effector_summaries: list[EffectorSummary]
    # Aggregate scores
    mean_r2: float
    mean_dw: float
    mean_cross_room_gain: float
    mean_direct_gain: float
    cross_direct_ratio: float
    # Specific target we're tracking
    target_ms_lr_to_tu: float | None  # mini_split_living_room → thermostat_upstairs_temp


# ── Metrics extraction ───────────────────────────────────────────────────


def _extract_report(variant: str = "baseline") -> ExperimentReport:
    """Run fit_sysid and extract structured metrics."""
    result, sensor_diagnostics = fit_sysid(verbose=False)

    # ── Per-sensor quality ──
    sensor_quality: list[SensorQuality] = []
    for sensor_name, diag in sensor_diagnostics.items():
        # Count significant gains for this sensor
        sig_gains = [
            g for g in result.effector_sensor_gains
            if g.sensor == sensor_name and not g.negligible
        ]
        # Count advisory betas
        adv_betas = len(result.environment_solar_betas.get(sensor_name, {}))
        for ft in result.fitted_taus:
            if ft.sensor == sensor_name:
                adv_betas += len(ft.environment_tau_betas)

        # Compute health grade
        n_segments = 0
        for ft in result.fitted_taus:
            if ft.sensor == sensor_name:
                n_segments = ft.n_segments
                break

        health = compute_sensor_health(
            sensor_name,
            r_squared=diag.r_squared,
            durbin_watson=diag.durbin_watson,
            n_segments=n_segments,
            n_gains=len(sig_gains),
            n_effectors=len(result.effectors),
            n_advisory_betas=adv_betas,
            n_unstable_kept=0,  # not tracked here
            holdout_degradation=diag.holdout_degradation,
            has_validation_errors=any(
                i.severity.value == "error" for i in diag.issues
            ),
        )
        sensor_quality.append(SensorQuality(
            sensor=sensor_name,
            r_squared=round(diag.r_squared, 4),
            durbin_watson=round(diag.durbin_watson, 4),
            holdout_degradation=round(diag.holdout_degradation, 4) if diag.holdout_degradation is not None else None,
            n_rows=diag.n_rows,
            n_features=diag.n_features,
            grade=health.grade.value,
        ))

    # ── Per-effector gain analysis ──
    # Group gains by effector
    gains_by_eff: dict[str, list[GainEntry]] = {}
    all_gain_entries: list[GainEntry] = []

    for g in result.effector_sensor_gains:
        entry = GainEntry(
            effector=g.effector,
            sensor=g.sensor,
            gain=g.gain_f_per_hour,
            t_stat=g.t_statistic,
            lag_min=g.best_lag_minutes,
            negligible=g.negligible,
            classification="negligible",  # will be reclassified below
        )
        gains_by_eff.setdefault(g.effector, []).append(entry)
        all_gain_entries.append(entry)

    # Classify: direct = highest |gain| for this effector; cross = others
    effector_summaries: list[EffectorSummary] = []
    all_cross_gains: list[float] = []
    all_direct_gains: list[float] = []

    for eff_name, entries in gains_by_eff.items():
        non_neg = [e for e in entries if not e.negligible]
        if not non_neg:
            # All negligible — still record
            effector_summaries.append(EffectorSummary(
                effector=eff_name,
                direct_sensor="(none)",
                direct_gain=0.0,
                direct_t=0.0,
                n_cross=0,
                mean_cross_gain=0.0,
                max_cross_gain=0.0,
                cross_direct_ratio=0.0,
            ))
            continue

        # Primary = highest |gain|
        best = max(non_neg, key=lambda e: abs(e.gain))
        best.classification = "direct"
        all_direct_gains.append(abs(best.gain))

        cross = [e for e in non_neg if e is not best]
        for e in cross:
            e.classification = "cross"
            all_cross_gains.append(abs(e.gain))

        cross_magnitudes = [abs(e.gain) for e in cross]
        mean_cross = float(np.mean(cross_magnitudes)) if cross_magnitudes else 0.0
        max_cross = float(np.max(cross_magnitudes)) if cross_magnitudes else 0.0
        ratio = mean_cross / abs(best.gain) if abs(best.gain) > 0 else 0.0

        effector_summaries.append(EffectorSummary(
            effector=eff_name,
            direct_sensor=best.sensor,
            direct_gain=round(best.gain, 4),
            direct_t=round(best.t_stat, 2),
            n_cross=len(cross),
            mean_cross_gain=round(mean_cross, 4),
            max_cross_gain=round(max_cross, 4),
            cross_direct_ratio=round(ratio, 3),
        ))

    # ── Aggregate metrics ──
    mean_r2 = float(np.mean([s.r_squared for s in sensor_quality])) if sensor_quality else 0.0
    mean_dw = float(np.mean([s.durbin_watson for s in sensor_quality])) if sensor_quality else 0.0
    mean_cross = float(np.mean(all_cross_gains)) if all_cross_gains else 0.0
    mean_direct = float(np.mean(all_direct_gains)) if all_direct_gains else 0.0
    overall_ratio = mean_cross / mean_direct if mean_direct > 0 else 0.0

    # Specific target
    target_gain = None
    for e in all_gain_entries:
        if e.effector == "mini_split_living_room" and e.sensor == "thermostat_upstairs_temp":
            target_gain = e.gain
            break

    return ExperimentReport(
        variant=variant,
        data_start=result.data_start,
        data_end=result.data_end,
        n_snapshots=result.n_snapshots,
        sensors=sensor_quality,
        gains=all_gain_entries,
        effector_summaries=effector_summaries,
        mean_r2=round(mean_r2, 4),
        mean_dw=round(mean_dw, 4),
        mean_cross_room_gain=round(mean_cross, 4),
        mean_direct_gain=round(mean_direct, 4),
        cross_direct_ratio=round(overall_ratio, 3),
        target_ms_lr_to_tu=round(target_gain, 4) if target_gain is not None else None,
    )


# ── Display ──────────────────────────────────────────────────────────────


def _print_report(report: ExperimentReport) -> None:
    """Print a human-readable metrics report."""
    print(f"\n{'=' * 72}")
    print(f"  Sysid Experiment: {report.variant}")
    print(f"  Data: {report.data_start[:10]} to {report.data_end[:10]} ({report.n_snapshots} snapshots)")
    print(f"{'=' * 72}")

    # Summary scores
    print(f"\n── Summary Metrics ──")
    print(f"  Mean R²:              {report.mean_r2:.4f}")
    print(f"  Mean DW:              {report.mean_dw:.4f}")
    print(f"  Mean direct gain:     {report.mean_direct_gain:.4f} °F/hr")
    print(f"  Mean cross-room gain: {report.mean_cross_room_gain:.4f} °F/hr")
    print(f"  Cross/direct ratio:   {report.cross_direct_ratio:.3f}")
    if report.target_ms_lr_to_tu is not None:
        print(f"  Target (ms_lr→tu):    {report.target_ms_lr_to_tu:+.4f} °F/hr")

    # Per-sensor quality
    print(f"\n── Sensor Regression Quality ──")
    print(f"  {'Sensor':<32s}   R²      DW    Holdout  Feats  Grade")
    print(f"  {'─' * 32}  ─────  ─────  ───────  ─────  ─────")
    for s in sorted(report.sensors, key=lambda x: x.sensor):
        ho = f"{s.holdout_degradation:6.1%}" if s.holdout_degradation is not None else "   n/a"
        print(f"  {s.sensor:<32s}  {s.r_squared:.3f}  {s.durbin_watson:.3f}  {ho}  {s.n_features:5d}  {s.grade:>5s}")

    # Per-effector gain analysis
    print(f"\n── Effector Gain Analysis ──")
    print(f"  {'Effector':<30s}  {'Direct Sensor':<32s}  {'Gain':>7s}  {'t':>5s}  {'#Xr':>3s}  {'MnXr':>6s}  {'Ratio':>5s}")
    print(f"  {'─' * 30}  {'─' * 32}  {'─' * 7}  {'─' * 5}  {'─' * 3}  {'─' * 6}  {'─' * 5}")
    for es in sorted(report.effector_summaries, key=lambda x: x.effector):
        print(
            f"  {es.effector:<30s}  {es.direct_sensor:<32s}"
            f"  {es.direct_gain:+.4f}  {es.direct_t:5.1f}"
            f"  {es.n_cross:3d}  {es.mean_cross_gain:.4f}  {es.cross_direct_ratio:.3f}"
        )

    # Detailed cross-room gains (sorted by magnitude)
    cross_gains = [g for g in report.gains if g.classification == "cross"]
    if cross_gains:
        cross_gains.sort(key=lambda g: abs(g.gain), reverse=True)
        print(f"\n── Cross-Room Gains (top 20) ──")
        print(f"  {'Effector':<28s}  →  {'Sensor':<30s}  {'Gain':>7s}  {'t':>6s}")
        print(f"  {'─' * 28}     {'─' * 30}  {'─' * 7}  {'─' * 6}")
        for g in cross_gains[:20]:
            print(f"  {g.effector:<28s}  →  {g.sensor:<30s}  {g.gain:+.4f}  {g.t_stat:6.2f}")


def _pct_change(old: float, new: float) -> str:
    """Format a percentage change with direction indicator."""
    if old == 0:
        return "  n/a"
    pct = (new - old) / abs(old) * 100
    arrow = "↓" if pct < -1 else "↑" if pct > 1 else "→"
    return f"{pct:+5.1f}% {arrow}"


def _compare_reports(baseline: ExperimentReport, variant: ExperimentReport) -> None:
    """Print a side-by-side comparison of two experiment reports."""
    print(f"\n{'=' * 80}")
    print(f"  Comparison: {variant.variant} vs {baseline.variant}")
    print(f"{'=' * 80}")

    # Summary comparison
    print(f"\n── Summary ──")
    print(f"  {'Metric':<25s}  {'Baseline':>10s}  {'Variant':>10s}  {'Change':>10s}  {'OK?':>3s}")
    print(f"  {'─' * 25}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 3}")

    rows = [
        ("Mean R²", baseline.mean_r2, variant.mean_r2, "≥"),
        ("Mean DW", baseline.mean_dw, variant.mean_dw, "≥"),
        ("Mean direct gain", baseline.mean_direct_gain, variant.mean_direct_gain, "~"),
        ("Mean cross-room gain", baseline.mean_cross_room_gain, variant.mean_cross_room_gain, "↓"),
        ("Cross/direct ratio", baseline.cross_direct_ratio, variant.cross_direct_ratio, "↓"),
    ]
    for label, b, v, want in rows:
        pct = _pct_change(b, v)
        if want == "↓":
            ok = "✓" if v <= b else "✗"
        elif want == "≥":
            ok = "✓" if v >= b * 0.9 else "✗"  # allow 10% drop
        else:
            ok = "✓" if abs(v - b) / max(abs(b), 0.001) < 0.2 else "?"  # ~20% stability
        print(f"  {label:<25s}  {b:10.4f}  {v:10.4f}  {pct:>10s}  {ok:>3s}")

    # Target gain
    if baseline.target_ms_lr_to_tu is not None and variant.target_ms_lr_to_tu is not None:
        b = baseline.target_ms_lr_to_tu
        v = variant.target_ms_lr_to_tu
        pct = _pct_change(b, v)
        ok = "✓" if abs(v) <= abs(b) else "✗"
        print(f"  {'Target (ms_lr→tu)':<25s}  {b:+10.4f}  {v:+10.4f}  {pct:>10s}  {ok:>3s}")

    # Per-sensor R² and DW comparison
    base_sensors = {s.sensor: s for s in baseline.sensors}
    var_sensors = {s.sensor: s for s in variant.sensors}
    common = sorted(set(base_sensors) & set(var_sensors))
    if common:
        print(f"\n── Per-Sensor Quality ──")
        print(f"  {'Sensor':<32s}  {'R²(b)':>6s}  {'R²(v)':>6s}  {'Δ':>7s}  {'DW(b)':>6s}  {'DW(v)':>6s}  {'Δ':>7s}  {'Grade':>5s}")
        print(f"  {'─' * 32}  {'─' * 6}  {'─' * 6}  {'─' * 7}  {'─' * 6}  {'─' * 6}  {'─' * 7}  {'─' * 5}")
        for name in common:
            bs, vs = base_sensors[name], var_sensors[name]
            r2_chg = _pct_change(bs.r_squared, vs.r_squared)
            dw_chg = _pct_change(bs.durbin_watson, vs.durbin_watson)
            print(
                f"  {name:<32s}  {bs.r_squared:.4f}  {vs.r_squared:.4f}  {r2_chg:>7s}"
                f"  {bs.durbin_watson:.4f}  {vs.durbin_watson:.4f}  {dw_chg:>7s}"
                f"  {bs.grade:>2s}→{vs.grade:<2s}"
            )

    # Per-effector direct gain stability
    base_effs = {e.effector: e for e in baseline.effector_summaries}
    var_effs = {e.effector: e for e in variant.effector_summaries}
    common_effs = sorted(set(base_effs) & set(var_effs))
    if common_effs:
        print(f"\n── Direct Gain Stability ──")
        print(f"  {'Effector':<28s}  {'Gain(b)':>8s}  {'Gain(v)':>8s}  {'Δ':>7s}  {'t(b)':>6s}  {'t(v)':>6s}  {'OK?':>3s}")
        print(f"  {'─' * 28}  {'─' * 8}  {'─' * 8}  {'─' * 7}  {'─' * 6}  {'─' * 6}  {'─' * 3}")
        for name in common_effs:
            be, ve = base_effs[name], var_effs[name]
            chg = _pct_change(be.direct_gain, ve.direct_gain)
            ok = "✓" if abs(ve.direct_gain - be.direct_gain) / max(abs(be.direct_gain), 0.001) < 0.2 else "?"
            print(
                f"  {name:<28s}  {be.direct_gain:+.4f}  {ve.direct_gain:+.4f}  {chg:>7s}"
                f"  {be.direct_t:6.2f}  {ve.direct_t:6.2f}  {ok:>3s}"
            )

    # Cross-room gain changes (top movers)
    base_gains = {(g.effector, g.sensor): g for g in baseline.gains if g.classification == "cross"}
    var_gains = {(g.effector, g.sensor): g for g in variant.gains if g.classification == "cross"}
    common_gains = sorted(set(base_gains) & set(var_gains))
    if common_gains:
        # Sort by absolute change in gain magnitude
        changes = []
        for key in common_gains:
            bg, vg = base_gains[key], var_gains[key]
            delta = abs(vg.gain) - abs(bg.gain)
            changes.append((key, bg, vg, delta))
        changes.sort(key=lambda x: x[3])  # biggest decreases first

        print(f"\n── Cross-Room Gain Changes (top 15) ──")
        print(f"  {'Effector → Sensor':<60s}  {'Base':>7s}  {'Var':>7s}  {'Δ':>7s}")
        print(f"  {'─' * 60}  {'─' * 7}  {'─' * 7}  {'─' * 7}")
        for (eff, sens), bg, vg, delta in changes[:15]:
            label = f"{eff} → {sens}"
            print(f"  {label:<60s}  {bg.gain:+.4f}  {vg.gain:+.4f}  {delta:+.4f}")


# ── CLI ──────────────────────────────────────────────────────────────────


def _to_dict(report: ExperimentReport) -> dict:
    """Convert report to a JSON-serializable dict."""
    return {
        "variant": report.variant,
        "data_start": report.data_start,
        "data_end": report.data_end,
        "n_snapshots": report.n_snapshots,
        "sensors": [asdict(s) for s in report.sensors],
        "gains": [asdict(g) for g in report.gains],
        "effector_summaries": [asdict(e) for e in report.effector_summaries],
        "mean_r2": report.mean_r2,
        "mean_dw": report.mean_dw,
        "mean_cross_room_gain": report.mean_cross_room_gain,
        "mean_direct_gain": report.mean_direct_gain,
        "cross_direct_ratio": report.cross_direct_ratio,
        "target_ms_lr_to_tu": report.target_ms_lr_to_tu,
    }


def _from_dict(d: dict) -> ExperimentReport:
    """Reconstruct report from JSON dict."""
    return ExperimentReport(
        variant=d["variant"],
        data_start=d["data_start"],
        data_end=d["data_end"],
        n_snapshots=d["n_snapshots"],
        sensors=[SensorQuality(**s) for s in d["sensors"]],
        gains=[GainEntry(**g) for g in d["gains"]],
        effector_summaries=[EffectorSummary(**e) for e in d["effector_summaries"]],
        mean_r2=d["mean_r2"],
        mean_dw=d["mean_dw"],
        mean_cross_room_gain=d["mean_cross_room_gain"],
        mean_direct_gain=d["mean_direct_gain"],
        cross_direct_ratio=d["cross_direct_ratio"],
        target_ms_lr_to_tu=d.get("target_ms_lr_to_tu"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sysid experiment harness")
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save metrics as baseline to experiments/baseline.json",
    )
    parser.add_argument(
        "--compare",
        type=str,
        metavar="BASELINE_JSON",
        help="Compare current results against a saved baseline",
    )
    parser.add_argument(
        "--output",
        type=str,
        metavar="PATH",
        help="Save metrics JSON to this path",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="baseline",
        help="Label for this experiment variant (default: baseline)",
    )
    args = parser.parse_args()

    # Run sysid and extract metrics
    variant_label = args.variant
    if args.save_baseline:
        variant_label = "baseline"

    print(f"\nRunning sysid ({variant_label})...")
    report = _extract_report(variant=variant_label)

    # Print standalone report
    _print_report(report)

    # Save if requested
    output_path = None
    if args.save_baseline:
        output_dir = Path("experiments")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "baseline.json"
    elif args.output:
        output_path = Path(args.output)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(_to_dict(report), f, indent=2)
        print(f"\nMetrics saved to {output_path}")

    # Compare if requested
    if args.compare:
        baseline_path = Path(args.compare)
        if not baseline_path.exists():
            print(f"\nError: baseline file not found: {baseline_path}", file=sys.stderr)
            sys.exit(1)
        with open(baseline_path) as f:
            baseline = _from_dict(json.load(f))
        _compare_reports(baseline, report)


if __name__ == "__main__":
    main()
