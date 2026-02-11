"""Visualization utility for weatherstat data.

Renders time-series plots of temperatures, weather, solar, and HVAC state
from the extracted historical data.

Run: uv run python -m weatherstat.visualize
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astral import LocationInfo
from astral.sun import elevation as sun_elevation

from weatherstat.config import DATA_DIR, LATITUDE, LONGITUDE, SNAPSHOTS_DIR

_LOCATION = LocationInfo(
    name="Home", region="US", timezone="America/Los_Angeles",
    latitude=LATITUDE, longitude=LONGITUDE,
)


def _compute_solar_elevation(timestamps: pd.DatetimeIndex) -> np.ndarray:
    elevations = []
    for ts in timestamps:
        dt = ts.to_pydatetime()
        try:
            elev = sun_elevation(_LOCATION.observer, dt)
        except ValueError:
            elev = 0.0
        elevations.append(max(elev, 0.0))  # clip below-horizon to 0
    return np.array(elevations)


def _heating_spans(ts: pd.DatetimeIndex, action: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Find contiguous spans where action == 'heating'."""
    is_heating = (action == "heating").values
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = None
    for i, h in enumerate(is_heating):
        if h and start is None:
            start = ts[i]
        elif not h and start is not None:
            spans.append((start, ts[i]))
            start = None
    if start is not None:
        spans.append((start, ts[-1]))
    return spans


def plot_floor(
    df: pd.DataFrame,
    ts: pd.DatetimeIndex,
    solar: np.ndarray,
    floor: str,
    ax_temp: plt.Axes,
    ax_env: plt.Axes,
    ax_solar: plt.Axes,
) -> None:
    """Plot a single floor's data across three axes."""
    if floor == "upstairs":
        temp_col = "thermostat_upstairs_temp"
        action_col = "thermostat_upstairs_action"
        agg_col = "upstairs_aggregate_temp"
        label = "Upstairs"
        color = "#d62728"
    else:
        temp_col = "thermostat_downstairs_temp"
        action_col = "thermostat_downstairs_action"
        agg_col = "downstairs_aggregate_temp"
        label = "Downstairs"
        color = "#1f77b4"

    # --- Temperature panel ---
    ax_temp.plot(ts, df[temp_col], color=color, linewidth=1, label=f"{label} thermostat", alpha=0.9)
    if agg_col in df.columns:
        ax_temp.plot(ts, df[agg_col], color=color, linewidth=0.8,
                     linestyle="--", label=f"{label} aggregate", alpha=0.6)
    ax_temp.plot(ts, df["outdoor_temp"], color="#2ca02c", linewidth=1, label="Outdoor", alpha=0.7)

    # Shade heating spans
    if action_col in df.columns:
        for start, end in _heating_spans(ts, df[action_col]):
            ax_temp.axvspan(start, end, alpha=0.10, color=color)

    ax_temp.set_ylabel("Temperature (°F)")
    ax_temp.legend(loc="upper left", fontsize=7, ncol=3)
    ax_temp.set_title(f"{label} — Temperature & Heating", fontsize=10, fontweight="bold")
    ax_temp.grid(True, alpha=0.3)

    # --- Environment panel (wind + humidity) ---
    if "wind_speed" in df.columns:
        ax_env.plot(ts, df["wind_speed"], color="#9467bd", linewidth=0.8, label="Wind speed (km/h)")
    ax_env_r = ax_env.twinx()
    if "outdoor_humidity" in df.columns:
        ax_env_r.plot(ts, df["outdoor_humidity"], color="#8c564b",
                      linewidth=0.8, label="Humidity (%)", alpha=0.7)
        ax_env_r.set_ylabel("Humidity (%)", fontsize=8)
        ax_env_r.set_ylim(0, 100)
    ax_env.set_ylabel("Wind (km/h)", fontsize=8)
    ax_env.legend(loc="upper left", fontsize=7)
    ax_env_r.legend(loc="upper right", fontsize=7)
    ax_env.set_title(f"{label} — Wind & Humidity", fontsize=10)
    ax_env.grid(True, alpha=0.3)

    # --- Solar panel ---
    ax_solar.fill_between(ts, 0, solar, color="#ff7f0e", alpha=0.4, label="Solar elevation")
    ax_solar.set_ylabel("Solar elevation (°)", fontsize=8)
    ax_solar.set_ylim(0, max(solar.max() * 1.1, 1))

    # Overlay weather condition as background colors
    if "weather_condition" in df.columns:
        condition_colors = {
            "sunny": "#fff9c4", "clear-night": "#e8eaf6",
            "partlycloudy": "#e0e0e0", "cloudy": "#bdbdbd",
            "rainy": "#bbdefb", "pouring": "#90caf9",
            "fog": "#f5f5f5", "snowy": "#e3f2fd",
        }
        prev_cond = None
        span_start = ts[0]
        for i in range(len(ts)):
            cond = df["weather_condition"].iloc[i]
            if cond != prev_cond and prev_cond is not None:
                c = condition_colors.get(str(prev_cond), "#ffffff")
                ax_solar.axvspan(span_start, ts[i], alpha=0.3, color=c)
                span_start = ts[i]
            prev_cond = cond
        if prev_cond is not None:
            c = condition_colors.get(str(prev_cond), "#ffffff")
            ax_solar.axvspan(span_start, ts[-1], alpha=0.3, color=c)

    ax_solar.legend(loc="upper left", fontsize=7)
    ax_solar.set_title(f"{label} — Solar Elevation & Weather", fontsize=10)
    ax_solar.grid(True, alpha=0.3)


def render(df: pd.DataFrame, output_path: Path, title: str = "Weatherstat Data") -> None:
    """Render the full visualization to a PNG file."""
    ts = pd.DatetimeIndex(pd.to_datetime(df["timestamp"]))
    solar = _compute_solar_elevation(ts)

    fig, axes = plt.subplots(6, 1, figsize=(18, 20), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    plot_floor(df, ts, solar, "upstairs", axes[0], axes[1], axes[2])
    plot_floor(df, ts, solar, "downstairs", axes[3], axes[4], axes[5])

    # Format x-axis
    axes[-1].set_xlabel("Time")
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize weatherstat data")
    parser.add_argument(
        "--source",
        choices=["full", "hourly"],
        default="full",
        help="Data source: full (5-min history) or hourly (statistics)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    parser.add_argument("--last-days", type=int, default=None, help="Only show the last N days")
    args = parser.parse_args()

    if args.source == "full":
        path = SNAPSHOTS_DIR / "historical_full.parquet"
        default_out = "weatherstat_full.png"
        title = "Weatherstat — 10-Day Full History (5-min intervals)"
    else:
        path = SNAPSHOTS_DIR / "historical_hourly.parquet"
        default_out = "weatherstat_hourly.png"
        title = "Weatherstat — Hourly Statistics"

    if not path.exists():
        print(f"Error: {path} not found. Run `just extract` first.")
        return

    df = pd.read_parquet(path)

    if args.last_days:
        ts = pd.to_datetime(df["timestamp"])
        cutoff = ts.max() - pd.Timedelta(days=args.last_days)
        df = df[ts >= cutoff].reset_index(drop=True)
        title += f" (last {args.last_days} days)"

    output_path = Path(args.output) if args.output else DATA_DIR / default_out
    render(df, output_path, title=title)


if __name__ == "__main__":
    main()
