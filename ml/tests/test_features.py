"""Tests for feature engineering pipeline."""

import numpy as np
import pandas as pd

from weatherstat.features import (
    add_delta_features,
    add_future_targets,
    add_time_features,
    build_features,
    encode_hvac_features,
)


def test_add_time_features_extracts_components() -> None:
    """Test that time features are correctly extracted from timestamps."""
    df = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-15T14:30:00",  # Monday, January, 2:30 PM
                "2024-07-04T06:00:00",  # Thursday, July, 6:00 AM
                "2024-12-25T23:59:00",  # Wednesday, December, 11:59 PM
            ]
        }
    )

    result = add_time_features(df)

    # Check basic time components
    assert list(result["hour"]) == [14, 6, 23]
    assert list(result["day_of_week"]) == [0, 3, 2]  # Mon=0, Thu=3, Wed=2
    assert list(result["month"]) == [1, 7, 12]

    # Check cyclical encoding is in [-1, 1]
    for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
        assert result[col].between(-1, 1).all(), f"{col} out of range"

    # Check specific cyclical values
    # Hour 6 -> sin(2pi * 6/24) = sin(pi/2) = 1.0
    assert np.isclose(result["hour_sin"].iloc[1], 1.0, atol=1e-10)

    # Hour 6 -> cos(2pi * 6/24) = cos(pi/2) ~ 0.0
    assert np.isclose(result["hour_cos"].iloc[1], 0.0, atol=1e-10)


def test_encode_hvac_features() -> None:
    """Test HVAC categorical encoding."""
    df = pd.DataFrame({
        "thermostat_upstairs_action": ["heating", "idle", "heating"],
        "thermostat_downstairs_action": ["idle", "idle", "heating"],
        "mini_split_bedroom_mode": ["heat", "off", "cool"],
        "mini_split_living_room_mode": ["off", "heat", "auto"],
        "blower_family_room_mode": ["off", "low", "high"],
        "blower_office_mode": ["high", "off", "low"],
        "navien_heating_mode": ["Space Heating", "Idle", "Space Heating"],
    })

    result = encode_hvac_features(df)

    assert list(result["thermostat_upstairs_action_enc"]) == [1, 0, 1]
    assert list(result["thermostat_downstairs_action_enc"]) == [0, 0, 1]
    assert list(result["mini_split_bedroom_mode_enc"]) == [1, 0, -1]
    assert list(result["blower_family_room_mode_enc"]) == [0, 1, 2]
    assert list(result["navien_heating_mode_enc"]) == [1, 0, 1]


def test_add_delta_features() -> None:
    """Test indoor-outdoor and target-current delta features."""
    df = pd.DataFrame({
        "thermostat_upstairs_temp": [72.0, 73.0, 71.0],
        "thermostat_downstairs_temp": [70.0, 71.0, 69.0],
        "outdoor_temp": [40.0, 42.0, 38.0],
        "thermostat_upstairs_target": [73.0, 73.0, 73.0],
        "thermostat_downstairs_target": [71.0, 71.0, 71.0],
    })

    result = add_delta_features(df)

    # Indoor-outdoor delta
    assert list(result["upstairs_outdoor_delta"]) == [32.0, 31.0, 33.0]
    assert list(result["downstairs_outdoor_delta"]) == [30.0, 29.0, 31.0]

    # Target-current delta
    assert list(result["upstairs_target_delta"]) == [1.0, 0.0, 2.0]
    assert list(result["downstairs_target_delta"]) == [1.0, 0.0, 2.0]


def test_add_future_targets() -> None:
    """Test future temperature target creation."""
    df = pd.DataFrame({
        "thermostat_upstairs_temp": [70.0, 71.0, 72.0, 73.0, 74.0],
        "thermostat_downstairs_temp": [68.0, 69.0, 70.0, 71.0, 72.0],
    })

    zones = {
        "upstairs": "thermostat_upstairs_temp",
        "downstairs": "thermostat_downstairs_temp",
    }
    result = add_future_targets(df, zones, horizons=[1, 2])

    # Upstairs T+1: shifted back by 1
    assert result["upstairs_temp_t+1"].iloc[0] == 71.0
    assert result["upstairs_temp_t+1"].iloc[3] == 74.0
    assert pd.isna(result["upstairs_temp_t+1"].iloc[4])  # last row has no future

    # Downstairs T+2: shifted back by 2
    assert result["downstairs_temp_t+2"].iloc[0] == 70.0
    assert pd.isna(result["downstairs_temp_t+2"].iloc[3])
    assert pd.isna(result["downstairs_temp_t+2"].iloc[4])


def test_build_features_baseline_mode() -> None:
    """Test that baseline mode produces expected feature columns."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-15 00:00", periods=10, freq="h").strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
        "thermostat_upstairs_temp": np.linspace(70, 74, 10),
        "thermostat_downstairs_temp": np.linspace(68, 72, 10),
        "outdoor_temp": np.linspace(35, 40, 10),
    })

    result = build_features(df, mode="baseline")

    # Should have time features
    assert "hour_sin" in result.columns
    assert "solar_elevation" in result.columns

    # Should have lag features at hourly intervals
    assert "thermostat_upstairs_temp_lag_1" in result.columns
    assert "thermostat_upstairs_temp_lag_6" in result.columns

    # Should have rolling features
    assert "outdoor_temp_rolling_3" in result.columns

    # Should have outdoor delta
    assert "upstairs_outdoor_delta" in result.columns

    # Should NOT have HVAC encoding (baseline mode)
    assert "thermostat_upstairs_action_enc" not in result.columns


def test_build_features_full_mode() -> None:
    """Test that full mode produces HVAC encoded features."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-15 00:00", periods=20, freq="5min").strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
        "thermostat_upstairs_temp": np.linspace(70, 74, 20),
        "thermostat_downstairs_temp": np.linspace(68, 72, 20),
        "outdoor_temp": np.linspace(35, 40, 20),
        "thermostat_upstairs_action": ["heating"] * 10 + ["idle"] * 10,
        "thermostat_downstairs_action": ["idle"] * 20,
        "thermostat_upstairs_target": [73.0] * 20,
        "thermostat_downstairs_target": [71.0] * 20,
        "navien_heating_mode": ["Space Heating"] * 10 + ["Idle"] * 10,
        "weather_condition": ["cloudy"] * 20,
        "outdoor_humidity": [80.0] * 20,
        "wind_speed": [5.0] * 20,
    })

    result = build_features(df, mode="full")

    # Should have HVAC encoding
    assert "thermostat_upstairs_action_enc" in result.columns
    assert "navien_heating_mode_enc" in result.columns

    # Should have delta features
    assert "upstairs_outdoor_delta" in result.columns
    assert "upstairs_target_delta" in result.columns

    # Should have weather features
    assert "weather_condition_code" in result.columns
    assert "wind_chill_approx" in result.columns

    # Should have 5-min lag features
    assert "thermostat_upstairs_temp_lag_1" in result.columns
    assert "thermostat_upstairs_temp_lag_12" in result.columns
