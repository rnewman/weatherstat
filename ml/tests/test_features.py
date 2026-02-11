"""Tests for feature engineering pipeline."""

import numpy as np
import pandas as pd

from weatherstat.features import add_time_features


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
    # Hour 6 → sin(2π * 6/24) = sin(π/2) = 1.0
    assert np.isclose(result["hour_sin"].iloc[1], 1.0, atol=1e-10)

    # Hour 6 → cos(2π * 6/24) = cos(π/2) ≈ 0.0
    assert np.isclose(result["hour_cos"].iloc[1], 0.0, atol=1e-10)
