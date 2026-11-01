"""
Unit tests for VoltIQ feature engineering and health indicators.
"""

import pytest
from src.feature_engineering import FeatureEngineer


def test_calculate_soh():
    # 2.0 Ah rated capacity, 1.6 Ah current -> 80% SOH (End of Life threshold)
    soh = FeatureEngineer.calculate_soh(current_capacity=1.6, rated_capacity=2.0)
    assert soh == pytest.approx(80.0, rel=1e-3)


def test_calculate_soh_invalid_rated_capacity():
    with pytest.raises(ValueError):
        FeatureEngineer.calculate_soh(current_capacity=1.5, rated_capacity=0.0)
