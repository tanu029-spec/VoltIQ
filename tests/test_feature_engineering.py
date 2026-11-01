"""
Unit and integration tests for VoltIQ Feature Engineering module.
Tests health indicators, sensor aggregation, chronological degradation deltas,
ICA feature stability, leakage prevention rules, and reproducibility.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import FeatureEngineer
from src.feature_config import (
    ML_FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    TARGET_COLUMNS,
    validate_ml_feature_set,
)


@pytest.fixture
def feature_engineer():
    return FeatureEngineer(
        clean_cycles_path="data/processed/clean_battery_cycles.csv",
        clean_measurements_path="data/processed/clean_battery_measurements.csv",
    )


@pytest.fixture
def sample_clean_data(feature_engineer):
    clean_c = pd.read_csv(feature_engineer.clean_cycles_path)
    clean_m = pd.read_csv(feature_engineer.clean_measurements_path)
    return clean_c, clean_m


def test_soh_and_fade_calculation():
    """Test SOH and capacity fade formulas."""
    # Exact 80% SOH (EOL boundary)
    soh = FeatureEngineer.calculate_soh(current_capacity=1.60, initial_capacity=2.00)
    assert soh == pytest.approx(80.0, rel=1e-4)

    fade = FeatureEngineer.calculate_capacity_fade(soh)
    assert fade == pytest.approx(20.0, rel=1e-4)

    # Invalid non-positive initial capacity
    with pytest.raises(ValueError):
        FeatureEngineer.calculate_soh(current_capacity=1.5, initial_capacity=0.0)
    with pytest.raises(ValueError):
        FeatureEngineer.calculate_soh(current_capacity=1.5, initial_capacity=-1.0)


def test_independent_battery_baselines(feature_engineer, sample_clean_data):
    """Test that each battery cell uses its own first discharge capacity as baseline."""
    clean_c, _ = sample_clean_data
    hi_df = feature_engineer.compute_health_indicators(clean_c)

    # Check each battery's first cycle SOH is exactly 100.0%
    for b_id, grp in hi_df.groupby("battery_id"):
        first_row = grp.iloc[0]
        assert first_row["soh_percentage"] == pytest.approx(100.0, rel=1e-4)
        assert first_row["capacity_fade_percentage"] == pytest.approx(0.0, abs=1e-4)
        assert first_row["initial_capacity_ah"] == pytest.approx(first_row["discharge_capacity_ah"], rel=1e-4)
        assert first_row["cycle_index"] == 1


def test_sensor_aggregation(feature_engineer, sample_clean_data):
    """Test descriptive statistics aggregation from high-frequency measurements."""
    _, clean_m = sample_clean_data
    # Test on single cycle
    sample_meas = clean_m[(clean_m["battery_id"] == "B0005") & (clean_m["cycle_number"] == 2)]
    sensor_df = feature_engineer.aggregate_sensor_features(sample_meas)

    assert len(sensor_df) == 1
    row = sensor_df.iloc[0]
    assert row["battery_id"] == "B0005"
    assert row["cycle_number"] == 2
    assert row["measurement_count"] == 197
    assert row["voltage_min"] <= row["voltage_mean"] <= row["voltage_max"]
    assert row["voltage_range"] == pytest.approx(row["voltage_max"] - row["voltage_min"], rel=1e-4)
    assert row["temperature_min"] <= row["temperature_mean"] <= row["temperature_max"]
    assert row["current_min"] <= row["current_mean"] <= row["current_max"]
    assert row["discharge_duration_seconds"] > 3000.0


def test_chronological_degradation_features(feature_engineer, sample_clean_data):
    """Test that cycle-to-cycle deltas respect chronological boundaries and never cross batteries."""
    clean_c, clean_m = sample_clean_data
    hi_df = feature_engineer.compute_health_indicators(clean_c)
    sensor_df = feature_engineer.aggregate_sensor_features(clean_m)
    merged = pd.merge(hi_df, sensor_df, on=["battery_id", "cycle_number"], how="inner")
    deg_df = feature_engineer.compute_degradation_features(merged)

    # For every battery, first cycle delta must be 0.0
    for b_id, grp in deg_df.groupby("battery_id"):
        first_row = grp.iloc[0]
        assert first_row["capacity_change_ah"] == 0.0
        assert first_row["capacity_change_percentage"] == 0.0
        assert first_row["soh_change_percentage"] == 0.0
        assert not first_row["capacity_recovery_flag"]
        assert first_row["cycle_duration_change"] == 0.0
        assert first_row["temperature_change_from_previous_cycle"] == 0.0

        # Second cycle delta must match exact subtraction
        second_row = grp.iloc[1]
        expected_cap_diff = second_row["discharge_capacity_ah"] - first_row["discharge_capacity_ah"]
        assert second_row["capacity_change_ah"] == pytest.approx(expected_cap_diff, rel=1e-4)


def test_ica_derivation(feature_engineer, sample_clean_data):
    """Test numerical ICA dQ/dV derivation on B0005 cycle 2."""
    _, clean_m = sample_clean_data
    sample_m = clean_m[(clean_m["battery_id"] == "B0005") & (clean_m["cycle_number"] == 2)]
    ica_summary, ica_curves = feature_engineer.compute_ica_features(sample_m)

    assert len(ica_summary) == 1
    row = ica_summary.iloc[0]
    # Peak ICA must be a realistic positive value (~4.5 to 5.5 Ah/V)
    assert 4.0 <= row["ica_peak_value"] <= 6.0
    # Peak voltage should lie within typical graphite/cathode phase plateau (3.40V - 3.65V)
    assert 3.40 <= row["ica_peak_voltage"] <= 3.65
    assert row["ica_area"] > 0.0

    # Test full curve points
    assert len(ica_curves) == len(FeatureEngineer.ICA_VOLTAGE_GRID)
    assert set(ica_curves.columns) == {"battery_id", "cycle_number", "voltage_grid_v", "dq_dv"}


def test_ml_target_leakage_prevention():
    """Test that ML_FEATURE_COLUMNS contains zero leakage columns."""
    validation = validate_ml_feature_set(ML_FEATURE_COLUMNS)
    assert validation["status"] == "PASS"
    assert validation["is_safe_for_training"]

    # Verify every leakage column is strictly rejected
    for leak_col in LEAKAGE_COLUMNS:
        assert leak_col not in ML_FEATURE_COLUMNS
        with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
            validate_ml_feature_set(ML_FEATURE_COLUMNS + [leak_col])


def test_full_feature_dataset_integrity(feature_engineer):
    """Test building complete feature dataset across all 636 cycles."""
    features_df, ica_curves = feature_engineer.build_feature_dataset()

    assert len(features_df) == 636
    assert set(features_df["battery_id"].unique()) == {"B0005", "B0006", "B0007", "B0018"}

    # Ensure all required ML features are present
    for col in ML_FEATURE_COLUMNS:
        assert col in features_df.columns, f"Missing ML feature: {col}"

    # Ensure all target columns are present
    for col in TARGET_COLUMNS:
        assert col in features_df.columns, f"Missing target column: {col}"

    # Ensure zero nulls in ML features
    ml_nulls = features_df[ML_FEATURE_COLUMNS].isnull().sum().to_dict()
    assert sum(ml_nulls.values()) == 0, f"Nulls found in ML features: {ml_nulls}"


def test_feature_engineering_reproducibility(feature_engineer):
    """Test that repeated feature extraction yields identical numerical results."""
    clean_c = pd.read_csv(feature_engineer.clean_cycles_path).head(10)
    clean_m = pd.read_csv(feature_engineer.clean_measurements_path)
    sample_m = clean_m[clean_m["cycle_number"].isin(clean_c["cycle_number"])]

    # Compute twice
    hi1 = feature_engineer.compute_health_indicators(clean_c)
    s1 = feature_engineer.aggregate_sensor_features(sample_m)
    m1 = pd.merge(hi1, s1, on=["battery_id", "cycle_number"])
    deg1 = feature_engineer.compute_degradation_features(m1)

    hi2 = feature_engineer.compute_health_indicators(clean_c)
    s2 = feature_engineer.aggregate_sensor_features(sample_m)
    m2 = pd.merge(hi2, s2, on=["battery_id", "cycle_number"])
    deg2 = feature_engineer.compute_degradation_features(m2)

    pd.testing.assert_frame_equal(deg1, deg2)
