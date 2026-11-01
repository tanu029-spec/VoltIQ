"""
Unit and integration tests for VoltIQ BatteryPreprocessor module.
Tests data validation, non-destructive cleaning, anomaly flagging,
schema consistency, and deterministic reproducibility.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import BatteryPreprocessor


@pytest.fixture
def preprocessor():
    return BatteryPreprocessor(
        processed_data_dir="data/processed",
        results_dir="results",
    )


@pytest.fixture
def sample_data(preprocessor):
    cycles_df, meas_df = preprocessor.load_extracted_data()
    return cycles_df, meas_df


def test_load_extracted_data(preprocessor, sample_data):
    """Test loading original extracted CSV datasets."""
    cycles_df, meas_df = sample_data
    assert len(cycles_df) == 636
    assert len(meas_df) == 185721
    assert set(cycles_df["battery_id"].unique()) == {"B0005", "B0006", "B0007", "B0018"}


def test_validation_clean_data(preprocessor, sample_data):
    """Test validation reports zero nulls, infs, and duplicates on authentic data."""
    cycles_df, meas_df = sample_data
    val = preprocessor.validate_data(cycles_df, meas_df)

    assert sum(val["null_counts"]["cycles"].values()) == 0
    assert sum(val["null_counts"]["measurements"].values()) == 0
    assert sum(val["infinite_counts"]["cycles"].values()) == 0
    assert sum(val["infinite_counts"]["measurements"].values()) == 0
    assert val["duplicates"]["exact_cycles"] == 0
    assert val["duplicates"]["cycle_keys"] == 0
    assert val["duplicates"]["timestamp_measurements"] == 0
    assert val["temporal_integrity"]["non_monotonic_cycles_count"] == 0
    assert val["cross_table_alignment"]["measurement_count_mismatches"] == 0
    assert val["capacity_validity"]["non_positive_count"] == 0


def test_validation_detects_injected_issues(preprocessor, sample_data):
    """Test validation correctly flags injected nulls, duplicates, and non-monotonic timestamps."""
    cycles_df, meas_df = sample_data

    # Inject a null, duplicate, and non-monotonic time
    corrupt_cycles = cycles_df.copy()
    corrupt_meas = meas_df.copy()

    # Inject null capacity at row 0
    corrupt_cycles.loc[0, "discharge_capacity_ah"] = np.nan
    # Inject duplicate cycle key using row 1 (which has valid capacity)
    duplicate_row = corrupt_cycles.iloc[[1]].copy()
    corrupt_cycles = pd.concat([corrupt_cycles, duplicate_row], ignore_index=True)
    # Inject non-monotonic timestamp
    corrupt_meas.loc[1, "time_seconds"] = corrupt_meas.loc[0, "time_seconds"] - 10.0

    val = preprocessor.validate_data(corrupt_cycles, corrupt_meas)
    assert val["null_counts"]["cycles"]["discharge_capacity_ah"] == 1
    assert val["duplicates"]["cycle_keys"] >= 1
    assert val["temporal_integrity"]["non_monotonic_cycles_count"] >= 1


def test_clean_data_preserves_records(preprocessor, sample_data):
    """Test that clean_data preserves every single row and enforces correct dtypes."""
    cycles_df, meas_df = sample_data
    clean_c, clean_m, audit = preprocessor.clean_data(cycles_df, meas_df)

    assert len(clean_c) == len(cycles_df)
    assert len(clean_m) == len(meas_df)
    assert clean_c["battery_id"].dtype == "string"
    assert clean_m["battery_id"].dtype == "string"
    assert clean_c["cycle_number"].dtype == "int64"
    assert clean_m["cycle_number"].dtype == "int64"
    assert clean_c["discharge_capacity_ah"].dtype == "float64"
    assert len(audit) >= 3


def test_anomaly_flagging(preprocessor, sample_data):
    """Test that anomaly detection attaches boolean flags and compiles summary correctly."""
    cycles_df, meas_df = sample_data
    clean_c, clean_m, _ = preprocessor.clean_data(cycles_df, meas_df)
    flagged_c, flagged_m, summary_df = preprocessor.flag_anomalies(clean_c, clean_m)

    # Check that original records are preserved
    assert len(flagged_c) == len(cycles_df)
    assert len(flagged_m) == len(meas_df)

    # Check new cycle flag columns
    expected_c_flags = ["is_capacity_iqr_outlier", "is_capacity_recovery", "is_duration_iqr_outlier"]
    for col in expected_c_flags:
        assert col in flagged_c.columns
        assert flagged_c[col].dtype == "bool"

    # Check new measurement flag columns
    expected_m_flags = [
        "is_voltage_physical_anomaly",
        "is_below_nominal_cutoff",
        "is_current_spike",
        "is_temperature_physical_anomaly",
        "is_load_sign_inverted",
    ]
    for col in expected_m_flags:
        assert col in flagged_m.columns
        assert flagged_m[col].dtype == "bool"

    # Verify electrochemical recovery jumps exist (> 0)
    assert flagged_c["is_capacity_recovery"].sum() > 0

    # Verify summary table contains all batteries and total row
    assert set(summary_df["battery_id"].tolist()) == {"B0005", "B0006", "B0007", "B0018", "ALL_BATTERIES"}


def test_export_and_reload(tmp_path, preprocessor, sample_data):
    """Test exporting clean datasets and reloading them cleanly."""
    cycles_df, meas_df = sample_data
    clean_c, clean_m, _ = preprocessor.clean_data(cycles_df, meas_df)
    flagged_c, flagged_m, _ = preprocessor.flag_anomalies(clean_c, clean_m)

    c_file, m_file = preprocessor.export_cleaned_data(flagged_c, flagged_m, output_dir=tmp_path)
    assert c_file.is_file()
    assert m_file.is_file()

    reloaded_c = pd.read_csv(c_file)
    reloaded_m = pd.read_csv(m_file)
    assert len(reloaded_c) == 636
    assert len(reloaded_m) == 185721
    assert "is_capacity_recovery" in reloaded_c.columns
    assert "is_below_nominal_cutoff" in reloaded_m.columns


def test_reproducibility(preprocessor, sample_data):
    """Test that running the cleaning and flagging pipeline twice yields identical results."""
    cycles_df, meas_df = sample_data
    c1, m1, _ = preprocessor.clean_data(cycles_df, meas_df)
    fc1, fm1, s1 = preprocessor.flag_anomalies(c1, m1)

    c2, m2, _ = preprocessor.clean_data(cycles_df, meas_df)
    fc2, fm2, s2 = preprocessor.flag_anomalies(c2, m2)

    pd.testing.assert_frame_equal(fc1, fc2)
    pd.testing.assert_frame_equal(fm1, fm2)
    pd.testing.assert_frame_equal(s1, s2)
