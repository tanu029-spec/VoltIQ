"""
Unit tests for VoltIQ BatteryDataLoader module.
Tests file discovery, struct parsing, discharge filtering, measurement integrity,
and error handling.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import scipy.io

from src.data_loader import BatteryDataLoader


@pytest.fixture
def data_loader():
    return BatteryDataLoader(raw_data_dir="data/raw")


def test_mat_file_discovery(data_loader):
    """Test discovering the expected NASA MATLAB files."""
    files = data_loader.discover_raw_files()
    assert len(files) == 4
    filenames = [f.name for f in files]
    for expected in ["B0005.mat", "B0006.mat", "B0007.mat", "B0018.mat"]:
        assert expected in filenames


def test_check_raw_files(data_loader):
    """Test check_raw_files dictionary check."""
    status = data_loader.check_raw_files()
    assert all(status.values())
    assert set(status.keys()) == {"B0005.mat", "B0006.mat", "B0007.mat", "B0018.mat"}


def test_battery_identification(data_loader):
    """Test root battery identification from loaded MATLAB dictionary."""
    battery_id, mat = data_loader.load_mat_file("B0005")
    assert battery_id == "B0005"
    assert "B0005" in mat


def test_extract_cycle_metadata(data_loader):
    """Test extraction of cycle counts and types."""
    meta = data_loader.extract_cycle_metadata("B0005")
    assert meta["battery_id"] == "B0005"
    assert meta["total_cycles"] == 616
    assert meta["cycle_types"]["charge"] == 170
    assert meta["cycle_types"]["discharge"] == 168
    assert meta["cycle_types"]["impedance"] == 278


def test_extract_battery_data_structure(data_loader):
    """Test extracting discharge data and validating DataFrame structures."""
    cycles_df, meas_df, skipped, counts = data_loader.extract_battery_data("B0005")

    # Cycle-level DataFrame checks
    assert isinstance(cycles_df, pd.DataFrame)
    assert len(cycles_df) == 168
    expected_cycle_cols = [
        "battery_id",
        "cycle_number",
        "cycle_type",
        "ambient_temperature",
        "discharge_capacity_ah",
        "measurement_count",
    ]
    assert list(cycles_df.columns) == expected_cycle_cols
    assert (cycles_df["cycle_type"] == "discharge").all()
    assert (cycles_df["ambient_temperature"] == 24.0).all()
    assert (cycles_df["discharge_capacity_ah"] > 0).all()

    # Measurement-level DataFrame checks
    assert isinstance(meas_df, pd.DataFrame)
    assert len(meas_df) == 50285
    expected_meas_cols = [
        "battery_id",
        "cycle_number",
        "time_seconds",
        "voltage_measured_v",
        "current_measured_a",
        "temperature_measured_c",
        "current_load_a",
        "voltage_load_v",
    ]
    assert list(meas_df.columns) == expected_meas_cols
    assert (meas_df["battery_id"] == "B0005").all()
    assert len(skipped) == 0


def test_measurement_array_consistency(data_loader):
    """Test that all measurement arrays within each cycle are aligned."""
    cycles_df, meas_df, skipped, _ = data_loader.extract_battery_data("B0018")
    assert len(skipped) == 0

    # Group measurements by cycle and verify count matches cycles_df
    meas_counts = meas_df.groupby("cycle_number").size().to_dict()
    for _, row in cycles_df.iterrows():
        cycle_num = row["cycle_number"]
        expected_count = row["measurement_count"]
        assert meas_counts[cycle_num] == expected_count


def test_missing_or_malformed_cycle_handling(tmp_path):
    """Test handling of malformed or missing data in cycles using mock structures."""
    # Build a mock MAT structure with 1 valid discharge, 1 charge, 1 malformed discharge
    mock_battery = {
        "BTEST": {
            "cycle": np.array([[
                # Cycle 0: charge
                {
                    "type": np.array(["charge"]),
                    "ambient_temperature": np.array([[24.0]]),
                    "time": np.array([[2008, 1, 1, 0, 0, 0]]),
                    "data": {"Time": np.array([[0.0, 1.0]])},
                },
                # Cycle 1: valid discharge
                {
                    "type": np.array(["discharge"]),
                    "ambient_temperature": np.array([[24.0]]),
                    "time": np.array([[2008, 1, 1, 1, 0, 0]]),
                    "data": {
                        "Capacity": np.array([[1.85]]),
                        "Time": np.array([[0.0, 1.0, 2.0]]),
                        "Voltage_measured": np.array([[4.2, 4.1, 4.0]]),
                        "Current_measured": np.array([[-2.0, -2.0, -2.0]]),
                        "Temperature_measured": np.array([[24.1, 24.2, 24.3]]),
                        "Current_load": np.array([[-2.0, -2.0, -2.0]]),
                        "Voltage_load": np.array([[4.2, 4.1, 4.0]]),
                    },
                },
                # Cycle 2: malformed discharge (missing Capacity)
                {
                    "type": np.array(["discharge"]),
                    "ambient_temperature": np.array([[24.0]]),
                    "time": np.array([[2008, 1, 1, 2, 0, 0]]),
                    "data": {"Time": np.array([[0.0, 1.0]])},
                },
            ]], dtype=object)
        }
    }

    mock_mat_path = tmp_path / "BTEST.mat"
    scipy.io.savemat(str(mock_mat_path), mock_battery)

    loader = BatteryDataLoader(raw_data_dir=tmp_path)
    cycles_df, meas_df, skipped, counts = loader.extract_battery_data(mock_mat_path)

    assert len(cycles_df) == 1
    assert len(meas_df) == 3
    assert len(skipped) == 1
    assert skipped[0]["cycle_number"] == 3
    assert "Capacity" in skipped[0]["reason"]
    assert counts["discharge"] == 2
    assert counts["charge"] == 1
