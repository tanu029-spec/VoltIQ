"""
Unit and integration tests for VoltIQ Exploratory Data Analysis (EDA) module.
Tests descriptive statistics, correlation matrices, collinearity detection,
battery difference metrics, feature classifications, and figure generations.
"""

from pathlib import Path
import json
import pytest
import pandas as pd
import numpy as np

from src.eda import BatteryEDA
from src.feature_config import (
    SENSOR_ONLY_FEATURES,
    CYCLE_AGE_FEATURES,
    TARGET_PROXY_FEATURES,
    FEATURE_SET_TIER_1_SENSOR_ONLY,
    FEATURE_SET_TIER_2_SENSOR_PLUS_AGE,
    FEATURE_SET_TIER_3_FULL_INFORMATIVE,
    LEAKAGE_COLUMNS,
    TARGET_COLUMNS,
)


@pytest.fixture
def eda_instance(tmp_path):
    """Fixture providing a BatteryEDA instance configured with project artifacts."""
    return BatteryEDA(
        features_path="data/processed/battery_health_features.csv",
        ica_path="data/processed/battery_ica_features.csv",
        figures_dir=tmp_path / "figures",
        results_dir=tmp_path / "results",
    )


def test_battery_eda_initialization(tmp_path):
    """Test initialization and non-existent file handling."""
    eda = BatteryEDA(
        features_path="data/processed/battery_health_features.csv",
        ica_path="data/processed/battery_ica_features.csv",
        figures_dir=tmp_path / "figs",
        results_dir=tmp_path / "res",
    )
    assert eda.features_path.exists()
    assert eda.ica_path.exists()
    assert isinstance(eda.figures_dir, Path)
    assert isinstance(eda.results_dir, Path)

    with pytest.raises(FileNotFoundError):
        invalid_eda = BatteryEDA(features_path="non_existent_features.csv")
        invalid_eda.load_data()


def test_load_data(eda_instance):
    """Test that load_data correctly loads features and ICA dataframes."""
    features_df, ica_df = eda_instance.load_data()

    assert isinstance(features_df, pd.DataFrame)
    assert isinstance(ica_df, pd.DataFrame)
    assert len(features_df) == 636
    assert len(ica_df) > 50000

    expected_cells = {"B0005", "B0006", "B0007", "B0018"}
    assert set(features_df["battery_id"].unique()) == expected_cells
    assert set(ica_df["battery_id"].unique()) == expected_cells


def test_compute_descriptive_statistics(eda_instance):
    """Test computation of global and per-battery descriptive statistics."""
    features_df, _ = eda_instance.load_data()
    stats = eda_instance.compute_descriptive_statistics(features_df)

    assert "total_records" in stats
    assert stats["total_records"] == 636
    assert "global_statistics" in stats
    assert "battery_wise_statistics" in stats

    global_stats = stats["global_statistics"]
    assert "soh_percentage" in global_stats
    assert "discharge_capacity_ah" in global_stats
    assert "voltage_mean" in global_stats
    assert "temperature_max" in global_stats

    # Check key distribution metrics
    soh_stats = global_stats["soh_percentage"]
    for k in ["mean", "std", "median", "min", "max", "q25", "q75"]:
        assert k in soh_stats

    assert soh_stats["max"] == pytest.approx(100.0, rel=1e-4)
    assert soh_stats["min"] > 50.0  # Lowest SOH observed is ~56.7%
    assert 80.0 < soh_stats["mean"] < 85.0

    # Per-battery statistics verification
    b_stats = stats["battery_wise_statistics"]
    for b_id in ["B0005", "B0006", "B0007", "B0018"]:
        assert b_id in b_stats
        assert b_stats[b_id]["soh_percentage"]["max"] == pytest.approx(100.0, rel=1e-4)


def test_compute_correlations(eda_instance):
    """Test computation of Pearson and Spearman correlations and SOH associations."""
    features_df, _ = eda_instance.load_data()
    corr_results = eda_instance.compute_correlations(features_df)

    assert "soh_pearson" in corr_results
    assert "soh_spearman" in corr_results
    assert "highly_collinear_pairs" in corr_results
    assert "highly_collinear_pairs_count" in corr_results

    soh_corr = corr_results["soh_pearson"]
    # Check physical correlation directions
    assert soh_corr["voltage_mean"] > 0.90  # Strong positive correlation with SOH
    assert soh_corr["temperature_std"] < -0.70  # Higher thermal dispersion as battery degrades
    assert soh_corr["cycle_index"] < -0.80  # Chronological negative correlation


def test_detect_high_collinearity(eda_instance):
    """Test identification of highly collinear feature pairs (|r| >= 0.90)."""
    features_df, _ = eda_instance.load_data()
    corr_results = eda_instance.compute_correlations(features_df)
    collinear_pairs = corr_results["highly_collinear_pairs"]

    assert isinstance(collinear_pairs, list)
    assert len(collinear_pairs) > 0
    assert corr_results["highly_collinear_pairs_count"] == len(collinear_pairs)

    pair_names = [(p["feature_1"], p["feature_2"]) for p in collinear_pairs]
    # Check that voltage_mean & voltage_median or duration & measurement_count are flagged
    has_voltage_collinearity = any(
        ("voltage_mean" in p and "voltage_median" in p) for p in pair_names
    )
    has_duration_collinearity = any(
        ("discharge_duration_seconds" in p and "cycle_duration_s" in p) or
        ("discharge_capacity_ah" in p and "cycle_duration_s" in p)
        for p in pair_names
    )
    assert has_voltage_collinearity
    assert has_duration_collinearity


def test_analyze_battery_differences(eda_instance):
    """Test comparative battery degradation and cycling kinetics analysis."""
    features_df, _ = eda_instance.load_data()
    diffs = eda_instance.analyze_battery_differences(features_df)

    assert set(diffs.keys()) == {"B0005", "B0006", "B0007", "B0018"}

    # Cycle counts
    assert diffs["B0005"]["discharge_cycles_count"] == 168
    assert diffs["B0006"]["discharge_cycles_count"] == 168
    assert diffs["B0007"]["discharge_cycles_count"] == 168
    assert diffs["B0018"]["discharge_cycles_count"] == 132

    # Cutoff voltages
    assert diffs["B0005"]["cutoff_voltage_v"] == 2.7
    assert diffs["B0006"]["cutoff_voltage_v"] == 2.5
    assert diffs["B0007"]["cutoff_voltage_v"] == 2.2
    assert diffs["B0018"]["cutoff_voltage_v"] == 2.5

    # Check capacity recoveries identified
    for b_id in diffs:
        assert diffs[b_id]["recovery_cycles_count"] > 0
        assert diffs[b_id]["initial_capacity_ah"] > diffs[b_id]["final_capacity_ah"]
        assert diffs[b_id]["absolute_capacity_loss_ah"] > 0


def test_feature_tier_configurations():
    """Test feature grouping classifications and audit rules."""
    # Ensure sensor-only, age, and proxies are mutually exclusive
    sensor_set = set(SENSOR_ONLY_FEATURES)
    age_set = set(CYCLE_AGE_FEATURES)
    proxy_set = set(TARGET_PROXY_FEATURES)

    assert sensor_set.isdisjoint(age_set)
    assert sensor_set.isdisjoint(proxy_set)
    assert age_set.isdisjoint(proxy_set)

    # Ensure Tier 1 contains no leakage columns and no target columns
    assert set(FEATURE_SET_TIER_1_SENSOR_ONLY).isdisjoint(set(LEAKAGE_COLUMNS))
    assert set(FEATURE_SET_TIER_1_SENSOR_ONLY).isdisjoint(set(TARGET_COLUMNS))

    # Ensure Tier 1 does not contain discharge_duration_seconds or cycle_index
    assert "discharge_duration_seconds" not in FEATURE_SET_TIER_1_SENSOR_ONLY
    assert "cycle_index" not in FEATURE_SET_TIER_1_SENSOR_ONLY

    # Ensure Tier 2 extends Tier 1 with cycle age features
    assert set(FEATURE_SET_TIER_2_SENSOR_PLUS_AGE) == sensor_set.union(age_set)

    # Ensure Tier 3 includes proxy features
    assert "discharge_duration_seconds" in FEATURE_SET_TIER_3_FULL_INFORMATIVE
    assert "ica_peak_value" in FEATURE_SET_TIER_3_FULL_INFORMATIVE


def test_export_eda_summary_json(eda_instance, tmp_path):
    """Test structured JSON summary export."""
    features_df, _ = eda_instance.load_data()
    stats = eda_instance.compute_descriptive_statistics(features_df)
    corrs = eda_instance.compute_correlations(features_df)
    diffs = eda_instance.analyze_battery_differences(features_df)

    out_file = tmp_path / "eda_test_summary.json"
    eda_instance.export_summary_json(stats, corrs, diffs, out_file)

    assert out_file.exists()
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "descriptive_statistics" in data
    assert "correlations" in data
    assert "battery_differences" in data
    assert "sample_counts" in data
    assert data["sample_counts"]["total_cycles"] == 636


def test_eda_figures_exist_and_non_empty():
    """Verify that all 9 required figures exist in results/figures/eda and are non-empty."""
    figures_dir = Path("results/figures/eda")
    required_plots = [
        "soh_degradation_curves.png",
        "capacity_fade_comparison.png",
        "battery_wise_soh_distributions.png",
        "sensor_trends_voltage_current_temperature.png",
        "capacity_vs_discharge_duration.png",
        "capacity_recovery_events.png",
        "correlation_heatmap_grouped.png",
        "sensor_feature_distributions.png",
        "b0018_comparative_degradation.png",
    ]

    for plot_name in required_plots:
        plot_path = figures_dir / plot_name
        assert plot_path.exists(), f"Missing required figure: {plot_name}"
        assert plot_path.stat().st_size > 10000, f"Figure {plot_name} is too small / incomplete"
