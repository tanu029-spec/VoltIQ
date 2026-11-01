"""
Unit and integration tests for VoltIQ Streamlit Dashboard module (Phase 7).
Verifies dataset accessibility, model loading, feature schema integrity,
predictive inference safety, and Plotly visualization generation.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.predict import SOHPredictor
from src.feature_config import FEATURE_SET_TIER_1_SENSOR_ONLY
from app.components.charts import (
    plot_fleet_soh_comparison,
    plot_battery_degradation_dual,
    plot_battery_sensor_dynamics,
    plot_ica_curves,
    plot_oof_true_vs_predicted,
    plot_feature_importance_bar,
    plot_soh_gauge,
)


@pytest.fixture
def dashboard_data():
    """Load core feature and metric datasets used by the dashboard."""
    features_path = Path("data/processed/battery_health_features.csv")
    ica_path = Path("data/processed/battery_ica_features.csv")
    model_comp_path = Path("results/metrics/model_comparison.csv")
    oof_path = Path("results/metrics/oof_predictions.csv")
    feat_imp_path = Path("results/metrics/feature_importance.csv")

    assert features_path.exists(), "Missing battery_health_features.csv"
    assert ica_path.exists(), "Missing battery_ica_features.csv"
    assert model_comp_path.exists(), "Missing model_comparison.csv"
    assert oof_path.exists(), "Missing oof_predictions.csv"
    assert feat_imp_path.exists(), "Missing feature_importance.csv"

    features_df = pd.read_csv(features_path)
    ica_df = pd.read_csv(ica_path)
    model_comp_df = pd.read_csv(model_comp_path)
    oof_df = pd.read_csv(oof_path)
    feat_imp_df = pd.read_csv(feat_imp_path)

    return features_df, ica_df, model_comp_df, oof_df, feat_imp_df


def test_dashboard_datasets_structure(dashboard_data):
    """Verify that required datasets have expected dimensions and schemas."""
    features_df, ica_df, model_comp_df, oof_df, feat_imp_df = dashboard_data

    # Features check
    assert len(features_df) == 636
    assert set(features_df["battery_id"].unique()) == {"B0005", "B0006", "B0007", "B0018"}
    for col in FEATURE_SET_TIER_1_SENSOR_ONLY:
        assert col in features_df.columns

    # ICA curves check
    assert len(ica_df) > 50000
    assert "voltage_grid_v" in ica_df.columns
    assert "dq_dv" in ica_df.columns

    # Model comparison metrics check
    assert len(model_comp_df) >= 15  # 5 models * 3 tiers
    assert "oof_rmse" in model_comp_df.columns
    assert "oof_r2" in model_comp_df.columns

    # OOF predictions check
    assert len(oof_df) == 636
    assert "soh_percentage" in oof_df.columns
    assert "pred_Tier 1_RandomForest" in oof_df.columns or "pred_Tier1_RandomForest" in oof_df.columns

    # Feature importance check
    assert len(feat_imp_df) == 21
    assert "importance_mdi" in feat_imp_df.columns


def test_model_loading_and_inference(dashboard_data):
    """Verify that the saved model artifact loads and performs bounded SOH inference."""
    features_df, _, _, _, _ = dashboard_data
    model_path = Path("models/tier1_randomforest_model.joblib")
    assert model_path.exists(), "Missing tier1_randomforest_model.joblib"

    predictor = SOHPredictor(model_path=model_path)
    predictor.load_model()

    assert predictor.pipeline is not None
    assert predictor.feature_names == FEATURE_SET_TIER_1_SENSOR_ONLY

    # Run inference on sample cycle
    sample_row = features_df.iloc[[0]]
    pred = predictor.predict(sample_row)

    assert isinstance(pred, np.ndarray)
    assert len(pred) == 1
    assert not np.isnan(pred[0])
    assert 50.0 <= pred[0] <= 105.0


def test_chart_generation_fleet_soh(dashboard_data):
    """Verify fleet SOH degradation chart generation."""
    features_df, _, _, _, _ = dashboard_data
    fig = plot_fleet_soh_comparison(features_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4  # 4 batteries


def test_chart_generation_battery_degradation(dashboard_data):
    """Verify single battery dual degradation chart generation."""
    features_df, _, _, _, _ = dashboard_data
    fig = plot_battery_degradation_dual(features_df, "B0005")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2  # SOH trace + Capacity trace


def test_chart_generation_sensor_dynamics(dashboard_data):
    """Verify sensor dynamics chart generation."""
    features_df, _, _, _, _ = dashboard_data
    fig = plot_battery_sensor_dynamics(features_df, "B0006")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # Voltage, Temperature, Load


def test_chart_generation_ica_curves(dashboard_data):
    """Verify ICA curve evolution chart generation."""
    _, ica_df, _, _, _ = dashboard_data
    fig = plot_ica_curves(ica_df, "B0007")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # Initial, mid, late traces


def test_chart_generation_oof_predictions(dashboard_data):
    """Verify Out-of-Fold true vs predicted scatter chart generation."""
    _, _, _, oof_df, _ = dashboard_data
    fig = plot_oof_true_vs_predicted(oof_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 4  # Traces for batteries + reference line


def test_chart_generation_feature_importance(dashboard_data):
    """Verify feature importance bar chart generation."""
    _, _, _, _, feat_imp_df = dashboard_data
    fig = plot_feature_importance_bar(feat_imp_df, top_n=10)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_chart_generation_soh_gauge():
    """Verify SOH circular gauge generation."""
    fig = plot_soh_gauge(predicted_soh=88.5, actual_soh=90.0)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "indicator"
