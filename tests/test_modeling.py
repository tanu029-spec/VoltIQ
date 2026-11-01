"""
Unit and integration tests for VoltIQ Machine Learning Modeling module (Phase 6).
Verifies GroupKFold Leave-One-Battery-Out data isolation, absence of target leakage,
reproducibility, metric accuracy, pipeline fitting safety, and inference stability.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from src.train import ModelTrainer
from src.predict import SOHPredictor
from src.evaluation import evaluate_soh_predictions, aggregate_fold_metrics
from src.feature_config import (
    FEATURE_SET_TIER_1_SENSOR_ONLY,
    FEATURE_SET_TIER_2_SENSOR_PLUS_AGE,
    FEATURE_SET_TIER_3_FULL_INFORMATIVE,
    LEAKAGE_COLUMNS,
    TARGET_COLUMNS,
    validate_ml_feature_set,
)


@pytest.fixture
def sample_dataset():
    """Load authentic engineered feature dataset."""
    df_path = Path("data/processed/battery_health_features.csv")
    assert df_path.exists(), "Feature dataset missing."
    return pd.read_csv(df_path)


@pytest.fixture
def trainer(tmp_path):
    """Instantiate ModelTrainer with temporary directory."""
    return ModelTrainer(models_dir=tmp_path / "models", random_state=42)


def test_group_kfold_no_battery_overlap(sample_dataset):
    """Verify strictly zero battery overlap between training and validation splits."""
    groups = sample_dataset["battery_id"].values
    unique_batteries = np.unique(groups)
    assert len(unique_batteries) == 4

    gkf = GroupKFold(n_splits=4)
    X_dummy = np.zeros((len(sample_dataset), 5))
    y_dummy = sample_dataset["soh_percentage"].values

    held_out_seen = set()
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_dummy, y_dummy, groups=groups)):
        train_bats = set(groups[train_idx])
        val_bats = set(groups[val_idx])

        # Strict disjoint set check
        assert train_bats.isdisjoint(val_bats), f"Fold {fold_idx} has overlapping batteries: {train_bats & val_bats}"
        assert len(val_bats) == 1, f"Fold {fold_idx} should have exactly 1 held-out battery, found {len(val_bats)}"

        held_out_bat = list(val_bats)[0]
        held_out_seen.add(held_out_bat)

    # All 4 batteries must appear as held-out group exactly once
    assert held_out_seen == set(unique_batteries)


def test_tier_1_feature_leakage_safety():
    """Verify that Tier 1 sensor features contain zero leakage or target columns."""
    validation = validate_ml_feature_set(FEATURE_SET_TIER_1_SENSOR_ONLY)
    assert validation["status"] == "PASS"
    assert validation["is_safe_for_training"] is True

    # Check against leakage and target lists
    assert set(FEATURE_SET_TIER_1_SENSOR_ONLY).isdisjoint(set(LEAKAGE_COLUMNS))
    assert set(FEATURE_SET_TIER_1_SENSOR_ONLY).isdisjoint(set(TARGET_COLUMNS))

    # Prohibited duration and age features
    assert "discharge_duration_seconds" not in FEATURE_SET_TIER_1_SENSOR_ONLY
    assert "cycle_index" not in FEATURE_SET_TIER_1_SENSOR_ONLY
    assert "cycle_number" not in FEATURE_SET_TIER_1_SENSOR_ONLY


def test_metric_calculations_match_manual():
    """Verify evaluate_soh_predictions matches exact analytical formulas."""
    y_true = np.array([100.0, 90.0, 80.0, 70.0])
    y_pred = np.array([98.0, 92.0, 78.0, 72.0])  # errors: -2, +2, -2, +2

    metrics = evaluate_soh_predictions(y_true, y_pred)

    expected_mae = 2.0
    expected_rmse = 2.0
    # SS_res = 4 * (2^2) = 16. SS_tot = (15^2 + 5^2 + (-5)^2 + (-15)^2) = 225 + 25 + 25 + 225 = 500
    expected_r2 = 1.0 - (16.0 / 500.0)

    assert metrics["mae"] == pytest.approx(expected_mae, rel=1e-5)
    assert metrics["rmse"] == pytest.approx(expected_rmse, rel=1e-5)
    assert metrics["r2"] == pytest.approx(expected_r2, rel=1e-5)
    assert metrics["mape"] == pytest.approx(np.mean([2/100, 2/90, 2/80, 2/70]) * 100.0, rel=1e-5)


def test_metric_calculation_error_handling():
    """Verify evaluation function raises appropriate errors on invalid inputs."""
    with pytest.raises(ValueError):
        evaluate_soh_predictions([], [])
    with pytest.raises(ValueError):
        evaluate_soh_predictions([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        evaluate_soh_predictions([1.0, 2.0], [np.nan, 2.0])
    with pytest.raises(ValueError):
        evaluate_soh_predictions([1.0, 2.0], [np.inf, 2.0])


def test_preprocessing_fitted_only_on_train_fold(sample_dataset, trainer):
    """
    Verify that transformers inside the pipeline (e.g. StandardScaler) are
    fitted exclusively on the training fold, not the validation fold.
    """
    registry = trainer.get_model_registry()
    pipeline_builder = registry["Ridge"]

    X = sample_dataset[FEATURE_SET_TIER_1_SENSOR_ONLY].values
    y = sample_dataset["soh_percentage"].values
    groups = sample_dataset["battery_id"].values

    gkf = GroupKFold(n_splits=4)
    train_idx, val_idx = next(gkf.split(X, y, groups=groups))

    pipeline = pipeline_builder()
    pipeline.fit(X[train_idx], y[train_idx])

    # Check fitted mean in scaler matches training mean, NOT overall mean
    scaler = pipeline.named_steps["scaler"]
    train_mean = np.nanmean(X[train_idx], axis=0)
    overall_mean = np.nanmean(X, axis=0)

    assert np.allclose(scaler.mean_, train_mean, rtol=1e-4)
    assert not np.allclose(scaler.mean_, overall_mean, rtol=1e-4)


def test_predictions_contain_no_nan_or_inf(sample_dataset, trainer):
    """Verify that predictions from all candidate models contain zero NaN or infinite values."""
    registry = trainer.get_model_registry()

    for m_name in ["DummyRegressor", "Ridge", "RandomForest"]:
        res = trainer.evaluate_model_group_kfold(
            model_name=m_name,
            pipeline_builder=registry[m_name],
            tier_name="Tier 1 (Sensor-Only)",
            feature_cols=FEATURE_SET_TIER_1_SENSOR_ONLY,
            df=sample_dataset,
        )
        preds = res["oof_predictions"]
        assert not np.isnan(preds).any()
        assert not np.isinf(preds).any()
        assert len(preds) == len(sample_dataset)


def test_training_reproducibility(sample_dataset):
    """Verify that models produce identical results with fixed random seed."""
    trainer1 = ModelTrainer(models_dir="models", random_state=42)
    trainer2 = ModelTrainer(models_dir="models", random_state=42)

    reg1 = trainer1.get_model_registry()["RandomForest"]
    reg2 = trainer2.get_model_registry()["RandomForest"]

    res1 = trainer1.evaluate_model_group_kfold(
        model_name="RandomForest",
        pipeline_builder=reg1,
        tier_name="Tier 1 (Sensor-Only)",
        feature_cols=FEATURE_SET_TIER_1_SENSOR_ONLY,
        df=sample_dataset,
    )
    res2 = trainer2.evaluate_model_group_kfold(
        model_name="RandomForest",
        pipeline_builder=reg2,
        tier_name="Tier 1 (Sensor-Only)",
        feature_cols=FEATURE_SET_TIER_1_SENSOR_ONLY,
        df=sample_dataset,
    )

    assert np.allclose(res1["oof_predictions"], res2["oof_predictions"], rtol=1e-5)
    assert res1["overall_oof_metrics"]["rmse"] == pytest.approx(res2["overall_oof_metrics"]["rmse"], rel=1e-5)


def test_soh_predictor_interface(sample_dataset):
    """Verify that SOHPredictor loads saved model artifacts and performs valid inference."""
    model_path = Path("models/tier1_randomforest_model.joblib")
    assert model_path.exists(), "Trained model artifact missing."

    predictor = SOHPredictor(model_path=model_path)
    predictor.load_model()

    assert predictor.pipeline is not None
    assert len(predictor.feature_names) == 21

    # Predict on test slice
    test_slice = sample_dataset[FEATURE_SET_TIER_1_SENSOR_ONLY].head(10)
    preds = predictor.predict(test_slice)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == 10
    assert not np.isnan(preds).any()
    assert (preds >= 0.0).all() and (preds <= 105.0).all()

    # Missing column validation check
    with pytest.raises(ValueError):
        incomplete_slice = test_slice.drop(columns=[FEATURE_SET_TIER_1_SENSOR_ONLY[0]])
        predictor.predict(incomplete_slice)
