"""
VoltIQ: Model Training & Evaluation Engine
Implements rigorous GroupKFold Leave-One-Battery-Out cross-validation across
multiple regression architectures and audited feature benchmark tiers.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import logging
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from src.feature_config import (
    FEATURE_SET_TIER_1_SENSOR_ONLY,
    FEATURE_SET_TIER_2_SENSOR_PLUS_AGE,
    FEATURE_SET_TIER_3_FULL_INFORMATIVE,
    validate_ml_feature_set,
)
from src.evaluation import evaluate_soh_predictions, aggregate_fold_metrics

logger = logging.getLogger("train")


class ModelTrainer:
    """
    Manages regression model training, cross-validation, feature importance analysis,
    and serialization for the VoltIQ system.
    """

    TIER_DEFINITIONS: Dict[str, List[str]] = {
        "Tier 1 (Sensor-Only)": FEATURE_SET_TIER_1_SENSOR_ONLY,
        "Tier 2 (Sensor + Age)": FEATURE_SET_TIER_2_SENSOR_PLUS_AGE,
        "Tier 3 (Full Diagnostic)": FEATURE_SET_TIER_3_FULL_INFORMATIVE,
    }

    def __init__(self, models_dir: Union[str, Path] = "models", random_state: int = 42):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

    def get_model_registry(self) -> Dict[str, Callable[[], Any]]:
        """
        Define candidate regression architectures configured with Scikit-learn Pipelines.
        Scalers and imputers are strictly encapsulated inside pipelines so they are
        fitted exclusively on the training folds.
        """
        rs = self.random_state
        return {
            "DummyRegressor": lambda: DummyRegressor(strategy="mean"),
            "LinearRegression": lambda: Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]),
            "Ridge": lambda: Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=1.0, random_state=rs)),
            ]),
            "RandomForest": lambda: Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("regressor", RandomForestRegressor(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_leaf=2,
                    random_state=rs,
                    n_jobs=-1,
                )),
            ]),
            "HistGradientBoosting": lambda: Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("regressor", HistGradientBoostingRegressor(
                    max_iter=100,
                    max_depth=4,
                    min_samples_leaf=5,
                    random_state=rs,
                )),
            ]),
        }

    def evaluate_model_group_kfold(
        self,
        model_name: str,
        pipeline_builder: Callable[[], Any],
        tier_name: str,
        feature_cols: List[str],
        df: pd.DataFrame,
        target_col: str = "soh_percentage",
        group_col: str = "battery_id",
    ) -> Dict[str, Any]:
        """
        Evaluate a single model on a specific feature tier using GroupKFold (Leave-One-Battery-Out).

        Args:
            model_name: Name of the regression architecture.
            pipeline_builder: Factory callable creating a fresh pipeline.
            tier_name: Benchmark tier identifier.
            feature_cols: List of input feature names.
            df: Input feature dataset containing features, target, and groups.
            target_col: Name of target column (default: 'soh_percentage').
            group_col: Name of grouping column (default: 'battery_id').

        Returns:
            Dict containing OOF predictions, per-battery metrics, aggregate statistics,
            and fold feature importances.
        """
        validate_ml_feature_set(feature_cols)

        X = df[feature_cols].values
        y = df[target_col].values.astype(np.float64)
        groups = df[group_col].values

        unique_groups = np.unique(groups)
        n_splits = len(unique_groups)
        gkf = GroupKFold(n_splits=n_splits)

        oof_predictions = np.zeros(len(y), dtype=np.float64)
        fold_metrics_list: List[Dict[str, Any]] = []
        fold_importances: List[np.ndarray] = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            held_out_battery = str(groups[val_idx][0])

            # Verify zero leakage across groups
            train_groups = set(groups[train_idx])
            val_groups = set(groups[val_idx])
            if train_groups.intersection(val_groups):
                raise RuntimeError(
                    f"DATA LEAKAGE DETECTED in Fold {fold_idx}: Overlapping groups {train_groups.intersection(val_groups)}"
                )

            # Build and fit pipeline strictly on training fold
            pipeline = pipeline_builder()
            pipeline.fit(X[train_idx], y[train_idx])

            # Predict on held-out battery
            val_preds = pipeline.predict(X[val_idx])
            oof_predictions[val_idx] = val_preds

            # Compute fold metrics
            metrics = evaluate_soh_predictions(y[val_idx], val_preds)
            metrics.update({
                "fold": fold_idx + 1,
                "held_out_battery": held_out_battery,
                "train_cycles": int(len(train_idx)),
                "val_cycles": int(len(val_idx)),
                "model_name": model_name,
                "feature_tier": tier_name,
            })
            fold_metrics_list.append(metrics)

            # Extract feature importance if supported
            regressor = pipeline.named_steps.get("regressor", pipeline) if hasattr(pipeline, "named_steps") else pipeline
            if hasattr(regressor, "feature_importances_"):
                fold_importances.append(regressor.feature_importances_)
            elif hasattr(regressor, "coef_"):
                fold_importances.append(np.abs(regressor.coef_))

        # Overall pooled Out-Of-Fold metrics
        overall_oof_metrics = evaluate_soh_predictions(y, oof_predictions)

        # Aggregate fold distributions (mean and std across the 4 batteries)
        fold_stats = aggregate_fold_metrics(fold_metrics_list)

        # Average feature importances across folds
        mean_importances = None
        if fold_importances:
            mean_importances = np.mean(fold_importances, axis=0)

        return {
            "model_name": model_name,
            "feature_tier": tier_name,
            "feature_columns": feature_cols,
            "overall_oof_metrics": overall_oof_metrics,
            "fold_metrics": fold_metrics_list,
            "fold_summary": fold_stats,
            "oof_predictions": oof_predictions,
            "feature_importances": mean_importances,
        }

    def run_benchmark_suite(
        self,
        df: pd.DataFrame,
        target_col: str = "soh_percentage",
        group_col: str = "battery_id",
    ) -> Dict[str, Any]:
        """
        Execute full benchmark suite across all 5 models and 3 feature tiers.
        """
        registry = self.get_model_registry()
        suite_results: Dict[str, Dict[str, Any]] = {}

        comparison_rows: List[Dict[str, Any]] = []
        all_fold_rows: List[Dict[str, Any]] = []
        oof_predictions_df = df[["battery_id", "cycle_index", "cycle_number", target_col]].copy()

        for tier_name, feature_cols in self.TIER_DEFINITIONS.items():
            logger.info("Executing benchmark for %s (%d features)...", tier_name, len(feature_cols))
            suite_results[tier_name] = {}

            for model_name, builder in registry.items():
                logger.info("  Evaluating %s...", model_name)
                res = self.evaluate_model_group_kfold(
                    model_name=model_name,
                    pipeline_builder=builder,
                    tier_name=tier_name,
                    feature_cols=feature_cols,
                    df=df,
                    target_col=target_col,
                    group_col=group_col,
                )
                suite_results[tier_name][model_name] = res

                # Track OOF column
                col_key = f"pred_{tier_name[:6].strip()}_{model_name}"
                oof_predictions_df[col_key] = res["oof_predictions"]

                # Track comparison row
                comparison_rows.append({
                    "feature_tier": tier_name,
                    "model_name": model_name,
                    "features_count": len(feature_cols),
                    "oof_rmse": res["overall_oof_metrics"]["rmse"],
                    "oof_mae": res["overall_oof_metrics"]["mae"],
                    "oof_r2": res["overall_oof_metrics"]["r2"],
                    "oof_mape": res["overall_oof_metrics"]["mape"],
                    "fold_rmse_mean": res["fold_summary"]["rmse_mean"],
                    "fold_rmse_std": res["fold_summary"]["rmse_std"],
                    "fold_mae_mean": res["fold_summary"]["mae_mean"],
                    "fold_mae_std": res["fold_summary"]["mae_std"],
                    "fold_r2_mean": res["fold_summary"]["r2_mean"],
                    "fold_r2_std": res["fold_summary"]["r2_std"],
                })

                # Track individual fold rows
                for fm in res["fold_metrics"]:
                    all_fold_rows.append(fm)

        comparison_df = pd.DataFrame(comparison_rows)
        fold_metrics_df = pd.DataFrame(all_fold_rows)

        return {
            "results": suite_results,
            "comparison_df": comparison_df,
            "fold_metrics_df": fold_metrics_df,
            "oof_predictions_df": oof_predictions_df,
        }

    def train_final_model(
        self,
        model_name: str,
        tier_name: str,
        df: pd.DataFrame,
        target_col: str = "soh_percentage",
    ) -> Tuple[Any, Path]:
        """
        Fit the selected pipeline architecture on the complete dataset (636 cycles)
        and serialize to disk with metadata.
        """
        feature_cols = self.TIER_DEFINITIONS[tier_name]
        validate_ml_feature_set(feature_cols)

        registry = self.get_model_registry()
        if model_name not in registry:
            raise ValueError(f"Unknown model architecture: {model_name}")

        pipeline = registry[model_name]()
        X = df[feature_cols].values
        y = df[target_col].values.astype(np.float64)

        pipeline.fit(X, y)

        slug = f"{tier_name[:6].lower().replace(' ', '')}_{model_name.lower()}"
        save_path = self.models_dir / f"{slug}_model.joblib"

        metadata = {
            "model_name": model_name,
            "feature_tier": tier_name,
            "features_count": len(feature_cols),
            "feature_names": feature_cols,
            "target_col": target_col,
            "training_cycles": len(df),
            "batteries": list(df["battery_id"].unique()),
        }

        joblib.dump({"pipeline": pipeline, "metadata": metadata}, save_path)
        logger.info("Saved final model artifact to: %s", save_path)

        return pipeline, save_path
