"""
VoltIQ: Inference and Prediction Pipeline
Loads serialized State of Health (SOH) models to perform robust inference on new cycle data.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import joblib

from src.feature_config import LEAKAGE_COLUMNS, validate_ml_feature_set


class SOHPredictor:
    """
    Inference interface for State of Health (SOH) estimation models.
    Provides strict feature validation, schema verification, and prediction bounding.
    """

    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        self.pipeline: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}
        self.feature_names: List[str] = []

    def load_model(self) -> None:
        """
        Load the serialized model artifact from disk and extract metadata.
        """
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        loaded_artifact = joblib.load(self.model_path)
        if isinstance(loaded_artifact, dict) and "pipeline" in loaded_artifact:
            self.pipeline = loaded_artifact["pipeline"]
            self.metadata = loaded_artifact.get("metadata", {})
            self.feature_names = self.metadata.get("feature_names", [])
        else:
            self.pipeline = loaded_artifact
            self.metadata = {}
            self.feature_names = []

    def predict(
        self,
        features: pd.DataFrame,
        clip_bounds: bool = True,
        min_soh: float = 0.0,
        max_soh: float = 105.0,
    ) -> np.ndarray:
        """
        Predict SOH (%) given an input feature DataFrame.

        Args:
            features: DataFrame containing cycle-level features.
            clip_bounds: If True, clip predictions to [min_soh, max_soh].
            min_soh: Physical lower bound (default: 0.0%).
            max_soh: Physical upper bound (default: 105.0%).

        Returns:
            np.ndarray of SOH percentage predictions.
        """
        if self.pipeline is None:
            self.load_model()

        if not isinstance(features, pd.DataFrame):
            raise TypeError("Input features must be provided as a pandas DataFrame.")

        # If model artifact recorded expected features, enforce strict ordering and presence
        if self.feature_names:
            missing_cols = [c for c in self.feature_names if c not in features.columns]
            if missing_cols:
                raise ValueError(
                    f"Missing required feature columns for model inference: {missing_cols}"
                )
            # Reorder columns to exact training schema
            X_input = features[self.feature_names].values
        else:
            X_input = features.values

        predictions = self.pipeline.predict(X_input)
        preds_arr = np.asarray(predictions, dtype=np.float64)

        if clip_bounds:
            preds_arr = np.clip(preds_arr, min_soh, max_soh)

        return preds_arr
