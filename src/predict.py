"""
VoltIQ: Inference and Prediction Pipeline
Loads trained SOH estimation models to perform predictions on new cycle data.
"""

from pathlib import Path
from typing import Any
import joblib
import pandas as pd


class SOHPredictor:
    """
    Inference interface for State of Health (SOH) models.
    """

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None

    def load_model(self) -> None:
        """
        Load the serialized model from disk.
        """
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        self.model = joblib.load(self.model_path)

    def predict(self, features: pd.DataFrame) -> Any:
        """
        Predict SOH values given an engineered feature DataFrame.
        """
        if self.model is None:
            self.load_model()
        return self.model.predict(features)
