"""
VoltIQ: Model Training Module
Handles model selection, cross-validation, hyperparameter tuning,
and serialization for battery State of Health (SOH) estimation.
"""

from pathlib import Path


class ModelTrainer:
    """
    Manages regression model training pipelines for SOH prediction.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
