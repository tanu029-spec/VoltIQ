"""
VoltIQ: Model Evaluation Module
Calculates standard regression performance metrics (RMSE, MAE, R2, MAPE)
and generates evaluation reports.
"""

from typing import Dict
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_soh_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute key regression evaluation metrics for SOH models.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
    }
