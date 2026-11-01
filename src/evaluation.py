"""
VoltIQ: Model Evaluation Module
Calculates standard regression performance metrics (RMSE, MAE, R2, MAPE),
aggregates fold-level distributions, and formats benchmark comparisons.
"""

from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_soh_predictions(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """
    Compute key regression evaluation metrics for State of Health (SOH) models.

    Args:
        y_true: Ground truth SOH percentages.
        y_pred: Model predicted SOH percentages.

    Returns:
        Dict with keys 'rmse', 'mae', 'r2', 'mape'.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    if len(y_true_arr) == 0:
        raise ValueError("Cannot evaluate empty predictions array.")
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"Shape mismatch: y_true length {len(y_true_arr)} != y_pred length {len(y_pred_arr)}"
        )
    if np.isnan(y_pred_arr).any() or np.isinf(y_pred_arr).any():
        raise ValueError("Predictions contain NaN or infinite values.")

    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    # Safe MAPE calculation avoiding division by zero
    non_zero_mask = y_true_arr != 0
    if np.any(non_zero_mask):
        mape = float(
            np.mean(np.abs((y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask]))
            * 100.0
        )
    else:
        mape = 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
    }


def aggregate_fold_metrics(fold_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute mean and standard deviation of evaluation metrics across folds.

    Args:
        fold_records: List of dicts, each containing 'rmse', 'mae', 'r2', 'mape'.

    Returns:
        Dict containing mean and std for each metric, plus raw fold records.
    """
    if not fold_records:
        raise ValueError("No fold records provided for aggregation.")

    df_folds = pd.DataFrame(fold_records)
    metrics = ["rmse", "mae", "r2", "mape"]

    summary: Dict[str, Any] = {"fold_count": len(fold_records)}
    for m in metrics:
        if m in df_folds.columns:
            summary[f"{m}_mean"] = float(df_folds[m].mean())
            summary[f"{m}_std"] = float(df_folds[m].std(ddof=1)) if len(df_folds) > 1 else 0.0

    summary["folds"] = fold_records
    return summary
