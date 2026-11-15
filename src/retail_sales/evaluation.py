"""Regression metrics (same set the original notebook reported)."""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def adjusted_r2(r2: float, n_rows: int, n_features: int) -> float:
    if n_rows - n_features - 1 <= 0:
        raise ValueError("Adjusted R² needs more rows than features.")
    return 1 - (1 - r2) * (n_rows - 1) / (n_rows - n_features - 1)


def regression_report(y_true, y_pred, n_features: int) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    r2 = float(r2_score(y_true, y_pred))
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": r2,
        "adj_r2": adjusted_r2(r2, len(y_true), n_features),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }
