from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared score."""
    return float(r2_score(y_true, y_pred))


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation."""
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(pearsonr(y_true, y_pred)[0])


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation."""
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(spearmanr(y_true, y_pred)[0])


def get_metric_functions() -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    """Return supported metric callables."""
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson_corr,
        "spearman": spearman_corr,
    }


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute regression metrics."""
    metric_fns = get_metric_functions()
    return {name: fn(y_true, y_pred) for name, fn in metric_fns.items()}