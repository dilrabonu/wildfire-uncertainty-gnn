from __future__ import annotations

import math
from typing import Dict

import numpy as np
from scipy.stats import pearsonr, spearmanr


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(y_true - y_pred)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else np.nan)

    pearson = float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else np.nan
    spearman = float(spearmanr(y_true, y_pred)[0]) if len(y_true) > 1 else np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
    }


def tail_rmse(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.10) -> float:
    mask = y_true.reshape(-1) >= threshold
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true.reshape(-1)[mask] - y_pred.reshape(-1)[mask]) ** 2))))