from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    pearson = float(pearsonr(y_true, y_pred)[0]) if len(np.unique(y_true)) > 1 else np.nan
    spearman = float(spearmanr(y_true, y_pred)[0]) if len(np.unique(y_true)) > 1 else np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
    }


def binwise_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: List[Tuple[float, float]],
) -> pd.DataFrame:
    rows = []
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    for low, high in bins:
        mask = (y_true >= low) & (y_true < high)
        if mask.sum() == 0:
            continue

        metrics = regression_metrics(y_true[mask], y_pred[mask])
        metrics["bin_low"] = low
        metrics["bin_high"] = high
        metrics["count"] = int(mask.sum())
        rows.append(metrics)

    return pd.DataFrame(rows)


def regression_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_unc: np.ndarray,
    num_bins: int = 10,
) -> float:
    """
    A simple regression calibration proxy:
    bin by predicted uncertainty and compare avg absolute error vs avg uncertainty.
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    y_unc = y_unc.reshape(-1)

    order = np.argsort(y_unc)
    y_true = y_true[order]
    y_pred = y_pred[order]
    y_unc = y_unc[order]

    bins = np.array_split(np.arange(len(y_true)), num_bins)
    ece = 0.0

    for idx in bins:
        if len(idx) == 0:
            continue
        mean_err = np.mean(np.abs(y_true[idx] - y_pred[idx]))
        mean_unc = np.mean(y_unc[idx])
        ece += (len(idx) / len(y_true)) * abs(mean_err - mean_unc)

    return float(ece)


def reliability_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_unc: np.ndarray,
    num_bins: int = 10,
) -> pd.DataFrame:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    y_unc = y_unc.reshape(-1)

    order = np.argsort(y_unc)
    y_true = y_true[order]
    y_pred = y_pred[order]
    y_unc = y_unc[order]

    rows = []
    bins = np.array_split(np.arange(len(y_true)), num_bins)
    for i, idx in enumerate(bins):
        if len(idx) == 0:
            continue
        rows.append(
            {
                "bin": i,
                "mean_uncertainty": float(np.mean(y_unc[idx])),
                "mean_abs_error": float(np.mean(np.abs(y_true[idx] - y_pred[idx]))),
                "count": int(len(idx)),
            }
        )

    return pd.DataFrame(rows)