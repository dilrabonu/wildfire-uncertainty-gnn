from __future__ import annotations

import numpy as np


def expected_calibration_error_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10,
) -> float:
    error = np.abs(y_true - y_pred)
    order = np.argsort(y_std)
    bins = np.array_split(order, n_bins)

    ece = 0.0
    total = len(y_true)

    for idx in bins:
        if len(idx) == 0:
            continue
        bin_conf = float(np.mean(y_std[idx]))
        bin_err = float(np.mean(error[idx]))
        ece += (len(idx) / total) * abs(bin_err - bin_conf)

    return float(ece)


def interval_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    z: float = 1.96,
) -> float:
    lower = y_pred - z * y_std
    upper = y_pred + z * y_std
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))