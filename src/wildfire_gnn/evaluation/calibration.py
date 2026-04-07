from __future__ import annotations

from typing import Dict

import numpy as np


def interval_coverage(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, z: float = 1.96) -> float:
    lower = mean - z * std
    upper = mean + z * std
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def regression_ece(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, n_bins: int = 10) -> float:
    std = np.clip(std, 1e-8, None)
    z = np.abs(y_true - mean) / std
    confidence = np.exp(-z)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidence >= bins[i]) & (confidence < bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = float(confidence[mask].mean())
        avg_acc = float((z[mask] <= 1.0).mean())
        ece += mask.mean() * abs(avg_conf - avg_acc)

    return float(ece)


def calibration_summary(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    return {
        "coverage_68": interval_coverage(y_true, mean, std, z=1.0),
        "coverage_95": interval_coverage(y_true, mean, std, z=1.96),
        "ece_reg": regression_ece(y_true, mean, std, n_bins=n_bins),
    }