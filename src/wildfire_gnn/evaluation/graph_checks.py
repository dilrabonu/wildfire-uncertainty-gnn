from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except ImportError: 
    torch = None


def tensor_basic_stats(tensor: Any) -> dict[str, float | int | tuple[int, ...]]:
    """Return basic statistics for a tensor-like object."""
    if torch is not None and isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)

    return {
        "shape": tuple(arr.shape),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def count_non_finite(tensor: Any) -> dict[str, int]:
    """Count NaN and Inf values in a tensor-like object."""
    if torch is not None and isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)

    return {
        "nan_count": int(np.isnan(arr).sum()),
        "posinf_count": int(np.isposinf(arr).sum()),
        "neginf_count": int(np.isneginf(arr).sum()),
    }


def edge_density(num_nodes: int, num_edges: int) -> float:
    """Compute average directed edges per node."""
    if num_nodes == 0:
        return 0.0
    return float(num_edges / num_nodes)


def unique_categorical_values(column: Any, max_show: int = 20) -> dict[str, Any]:
    """Return unique values and counts for a categorical feature column."""
    if torch is not None and isinstance(column, torch.Tensor):
        arr = column.detach().cpu().numpy()
    else:
        arr = np.asarray(column)

    values, counts = np.unique(arr, return_counts=True)

    return {
        "num_unique": int(len(values)),
        "sample_values": values[:max_show].tolist(),
        "sample_counts": counts[:max_show].tolist(),
    }