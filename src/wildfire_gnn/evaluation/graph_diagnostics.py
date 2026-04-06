from __future__ import annotations

import numpy as np
import pandas as pd


def _summarize_target(y: np.ndarray) -> dict:
    return {
        "count": int(len(y)),
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "min": float(np.min(y)),
        "q25": float(np.quantile(y, 0.25)),
        "median": float(np.quantile(y, 0.50)),
        "q75": float(np.quantile(y, 0.75)),
        "max": float(np.max(y)),
    }


def make_split_diagnostics(data) -> pd.DataFrame:
    rows = []
    split_map = {
        "train": data.train_mask.cpu().numpy(),
        "val": data.val_mask.cpu().numpy(),
        "test": data.test_mask.cpu().numpy(),
    }

    y = data.y.cpu().numpy().reshape(-1)

    for split_name, mask in split_map.items():
        y_split = y[mask]
        stats = _summarize_target(y_split)
        stats["split"] = split_name
        rows.append(stats)

    return pd.DataFrame(rows)


def make_binned_target_distribution(data) -> pd.DataFrame:
    bins = [0.0, 0.01, 0.05, 0.10, 0.25, np.inf]
    labels = ["0-0.01", "0.01-0.05", "0.05-0.10", "0.10-0.25", "0.25+"]

    rows = []
    split_map = {
        "train": data.train_mask.cpu().numpy(),
        "val": data.val_mask.cpu().numpy(),
        "test": data.test_mask.cpu().numpy(),
    }

    y = data.y.cpu().numpy().reshape(-1)

    for split_name, mask in split_map.items():
        y_split = y[mask]
        cats = pd.cut(y_split, bins=bins, labels=labels, include_lowest=True, right=False)
        counts = pd.Series(cats).value_counts().reindex(labels, fill_value=0)
        proportions = counts / max(len(y_split), 1)

        for label in labels:
            rows.append({
                "split": split_name,
                "bin": label,
                "count": int(counts[label]),
                "proportion": float(proportions[label]),
            })

    return pd.DataFrame(rows)