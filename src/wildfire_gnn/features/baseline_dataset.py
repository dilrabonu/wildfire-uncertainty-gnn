from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_graph_as_dataframe(
    graph_path: str | Path,
    clip_target: bool = True,
    target_min: float = 0.0,
    target_max: float = 1.0,
    drop_constant_features: bool = True,
) -> pd.DataFrame:
    """Load saved PyG graph and convert node data into a dataframe."""
    graph = torch.load(graph_path, weights_only=False, map_location="cpu")

    x = graph.x.detach().cpu().numpy()
    y = graph.y.detach().cpu().numpy().reshape(-1)
    pos = graph.pos.detach().cpu().numpy()

    feature_names = list(graph.feature_names)
    df = pd.DataFrame(x, columns=feature_names)

    df["target"] = y
    df["row_index"] = pos[:, 0].astype(int)
    df["col_index"] = pos[:, 1].astype(int)

    # Replace inf with NaN then drop bad rows
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0).reset_index(drop=True)

    if clip_target:
        df["target"] = df["target"].clip(lower=target_min, upper=target_max)

    if drop_constant_features:
        protected_cols = {"target", "row_index", "col_index"}
        constant_cols = [
            col for col in df.columns
            if col not in protected_cols and df[col].nunique(dropna=False) <= 1
        ]
        if constant_cols:
            df = df.drop(columns=constant_cols)

    return df


def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save dataframe to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def validate_baseline_dataframe(df: pd.DataFrame) -> dict[str, float]:
    """Return sanity-check statistics for the dataframe."""
    stats = {
        "num_rows": int(len(df)),
        "num_columns": int(df.shape[1]),
        "target_min": float(df["target"].min()),
        "target_max": float(df["target"].max()),
        "target_mean": float(df["target"].mean()),
        "target_std": float(df["target"].std()),
        "missing_values_total": int(df.isna().sum().sum()),
    }
    return stats