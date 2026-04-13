from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree


@dataclass
class FeatureArtifacts:
    continuous_stats: Dict[str, Dict[str, float]]
    fuel_mapping: Dict[int, int]
    feature_names_out: List[str]


def _clip_series(s: pd.Series, q_low: float, q_high: float) -> pd.Series:
    low = s.quantile(q_low)
    high = s.quantile(q_high)
    return s.clip(lower=low, upper=high)


def _sqrt_scaled(s: pd.Series) -> pd.Series:
    s = s - s.min()
    return np.sqrt(s + 1e-8)


def _log1p_scaled(s: pd.Series) -> pd.Series:
    s = s - s.min()
    return np.log1p(s)


def _robust_zscore(s: pd.Series) -> pd.Series:
    median = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    iqr = max(iqr, 1e-8)
    return (s - median) / iqr


def transform_continuous_features(
    df: pd.DataFrame,
    continuous_cols: List[str],
    method: str = "sqrt_scaled",
    clip_quantile_low: float = 0.001,
    clip_quantile_high: float = 0.999,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Transform skewed continuous features before modeling.
    """
    df = df.copy()
    stats: Dict[str, Dict[str, float]] = {}

    for col in continuous_cols:
        s = df[col].astype(float)
        s = _clip_series(s, clip_quantile_low, clip_quantile_high)

        if method == "sqrt_scaled":
            s = _sqrt_scaled(s)
        elif method == "log1p":
            s = _log1p_scaled(s)
        elif method == "robust_zscore":
            s = _robust_zscore(s)
        else:
            raise ValueError(f"Unsupported transform method: {method}")

        mean = float(s.mean())
        std = float(s.std())
        std = max(std, 1e-8)

        s = (s - mean) / std
        df[col] = s
        stats[col] = {"mean": mean, "std": std}

    return df, stats


def encode_fuel_onehot(
    df: pd.DataFrame,
    fuel_col: str = "Fuel_Models.img",
) -> Tuple[pd.DataFrame, Dict[int, int], List[str]]:
    """
    One-hot encode fuel category IDs.
    """
    df = df.copy()
    unique_vals = sorted(df[fuel_col].dropna().astype(int).unique().tolist())
    mapping = {val: idx for idx, val in enumerate(unique_vals)}

    df[fuel_col] = df[fuel_col].astype(int).map(mapping)
    onehot = pd.get_dummies(df[fuel_col], prefix="fuel", dtype=float)

    df = pd.concat([df.drop(columns=[fuel_col]), onehot], axis=1)
    feature_names = onehot.columns.tolist()

    return df, mapping, feature_names


def build_engineered_feature_table(
    df: pd.DataFrame,
    continuous_cols: List[str],
    categorical_cols: List[str],
    transform_method: str = "sqrt_scaled",
    clip_quantile_low: float = 0.001,
    clip_quantile_high: float = 0.999,
    encode_fuel: str = "onehot",
) -> Tuple[pd.DataFrame, FeatureArtifacts]:
    """
    Build engineered tabular feature table.
    Useful for analysis and possible future ablations.
    """
    df = df.copy()

    df, cont_stats = transform_continuous_features(
        df=df,
        continuous_cols=continuous_cols,
        method=transform_method,
        clip_quantile_low=clip_quantile_low,
        clip_quantile_high=clip_quantile_high,
    )

    fuel_mapping: Dict[int, int] = {}
    extra_feature_names: List[str] = []

    if "Fuel_Models.img" in categorical_cols and encode_fuel == "onehot":
        df, fuel_mapping, extra_feature_names = encode_fuel_onehot(df, fuel_col="Fuel_Models.img")

    feature_names_out = [
        c for c in df.columns
        if c not in ["target", "split", "row_index", "col_index"]
    ]

    artifacts = FeatureArtifacts(
        continuous_stats=cont_stats,
        fuel_mapping=fuel_mapping,
        feature_names_out=feature_names_out + extra_feature_names,
    )
    return df, artifacts


def add_degree_feature(data: Data) -> Data:
    """
    Add node degree as an extra graph feature.
    """
    data = data.clone()
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes).view(-1, 1)
    data.x = torch.cat([data.x, deg], dim=1)
    return data


def add_neighborhood_summary_features(
    data: Data,
    add_mean: bool = True,
    add_std: bool = True,
    add_max: bool = False,
    add_residual: bool = False,
) -> Data:
    """
    Add neighborhood summary features from incoming neighbors.

    Optional features:
    - neighbor mean
    - neighbor std
    - neighbor max
    - residual = node feature - neighbor mean
    """
    data = data.clone()
    x = data.x
    edge_index = data.edge_index
    src, dst = edge_index[0], edge_index[1]

    num_nodes, num_feats = x.shape
    device = x.device

    counts = torch.zeros((num_nodes, 1), device=device)
    neigh_sum = torch.zeros((num_nodes, num_feats), device=device)
    neigh_sq_sum = torch.zeros((num_nodes, num_feats), device=device)
    neigh_max = torch.full((num_nodes, num_feats), -torch.inf, device=device)

    counts.index_add_(0, dst, torch.ones((dst.shape[0], 1), device=device))
    neigh_sum.index_add_(0, dst, x[src])
    neigh_sq_sum.index_add_(0, dst, x[src] ** 2)

    for i in range(src.shape[0]):
        d = dst[i]
        s = src[i]
        neigh_max[d] = torch.maximum(neigh_max[d], x[s])

    counts = torch.clamp(counts, min=1.0)
    neigh_mean = neigh_sum / counts
    neigh_var = torch.clamp((neigh_sq_sum / counts) - (neigh_mean ** 2), min=0.0)
    neigh_std = torch.sqrt(neigh_var + 1e-8)

    neigh_max = torch.where(torch.isinf(neigh_max), x, neigh_max)

    feats = [x]

    if add_mean:
        feats.append(neigh_mean)
    if add_std:
        feats.append(neigh_std)
    if add_max:
        feats.append(neigh_max)
    if add_residual:
        feats.append(x - neigh_mean)

    data.x = torch.cat(feats, dim=1)
    return data