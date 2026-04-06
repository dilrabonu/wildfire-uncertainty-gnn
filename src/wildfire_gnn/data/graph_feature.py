from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def _normalize_pos(pos: torch.Tensor) -> torch.Tensor:
    pos = pos.float()
    mins = pos.min(dim=0).values
    maxs = pos.max(dim=0).values
    denom = (maxs - mins).clamp_min(1e-8)
    return (pos - mins) / denom


def _one_hot_fuel_column(x: torch.Tensor, fuel_idx: int) -> torch.Tensor:
    fuel = x[:, fuel_idx].round().long()
    fuel = fuel - fuel.min()
    num_classes = int(fuel.max().item()) + 1
    fuel_oh = F.one_hot(fuel, num_classes=num_classes).float()

    keep_cols = [i for i in range(x.shape[1]) if i != fuel_idx]
    x_keep = x[:, keep_cols].float()
    return torch.cat([x_keep, fuel_oh], dim=1)


def _grouped_fuel_column(x: torch.Tensor, fuel_idx: int) -> torch.Tensor:
    fuel = x[:, fuel_idx].float()
    fuel = torch.bucketize(
        fuel,
        boundaries=torch.tensor([2, 4, 6, 8, 10], dtype=torch.float32, device=fuel.device),
    )
    fuel_oh = F.one_hot(fuel.long(), num_classes=6).float()

    keep_cols = [i for i in range(x.shape[1]) if i != fuel_idx]
    x_keep = x[:, keep_cols].float()
    return torch.cat([x_keep, fuel_oh], dim=1)


def _neighbor_mean_features(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    feature_indices: Sequence[int],
) -> torch.Tensor:
    src, dst = edge_index
    selected = x[src][:, list(feature_indices)]

    out = torch.zeros(
        (x.shape[0], len(feature_indices)),
        dtype=x.dtype,
        device=x.device,
    )
    out.index_add_(0, dst, selected)

    deg = torch.bincount(dst, minlength=x.shape[0]).float().unsqueeze(1).to(x.device)
    deg = deg.clamp_min(1.0)
    return out / deg


def _neighbor_std_features(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    feature_indices: Sequence[int],
) -> torch.Tensor:
    mean = _neighbor_mean_features(x, edge_index, feature_indices)

    src, dst = edge_index
    selected = x[src][:, list(feature_indices)]
    sq_diff = (selected - mean[dst]) ** 2

    out = torch.zeros(
        (x.shape[0], len(feature_indices)),
        dtype=x.dtype,
        device=x.device,
    )
    out.index_add_(0, dst, sq_diff)

    deg = torch.bincount(dst, minlength=x.shape[0]).float().unsqueeze(1).to(x.device)
    deg = deg.clamp_min(1.0)
    return torch.sqrt(out / deg + 1e-8)


def add_recovery_features(data, config: dict):
    if not config["feature_recovery"]["enabled"]:
        return data

    x = data.x.float()
    fr_cfg = config["feature_recovery"]

    fuel_idx = int(fr_cfg["fuel_model_index"])
    fuel_encoding = fr_cfg["fuel_encoding"]

    if fuel_encoding == "one_hot":
        x = _one_hot_fuel_column(x, fuel_idx)
    elif fuel_encoding == "grouped":
        x = _grouped_fuel_column(x, fuel_idx)
    elif fuel_encoding == "raw":
        pass
    else:
        raise ValueError(f"Unsupported fuel_encoding: {fuel_encoding}")

    if fr_cfg["add_positional_features"]:
        if not hasattr(data, "pos") or data.pos is None:
            raise AttributeError("Cannot add positional features because data.pos is missing.")
        pos_norm = _normalize_pos(data.pos)
        x = torch.cat([x, pos_norm], dim=1)

    if fr_cfg["add_degree_feature"]:
        dst = data.edge_index[1]
        deg = torch.bincount(dst, minlength=x.shape[0]).float().unsqueeze(1).to(x.device)
        deg = deg / deg.max().clamp_min(1.0)
        x = torch.cat([x, deg], dim=1)

    summary_indices = [int(i) for i in fr_cfg["neighbor_summary_feature_indices"]]

    if fr_cfg["add_neighbor_mean_features"]:
        mean_feats = _neighbor_mean_features(x, data.edge_index, summary_indices)
        x = torch.cat([x, mean_feats], dim=1)

    if fr_cfg["add_neighbor_std_features"]:
        std_feats = _neighbor_std_features(x, data.edge_index, summary_indices)
        x = torch.cat([x, std_feats], dim=1)

    data.x = x
    return data