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
    fuel = x[:, fuel_idx].long()
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
    x_source: torch.Tensor,
    edge_index: torch.Tensor,
    feature_indices: Sequence[int],
) -> torch.Tensor:
    src, dst = edge_index
    selected = x_source[src][:, list(feature_indices)]

    out = torch.zeros(
        (x_source.shape[0], len(feature_indices)),
        dtype=x_source.dtype,
        device=x_source.device,
    )
    out.index_add_(0, dst, selected)

    deg = torch.bincount(dst, minlength=x_source.shape[0]).float().unsqueeze(1).to(x_source.device)
    deg = deg.clamp_min(1.0)
    return out / deg


def add_recovery_features(data, config: dict):
    if not config["feature_recovery"]["enabled"]:
        return data

    fr_cfg = config["feature_recovery"]

    x_orig = data.x.float()
    x_work = x_orig.clone()

    fuel_idx = int(fr_cfg["fuel_model_index"])
    fuel_encoding = fr_cfg["fuel_encoding"]

    if fuel_encoding == "one_hot":
        x_work = _one_hot_fuel_column(x_work, fuel_idx)
    elif fuel_encoding == "grouped":
        x_work = _grouped_fuel_column(x_work, fuel_idx)
    elif fuel_encoding == "raw":
        pass
    else:
        raise ValueError(f"Unsupported fuel_encoding: {fuel_encoding}")

    if fr_cfg["add_positional_features"]:
        if not hasattr(data, "pos") or data.pos is None:
            raise AttributeError("Cannot add positional features because data.pos is missing.")
        pos_norm = _normalize_pos(data.pos)
        x_work = torch.cat([x_work, pos_norm], dim=1)

    if fr_cfg["add_degree_feature"]:
        if not hasattr(data, "edge_index") or data.edge_index is None:
            raise AttributeError("Cannot add degree feature because data.edge_index is missing.")
        dst = data.edge_index[1]
        deg = torch.bincount(dst, minlength=x_work.shape[0]).float().unsqueeze(1).to(x_work.device)
        deg = deg / deg.max().clamp_min(1.0)
        x_work = torch.cat([x_work, deg], dim=1)

    summary_indices = [int(i) for i in fr_cfg["neighbor_summary_feature_indices"]]

    if fr_cfg["add_neighbor_mean_features"]:
        mean_feats = _neighbor_mean_features(x_orig, data.edge_index, summary_indices)
        x_work = torch.cat([x_work, mean_feats], dim=1)

    data.x = x_work.float()
    return data