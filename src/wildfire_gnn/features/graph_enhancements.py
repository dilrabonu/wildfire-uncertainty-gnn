from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import degree


@dataclass
class TargetTransform:
    name: str = "sqrt_scaled"
    target_max: float = 0.25
    eps: float = 1e-8

    def forward(self, y: Tensor) -> Tensor:
        y = y.clamp(min=0.0, max=self.target_max)
        if self.name == "sqrt_scaled":
            return torch.sqrt((y / self.target_max).clamp(min=0.0) + self.eps)
        if self.name == "identity":
            return y
        raise ValueError(f"Unsupported target transform: {self.name}")

    def inverse(self, y_t: Tensor) -> Tensor:
        if self.name == "sqrt_scaled":
            return (y_t.clamp(min=0.0) ** 2) * self.target_max
        if self.name == "identity":
            return y_t
        raise ValueError(f"Unsupported target transform: {self.name}")


def build_edge_weights(data: Data, sigma: float = 0.15) -> Tensor:
    """
    Build scalar edge weights from normalized node positions.
    Assumes data.pos is in [0, 1] range.
    """
    src, dst = data.edge_index
    pos_src = data.pos[src]
    pos_dst = data.pos[dst]
    dist = torch.norm(pos_src - pos_dst, dim=1)
    weights = torch.exp(-(dist ** 2) / max(sigma ** 2, 1e-8))
    return weights.unsqueeze(-1)


def add_degree_feature(data: Data) -> Data:
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=data.x.dtype)
    deg = deg.unsqueeze(-1)
    data.x = torch.cat([data.x, deg], dim=1)
    return data


def add_neighborhood_statistics(data: Data) -> Data:
    """
    Adds 1-hop neighborhood mean and std for each node feature.
    This gives local context closer to patch-based CNNs.
    """
    x = data.x
    src, dst = data.edge_index
    num_nodes, num_feats = x.shape

    neigh_sum = torch.zeros((num_nodes, num_feats), dtype=x.dtype, device=x.device)
    neigh_sq_sum = torch.zeros((num_nodes, num_feats), dtype=x.dtype, device=x.device)
    counts = torch.zeros((num_nodes, 1), dtype=x.dtype, device=x.device)

    neigh_sum.index_add_(0, dst, x[src])
    neigh_sq_sum.index_add_(0, dst, x[src] ** 2)
    counts.index_add_(0, dst, torch.ones((src.shape[0], 1), dtype=x.dtype, device=x.device))

    counts = counts.clamp_min(1.0)
    neigh_mean = neigh_sum / counts
    neigh_var = (neigh_sq_sum / counts) - (neigh_mean ** 2)
    neigh_std = torch.sqrt(neigh_var.clamp_min(0.0) + 1e-8)
    local_contrast = x - neigh_mean

    data.x = torch.cat([x, neigh_mean, neigh_std, local_contrast], dim=1)
    return data


def prepare_graph_for_gnn(
    data: Data,
    transform_name: str = "sqrt_scaled",
    target_max: float = 0.25,
) -> Tuple[Data, TargetTransform]:
    """
    Main enhancement function before training.
    """
    transformer = TargetTransform(name=transform_name, target_max=target_max)

    data = add_degree_feature(data)
    data = add_neighborhood_statistics(data)
    data.edge_attr = build_edge_weights(data)

    if data.y.dim() == 1:
        data.y = data.y.unsqueeze(-1)

    data.y_transformed = transformer.forward(data.y)
    return data, transformer