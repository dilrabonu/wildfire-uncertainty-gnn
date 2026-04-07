from __future__ import annotations

import copy
from typing import Iterable

import torch
from torch_geometric.data import Data


def intervene_reduce_feature(
    data: Data,
    feature_index: int,
    node_indices: Iterable[int],
    reduction_factor: float,
) -> Data:
    """
    Example:
    reduction_factor=0.30 means reduce feature by 30% on selected nodes.
    """
    data_new = copy.deepcopy(data)
    node_indices = torch.as_tensor(list(node_indices), dtype=torch.long)
    data_new.x[node_indices, feature_index] = data_new.x[node_indices, feature_index] * (1.0 - reduction_factor)
    return data_new


def intervene_firebreak(
    data: Data,
    blocked_node_indices: Iterable[int],
) -> Data:
    """
    Removes edges touching selected nodes to simulate a hard firebreak barrier.
    """
    data_new = copy.deepcopy(data)
    blocked = set(int(i) for i in blocked_node_indices)

    src = data_new.edge_index[0].cpu().numpy()
    dst = data_new.edge_index[1].cpu().numpy()

    keep = [(i not in blocked and j not in blocked) for i, j in zip(src, dst)]
    keep = torch.as_tensor(keep, dtype=torch.bool)

    data_new.edge_index = data_new.edge_index[:, keep]
    if getattr(data_new, "edge_attr", None) is not None:
        data_new.edge_attr = data_new.edge_attr[keep]

    return data_new