from __future__ import annotations

from math import sqrt

import torch


def rebuild_spatial_edges_from_pos(
    data,
    connectivity: int = 8,
    use_distance_weights: bool = True,
):
    """
    Rebuild graph topology from integer raster coordinates in data.pos.

    Assumes:
        data.pos[:, 0] -> row_index
        data.pos[:, 1] -> col_index
    """
    if not hasattr(data, "pos") or data.pos is None:
        raise AttributeError("Graph data must contain data.pos to rebuild topology.")

    pos = data.pos.detach().cpu().long()
    n = pos.shape[0]

    coord_to_idx = {
        (int(pos[i, 0].item()), int(pos[i, 1].item())): i
        for i in range(n)
    }

    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        offsets = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
    else:
        raise ValueError(f"Unsupported connectivity: {connectivity}")

    src_list = []
    dst_list = []
    weight_list = []

    for i in range(n):
        r = int(pos[i, 0].item())
        c = int(pos[i, 1].item())

        for dr, dc in offsets:
            j = coord_to_idx.get((r + dr, c + dc))
            if j is None:
                continue

            src_list.append(i)
            dst_list.append(j)

            if use_distance_weights:
                if abs(dr) + abs(dc) == 1:
                    weight_list.append(1.0)
                else:
                    weight_list.append(1.0 / sqrt(2.0))

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    data.edge_index = edge_index
    if use_distance_weights:
        data.edge_weight = torch.tensor(weight_list, dtype=torch.float32)
    else:
        data.edge_weight = None

    return data