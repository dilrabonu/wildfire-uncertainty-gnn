from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    delta: float = 0.02,
) -> torch.Tensor:
    loss = F.huber_loss(pred, target, delta=delta, reduction="none")
    return (loss * weights).mean()


def weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    loss = (pred - target) ** 2
    return (loss * weights).mean()


def gaussian_nll_loss(
    mean: torch.Tensor,
    var: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return F.gaussian_nll_loss(mean, target, var, full=True)


def build_target_weights(
    target_raw: torch.Tensor,
    bin_edges: list[float],
    bin_weights: list[float],
) -> torch.Tensor:
    """
    Build weights from RAW target values, not transformed target.
    target_raw shape: [N, 1]
    """
    assert len(bin_edges) >= 2
    assert len(bin_weights) == len(bin_edges) - 1

    weights = torch.ones_like(target_raw)

    for i in range(len(bin_weights)):
        low = bin_edges[i]
        high = bin_edges[i + 1]
        mask = (target_raw >= low) & (target_raw < high)
        weights[mask] = bin_weights[i]

    last_mask = target_raw >= bin_edges[-1]
    if last_mask.any():
        weights[last_mask] = bin_weights[-1]

    return weights