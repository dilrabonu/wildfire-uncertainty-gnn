from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_tail_weights(y_true: Tensor, power: float = 1.5, max_weight: float = 6.0) -> Tensor:
    """
    Higher target values receive larger weight.
    Assumes transformed target is in [0, 1] approximately.
    """
    y = y_true.detach().clamp(min=0.0)
    weights = 1.0 + (y ** power) * (max_weight - 1.0)
    return weights


def weighted_huber_loss(
    y_pred: Tensor,
    y_true: Tensor,
    delta: float = 0.02,
    power: float = 1.5,
    max_weight: float = 6.0,
) -> Tensor:
    weights = compute_tail_weights(y_true, power=power, max_weight=max_weight)
    loss = F.huber_loss(y_pred, y_true, delta=delta, reduction="none")
    return (weights * loss).mean()


def gaussian_nll_loss(
    mean: Tensor,
    var: Tensor,
    y_true: Tensor,
    power: float = 1.5,
    max_weight: float = 6.0,
) -> Tensor:
    weights = compute_tail_weights(y_true, power=power, max_weight=max_weight)
    nll = 0.5 * (torch.log(var) + ((y_true - mean) ** 2) / var)
    return (weights * nll).mean()


def hybrid_gaussian_nll(
    mean: Tensor,
    var: Tensor,
    y_true: Tensor,
    delta: float = 0.02,
    power: float = 1.5,
    max_weight: float = 6.0,
    alpha: float = 0.8,
) -> Tensor:
    nll = gaussian_nll_loss(mean, var, y_true, power=power, max_weight=max_weight)
    huber = weighted_huber_loss(mean, y_true, delta=delta, power=power, max_weight=max_weight)
    return alpha * nll + (1.0 - alpha) * huber