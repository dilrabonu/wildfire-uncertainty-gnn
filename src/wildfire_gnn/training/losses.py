from __future__ import annotations

import torch
import torch.nn as nn


class GaussianNLLLossStable(nn.Module):
    def __init__(self, min_variance: float = 1e-6):
        super().__init__()
        self.min_variance = min_variance

    def forward(self, mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        var = torch.exp(log_var).clamp_min(self.min_variance)
        loss = 0.5 * (torch.log(var) + ((target - mean) ** 2) / var)
        return loss.mean()