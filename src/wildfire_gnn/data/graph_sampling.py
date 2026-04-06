from __future__ import annotations

import numpy as np
import torch


def rebalance_train_mask_only(
    data,
    bins: list[float],
    max_train_samples_per_bin: int,
    random_seed: int = 42,
):
    """
    Rebalance ONLY the training nodes.
    Validation and test masks remain untouched.
    This avoids leakage into model selection or final evaluation.
    """
    rng = np.random.default_rng(random_seed)

    train_mask_np = data.train_mask.cpu().numpy()
    y_np = data.y.cpu().numpy().reshape(-1)

    train_idx = np.where(train_mask_np)[0]
    y_train = y_np[train_idx]

    keep_idx = []

    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]

        if i < len(bins) - 2:
            bin_mask = (y_train >= lo) & (y_train < hi)
        else:
            bin_mask = (y_train >= lo) & (y_train <= hi)

        bin_idx = train_idx[bin_mask]

        if len(bin_idx) <= max_train_samples_per_bin:
            keep_idx.extend(bin_idx.tolist())
        else:
            sampled = rng.choice(bin_idx, size=max_train_samples_per_bin, replace=False)
            keep_idx.extend(sampled.tolist())

    keep_idx = np.array(sorted(keep_idx), dtype=np.int64)

    new_train_mask = torch.zeros_like(data.train_mask)
    new_train_mask[torch.as_tensor(keep_idx, dtype=torch.long)] = True
    data.train_mask = new_train_mask

    return data