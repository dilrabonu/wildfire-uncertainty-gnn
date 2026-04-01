from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_random_split(
    df: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    random_seed: int,
) -> dict[str, np.ndarray]:
    """Create random train/val/test splits."""
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    indices = np.arange(len(df))

    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_size,
        random_state=random_seed,
        shuffle=True,
    )

    relative_val_size = val_size / (val_size + test_size)

    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=relative_val_size,
        random_state=random_seed,
        shuffle=True,
    )

    return {
        "train_idx": np.sort(train_idx),
        "val_idx": np.sort(val_idx),
        "test_idx": np.sort(test_idx),
    }


def make_spatial_block_split(
    df: pd.DataFrame,
    n_row_blocks: int,
    n_col_blocks: int,
    train_blocks: float,
    val_blocks: float,
    test_blocks: float,
    random_seed: int,
) -> dict[str, np.ndarray]:
    """Create spatial block train/val/test split using center coordinates."""
    if not np.isclose(train_blocks + val_blocks + test_blocks, 1.0):
        raise ValueError("train_blocks + val_blocks + test_blocks must sum to 1.0")

    df_local = df.copy()

    row_bins = np.linspace(df_local["row_index"].min(), df_local["row_index"].max() + 1, n_row_blocks + 1)
    col_bins = np.linspace(df_local["col_index"].min(), df_local["col_index"].max() + 1, n_col_blocks + 1)

    df_local["row_block"] = pd.cut(
        df_local["row_index"],
        bins=row_bins,
        labels=False,
        include_lowest=True,
        right=False,
    )
    df_local["col_block"] = pd.cut(
        df_local["col_index"],
        bins=col_bins,
        labels=False,
        include_lowest=True,
        right=False,
    )

    df_local["spatial_block"] = (
        df_local["row_block"].astype(str) + "_" + df_local["col_block"].astype(str)
    )

    unique_blocks = np.array(sorted(df_local["spatial_block"].dropna().unique()))
    rng = np.random.default_rng(random_seed)
    rng.shuffle(unique_blocks)

    n_blocks = len(unique_blocks)
    n_train = int(round(train_blocks * n_blocks))
    n_val = int(round(val_blocks * n_blocks))
    n_test = n_blocks - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError("Too few blocks for requested proportions.")

    train_block_ids = set(unique_blocks[:n_train])
    val_block_ids = set(unique_blocks[n_train:n_train + n_val])
    test_block_ids = set(unique_blocks[n_train + n_val:])

    train_idx = df_local.index[df_local["spatial_block"].isin(train_block_ids)].to_numpy()
    val_idx = df_local.index[df_local["spatial_block"].isin(val_block_ids)].to_numpy()
    test_idx = df_local.index[df_local["spatial_block"].isin(test_block_ids)].to_numpy()

    return {
        "train_idx": np.sort(train_idx),
        "val_idx": np.sort(val_idx),
        "test_idx": np.sort(test_idx),
    }


def save_splits(splits: dict[str, np.ndarray], output_path: str | Path) -> None:
    """Save split arrays to NPZ."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **splits)