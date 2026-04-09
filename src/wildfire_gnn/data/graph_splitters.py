from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


def make_random_node_split(
    node_df: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    random_seed: int,
) -> dict[str, np.ndarray]:
    """Create random train/val/test node splits."""
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    indices = np.arange(len(node_df))

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


def make_spatial_block_node_split(
    node_df: pd.DataFrame,
    n_row_blocks: int,
    n_col_blocks: int,
    train_blocks: float,
    val_blocks: float,
    test_blocks: float,
    random_seed: int,
) -> dict[str, np.ndarray]:
    """Create spatial block split for graph nodes using row/col coordinates."""
    if not np.isclose(train_blocks + val_blocks + test_blocks, 1.0):
        raise ValueError("train_blocks + val_blocks + test_blocks must sum to 1.0")

    df_local = node_df.copy()

    row_bins = np.linspace(
        df_local["row_index"].min(),
        df_local["row_index"].max() + 1,
        n_row_blocks + 1,
    )
    col_bins = np.linspace(
        df_local["col_index"].min(),
        df_local["col_index"].max() + 1,
        n_col_blocks + 1,
    )

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


def load_splits(path: str | Path) -> dict[str, np.ndarray]:
    """Load split arrays from NPZ."""
    arr = np.load(path)
    return {
        "train_idx": arr["train_idx"],
        "val_idx": arr["val_idx"],
        "test_idx": arr["test_idx"],
    }


def attach_masks_to_graph(
    data: Data,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Data:
    """Attach boolean train/val/test masks to a PyG Data object."""
    n = data.num_nodes
    data.train_mask = torch.zeros(n, dtype=torch.bool)
    data.val_mask = torch.zeros(n, dtype=torch.bool)
    data.test_mask = torch.zeros(n, dtype=torch.bool)

    data.train_mask[torch.as_tensor(train_idx, dtype=torch.long)] = True
    data.val_mask[torch.as_tensor(val_idx, dtype=torch.long)] = True
    data.test_mask[torch.as_tensor(test_idx, dtype=torch.long)] = True
    return data


def attach_masks_from_split_file(data: Data, split_path: str | Path) -> Data:
    """Load split arrays from NPZ and attach boolean masks to graph."""
    splits = load_splits(split_path)
    return attach_masks_to_graph(
        data=data,
        train_idx=splits["train_idx"],
        val_idx=splits["val_idx"],
        test_idx=splits["test_idx"],
    )


def print_mask_summary(data: Data) -> None:
    """Print train/val/test mask summary."""
    print("Mask summary:")
    print("train:", int(data.train_mask.sum()))
    print("val  :", int(data.val_mask.sum()))
    print("test :", int(data.test_mask.sum()))
    print("total:", int(data.train_mask.sum() + data.val_mask.sum() + data.test_mask.sum()))
    print("nodes:", int(data.num_nodes))