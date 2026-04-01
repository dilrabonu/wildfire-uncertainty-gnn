from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset


def _read_raster(path: str | Path) -> tuple[np.ndarray, Any, dict]:
    """Read a single-band raster as float32."""
    path = Path(path)
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        profile = src.profile.copy()
    return arr, nodata, profile


def load_aligned_rasters(
    feature_raster_paths: list[str | Path],
    target_raster_path: str | Path,
    target_min: float = 0.0,
    target_max: float = 1.0,
    standardize_continuous_channels: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Load aligned feature rasters and target raster.

    Returns:
        x_stack: np.ndarray of shape [C, H, W]
        y: np.ndarray of shape [H, W]
        channel_stats_df: DataFrame with normalization stats
        valid_mask: boolean mask of valid target pixels
    """
    target, target_nodata, target_profile = _read_raster(target_raster_path)
    target = np.clip(target, target_min, target_max)

    valid_mask = np.isfinite(target)
    if target_nodata is not None:
        valid_mask &= target != target_nodata

    feature_arrays: list[np.ndarray] = []
    rows = []

    for path in feature_raster_paths:
        arr, nodata, profile = _read_raster(path)

        if arr.shape != target.shape:
            raise ValueError(f"Feature raster shape mismatch: {path} -> {arr.shape}, target -> {target.shape}")

        feature_name = Path(path).name
        current_valid = np.isfinite(arr)
        if nodata is not None:
            current_valid &= arr != nodata
        valid_mask &= current_valid

        feature_arrays.append(arr)

    x_stack = np.stack(feature_arrays, axis=0).astype(np.float32)

    # Standardize continuous channels only.
    # Here we treat Fuel_Models as categorical-like, so we only standardize first 4 channels.
    if standardize_continuous_channels:
        for ch in range(x_stack.shape[0]):
            feature_name = Path(feature_raster_paths[ch]).name
            is_fuel = feature_name.lower().startswith("fuel_models")

            values = x_stack[ch][valid_mask]
            mean = float(values.mean())
            std = float(values.std()) if float(values.std()) > 1e-8 else 1.0

            if not is_fuel:
                x_stack[ch] = (x_stack[ch] - mean) / std
            else:
                # Light scaling for categorical-like raster
                min_v = float(values.min())
                max_v = float(values.max())
                denom = max(max_v - min_v, 1.0)
                x_stack[ch] = (x_stack[ch] - min_v) / denom

            rows.append(
                {
                    "feature": feature_name,
                    "mean_or_min": mean if not is_fuel else min_v,
                    "std_or_range": std if not is_fuel else denom,
                    "mode": "zscore" if not is_fuel else "minmax",
                }
            )
    else:
        for path in feature_raster_paths:
            rows.append(
                {
                    "feature": Path(path).name,
                    "mean_or_min": np.nan,
                    "std_or_range": np.nan,
                    "mode": "none",
                }
            )

    channel_stats_df = pd.DataFrame(rows)
    return x_stack, target.astype(np.float32), channel_stats_df, valid_mask


def build_patch_metadata(
    target: np.ndarray,
    valid_mask: np.ndarray,
    patch_size: int,
) -> pd.DataFrame:
    """Build metadata table for valid patch centers."""
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd")

    radius = patch_size // 2
    h, w = target.shape

    rows = []
    valid_positions = np.argwhere(valid_mask)

    for r, c in valid_positions:
        if r < radius or r >= h - radius:
            continue
        if c < radius or c >= w - radius:
            continue

        rows.append(
            {
                "row_index": int(r),
                "col_index": int(c),
                "target": float(target[r, c]),
            }
        )

    return pd.DataFrame(rows)


def save_patch_metadata(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save patch metadata to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


class RasterPatchDataset(Dataset):
    """Patch dataset for CNN regression."""

    def __init__(
        self,
        x_stack: np.ndarray,
        target: np.ndarray,
        metadata: pd.DataFrame,
        patch_size: int,
    ) -> None:
        self.x_stack = x_stack
        self.target = target
        self.metadata = metadata.reset_index(drop=True).copy()
        self.patch_size = patch_size
        self.radius = patch_size // 2

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.metadata.iloc[index]
        r = int(row["row_index"])
        c = int(row["col_index"])

        patch = self.x_stack[
            :,
            r - self.radius:r + self.radius + 1,
            c - self.radius:c + self.radius + 1,
        ]
        y = np.float32(row["target"])

        return {
            "x": torch.tensor(patch, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "row_index": torch.tensor(r, dtype=torch.long),
            "col_index": torch.tensor(c, dtype=torch.long),
        }