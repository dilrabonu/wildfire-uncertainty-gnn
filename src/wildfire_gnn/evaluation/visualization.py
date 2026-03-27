from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_raster(
    array: np.ndarray,
    title: str,
    cmap: str = "viridis",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot a raster array.

    Args:
        array: 2D raster array.
        title: Plot title.
        cmap: Colormap.
        save_path: Optional save path.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(array, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.axis("off")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_histogram(
    values: np.ndarray,
    title: str,
    bins: int = 100,
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot histogram of values."""
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()