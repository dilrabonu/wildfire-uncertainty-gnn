from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import xy


FLOAT32_NODATA_SENTINEL = np.float32(-3.4028235e38)


def read_single_band_raster(
    raster_path: str | Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a single-band raster and return data plus metadata.

    Args:
        raster_path: Path to the raster file.

    Returns:
        Tuple of raster array and metadata dictionary.
    """
    path = Path(raster_path)
    with rasterio.open(path) as src:
        data = src.read(1)
        meta = src.meta.copy()
        meta.update(
            {
                "bounds": src.bounds,
                "crs": src.crs,
                "transform": src.transform,
                "nodata": src.nodata,
                "width": src.width,
                "height": src.height,
                "dtype": str(data.dtype),
            }
        )
    return data, meta


def build_valid_data_mask(
    arr: np.ndarray,
    nodata: float | int | None = None,
) -> np.ndarray:
    """Build a robust validity mask for geospatial rasters.

    Handles:
    - explicit nodata values
    - NaNs
    - infinities
    - common float32 nodata sentinel values used in GIS rasters

    Args:
        arr: Input array.
        nodata: Optional nodata value from raster metadata.

    Returns:
        Boolean mask where True means valid data.
    """
    mask = np.ones(arr.shape, dtype=bool)

    if np.issubdtype(arr.dtype, np.floating):
        mask &= np.isfinite(arr)

        # Common GIS float nodata sentinel
        mask &= arr > (FLOAT32_NODATA_SENTINEL / 10)

    if nodata is not None:
        if np.issubdtype(arr.dtype, np.floating):
            mask &= ~np.isclose(arr, nodata, rtol=0.0, atol=1e-6)
        else:
            mask &= arr != nodata

    return mask


def raster_to_flat_table(
    raster_array: np.ndarray,
    transform: Any,
    nodata: float | int | None = None,
) -> dict[str, np.ndarray]:
    """Flatten raster into x, y, value arrays for valid cells only."""
    rows, cols = np.indices(raster_array.shape)
    xs, ys = xy(transform, rows, cols)
    xs = np.asarray(xs).reshape(-1)
    ys = np.asarray(ys).reshape(-1)
    values = raster_array.reshape(-1)

    valid_mask = build_valid_data_mask(raster_array, nodata=nodata).reshape(-1)

    return {
        "x": xs[valid_mask],
        "y": ys[valid_mask],
        "value": values[valid_mask],
    }


def summarize_array(
    arr: np.ndarray,
    nodata: float | int | None = None,
) -> dict[str, float | int]:
    """Compute basic summary statistics for valid raster values only.

    Args:
        arr: Raster array.
        nodata: Optional nodata value.

    Returns:
        Dictionary of summary statistics.

    Raises:
        ValueError: If no valid values remain after filtering.
    """
    valid_mask = build_valid_data_mask(arr, nodata=nodata)
    flat = arr[valid_mask]

    if flat.size == 0:
        raise ValueError("No valid values found after filtering nodata and invalid values.")

    flat = flat.astype(np.float64, copy=False)

    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "median": float(np.median(flat)),
        "count": int(flat.size),
    }