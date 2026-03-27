from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import xy


def read_single_band_raster(raster_path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a single-band raster and return data + metadata."""
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
            }
        )
    return data, meta


def raster_to_flat_table(
    raster_array: np.ndarray,
    transform: Any,
    nodata: float | int | None = None,
) -> dict[str, np.ndarray]:
    """Flatten raster into x, y, value table."""
    rows, cols = np.indices(raster_array.shape)
    xs, ys = xy(transform, rows, cols)
    xs = np.asarray(xs).reshape(-1)
    ys = np.asarray(ys).reshape(-1)
    values = raster_array.reshape(-1)

    if nodata is not None:
        mask = values != nodata
    else:
        mask = ~np.isnan(values) if np.issubdtype(values.dtype, np.floating) else np.ones_like(values, dtype=bool)

    return {
        "x": xs[mask],
        "y": ys[mask],
        "value": values[mask],
    }


def summarize_array(arr: np.ndarray, nodata: float | int | None = None) -> dict[str, float]:
    """Compute basic summary statistics for a raster array."""
    flat = arr.reshape(-1)

    if nodata is not None:
        flat = flat[flat != nodata]

    if np.issubdtype(flat.dtype, np.floating):
        flat = flat[~np.isnan(flat)]

    if flat.size == 0:
        raise ValueError("No valid values found after filtering nodata.")

    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "median": float(np.median(flat)),
        "count": int(flat.size),
    }