from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


FLOAT32_NODATA_SENTINEL = np.float32(-3.4028235e38)


def read_single_band_raster(
    raster_path: str | Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a single-band raster and return data plus metadata."""
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
    """Build a robust validity mask for geospatial rasters."""
    mask = np.ones(arr.shape, dtype=bool)

    if np.issubdtype(arr.dtype, np.floating):
        mask &= np.isfinite(arr)
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
    from rasterio.transform import xy

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
    """Compute basic summary statistics for valid raster values only."""
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


def align_raster_to_reference(
    src_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    resampling: Resampling,
) -> None:
    """
    Align a raster to the reference raster grid and save it.

    Args:
        src_path: Source raster path.
        reference_path: Reference raster path, usually Burn_Prob.img.
        output_path: Output aligned raster path.
        resampling: Rasterio resampling method.
    """
    src_path = Path(src_path)
    reference_path = Path(reference_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(reference_path) as ref:
        ref_meta = ref.meta.copy()
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height
        ref_nodata = ref.nodata

    with rasterio.open(src_path) as src:
        src_arr = src.read(1)

        # destination array uses float32 for continuous rasters
        if resampling == Resampling.nearest and np.issubdtype(src_arr.dtype, np.integer):
            dst_dtype = src_arr.dtype
        else:
            dst_dtype = np.float32

        dst_arr = np.full(
            (ref_height, ref_width),
            ref_nodata if ref_nodata is not None else 0,
            dtype=dst_dtype,
        )

        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=ref_nodata,
            resampling=resampling,
        )

    out_meta = ref_meta.copy()
    out_meta.update(
        {
            "driver": "HFA",   # creates .img
            "count": 1,
            "dtype": str(dst_arr.dtype),
            "nodata": ref_nodata,
        }
    )

    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(dst_arr, 1)


def align_feature_stack_to_reference(
    raw_dir: str | Path,
    output_dir: str | Path,
    reference_filename: str = "Burn_Prob.img",
) -> list[Path]:
    """
    Align all required wildfire rasters to the reference grid and save them.

    Resampling strategy:
    - bilinear for continuous rasters
    - nearest for categorical rasters (Fuel_Models)

    Returns:
        List of saved aligned raster paths.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_path = raw_dir / reference_filename
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference raster not found: {reference_path}")

    raster_specs = {
        "Burn_Prob.img": Resampling.bilinear,
        "CFL.img": Resampling.bilinear,
        "FSP_Index.img": Resampling.bilinear,
        "Ignition_Prob.img": Resampling.bilinear,
        "Struct_Exp_Index.img": Resampling.bilinear,
        "Fuel_Models.img": Resampling.nearest,
    }

    saved_paths: list[Path] = []

    for filename, method in raster_specs.items():
        src_path = raw_dir / filename
        if not src_path.exists():
            raise FileNotFoundError(f"Missing required raster: {src_path}")

        out_path = output_dir / filename
        align_raster_to_reference(
            src_path=src_path,
            reference_path=reference_path,
            output_path=out_path,
            resampling=method,
        )
        saved_paths.append(out_path)

    return saved_paths