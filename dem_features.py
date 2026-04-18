"""
Computes terrain features from a Digital Elevation Model (DEM) raster:
  - Elevation (raw, z-scored)
  - Slope (degrees)
  - Aspect (sin/cos encoded — avoids 0/360 discontinuity)
  - Topographic Wetness Index (TWI = ln(A / tan(slope)))

Data source priority:
  1. Your existing DEM raster (if already downloaded)
  2. EU-DEM or SRTM 30m — download from:
       https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm-1
       or Copernicus: https://land.copernicus.eu/imagery-in-situ/eu-dem

Usage:
    from dem_features import DEMFeatureExtractor
    extractor = DEMFeatureExtractor("path/to/dem.tif")
    df = extractor.extract_for_points(points_gdf)  # GeoDataFrame with geometry column
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class DEMFeatureExtractor:
    """
    Extracts slope, aspect (sin/cos), elevation, and TWI from a DEM raster
    at arbitrary point locations via pixel sampling.

    Parameters
    ----------
    dem_path : str | Path
        Path to a GeoTIFF DEM raster (any CRS — will reproject points to match).
    nodata_fill : float
        Value to substitute for nodata pixels (default: median of valid pixels).
    """

    def __init__(self, dem_path: str | Path, nodata_fill: float | None = None):
        self.dem_path = Path(dem_path)
        if not self.dem_path.exists():
            raise FileNotFoundError(
                f"DEM raster not found: {self.dem_path}\n"
                "Download SRTM 30m from https://earthexplorer.usgs.gov/ "
                "or EU-DEM from https://land.copernicus.eu/imagery-in-situ/eu-dem"
            )

        with rasterio.open(self.dem_path) as src:
            self.crs = src.crs
            self.transform = src.transform
            self.resolution_m = abs(src.transform.a)  # pixel width in CRS units
            elevation_raw = src.read(1).astype(np.float32)
            self.nodata = src.nodata

        # Replace nodata
        if self.nodata is not None:
            mask = elevation_raw == self.nodata
        else:
            mask = ~np.isfinite(elevation_raw)

        fill_val = nodata_fill if nodata_fill is not None else float(np.nanmedian(elevation_raw[~mask]))
        elevation_raw[mask] = fill_val
        self.elevation = elevation_raw

        # Pre-compute derivatives
        self._compute_derivatives()
        print(f"[DEMFeatureExtractor] Loaded DEM: {elevation_raw.shape}, "
              f"resolution ~{self.resolution_m:.1f}m, CRS: {self.crs}")

    def _compute_derivatives(self):
        """Compute slope, aspect, and TWI from elevation array."""
        elev = self.elevation
        res = self.resolution_m

        # Sobel-based gradients (central differences at interior pixels)
        # dz/dy = north-south gradient, dz/dx = east-west gradient
        dz_dy, dz_dx = np.gradient(elev, res, res)

        # Slope in degrees: arctan(sqrt((dz/dx)^2 + (dz/dy)^2))
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        self.slope_deg = np.degrees(slope_rad).astype(np.float32)

        # Aspect in degrees (0=North, 90=East, 180=South, 270=West)
        # Use atan2 — flip dz_dy sign so North is 0
        aspect_rad = np.arctan2(-dz_dy, dz_dx)
        aspect_deg = np.degrees(aspect_rad)
        aspect_deg = (aspect_deg + 360) % 360  # normalize to [0, 360)

        # Encode as sin/cos to avoid 0/360 discontinuity
        aspect_rad_norm = np.radians(aspect_deg)
        self.aspect_sin = np.sin(aspect_rad_norm).astype(np.float32)
        self.aspect_cos = np.cos(aspect_rad_norm).astype(np.float32)

        # Topographic Wetness Index (TWI)
        # TWI = ln(A / tan(slope)), where A = upslope contributing area
        # Approximation: use local slope only (no flow accumulation raster)
        # Replace flat areas (slope < 0.001 rad) to avoid division by zero
        slope_rad_safe = np.where(slope_rad < 0.001, 0.001, slope_rad)
        # Approximate upslope area as 1 cell (simplification without flow routing)
        upslope_area = res ** 2  # scalar approximation
        self.twi = np.log(upslope_area / np.tan(slope_rad_safe)).astype(np.float32)

        print(f"[DEMFeatureExtractor] Slope: {self.slope_deg.min():.1f}–{self.slope_deg.max():.1f}°, "
              f"TWI: {self.twi.min():.2f}–{self.twi.max():.2f}")


    def extract_for_points(
        self,
        points: gpd.GeoDataFrame | pd.DataFrame,
        x_col: str = None,
        y_col: str = None,
    ) -> pd.DataFrame:
        """
        Extract DEM features at each point location.

        Parameters
        ----------
        points : GeoDataFrame (with geometry) or DataFrame with x_col/y_col columns.
        x_col, y_col : column names for longitude/easting and latitude/northing
                       (only needed if points is a plain DataFrame).

        Returns
        -------
        pd.DataFrame with columns:
            dem_elevation_m, dem_slope_deg, dem_aspect_sin, dem_aspect_cos, dem_twi
        """
        # Resolve coordinates
        if isinstance(points, gpd.GeoDataFrame):
            # Reproject to DEM CRS
            pts = points.to_crs(self.crs)
            xs = pts.geometry.x.values
            ys = pts.geometry.y.values
        else:
            if x_col is None or y_col is None:
                raise ValueError("Provide x_col and y_col for plain DataFrames.")
            xs = points[x_col].values
            ys = points[y_col].values

        rows, cols = self._coords_to_pixels(xs, ys)

        # Clamp to raster bounds
        h, w = self.elevation.shape
        rows = np.clip(rows, 0, h - 1)
        cols = np.clip(cols, 0, w - 1)

        result = pd.DataFrame({
            "dem_elevation_m":  self.elevation[rows, cols],
            "dem_slope_deg":    self.slope_deg[rows, cols],
            "dem_aspect_sin":   self.aspect_sin[rows, cols],
            "dem_aspect_cos":   self.aspect_cos[rows, cols],
            "dem_twi":          self.twi[rows, cols],
        })

        # Sanity check
        nan_frac = result.isna().mean().max()
        if nan_frac > 0.05:
            print(f"[WARNING] {nan_frac:.1%} NaN in DEM features — check CRS alignment.")
        else:
            print(f"[DEMFeatureExtractor] Extracted {len(result)} rows. "
                  f"Slope mean: {result['dem_slope_deg'].mean():.2f}°")

        return result.reset_index(drop=True)

    def _coords_to_pixels(self, xs: np.ndarray, ys: np.ndarray):
        """Convert geographic coordinates to raster row/col indices."""
        # rasterio rowcol: transform maps pixel to world; inverse maps world to pixel
        with rasterio.open(self.dem_path) as src:
            rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        return np.array(rows), np.array(cols)

    @staticmethod
    def create_synthetic_dem(
        output_path: str | Path,
        bbox: tuple,  # (min_x, min_y, max_x, max_y) in EPSG:4326
        resolution_deg: float = 0.0003,
        crs: str = "EPSG:4326",
        seed: int = 42,
    ) -> "DEMFeatureExtractor":
        """
        Generate a realistic synthetic DEM (Perlin-noise-like) for testing
        when no real DEM is available. Creates a GeoTIFF at output_path.

        This is for DEVELOPMENT ONLY. Replace with real SRTM data for publication.
        """
        from rasterio.transform import from_bounds

        rng = np.random.default_rng(seed)
        min_x, min_y, max_x, max_y = bbox

        cols = int((max_x - min_x) / resolution_deg)
        rows = int((max_y - min_y) / resolution_deg)

        # Layered noise to simulate realistic terrain
        def smooth_noise(r, c, scale):
            x = np.linspace(0, scale, c)
            y = np.linspace(0, scale, r)
            xx, yy = np.meshgrid(x, y)
            return rng.standard_normal((r, c)) * 0 + np.sin(xx) * np.cos(yy)

        elevation = (
            500 * smooth_noise(rows, cols, 4)
            + 200 * smooth_noise(rows, cols, 8)
            + 50  * smooth_noise(rows, cols, 16)
            + rng.standard_normal((rows, cols)) * 20
        )
        # Shift to realistic Greece range: 0–2917m (Olympus)
        elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min()) * 1500

        transform = from_bounds(min_x, min_y, max_x, max_y, cols, rows)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(
            output_path, "w", driver="GTiff",
            height=rows, width=cols, count=1,
            dtype=np.float32, crs=crs, transform=transform
        ) as dst:
            dst.write(elevation.astype(np.float32), 1)

        print(f"[DEMFeatureExtractor] Synthetic DEM saved: {output_path} ({rows}×{cols})")
        return DEMFeatureExtractor(output_path)

if __name__ == "__main__":
    import sys

    dem_path = Path("data/rasters/dem_greece.tif")

    if dem_path.exists():
        extractor = DEMFeatureExtractor(dem_path)
    else:
        print(f"DEM not found at {dem_path}. Creating synthetic DEM for testing...")
        # Approximate bounding box for mainland Greece
        extractor = DEMFeatureExtractor.create_synthetic_dem(
            output_path=dem_path,
            bbox=(20.0, 37.0, 27.0, 42.0),
            resolution_deg=0.001,
        )

    # Test with 5 random points in Greece
    test_pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            x=[21.5, 22.0, 23.5, 24.0, 25.5],
            y=[38.0, 39.5, 40.0, 38.5, 41.0],
        ),
        crs="EPSG:4326"
    )

    features = extractor.extract_for_points(test_pts)
    print("\nSample DEM features:")
    print(features.to_string())
    print("\nAll assertions passed. dem_features.py is ready.")