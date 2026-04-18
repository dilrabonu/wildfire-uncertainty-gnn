import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class DEMFeatureExtractor:
    """
    Extract slope, aspect, elevation, and TWI from DEM raster.
    Handles both geographic (degrees) and projected (metres) CRS correctly.
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
            elevation_raw = src.read(1).astype(np.float32)
            self.nodata = src.nodata

            # ── FIX: CRS-aware resolution ──────────────────────────────────────
            # src.transform.a = pixel width in CRS units
            # If CRS is geographic (degrees), multiply by ~111,000 m/degree
            # so np.gradient produces elevation-per-meter, not elevation-per-degree
            res_raw = abs(src.transform.a)
            if src.crs and src.crs.is_geographic:
                # 1° ≈ 111,000 m at equator. This approximation is fine for Greece
                # (lat 37-42°): error < 10% vs true arc length, negligible for slope.
                self.resolution_m = res_raw * 111_000.0
            else:
                # Already in metres (e.g. EPSG:2100 Greek Grid)
                self.resolution_m = res_raw
            
        if self.nodata is not None:
            mask = elevation_raw == self.nodata
        else:
            mask = ~np.isfinite(elevation_raw)

        fill_val = (
            nodata_fill
            if nodata_fill is not None
            else float(np.nanmedian(elevation_raw[~mask]))
        )
        elevation_raw[mask] = fill_val
        self.elevation = elevation_raw

        self._compute_derivatives()
        print(
            f"[DEMFeatureExtractor] Loaded DEM: {elevation_raw.shape}, "
            f"resolution ~{self.resolution_m:.1f}m, CRS: {self.crs}"
        )

    def _compute_derivatives(self):
        """Compute slope (°), aspect (sin/cos), TWI from elevation array."""
        elev = self.elevation
        # Guard against zero resolution (should never happen after CRS fix)
        res = max(float(self.resolution_m), 1.0)

        # Central-difference gradients in m/m
        dz_dy, dz_dx = np.gradient(elev, res, res)

        # Slope in degrees
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        self.slope_deg = np.degrees(slope_rad).astype(np.float32)

        # Aspect: 0° = North, clockwise. Encoded as sin/cos to avoid discontinuity.
        aspect_rad = np.arctan2(-dz_dy, dz_dx)
        aspect_deg = (np.degrees(aspect_rad) + 360) % 360
        aspect_rad_norm = np.radians(aspect_deg)
        self.aspect_sin = np.sin(aspect_rad_norm).astype(np.float32)
        self.aspect_cos = np.cos(aspect_rad_norm).astype(np.float32)

        # Topographic Wetness Index: ln(A / tan(slope))
        # Using local approximation: upslope area ≈ one cell
        slope_rad_safe = np.where(slope_rad < 0.001, 0.001, slope_rad)
        upslope_area = res**2  # 1-cell approximation
        self.twi = np.log(upslope_area / np.tan(slope_rad_safe)).astype(np.float32)

        print(
            f"[DEMFeatureExtractor] Slope: {self.slope_deg.min():.1f}–"
            f"{self.slope_deg.max():.1f}°  "
            f"(mean {self.slope_deg.mean():.1f}°), "
            f"TWI: {self.twi.min():.2f}–{self.twi.max():.2f}"
        )

        # Sanity check: if mean slope is still >80°, fix didn't work
        if self.slope_deg.mean() > 80:
            print(
                "[DEMFeatureExtractor] WARNING: mean slope > 80° — "
                "CRS resolution fix may not have taken effect. "
                f"resolution_m used = {self.resolution_m:.2f}"
            )

    def extract_for_points(
        self,
        points: gpd.GeoDataFrame | pd.DataFrame,
        x_col: str = None,
        y_col: str = None,
        points_crs=None,
    ) -> pd.DataFrame:
        """
        Extract DEM-derived features at point locations.

        Parameters
        ----------
        points     : GeoDataFrame (geometry column) or plain DataFrame.
        x_col      : easting / longitude column name (plain DataFrame only).
        y_col      : northing / latitude column name (plain DataFrame only).
        points_crs : CRS of the input coordinates. If different from DEM CRS,
                     coordinates are reprojected automatically.
        """
        if isinstance(points, gpd.GeoDataFrame):
            pts = points.to_crs(self.crs)
            xs = pts.geometry.x.values
            ys = pts.geometry.y.values
        else:
            if x_col is None or y_col is None:
                raise ValueError("Provide x_col and y_col for plain DataFrames.")

            if points_crs is not None and str(points_crs) != str(self.crs):
                gdf = gpd.GeoDataFrame(
                    points.copy(),
                    geometry=gpd.points_from_xy(points[x_col], points[y_col]),
                    crs=points_crs,
                ).to_crs(self.crs)
                xs = gdf.geometry.x.values
                ys = gdf.geometry.y.values
            else:
                xs = points[x_col].values
                ys = points[y_col].values

        rows, cols = self._coords_to_pixels(xs, ys)

        h, w = self.elevation.shape
        rows = np.clip(rows, 0, h - 1)
        cols = np.clip(cols, 0, w - 1)

        result = pd.DataFrame(
            {
                "dem_elevation_m": self.elevation[rows, cols],
                "dem_slope_deg":   self.slope_deg[rows, cols],
                "dem_aspect_sin":  self.aspect_sin[rows, cols],
                "dem_aspect_cos":  self.aspect_cos[rows, cols],
                "dem_twi":         self.twi[rows, cols],
            }
        )

        nan_frac = result.isna().mean().max()
        if nan_frac > 0.05:
            print(
                f"[DEMFeatureExtractor] WARNING: {nan_frac:.1%} NaN in DEM features "
                "— check CRS alignment."
            )
        else:
            print(
                f"[DEMFeatureExtractor] Extracted {len(result)} rows. "
                f"Slope mean: {result['dem_slope_deg'].mean():.2f}°"
            )
            # Validation hint
            if result["dem_slope_deg"].mean() > 80:
                print(
                    "[DEMFeatureExtractor] WARNING: slope mean still > 80° after extraction. "
                    "Verify that x_col/y_col are in the same CRS as the DEM or that "
                    "points_crs was passed correctly."
                )

        return result.reset_index(drop=True)

    def _coords_to_pixels(self, xs: np.ndarray, ys: np.ndarray):
        with rasterio.open(self.dem_path) as src:
            rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        return np.array(rows), np.array(cols)

    @staticmethod
    def create_synthetic_dem(
        output_path: str | Path,
        bbox: tuple,
        resolution_deg: float = 0.001,
        crs: str = "EPSG:4326",
        seed: int = 42,
    ) -> "DEMFeatureExtractor":
        """Synthetic DEM for development when real SRTM is unavailable."""
        from rasterio.transform import from_bounds

        rng = np.random.default_rng(seed)
        min_x, min_y, max_x, max_y = bbox

        n_cols = int((max_x - min_x) / resolution_deg)
        n_rows = int((max_y - min_y) / resolution_deg)

        def smooth_noise(r, c, scale):
            x = np.linspace(0, scale, c)
            y = np.linspace(0, scale, r)
            xx, yy = np.meshgrid(x, y)
            return np.sin(xx) * np.cos(yy)

        elevation = (
            500 * smooth_noise(n_rows, n_cols, 4)
            + 200 * smooth_noise(n_rows, n_cols, 8)
            + 50  * smooth_noise(n_rows, n_cols, 16)
            + rng.standard_normal((n_rows, n_cols)) * 20
        )
        elevation = (
            (elevation - elevation.min())
            / (elevation.max() - elevation.min())
            * 1500
        )

        transform = from_bounds(min_x, min_y, max_x, max_y, n_cols, n_rows)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(
            output_path, "w", driver="GTiff",
            height=n_rows, width=n_cols, count=1,
            dtype=np.float32, crs=crs, transform=transform,
        ) as dst:
            dst.write(elevation.astype(np.float32), 1)

        print(
            f"[DEMFeatureExtractor] Synthetic DEM saved: {output_path} "
            f"({n_rows}×{n_cols})"
        )
        return DEMFeatureExtractor(output_path)