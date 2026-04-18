"""
Unified feature engineering pipeline for the BlazeVeritas wildfire GNN.

Integrates:
  - DEM terrain features (slope, aspect, TWI, elevation) via dem_features.py
  - All raster transforms (log1p, one-hot fuel, interactions, multi-scale,
    gradients, connectivity) via feature_transforms.py
  - NDVI integration (from Sentinel-2 / MODIS if available)
  - Historical fire frequency integration (EFFIS burned area count)
  - Pyrome-level aggregation features
  - Target transformation via QuantileTargetTransformer
  - Pearson correlation report to verify feature quality

Output:
  - df_train_features.parquet
  - df_test_features.parquet
  - target_transformer.pkl
  - feature_names.json
  - correlation_report.csv

Entry point:
    python feature_engineering.py --config feature_config.yaml
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import pearsonr
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

# Local modules
from dem_features import DEMFeatureExtractor
from feature_transforms import build_default_pipeline, QuantileTargetTransformer


# Feature Engineering Master Class


class WildfireFeatureEngineer:
    """
    Orchestrates all feature engineering steps for the wildfire GNN dataset.

    Parameters
    ----------
    config : dict — loaded from feature_config.yaml (see that file for keys)
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_names_: List[str] = []
        self.raster_shape_: Optional[Tuple[int, int]] = None

    # Main public method
    
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Full pipeline. Returns (df_train, df_test) with all features.
        Also saves artefacts to output_dir.
        """
        print("=" * 60)
        print("WILDFIRE FEATURE ENGINEERING PIPELINE")
        print("=" * 60)

        # Step 1 — Load raw data
        df = self._load_raw_data()

        # Step 2 — Infer raster shape
        self.raster_shape_ = self._infer_raster_shape(df)
        print(f"\n[Step 2] Raster grid shape: {self.raster_shape_}")

        # Step 3 — DEM features
        df = self._add_dem_features(df)

        # Step 4 — NDVI (optional)
        df = self._add_ndvi_features(df)

        # Step 5 — Historical fire frequency (optional)
        df = self._add_fire_frequency(df)

        # Step 6 — Pyrome-level aggregation
        df = self._add_pyrome_aggregations(df)

        # Step 7 — Core transform pipeline (log, one-hot, interactions,
        #           multi-scale, gradients, connectivity)
        df, pipeline_feature_names = self._apply_transform_pipeline(df)

        # Step 8 — Spatial train/test split
        df_train, df_test = self._spatial_split(df)

        # Step 9 — Target transformation
        df_train, df_test = self._transform_target(df_train, df_test)

        # Step 10 — Correlation report
        self._correlation_report(df_train)

        # Step 11 — Save everything
        self._save(df_train, df_test)

        print("\n" + "=" * 60)
        print(f"DONE. Features: {len(self.feature_names_)}, "
              f"Train: {len(df_train)}, Test: {len(df_test)}")
        print("=" * 60)

        return df_train, df_test

    # Step 1: Load raw data
    
    def _load_raw_data(self) -> pd.DataFrame:
        raw_path = Path(self.cfg["raw_data_path"])
        print(f"\n[Step 1] Loading raw data from {raw_path}")

        if raw_path.suffix == ".parquet":
            df = pd.read_parquet(raw_path)
        elif raw_path.suffix == ".csv":
            df = pd.read_csv(raw_path)
        elif raw_path.suffix in [".gpkg", ".shp"]:
            gdf = gpd.read_file(raw_path)
            df = pd.DataFrame(gdf.drop(columns="geometry"))
            self._geometry = gdf.geometry
        else:
            raise ValueError(f"Unsupported format: {raw_path.suffix}")

        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"  Original columns: {list(df.columns)}")
        
    # STANDARDIZE RAW COLUMN NAMES
    
        rename_map = {
            "target": "Burn_Prob",
            "Ignition_Prob.img": "Ignition_Prob",
            "CFL.img": "CFL",
            "FSP_Index.img": "FSP_Index",
            "Struct_Exp_Index.img": "Struct_Exp_Index",
            "Fuel_Models.img": "Fuel_Models",
            "row_index": "row",
            "col_index": "col",
        }

        cols_before = set(df.columns)
        applicable_map = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=applicable_map)

        if applicable_map:
            print("  Applied column renaming:")
            for old_name, new_name in applicable_map.items():
                print(f"    {old_name} -> {new_name}")
        else:
            print("  No column renaming applied.")

        print(f"  Standardized columns: {list(df.columns)}")

    # REQUIRED COLUMN CHECK
    
        required = self.cfg.get("required_columns", [])
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns after renaming: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

    # FILL NUMERIC NaNs
   
        feat_cols = [
            c for c in df.columns
            if c not in [self.cfg["target_col"], self.cfg.get("split_col", "")]
        ]

        for c in feat_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                med = df[c].median()
                n_nan = df[c].isna().sum()
                if n_nan > 0:
                    df[c] = df[c].fillna(med)
                    print(f"  Filled {n_nan} NaNs in {c} with median {med:.4f}")

        return df

    # Step 2: Infer raster shape
    
    def _infer_raster_shape(self, df: pd.DataFrame) -> Tuple[int, int]:
        row_col = self.cfg.get("row_col"), self.cfg.get("col_col")
        if row_col[0] and row_col[1] and row_col[0] in df and row_col[1] in df:
            H = int(df[row_col[0]].max()) + 1
            W = int(df[row_col[1]].max()) + 1
            return (H, W)
        else:
            # Try to estimate from coordinates
            n = len(df)
            side = int(np.sqrt(n))
            print(f"  WARNING: row/col columns not found. "
                  f"Estimating shape as ~({side}, {side}).")
            return (side, side)

    # Step 3: DEM features
    
    def _add_dem_features(self, df: pd.DataFrame) -> pd.DataFrame:
        dem_path = self.cfg.get("dem_path")
        if not dem_path:
            print("\n[Step 3] DEM path not configured — skipping.")
            return df

        print(f"\n[Step 3] Extracting DEM features from {dem_path}")
        try:
            extractor = DEMFeatureExtractor(dem_path)
        except FileNotFoundError:
            bbox = self.cfg.get("bbox_greece", (20.0, 37.0, 27.0, 42.0))
            print(f"  DEM not found. Creating synthetic DEM for bbox {bbox}...")
            extractor = DEMFeatureExtractor.create_synthetic_dem(
                output_path=dem_path,
                bbox=bbox,
                resolution_deg=0.001,
            )

        # Extract using geometry or coordinate columns
        x_col = self.cfg.get("lon_col", "lon")
        y_col = self.cfg.get("lat_col", "lat")

        if hasattr(self, "_geometry"):
            gdf_pts = gpd.GeoDataFrame(geometry=self._geometry, crs="EPSG:4326")
            dem_feats = extractor.extract_for_points(gdf_pts)
        elif x_col in df.columns and y_col in df.columns:
            dem_feats = extractor.extract_for_points(df, x_col=x_col, y_col=y_col)
        else:
            print(f"  No coordinates found (tried {x_col}, {y_col}) — skipping DEM.")
            return df

        dem_feats.index = df.index
        df = pd.concat([df, dem_feats], axis=1)
        print(f"  Added columns: {list(dem_feats.columns)}")
        return df

    # Step 4: NDVI
    
    def _add_ndvi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ndvi_path = self.cfg.get("ndvi_path")
        if not ndvi_path or not Path(ndvi_path).exists():
            print(f"\n[Step 4] NDVI raster not found at '{ndvi_path}' — generating proxy.")
            # Proxy: if DEM slope is available, use inverse as vegetation proxy
            if "dem_slope_deg" in df.columns:
                # Steep slopes → less vegetation → lower NDVI
                slope_norm = (df["dem_slope_deg"] - df["dem_slope_deg"].mean()) / (
                    df["dem_slope_deg"].std() + 1e-8)
                df["ndvi_summer"] = np.clip(0.4 - 0.1 * slope_norm, -1, 1)
                print("  Generated NDVI proxy from slope. "
                      "Replace with real Sentinel-2 NDVI for publication.")
            return df

        print(f"\n[Step 4] Loading NDVI from {ndvi_path}")
        # Same pattern as DEM — use DEMFeatureExtractor with the NDVI raster
        ndvi_extractor = DEMFeatureExtractor(ndvi_path)
        x_col = self.cfg.get("lon_col", "lon")
        y_col = self.cfg.get("lat_col", "lat")

        if hasattr(self, "_geometry"):
            gdf_pts = gpd.GeoDataFrame(geometry=self._geometry, crs="EPSG:4326")
            ndvi_raw = ndvi_extractor.extract_for_points(gdf_pts)
        else:
            ndvi_raw = ndvi_extractor.extract_for_points(df, x_col=x_col, y_col=y_col)

        # NDVI raster stores values in band 1 as "elevation"
        df["ndvi_summer"] = ndvi_raw["dem_elevation_m"].values
        print(f"  NDVI range: [{df['ndvi_summer'].min():.3f}, "
              f"{df['ndvi_summer'].max():.3f}]")
        return df

    
    # Step 5: Historical fire frequency
    
    def _add_fire_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        fire_freq_path = self.cfg.get("fire_frequency_path")
        if not fire_freq_path or not Path(fire_freq_path).exists():
            print(f"\n[Step 5] Fire frequency raster not found — skipping.")
            print("  Download from EFFIS: https://effis.jrc.ec.europa.eu/")
            return df

        print(f"\n[Step 5] Loading historical fire frequency from {fire_freq_path}")
        ff_extractor = DEMFeatureExtractor(fire_freq_path)
        x_col = self.cfg.get("lon_col", "lon")
        y_col = self.cfg.get("lat_col", "lat")

        if hasattr(self, "_geometry"):
            gdf_pts = gpd.GeoDataFrame(geometry=self._geometry, crs="EPSG:4326")
            ff_raw = ff_extractor.extract_for_points(gdf_pts)
        else:
            ff_raw = ff_extractor.extract_for_points(df, x_col=x_col, y_col=y_col)

        df["fire_freq_20yr"] = np.clip(ff_raw["dem_elevation_m"].values, 0, 20)
        print(f"  Fire frequency: mean {df['fire_freq_20yr'].mean():.2f} "
              f"burns over 20 years")
        return df

    # Step 6: Pyrome-level aggregation
   
    def _add_pyrome_aggregations(self, df: pd.DataFrame) -> pd.DataFrame:
        pyrome_col = self.cfg.get("pyrome_col", "Pyrome_ID")
        target_col = self.cfg["target_col"]

        if pyrome_col not in df.columns:
            print(f"\n[Step 6] Pyrome column '{pyrome_col}' not found — skipping.")
            return df

        print(f"\n[Step 6] Computing pyrome-level aggregations...")

        agg_cols = [target_col, "Ignition_Prob", "CFL"]
        agg_cols = [c for c in agg_cols if c in df.columns]

        pyrome_agg = df.groupby(pyrome_col)[agg_cols].agg(["mean", "std", "max"]).reset_index()
        pyrome_agg.columns = [pyrome_col] + [
            f"pyrome_{col}_{stat}"
            for col, stat in pyrome_agg.columns[1:]
        ]

        df = df.merge(pyrome_agg, on=pyrome_col, how="left")

        # Residual target: how much riskier than pyrome mean?
        pyrome_mean_col = f"pyrome_{target_col}_mean"
        if pyrome_mean_col in df.columns:
            df["target_residual_from_pyrome"] = (
                df[target_col] - df[pyrome_mean_col]
            )
            print(f"  Added target_residual_from_pyrome "
                  f"(novel framing from diagnosis doc)")

        new_cols = [c for c in df.columns if c.startswith("pyrome_")]
        print(f"  Added {len(new_cols)} pyrome features: {new_cols[:4]} ...")
        return df

    # Step 7: Transform pipeline
   
    def _apply_transform_pipeline(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        print(f"\n[Step 7] Applying transform pipeline...")
        target_col = self.cfg["target_col"]
        split_col  = self.cfg.get("split_col")
        row_col    = self.cfg.get("row_col")
        col_col    = self.cfg.get("col_col")
        pyrome_col = self.cfg.get("pyrome_col")

        # Columns to exclude from transforms
        exclude_cols = [target_col] + [
            c for c in [split_col, row_col, col_col, pyrome_col, "target_residual_from_pyrome"]
            if c and c in df.columns
        ]

        df_features = df.drop(columns=exclude_cols)

        # Pixel indices for raster-based transforms
        if row_col and col_col and row_col in df and col_col in df:
            H, W = self.raster_shape_
            pixel_indices = (df[row_col].values * W + df[col_col].values).astype(int)
        else:
            pixel_indices = None

        pipeline = build_default_pipeline(raster_shape=self.raster_shape_)
        df_transformed, new_cols = pipeline.fit_transform(
            df_features,
            raster_shape=self.raster_shape_,
            pixel_indices=pixel_indices,
        )

        # Re-attach excluded columns
        for c in exclude_cols:
            if c in df.columns:
                df_transformed[c] = df[c].values

        # Store feature names (exclude non-feature cols)
        self.feature_names_ = [
            c for c in df_transformed.columns
            if c not in exclude_cols and df_transformed[c].dtype in [
                np.float32, np.float64, np.float16, float, np.int32, np.int64, int
            ]
        ]
        print(f"  Total features after pipeline: {len(self.feature_names_)}")
        return df_transformed, new_cols

    # Step 8: Spatial split
   
    def _spatial_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print(f"\n[Step 8] Spatial train/test split...")
        split_col  = self.cfg.get("split_col")
        pyrome_col = self.cfg.get("pyrome_col", "Pyrome_ID")
        test_size  = self.cfg.get("test_size", 0.2)
        seed       = self.cfg.get("random_seed", 42)

        if split_col and split_col in df.columns:
            # Pre-defined split column (e.g. "split" = "train"/"test")
            df_train = df[df[split_col] == "train"].copy()
            df_test  = df[df[split_col] == "test"].copy()
            print(f"  Using pre-defined split: train={len(df_train)}, test={len(df_test)}")
        elif pyrome_col in df.columns:
            # Spatially disjoint: split on pyrome groups
            pyromes = df[pyrome_col].unique()
            rng = np.random.default_rng(seed)
            n_test = max(1, int(len(pyromes) * test_size))
            test_pyromes = rng.choice(pyromes, n_test, replace=False)
            mask_test = df[pyrome_col].isin(test_pyromes)
            df_train = df[~mask_test].copy()
            df_test  = df[mask_test].copy()
            print(f"  Pyrome-split: {len(pyromes)-n_test} train pyromes, "
                  f"{n_test} test pyromes")
            print(f"  Rows: train={len(df_train)}, test={len(df_test)}")
        else:
            # Fallback: geographic block split using row/col
            row_col = self.cfg.get("row_col", "row")
            if row_col in df.columns:
                H, _ = self.raster_shape_
                cutoff = int(H * (1 - test_size))
                mask_test = df[row_col] >= cutoff
            else:
                # Random as last resort
                mask_test = np.zeros(len(df), dtype=bool)
                test_idx = np.random.default_rng(seed).choice(
                    len(df), int(len(df) * test_size), replace=False)
                mask_test[test_idx] = True
            df_train = df[~mask_test].copy()
            df_test  = df[mask_test].copy()
            print(f"  Geographic block split: train={len(df_train)}, test={len(df_test)}")

        return df_train.reset_index(drop=True), df_test.reset_index(drop=True)

  
    # Step 9: Target transformation
   
    def _transform_target(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print(f"\n[Step 9] Quantile target transformation...")
        target_col = self.cfg["target_col"]
        qt = QuantileTargetTransformer()

        y_train = df_train[target_col].values
        y_test  = df_test[target_col].values

        y_train_t = qt.fit_transform(y_train)
        y_test_t  = qt.transform(y_test)

        df_train[f"{target_col}_raw"] = y_train
        df_train[target_col]          = y_train_t
        df_test[f"{target_col}_raw"]  = y_test
        df_test[target_col]           = y_test_t

        # Save transformer
        qt_path = self.output_dir / "target_transformer.pkl"
        with open(qt_path, "wb") as f:
            pickle.dump(qt, f)

        print(f"  Target transformed. "
              f"Train mean: {y_train.mean():.4f} → {y_train_t.mean():.4f}")
        print(f"  QuantileTransformer saved: {qt_path}")
        return df_train, df_test
    # Step 10: Correlation report
   
    def _correlation_report(self, df_train: pd.DataFrame):
        print(f"\n[Step 10] Pearson correlation report...")
        target_col = self.cfg["target_col"]
        if target_col not in df_train.columns:
            return

        y = df_train[target_col].values
        records = []
        for col in self.feature_names_:
            if col not in df_train.columns:
                continue
            x = df_train[col].values
            try:
                r, p = pearsonr(x, y)
            except Exception:
                r, p = 0.0, 1.0
            records.append({"feature": col, "pearson_r": r, "p_value": p})

        corr_df = pd.DataFrame(records).sort_values("pearson_r", key=abs, ascending=False)
        report_path = self.output_dir / "correlation_report.csv"
        corr_df.to_csv(report_path, index=False)

        print(f"\n  Top 15 features by |Pearson r| with {target_col}:")
        print(corr_df.head(15).to_string(index=False))
        print(f"\n  Full report saved: {report_path}")

        # Diagnosis: flag if top correlation < 0.15
        top_r = corr_df["pearson_r"].abs().max()
        if top_r < 0.15:
            print("\n    WARNING: Max |r| < 0.15 — features may not be discriminating.")
            print("     Check DEM extraction, NDVI, and raster alignment.")
        else:
            print(f"\n   Max |r| = {top_r:.3f} — feature space looks healthy.")

        # Step 11: Save
   

    def _save(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        print(f"\n[Step 11] Saving outputs to {self.output_dir}...")

        df_train.to_parquet(self.output_dir / "df_train_features.parquet", index=False)
        df_test.to_parquet(self.output_dir / "df_test_features.parquet",  index=False)

        with open(self.output_dir / "feature_names.json", "w") as f:
            json.dump(self.feature_names_, f, indent=2)

        print(f"  Saved: df_train_features.parquet ({len(df_train)} rows)")
        print(f"  Saved: df_test_features.parquet  ({len(df_test)} rows)")
        print(f"  Saved: feature_names.json ({len(self.feature_names_)} features)")

# CLI entry point

def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Wildfire Feature Engineering Pipeline")
    parser.add_argument(
        "--config", type=str, default="feature_config.yaml",
        help="Path to feature_config.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    engineer = WildfireFeatureEngineer(config)
    engineer.run()


if __name__ == "__main__":
    main()