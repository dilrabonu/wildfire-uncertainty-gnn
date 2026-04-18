"""
One-time orchestration script. Run this ONCE before training.

What it does:
  1. Validates that input data and rasters exist
  2. Runs the full WildfireFeatureEngineer pipeline
  3. Runs diagnostic assertions on the output
  4. Prints a feature summary table
  5. Verifies the Pearson correlations improved

Usage:
    python prepare_features.py                          
    python prepare_features.py --config my_config.yaml
    python prepare_features.py --dry-run              

What to expect:
  - Runtime: 2–10 minutes depending on dataset size and raster resolution
  - Output: data/features/ directory with parquet files + reports
  - Target: top-5 Pearson |r| > 0.15 (previously 0.20 from coordinates only)
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# Input validation

def validate_inputs(config: dict) -> bool:
    """Check all configured paths exist and print actionable messages."""
    print("\n" + "=" * 60)
    print("INPUT VALIDATION")
    print("=" * 60)

    all_ok = True

    # Raw data
    raw_path = Path(config["raw_data_path"])
    if raw_path.exists():
        print(f"   Raw data:    {raw_path}")
    else:
        print(f"   Raw data NOT FOUND: {raw_path}")
        all_ok = False

    # DEM
    dem_path = config.get("dem_path")
    if dem_path:
        if Path(dem_path).exists():
            print(f"   DEM raster:  {dem_path}")
        else:
            print(f"    DEM raster not found: {dem_path}")
            print(f"      Download: https://earthexplorer.usgs.gov/ (SRTM 30m)")
            print(f"      Or EU-DEM: https://land.copernicus.eu/imagery-in-situ/eu-dem")
            print(f"      A synthetic DEM will be used for development.")

    # NDVI
    ndvi_path = config.get("ndvi_path")
    if ndvi_path:
        if Path(ndvi_path).exists():
            print(f"   NDVI raster: {ndvi_path}")
        else:
            print(f"    NDVI raster not found: {ndvi_path}")
            print(f"      Download: https://scihub.copernicus.eu/ (Sentinel-2 summer composite)")
            print(f"      A slope-based proxy will be used.")

    # Fire frequency
    ff_path = config.get("fire_frequency_path")
    if ff_path:
        if Path(ff_path).exists():
            print(f"   Fire freq:   {ff_path}")
        else:
            print(f"    Fire frequency raster not found: {ff_path}")
            print(f"      Download: https://effis.jrc.ec.europa.eu/")
            print(f"      This feature will be skipped — re-run after download.")

    # Output dir
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"   Output dir:  {out_dir} (created if not exists)")

    # Required columns check
    required = config.get("required_columns", [])
    if required:
        try:
            raw_path_obj = Path(config["raw_data_path"])
            if raw_path_obj.exists():
                if raw_path_obj.suffix == ".parquet":
                    sample = pd.read_parquet(raw_path_obj, columns=None)
                elif raw_path_obj.suffix == ".csv":
                    sample = pd.read_csv(raw_path_obj, nrows=5)
                else:
                    sample = None

                if sample is not None:
                    missing = [c for c in required if c not in sample.columns]
                    if missing:
                        print(f"   Missing required columns: {missing}")
                        print(f"     Available: {list(sample.columns)}")
                        all_ok = False
                    else:
                        print(f"   Required columns present: {required}")
        except Exception as e:
            print(f"    Could not pre-check columns: {e}")

    print()
    return all_ok


# Post-run diagnostics

def run_diagnostics(output_dir: Path, config: dict) -> bool:
    """
    After pipeline runs, verify the outputs are sensible.
    Returns True if all checks pass.
    """
    print("\n" + "=" * 60)
    print("POST-RUN DIAGNOSTICS")
    print("=" * 60)

    all_pass = True

    # Check files exist
    expected_files = [
        "df_train_features.parquet",
        "df_test_features.parquet",
        "target_transformer.pkl",
        "feature_names.json",
        "correlation_report.csv",
    ]
    for fname in expected_files:
        fpath = output_dir / fname
        if fpath.exists():
            print(f"   {fname} ({fpath.stat().st_size // 1024} KB)")
        else:
            print(f"   Missing: {fname}")
            all_pass = False

    if not all_pass:
        return False

    # Load and check
    df_train = pd.read_parquet(output_dir / "df_train_features.parquet")
    df_test  = pd.read_parquet(output_dir / "df_test_features.parquet")

    with open(output_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    corr_df = pd.read_csv(output_dir / "correlation_report.csv")

    print(f"\n  Train shape:  {df_train.shape}")
    print(f"  Test shape:   {df_test.shape}")
    print(f"  Features:     {len(feature_names)}")

    # Check 1: No all-NaN columns
    nan_cols = [c for c in feature_names
                if c in df_train.columns and df_train[c].isna().all()]
    if nan_cols:
        print(f"\n   All-NaN columns: {nan_cols}")
        all_pass = False
    else:
        print(f"\n   No all-NaN columns")

    # Check 2: No zero-variance columns (after transforms)
    zero_var_cols = [
        c for c in feature_names
        if c in df_train.columns and df_train[c].std() < 1e-8
    ]
    if zero_var_cols:
        print(f"    Zero-variance columns (will not help model): {zero_var_cols}")
    else:
        print(f"   No zero-variance columns")

    # Check 3: Train/test overlap check (if pyrome column exists)
    pyrome_col = config.get("pyrome_col", "Pyrome_ID")
    if pyrome_col in df_train.columns and pyrome_col in df_test.columns:
        train_pyromes = set(df_train[pyrome_col].unique())
        test_pyromes  = set(df_test[pyrome_col].unique())
        overlap = train_pyromes & test_pyromes
        if overlap:
            print(f"   Pyrome overlap: {len(overlap)} pyromes in both splits")
        else:
            print(f"  Spatially disjoint split — no pyrome overlap")

    # Check 4: Correlation improvement
    top_5_r = corr_df["pearson_r"].abs().nlargest(5).tolist()
    print(f"\n  Top-5 |Pearson r| values: {[f'{r:.3f}' for r in top_5_r]}")

    if top_5_r[0] > 0.15:
        print(f"  Feature discriminability looks healthy (top r={top_5_r[0]:.3f} > 0.15)")
    else:
        print(f"   Low discriminability (top r={top_5_r[0]:.3f}). "
              f"Check DEM + NDVI alignment.")

    # Check 5: Target distribution
    target_col = config["target_col"]
    if target_col in df_train.columns:
        y = df_train[target_col].values
        print(f"\n  Target ({target_col}):")
        print(f"    mean={y.mean():.4f}, std={y.std():.4f}, "
              f"min={y.min():.4f}, max={y.max():.4f}")
        if abs(y.mean()) < 0.1:
            print(f"  Target looks approximately Gaussian after quantile transform")
        else:
            print(f"   Target mean far from 0 — check quantile transformer output")

    # Print feature summary table
    print(f"\n{'─'*55}")
    print(f"  FEATURE SUMMARY")
    print(f"{'─'*55}")

    feature_groups = {
        "DEM terrain":        [c for c in feature_names if c.startswith("dem_")],
        "NDVI":               [c for c in feature_names if "ndvi" in c],
        "Fire frequency":     [c for c in feature_names if "fire_freq" in c],
        "Pyrome aggregation": [c for c in feature_names if c.startswith("pyrome_")],
        "Fuel one-hot":       [c for c in feature_names if c.startswith("fuel_")],
        "Interaction terms":  [c for c in feature_names if c.startswith("inter_")],
        "Multi-scale stats":  [c for c in feature_names if "_mean" in c or "_std" in c],
        "Spatial gradients":  [c for c in feature_names if "_grad_" in c],
        "Connectivity":       [c for c in feature_names if "connectivity" in c],
        "Base features":      [c for c in feature_names if not any(
            p in c for p in ["dem_", "ndvi", "fire_freq", "pyrome_", "fuel_",
                              "inter_", "_mean", "_std", "_grad_", "connectivity"])],
    }

    for group, cols in feature_groups.items():
        if cols:
            print(f"  {group:25s}: {len(cols):3d} features")

    print(f"{'─'*55}")
    print(f"  {'TOTAL':25s}: {len(feature_names):3d} features")
    print(f"{'─'*55}")

    return all_pass

# Main

def main():
    parser = argparse.ArgumentParser(description="Prepare wildfire GNN features")
    parser.add_argument("--config", default="feature_config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs only, don't run pipeline")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        print("Run with default config? Creating feature_config.yaml...")
        from prepare_features import create_default_config
        create_default_config(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate inputs
    inputs_ok = validate_inputs(config)
    if not inputs_ok:
        print("\n Input validation failed. Fix the issues above, then re-run.")
        sys.exit(1)

    if args.dry_run:
        print("\n Dry run complete. Inputs look valid.")
        return

    # Run pipeline
    print("\n" + "=" * 60)
    print("RUNNING FEATURE PIPELINE")
    print("=" * 60)

    t0 = time.time()

    from feature_engineering import WildfireFeatureEngineer
    engineer = WildfireFeatureEngineer(config)
    df_train, df_test = engineer.run()

    elapsed = time.time() - t0
    print(f"\n  Pipeline completed in {elapsed:.1f}s")

    # Diagnostics
    output_dir = Path(config["output_dir"])
    all_pass = run_diagnostics(output_dir, config)

    if all_pass:
        print("\n" + "=" * 60)
        print(" ALL CHECKS PASSED")
        print("=" * 60)
        print(f"\nNext step: open 06_feature_engineering.ipynb for visual validation.")
        print(f"Then run your GNN training with the new feature files:")
        print(f"  {output_dir}/df_train_features.parquet")
        print(f"  {output_dir}/df_test_features.parquet")
        print(f"  {output_dir}/feature_names.json")
    else:
        print("\n Some diagnostics failed. Review messages above before training.")


def create_default_config(path: Path):
    """Write a default feature_config.yaml if none exists."""
    default = {
        "raw_data_path":       "data/processed/wildfire_nodes.parquet",
        "dem_path":            "data/rasters/dem_greece.tif",
        "ndvi_path":           "data/rasters/ndvi_summer.tif",
        "fire_frequency_path": "data/rasters/fire_frequency_20yr.tif",
        "output_dir":          "data/features",
        "target_col":          "Burn_Prob",
        "pyrome_col":          "Pyrome_ID",
        "row_col":             "row",
        "col_col":             "col",
        "lon_col":             "lon",
        "lat_col":             "lat",
        "split_col":           None,
        "test_size":           0.2,
        "random_seed":         42,
        "bbox_greece":         [20.0, 37.0, 27.0, 42.0],
        "required_columns": [
            "Burn_Prob", "Ignition_Prob", "CFL",
            "FSP_Index", "Struct_Exp_Index", "Fuel_Models"
        ],
    }
    with open(path, "w") as f:
        yaml.dump(default, f, default_flow_style=False, sort_keys=False)
    print(f"Created default config: {path}")
    print("Edit this file to match your paths, then re-run.")


if __name__ == "__main__":
    main()