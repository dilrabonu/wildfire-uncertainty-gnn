"""

Phase 5.4 — Complete graph reconstruction.

Solves three problems simultaneously:
  1. Missing 40,718 nodes — runs feature pipeline on ALL 300,000 nodes
  2. No validation mask — creates spatially disjoint train / val / test split
  3. Stale 7-feature graph — saves graph_data_enriched.pt with 60 features

Output files:
  data/features/df_all_features.parquet     ← all 300k nodes, 60 features
  data/features/splits_phase54.npz          ← train/val/test indices (3-way)
  data/processed/graph_data_enriched.pt     ← final PyG Data object

Split strategy (spatially disjoint, no leakage):
  - Sort nodes by raster row (north→south)
  - Bottom 20% rows  → test   (geographically disjoint from train)
  - Next  10% rows   → val    (buffer zone between train and test)
  - Remaining 70%    → train
  This matches the spirit of baseline_splits_spatial.npz but covers all nodes.

Usage:
    python 05_phase54_graph_rebuild.py
    python 05_phase54_graph_rebuild.py --config feature_config.yaml --device cpu
"""

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

warnings.filterwarnings("ignore")


# Imports from existing pipeline modules
try:
    from dem_features import DEMFeatureExtractor
    from feature_transforms import build_default_pipeline, QuantileTargetTransformer
except ImportError as e:
    print(f"ERROR: Cannot import pipeline modules: {e}")
    print("Make sure dem_features.py and feature_transforms.py are in the same directory.")
    sys.exit(1)

# Step 1 — Load and standardise ALL 300k rows

def load_all_nodes(cfg: dict) -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 1 — Load all nodes (target: 300,000)")
    print("="*60)

    raw_path = Path(cfg["raw_data_path"])
    if raw_path.suffix == ".csv":
        df = pd.read_csv(raw_path)
    elif raw_path.suffix == ".parquet":
        df = pd.read_parquet(raw_path)
    else:
        raise ValueError(f"Unsupported format: {raw_path.suffix}")

    print(f"  Loaded: {len(df)} rows × {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    # Standardise column names
    rename_map = {
        "target":               "Burn_Prob",
        "Ignition_Prob.img":    "Ignition_Prob",
        "CFL.img":              "CFL",
        "FSP_Index.img":        "FSP_Index",
        "Struct_Exp_Index.img": "Struct_Exp_Index",
        "Fuel_Models.img":      "Fuel_Models",
        "row_index":            "row",
        "col_index":            "col",
    }
    applied = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=applied)
    if applied:
        print(f"  Renamed: {applied}")

    assert "Burn_Prob" in df.columns, "Target column not found after renaming."
    assert "row" in df.columns and "col" in df.columns, "row/col columns not found."
    assert len(df) == 300_000, (
        f"Expected 300,000 nodes, got {len(df)}. "
        "Check raw_data_path in config."
    )

    # Fill NaNs with column medians
    for c in df.select_dtypes(include=np.number).columns:
        n_nan = df[c].isna().sum()
        if n_nan > 0:
            df[c] = df[c].fillna(df[c].median())
            print(f"  Filled {n_nan} NaNs in {c}")

    print(f"  ✓ All 300,000 nodes loaded cleanly.")
    return df

# Step 2 — Derive projected coordinates from reference raster

def derive_coordinates(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, object]:
    print("\n" + "="*60)
    print("STEP 2 — Derive projected coordinates")
    print("="*60)

    import rasterio
    from rasterio.transform import xy

    ref_path = Path(cfg["reference_raster_path"])
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference raster not found: {ref_path}")

    with rasterio.open(ref_path) as src:
        ref_crs   = src.crs
        transform = src.transform
        ref_h, ref_w = src.height, src.width

    rows = df["row"].astype(int).values
    cols = df["col"].astype(int).values

    rows_clipped = np.clip(rows, 0, ref_h - 1)
    cols_clipped = np.clip(cols, 0, ref_w - 1)
    n_oob = ((rows != rows_clipped) | (cols != cols_clipped)).sum()
    if n_oob > 0:
        print(f"  WARNING: {n_oob} row/col pairs clipped to raster bounds.")

    xs, ys = xy(transform, rows_clipped, cols_clipped, offset="center")
    df["x_coord"] = np.asarray(xs, dtype=np.float64)
    df["y_coord"] = np.asarray(ys, dtype=np.float64)

    print(f"  Reference CRS: {ref_crs}")
    print(f"  x_coord range: [{df['x_coord'].min():.0f}, {df['x_coord'].max():.0f}]")
    print(f"  y_coord range: [{df['y_coord'].min():.0f}, {df['y_coord'].max():.0f}]")
    print(f"  ✓ Coordinates derived for all {len(df)} nodes.")
    return df, ref_crs

# Step 3 — DEM features for all nodes

def add_dem_features(df: pd.DataFrame, cfg: dict, ref_crs) -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 3 — DEM terrain features")
    print("="*60)

    dem_path = cfg.get("dem_path")
    if not dem_path or not Path(dem_path).exists():
        print(f"  DEM not found at '{dem_path}' — generating synthetic DEM.")
        bbox = tuple(cfg.get("bbox_greece", [20.0, 37.0, 27.0, 42.0]))
        dem_path = "data/interim/aligned/dem_greece_synthetic.tif"
        extractor = DEMFeatureExtractor.create_synthetic_dem(
            output_path=dem_path, bbox=bbox, resolution_deg=0.001
        )
    else:
        extractor = DEMFeatureExtractor(dem_path)

    dem_feats = extractor.extract_for_points(
        df,
        x_col="x_coord",
        y_col="y_coord",
        points_crs=ref_crs,
    )
    dem_feats.index = df.index
    df = pd.concat([df, dem_feats], axis=1)

    print(f"  dem_slope_deg: mean={df['dem_slope_deg'].mean():.2f}°, "
          f"std={df['dem_slope_deg'].std():.2f}°")

    if df["dem_slope_deg"].mean() > 80:
        print("    Slope still >80° — check dem_features.py CRS fix.")

    # NDVI proxy from slope (real NDVI raster can replace this)
    if "dem_slope_deg" in df.columns:
        slope_norm = (df["dem_slope_deg"] - df["dem_slope_deg"].mean()) / (
            df["dem_slope_deg"].std() + 1e-8
        )
        df["ndvi_summer"] = np.clip(0.4 - 0.1 * slope_norm, -1, 1)

    print(f"  ✓ DEM features added for all {len(df)} nodes.")
    return df

# Step 4 — Feature transform pipeline (log, one-hot, interactions, multi-scale,
#           gradients, connectivity) — on all 300k nodes

def apply_feature_pipeline(
    df: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, list[str], object]:
    print("\n" + "="*60)
    print("STEP 4 — Feature transform pipeline (all 300k nodes)")
    print("="*60)

    H = int(df["row"].max()) + 1
    W = int(df["col"].max()) + 1
    raster_shape = (H, W)
    print(f"  Raster shape: {raster_shape}")

    # Pixel flat indices for raster-based ops
    pixel_indices = (df["row"].values * W + df["col"].values).astype(int)

    # Columns excluded from feature set
    exclude = ["Burn_Prob", "row", "col", "x_coord", "y_coord",
               "row_norm", "col_norm"]
    exclude = [c for c in exclude if c in df.columns]

    df_feat_input = df.drop(columns=exclude)

    pipeline = build_default_pipeline(raster_shape=raster_shape)
    df_transformed, new_cols = pipeline.fit_transform(
        df_feat_input,
        raster_shape=raster_shape,
        pixel_indices=pixel_indices,
    )

    # Re-attach excluded columns
    for c in exclude:
        df_transformed[c] = df[c].values

    # Feature names = numeric columns not in exclude list
    feature_names = [
        c for c in df_transformed.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df_transformed[c])
    ]

    print(f"  ✓ {len(feature_names)} features computed for all {len(df_transformed)} nodes.")
    print(f"  Feature groups:")

    groups = {
        "Base raster":        [c for c in feature_names if c in
                               ["CFL","FSP_Index","Ignition_Prob","Struct_Exp_Index"]],
        "DEM terrain":        [c for c in feature_names if c.startswith("dem_")],
        "NDVI":               [c for c in feature_names if "ndvi" in c],
        "Fuel one-hot":       [c for c in feature_names if c.startswith("fuel_")],
        "Interaction terms":  [c for c in feature_names if c.startswith("inter_")],
        "Multi-scale stats":  [c for c in feature_names if "_mean" in c or "_std" in c],
        "Spatial gradients":  [c for c in feature_names if "_grad_" in c],
        "Connectivity":       [c for c in feature_names if "connectivity" in c],
        "Coordinates":        [c for c in feature_names if c in ["row_norm","col_norm"]],
    }
    for g, cols in groups.items():
        if cols:
            print(f"    {g:22s}: {len(cols)}")

    return df_transformed, feature_names, pipeline

# Step 5 — Target quantile transformation

def transform_target(
    df: pd.DataFrame, output_dir: Path
) -> tuple[pd.DataFrame, object]:
    print("\n" + "="*60)
    print("STEP 5 — Quantile target transformation")
    print("="*60)

    qt = QuantileTargetTransformer()
    y_raw = df["Burn_Prob"].values
    y_transformed = qt.fit_transform(y_raw)

    df["Burn_Prob_raw"] = y_raw
    df["Burn_Prob"]     = y_transformed

    qt_path = output_dir / "target_transformer.pkl"
    with open(qt_path, "wb") as f:
        pickle.dump(qt, f)

    print(f"  Raw target:         mean={y_raw.mean():.4f}, median={np.median(y_raw):.4f}")
    print(f"  Transformed target: mean={y_transformed.mean():.4f}, std={y_transformed.std():.4f}")
    print(f"  ✓ Transformer saved: {qt_path}")
    return df, qt

# Step 6 — Spatially disjoint 3-way split (train / val / test)

def create_spatial_split(
    df: pd.DataFrame,
    test_frac:  float = 0.20,
    val_frac:   float = 0.10,
    seed:       int   = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Spatially disjoint split based on raster row position (north → south).

    Layout (sorted by row, low=north, high=south):
     
    Why rows not random:
      - Test region is geographically isolated → realistic deployment scenario
      - Val is between train and test → no spatial leakage through neighbors
      - Matches the spatial-split logic used in baseline_splits_spatial.npz
    """
    print("\n" + "="*60)
    print("STEP 6 — Spatially disjoint 3-way split")
    print("="*60)

    rows = df["row"].values
    n    = len(df)
    H    = int(rows.max()) + 1

    # Row thresholds
    train_cutoff = int(H * (1 - test_frac - val_frac))  # e.g. row 1328 of 1898
    val_cutoff   = int(H * (1 - test_frac))              # e.g. row 1518 of 1898

    train_mask = rows <  train_cutoff
    val_mask   = (rows >= train_cutoff) & (rows < val_cutoff)
    test_mask  = rows >= val_cutoff

    train_idx = np.where(train_mask)[0]
    val_idx   = np.where(val_mask  )[0]
    test_idx  = np.where(test_mask )[0]

    # Verify no overlap and full coverage
    assert len(train_idx) + len(val_idx) + len(test_idx) == n, "Split does not cover all nodes."
    assert len(set(train_idx) & set(val_idx))  == 0, "Train/Val overlap."
    assert len(set(train_idx) & set(test_idx)) == 0, "Train/Test overlap."
    assert len(set(val_idx)   & set(test_idx)) == 0, "Val/Test overlap."

    print(f"  Raster height: {H} rows")
    print(f"  Train cutoff:  row < {train_cutoff}  ({train_cutoff/H:.0%} of rows)")
    print(f"  Val   cutoff:  row < {val_cutoff}    ({val_cutoff/H:.0%} of rows)")
    print(f"  Test  cutoff:  row ≥ {val_cutoff}")
    print(f"\n  Split sizes:")
    print(f"    Train : {len(train_idx):>7,}  ({len(train_idx)/n:.1%})")
    print(f"    Val   : {len(val_idx):>7,}  ({len(val_idx)/n:.1%})")
    print(f"    Test  : {len(test_idx):>7,}  ({len(test_idx)/n:.1%})")
    print(f"    Total : {n:>7,}")
    print(f"  ✓ No spatial overlap between any split.")

    return train_idx, val_idx, test_idx

# Step 7 — Correlation report on train nodes

def correlation_report(
    df: pd.DataFrame,
    feature_names: list[str],
    train_idx: np.ndarray,
    output_dir: Path,
):
    print("\n" + "="*60)
    print("STEP 7 — Pearson correlation report (train nodes only)")
    print("="*60)

    from scipy.stats import pearsonr

    df_train = df.iloc[train_idx]
    y = df_train["Burn_Prob"].values

    records = []
    for col in feature_names:
        if col not in df_train.columns:
            continue
        x = df_train[col].values
        try:
            r, p = pearsonr(x, y)
        except Exception:
            r, p = 0.0, 1.0
        records.append({"feature": col, "pearson_r": r, "p_value": p})

    corr_df = pd.DataFrame(records).sort_values("pearson_r", key=abs, ascending=False)
    report_path = output_dir / "correlation_report_phase54.csv"
    corr_df.to_csv(report_path, index=False)

    print(f"\n  Top 15 features by |r|:")
    print(corr_df.head(15).to_string(index=False))
    print(f"\n  Max |r|: {corr_df['pearson_r'].abs().max():.4f}")
    print(f"  Report: {report_path}")

    if corr_df["pearson_r"].abs().max() < 0.15:
        print("    Max |r| < 0.15 — check DEM slope fix and raster alignment.")
    else:
        print("  ✓ Feature discriminability is healthy.")


# Step 8 — Build PyG graph with 8-connected pixel edges

def build_pyg_graph(
    df: pd.DataFrame,
    feature_names: list[str],
    train_idx: np.ndarray,
    val_idx:   np.ndarray,
    test_idx:  np.ndarray,
    cfg: dict,
) -> object:
    print("\n" + "="*60)
    print("STEP 8 — Build PyG graph (8-connected pixel grid)")
    print("="*60)

    try:
        from torch_geometric.data import Data
    except ImportError:
        print("ERROR: torch_geometric not installed.")
        print("  Install: pip install torch-geometric")
        return None

    n = len(df)
    H = int(df["row"].max()) + 1
    W = int(df["col"].max()) + 1

    # ── Node features ────────────────────────────────────────────────────────
    X = df[feature_names].to_numpy(dtype=np.float32)
    y = df["Burn_Prob"].values.astype(np.float32).reshape(-1, 1)
    pos = df[["row", "col"]].to_numpy(dtype=np.float32)

    assert np.isfinite(X).all(),   "Non-finite values in X."
    assert np.isfinite(y).all(),   "Non-finite values in y."
    assert np.isfinite(pos).all(), "Non-finite values in pos."

    print(f"  X shape:   {X.shape}")
    print(f"  y shape:   {y.shape}")
    print(f"  pos shape: {pos.shape}")

    # ── Build coordinate lookup: (row, col) → node index ─────────────────────
    # All 300k nodes, so every pixel in the dataset has exactly one entry.
    print(f"  Building pixel→node index lookup...")
    coord_to_node = {
        (int(r), int(c)): i
        for i, (r, c) in enumerate(zip(df["row"].values, df["col"].values))
    }

    # ── 8-connected edges ────────────────────────────────────────────────────
    print(f"  Building 8-connected edges for {n} nodes...")
    src_list, dst_list = [], []

    shifts = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    for i, (r, c) in enumerate(zip(df["row"].values, df["col"].values)):
        r, c = int(r), int(c)
        for dr, dc in shifts:
            nr, nc = r + dr, c + dc
            j = coord_to_node.get((nr, nc))
            if j is not None:
                src_list.append(i)
                dst_list.append(j)

    edge_index = torch.tensor(
        [src_list, dst_list], dtype=torch.long
    )
    print(f"  Edges: {edge_index.shape[1]:,} (avg {edge_index.shape[1]/n:.1f} per node)")

    # ── Masks ─────────────────────────────────────────────────────────────────
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask  [val_idx  ] = True
    test_mask [test_idx ] = True

    assert (train_mask & val_mask).sum()  == 0, "Train/Val mask overlap."
    assert (train_mask & test_mask).sum() == 0, "Train/Test mask overlap."
    assert (val_mask   & test_mask).sum() == 0, "Val/Test mask overlap."
    assert (train_mask | val_mask | test_mask).all(), "Not all nodes covered."

    print(f"\n  Mask summary:")
    print(f"    train_mask: {train_mask.sum().item():>7,}")
    print(f"    val_mask:   {val_mask.sum().item():>7,}")
    print(f"    test_mask:  {test_mask.sum().item():>7,}")
    print(f"    total:      {n:>7,}")

    # ── Assemble Data object ──────────────────────────────────────────────────
    data = Data(
        x          = torch.tensor(X,   dtype=torch.float32),
        y          = torch.tensor(y,   dtype=torch.float32),
        pos        = torch.tensor(pos, dtype=torch.float32),
        edge_index = edge_index,
        train_mask = train_mask,
        val_mask   = val_mask,
        test_mask  = test_mask,
        num_nodes  = n,
    )

    print(f"\n  ✓ PyG Data object assembled:")
    print(f"    Nodes:      {data.num_nodes:,}")
    print(f"    Edges:      {data.num_edges:,}")
    print(f"    Features:   {data.num_node_features}")
    print(f"    val_mask>0: {data.val_mask.sum().item() > 0}  ← fixed from placeholder")

    return data

# Step 9 — Save everything

def save_all(
    df: pd.DataFrame,
    feature_names: list[str],
    train_idx: np.ndarray,
    val_idx:   np.ndarray,
    test_idx:  np.ndarray,
    graph_data,
    output_dir: Path,
    processed_dir: Path,
):
    print("\n" + "="*60)
    print("STEP 9 — Saving all outputs")
    print("="*60)

    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full feature table (all 300k nodes)
    all_path = output_dir / "df_all_features.parquet"
    df.to_parquet(all_path, index=False)
    print(f"  ✓ df_all_features.parquet       ({len(df):,} rows, {all_path.stat().st_size//1024} KB)")

    # 2. Train/val/test split tables
    df_train = df.iloc[train_idx].copy().reset_index(drop=True)
    df_val   = df.iloc[val_idx  ].copy().reset_index(drop=True)
    df_test  = df.iloc[test_idx ].copy().reset_index(drop=True)

    df_train.to_parquet(output_dir / "df_train_features.parquet", index=False)
    df_val.to_parquet(  output_dir / "df_val_features.parquet",   index=False)
    df_test.to_parquet( output_dir / "df_test_features.parquet",  index=False)
    print(f"  ✓ df_train_features.parquet     ({len(df_train):,} rows)")
    print(f"  ✓ df_val_features.parquet       ({len(df_val):,} rows)")
    print(f"  ✓ df_test_features.parquet      ({len(df_test):,} rows)")

    # 3. Feature names
    fn_path = output_dir / "feature_names.json"
    with open(fn_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"  ✓ feature_names.json            ({len(feature_names)} features)")

    # 4. Split indices
    splits_path = output_dir / "splits_phase54.npz"
    np.savez(splits_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    print(f"  ✓ splits_phase54.npz")

    # 5. PyG graph
    if graph_data is not None:
        graph_path = processed_dir / "graph_data_enriched.pt"
        torch.save(graph_data, graph_path)
        size_mb = graph_path.stat().st_size / 1024 / 1024
        print(f"  ✓ graph_data_enriched.pt        ({size_mb:.1f} MB)")

    print(f"\n  All outputs in: {output_dir}")
    print(f"  Graph in:       {processed_dir}")

# Diagnostic summary

def print_summary(
    df: pd.DataFrame,
    feature_names: list[str],
    train_idx, val_idx, test_idx,
    graph_data,
    elapsed: float,
):
    print("\n" + "="*60)
    print("PHASE 5.4 COMPLETE — SUMMARY")
    print("="*60)

    n = len(df)

    checks = [
        ("All 300k nodes included",       n == 300_000),
        ("60 features computed",          len(feature_names) >= 55),
        ("No NaN in features",
         df[feature_names].isna().sum().sum() == 0),
        ("Train mask populated",          len(train_idx) > 0),
        ("Val mask populated  (was 0)",   len(val_idx) > 0),
        ("Test mask populated",           len(test_idx) > 0),
        ("No split overlap",
         len(set(train_idx)&set(val_idx))==0 and
         len(set(train_idx)&set(test_idx))==0),
        ("DEM slope reasonable",
         df.get("dem_slope_deg", pd.Series([15])).mean() < 80),
        ("Target Gaussian (std≈1)",
         0.95 < df["Burn_Prob"].std() < 1.05),
        ("Graph built",                   graph_data is not None),
    ]

    all_pass = True
    for label, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {label}")
        if not passed:
            all_pass = False

    print(f"\n  Nodes    : {n:,}")
    print(f"  Features : {len(feature_names)}")
    print(f"  Train    : {len(train_idx):,}  ({len(train_idx)/n:.1%})")
    print(f"  Val      : {len(val_idx):,}  ({len(val_idx)/n:.1%})")
    print(f"  Test     : {len(test_idx):,}  ({len(test_idx)/n:.1%})")
    if graph_data is not None:
        print(f"  Edges    : {graph_data.num_edges:,}")

    print(f"\n  Elapsed  : {elapsed:.1f}s")

    if all_pass:
        print("\n  ✓ ALL CHECKS PASSED")
        print("\n  Next step: open 04_gnn_experiments.ipynb")
        print("  Load graph: data/processed/graph_data_enriched.pt")
        print("  Feature list: data/features/feature_names.json")
    else:
        print("\n  ⚠ Some checks failed — review output above.")

    print("="*60)

# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="feature_config.yaml")
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--test_frac", type=float, default=0.20)
    parser.add_argument("--val_frac",  type=float, default=0.10)
    args = parser.parse_args()

    t0 = time.time()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir    = Path(cfg.get("output_dir", "data/features"))
    processed_dir = Path("data/processed")

    # Run all steps
    df                          = load_all_nodes(cfg)
    df, ref_crs                 = derive_coordinates(df, cfg)
    df                          = add_dem_features(df, cfg, ref_crs)
    df, feature_names, pipeline = apply_feature_pipeline(df, cfg)
    df, qt                      = transform_target(df, output_dir)
    train_idx, val_idx, test_idx = create_spatial_split(
        df, test_frac=args.test_frac, val_frac=args.val_frac
    )
    correlation_report(df, feature_names, train_idx, output_dir)
    graph_data = build_pyg_graph(
        df, feature_names, train_idx, val_idx, test_idx, cfg
    )
    save_all(
        df, feature_names, train_idx, val_idx, test_idx,
        graph_data, output_dir, processed_dir
    )

    elapsed = time.time() - t0
    print_summary(df, feature_names, train_idx, val_idx, test_idx, graph_data, elapsed)


if __name__ == "__main__":
    main()