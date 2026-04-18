"""
Run this AFTER python prepare_features.py completes.
Produces 5 diagnostic figures + R² comparison table.

Usage:
    python 06_feature_engineering_validation.py
    python 06_feature_engineering_validation.py --output_dir data/features
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # works without a display / Jupyter
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, probplot
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "font.size": 11})

# Config

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="data/features",
                   help="Where prepare_features.py saved its outputs")
    p.add_argument("--target_col", default="Burn_Prob")
    return p.parse_args()

# Load

def load_outputs(output_dir: Path, target_col: str):
    print(f"\n{'='*60}")
    print("LOADING PIPELINE OUTPUTS")
    print(f"{'='*60}")

    df_train = pd.read_parquet(output_dir / "df_train_features.parquet")
    df_test  = pd.read_parquet(output_dir / "df_test_features.parquet")
    corr_df  = pd.read_csv(output_dir / "correlation_report.csv")

    with open(output_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    with open(output_dir / "target_transformer.pkl", "rb") as f:
        qt = pickle.load(f)

    print(f"  Train : {df_train.shape}")
    print(f"  Test  : {df_test.shape}")
    print(f"  Features: {len(feature_names)}")
    return df_train, df_test, corr_df, feature_names, qt

# Figure 1 — Target Distribution

def fig_target_distribution(df_train, target_col, out_dir):
    print("\n[Fig 1] Target distribution...")
    raw_col = f"{target_col}_raw"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Target Variable: Burn_Prob", fontsize=13, fontweight="bold")

    # Panel 1 — Raw
    if raw_col in df_train.columns:
        y_raw = df_train[raw_col].values
        axes[0].hist(y_raw, bins=80, color="#e74c3c", alpha=0.85, edgecolor="none")
        axes[0].axvline(np.median(y_raw), color="black", linestyle="--",
                        label=f"median={np.median(y_raw):.4f}")
        axes[0].set_title("Raw Burn_Prob\n(original — heavily skewed)")
        axes[0].set_xlabel("Burn Probability")
        axes[0].set_ylabel("Count")
        axes[0].legend(fontsize=9)
    else:
        axes[0].text(0.5, 0.5, f"{raw_col}\nnot found",
                     ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title("Raw Burn_Prob\n(not available)")

    # Panel 2 — Transformed
    y_t = df_train[target_col].values
    axes[1].hist(y_t, bins=80, color="#2ecc71", alpha=0.85, edgecolor="none")
    axes[1].axvline(0, color="black", linestyle="--", label="mean≈0")
    axes[1].set_title("After Quantile Transform\n(target for model training)")
    axes[1].set_xlabel("Transformed value")
    axes[1].set_ylabel("Count")
    axes[1].legend(fontsize=9)

    # Panel 3 — Q-Q plot
    (osm, osr), (slope, intercept, r) = probplot(y_t, dist="norm")
    axes[2].plot(osm, osr, "o", color="#3498db", markersize=2, alpha=0.4)
    axes[2].plot(osm, slope * np.array(osm) + intercept,
                 color="red", linewidth=1.5, label=f"r²={r**2:.3f}")
    axes[2].set_title("Q-Q Plot (Transformed)\nShould lie on diagonal")
    axes[2].set_xlabel("Theoretical quantiles")
    axes[2].set_ylabel("Ordered values")
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    fpath = out_dir / "fig1_target_distribution.png"
    plt.savefig(fpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")

# Figure 2 — Top-30 Pearson Correlation Bar Chart

def fig_correlation(corr_df, target_col, out_dir):
    print("\n[Fig 2] Correlation chart...")
    top30 = corr_df.head(30).copy()

    fig, ax = plt.subplots(figsize=(11, 8))
    colors = ["#e74c3c" if r > 0 else "#3498db" for r in top30["pearson_r"]]
    ax.barh(top30["feature"], top30["pearson_r"], color=colors, alpha=0.85)
    ax.axvline(0,    color="black",  linewidth=0.8)
    ax.axvline(0.15, color="#7f8c8d", linewidth=1.3, linestyle="--",
               label="|r|=0.15 threshold")
    ax.axvline(-0.15, color="#7f8c8d", linewidth=1.3, linestyle="--")
    ax.set_xlabel(f"Pearson r with {target_col} (quantile-transformed)", fontsize=11)
    ax.set_title("Top 30 Features by |Pearson r|", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    n_above = (corr_df["pearson_r"].abs() > 0.15).sum()
    ax.text(0.98, 0.02,
            f"{n_above}/{len(corr_df)} features above |r|=0.15",
            ha="right", va="bottom", transform=ax.transAxes,
            fontsize=9, color="#7f8c8d")

    plt.tight_layout()
    fpath = out_dir / "fig2_correlation_top30.png"
    plt.savefig(fpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")

    top_r = corr_df["pearson_r"].abs().max()
    print(f"  Max |r|: {top_r:.4f} | n_above_threshold: {n_above}")
    if top_r < 0.15:
        print("   WARNING: very low correlations — check DEM/NDVI alignment.")
    else:
        print(f"   Feature discriminability looks healthy.")


# Figure 3 — Multi-Scale Stats Distribution

def fig_multiscale(df_train, target_col, out_dir):
    print("\n[Fig 3] Multi-scale distributions...")
    base_col  = "Ignition_Prob"
    scales    = [3, 7, 15]
    scale_cols = [f"{base_col}_mean{k}" for k in scales
                  if f"{base_col}_mean{k}" in df_train.columns]

    if not scale_cols:
        print("    No multi-scale columns found — check raster_shape in config.")
        return

    all_cols = [base_col] + scale_cols
    all_cols = [c for c in all_cols if c in df_train.columns]

    fig, axes = plt.subplots(1, len(all_cols), figsize=(4 * len(all_cols), 4))
    if len(all_cols) == 1:
        axes = [axes]

    for i, col in enumerate(all_cols):
        ax = axes[i]
        data = df_train[col].values
        ax.hist(data, bins=60, color="#9b59b6", alpha=0.85, edgecolor="none")
        try:
            r, _ = pearsonr(data, df_train[target_col].values)
        except Exception:
            r = 0.0
        ax.set_title(f"{col}\n|r|={abs(r):.3f}", fontsize=10)
        ax.set_xlabel("Value", fontsize=9)
        if i == 0:
            ax.set_ylabel("Count")

    fig.suptitle("Multi-Scale Neighborhood Stats — Ignition Prob",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fpath = out_dir / "fig3_multiscale_stats.png"
    plt.savefig(fpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")
    if len(all_cols) > 1:
        print(f"  Shows how spatial smoothing at different scales adds new signal.")

# Figure 4 — DEM Feature Sanity

def fig_dem(df_train, target_col, out_dir):
    print("\n[Fig 4] DEM feature check...")
    dem_cols = [c for c in df_train.columns if c.startswith("dem_")]

    if not dem_cols:
        print("    No DEM columns found. Set dem_path in feature_config.yaml.")
        return

    n = len(dem_cols)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for i, col in enumerate(dem_cols):
        ax = axes[i]
        data = df_train[col].fillna(0).values
        ax.hist(data, bins=60, color="#f39c12", alpha=0.85, edgecolor="none")
        try:
            r, _ = pearsonr(data, df_train[target_col].values)
        except Exception:
            r = 0.0
        ax.set_title(f"{col}\n|r|={abs(r):.3f}", fontsize=10)
        ax.set_xlabel("Value", fontsize=9)
        if i == 0:
            ax.set_ylabel("Count")

    fig.suptitle("DEM Terrain Features — Distribution & Correlation",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fpath = out_dir / "fig4_dem_features.png"
    plt.savefig(fpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")

    if "dem_slope_deg" in df_train.columns:
        r_s, _ = pearsonr(
            df_train["dem_slope_deg"].fillna(0).values,
            df_train[target_col].values
        )
        print(f"  Slope ↔ {target_col}: r={r_s:.4f}")
        if abs(r_s) > 0.10:
            print("   DEM slope is contributing signal.")
        else:
            print("    Low slope correlation — verify DEM CRS matches raster CRS.")
# Figure 5 — Feature Inventory

def fig_feature_inventory(feature_names, out_dir):
    print("\n[Fig 5] Feature inventory...")

    groups = {
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
                                p in c for p in [
                                    "dem_", "ndvi", "fire_freq", "pyrome_",
                                    "fuel_", "inter_", "_mean", "_std",
                                    "_grad_", "connectivity"])],
    }

    active = {k: v for k, v in groups.items() if v}
    names  = list(active.keys())
    counts = [len(v) for v in active.values()]
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, counts, color=colors, alpha=0.9, edgecolor="white")
    ax.bar_label(bars, padding=4, fontsize=10)
    ax.set_xlabel("Number of features")
    ax.set_title(f"Feature Inventory — {len(feature_names)} total",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts) * 1.15)

    plt.tight_layout()
    fpath = out_dir / "fig5_feature_inventory.png"
    plt.savefig(fpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")
    for name, count in zip(names, counts):
        print(f"    {name:25s}: {count:3d}")
    print(f"    {'─'*32}")
    print(f"    {'TOTAL':25s}: {len(feature_names):3d}")

# Section 9 — Ridge R² comparison

def section_ridge_comparison(df_train, df_test, feature_names, target_col):
    print(f"\n{'='*60}")
    print("SECTION 9 — RIDGE REGRESSION R² COMPARISON")
    print(f"{'='*60}")
    print("(Proxy test: if Ridge improves, GNN will too)\n")

    baseline_cols = [
        c for c in [
            "Ignition_Prob", "CFL", "FSP_Index",
            "Struct_Exp_Index", "Fuel_Models", "row", "col"
        ] if c in df_train.columns
    ]

    def evaluate(cols, label):
        valid = [c for c in cols
                 if c in df_train.columns and c in df_test.columns]
        if not valid:
            print(f"  {label:40s}: no valid columns")
            return None

        X_tr = df_train[valid].fillna(0).values
        X_te = df_test[valid].fillna(0).values
        y_tr = df_train[target_col].values
        y_te = df_test[target_col].values

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

        m = Ridge(alpha=1.0)
        m.fit(X_tr, y_tr)
        r2_tr = r2_score(y_tr, m.predict(X_tr))
        r2_te = r2_score(y_te, m.predict(X_te))
        print(f"  {label:40s}  "
              f"train R²={r2_tr:+.4f}  |  test R²={r2_te:+.4f}  "
              f"| n_feat={len(valid)}")
        return r2_te

    r2_base = evaluate(baseline_cols, "Baseline (original 7 features)")

    dem_cols = [c for c in feature_names if c.startswith("dem_")]
    r2_dem   = evaluate(baseline_cols + dem_cols, "Baseline + DEM terrain")

    ms_cols  = [c for c in feature_names if "_mean" in c or "_std" in c]
    r2_ms    = evaluate(baseline_cols + ms_cols, "Baseline + Multi-scale stats")

    grad_cols = [c for c in feature_names if "_grad_" in c]
    r2_grad   = evaluate(baseline_cols + grad_cols, "Baseline + Spatial gradients")

    r2_full  = evaluate(feature_names, "Full feature set (all groups)")

    print(f"\n{'─'*60}")
    if r2_full is not None and r2_base is not None:
        delta = r2_full - r2_base
        if delta > 0:
            print(f"   Full features improve test R² by +{delta:.4f}")
            print("     → New features carry real signal. GNN R² should turn positive.")
        else:
            print(f"    Full features did NOT improve over baseline (Δ={delta:.4f})")
            print("     Likely causes:")
            print("       1. DEM CRS mismatch  — all dem_* values identical → zero variance")
            print("       2. raster_shape wrong — multi-scale windows misaligned")
            print("       3. NDVI proxy poor    — verify slope proxy is correlated")
            print("     Action: open correlation_report.csv and check dem_slope_deg |r|")
    print(f"{'─'*60}")

# Main

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    target_col = args.target_col

    # Check outputs exist
    required = [
        "df_train_features.parquet",
        "df_test_features.parquet",
        "correlation_report.csv",
        "feature_names.json",
        "target_transformer.pkl",
    ]
    missing = [f for f in required if not (out_dir / f).exists()]
    if missing:
        print(f"\n Missing files in {out_dir}:")
        for f in missing:
            print(f"   {f}")
        print("\nRun first: python prepare_features.py")
        return

    df_train, df_test, corr_df, feature_names, qt = load_outputs(out_dir, target_col)

    fig_target_distribution(df_train, target_col, out_dir)
    fig_correlation(corr_df, target_col, out_dir)
    fig_multiscale(df_train, target_col, out_dir)
    fig_dem(df_train, target_col, out_dir)
    fig_feature_inventory(feature_names, out_dir)
    section_ridge_comparison(df_train, df_test, feature_names, target_col)

    print(f"\n{'='*60}")
    print("ALL DONE")
    print(f"{'='*60}")
    print(f"5 figures saved to: {out_dir}/")
    print(f"  fig1_target_distribution.png")
    print(f"  fig2_correlation_top30.png")
    print(f"  fig3_multiscale_stats.png")
    print(f"  fig4_dem_features.png")
    print(f"  fig5_feature_inventory.png")
    print(f"\nNext: retrain GraphSAGE with the new features.")
    print(f"       Load: {out_dir}/df_train_features.parquet")
    print(f"       Features: {out_dir}/feature_names.json")


if __name__ == "__main__":
    main()