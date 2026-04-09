import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def compute_feature_summary(df: pd.DataFrame, save_path: str):
    summary = df.describe().T
    summary["median"] = df.median()
    summary["skew"] = df.skew()
    summary["missing"] = df.isnull().sum()

    summary.to_csv(save_path)
    return summary


def plot_feature_distributions(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    for col in df.columns:
        plt.figure(figsize=(5, 4))
        sns.histplot(df[col], bins=50, kde=True)
        plt.title(f"Distribution: {col}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_dist_{col}.png")
        plt.close()


def compute_feature_target_correlation(df: pd.DataFrame, target_col: str, save_path: str):
    results = []

    for col in df.columns:
        if col == target_col:
            continue

        pearson = df[col].corr(df[target_col], method="pearson")
        spearman = df[col].corr(df[target_col], method="spearman")

        results.append({
            "feature": col,
            "pearson": pearson,
            "spearman": spearman
        })

    corr_df = pd.DataFrame(results).sort_values(by="spearman", ascending=False)
    corr_df.to_csv(save_path, index=False)
    return corr_df

def plot_feature_vs_target(df: pd.DataFrame, target_col: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    for col in df.columns:
        if col == target_col:
            continue

        plt.figure(figsize=(5, 5))
        plt.scatter(df[col], df[target_col], alpha=0.1)
        plt.xlabel(col)
        plt.ylabel(target_col)
        plt.title(f"{col} vs {target_col}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_vs_target_{col}.png")
        plt.close()


def compute_split_shift(df: pd.DataFrame, split_col: str, save_path: str):
    stats = []

    for col in df.columns:
        if col == split_col:
            continue

        train = df[df[split_col] == "train"][col]
        val = df[df[split_col] == "val"][col]
        test = df[df[split_col] == "test"][col]

        stats.append({
            "feature": col,
            "train_mean": train.mean(),
            "val_mean": val.mean(),
            "test_mean": test.mean(),
            "train_std": train.std(),
            "val_std": val.std(),
            "test_std": test.std(),
        })

    shift_df = pd.DataFrame(stats)
    shift_df.to_csv(save_path, index=False)
    return shift_df