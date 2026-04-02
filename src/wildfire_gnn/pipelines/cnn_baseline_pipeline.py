from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from wildfire_gnn.evaluation.metrics import compute_regression_metrics
from wildfire_gnn.features.cnn_dataset import (
    RasterPatchDataset,
    build_patch_metadata,
    load_aligned_rasters,
    save_patch_metadata,
)
from wildfire_gnn.features.cnn_splitters import (
    make_random_split,
    make_spatial_block_split,
    save_splits,
)
from wildfire_gnn.models.cnn_baseline import CNNBaselineRegressor
from wildfire_gnn.utils.logger import get_logger


class CNNBaselinePipeline:
    """Train and evaluate CNN baseline on random and spatial splits."""

    def __init__(self, config: dict) -> None:
        self.config = config

        log_file = Path(config["outputs"]["logs_dir"]) / "run_cnn_baseline.log"
        self.logger = get_logger("cnn_baseline_pipeline", log_file=str(log_file))

        self.device = self._resolve_device(config["train"]["device"])

    def _resolve_device(self, device_cfg: str) -> torch.device:
        if device_cfg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_cfg)

    def _subsample_metadata(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """Subsample patch metadata using random or stratified sampling."""
        max_patch_samples = self.config["data"].get("max_patch_samples")
        if max_patch_samples is None or len(metadata_df) <= int(max_patch_samples):
            self.logger.info(
                "No subsampling applied: using all %d patch centers.",
                len(metadata_df),
            )
            return metadata_df.reset_index(drop=True)

        sampling_strategy = str(self.config["data"].get("sampling_strategy", "random")).lower()
        random_seed = int(self.config["project"]["random_seed"])

        if sampling_strategy == "random":
            sampled_df = metadata_df.sample(
                n=int(max_patch_samples),
                random_state=random_seed,
            ).reset_index(drop=True)

            self.logger.info(
                "Applied random subsampling: kept %d / %d patches",
                len(sampled_df),
                len(metadata_df),
            )
            return sampled_df

        if sampling_strategy == "stratified":
            bin_edges = self.config["data"]["stratify_bins"]
            if len(bin_edges) < 2:
                raise ValueError("stratify_bins must contain at least two bin edges.")

            df = metadata_df.copy()
            df["target_bin"] = pd.cut(
                df["target"],
                bins=bin_edges,
                include_lowest=True,
                right=False,
                duplicates="drop",
            )

            valid_df = df.dropna(subset=["target_bin"]).copy()
            if valid_df.empty:
                raise ValueError("No valid rows remained after target binning.")

            grouped = list(valid_df.groupby("target_bin", observed=False))
            n_total = int(max_patch_samples)
            n_groups = len(grouped)
            per_group = max(1, n_total // n_groups)

            sampled_parts = []
            for bin_name, group_df in grouped:
                n_take = min(len(group_df), per_group)
                sampled = group_df.sample(n=n_take, random_state=random_seed)
                sampled_parts.append(sampled)

                self.logger.info(
                    "Stratified bin %s: available=%d sampled=%d",
                    bin_name,
                    len(group_df),
                    n_take,
                )

            sampled_df = pd.concat(sampled_parts, axis=0)

            remaining = n_total - len(sampled_df)
            if remaining > 0:
                remaining_pool = valid_df.drop(index=sampled_df.index, errors="ignore")
                if len(remaining_pool) > 0:
                    extra_n = min(remaining, len(remaining_pool))
                    extra_df = remaining_pool.sample(n=extra_n, random_state=random_seed)
                    sampled_df = pd.concat([sampled_df, extra_df], axis=0)

            sampled_df = sampled_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
            sampled_df = sampled_df.drop(columns=["target_bin"], errors="ignore")

            self.logger.info(
                "Applied stratified subsampling: kept %d / %d patches",
                len(sampled_df),
                len(metadata_df),
            )
            return sampled_df

        raise ValueError(f"Unsupported sampling_strategy: {sampling_strategy}")

    def prepare_dataset(self) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Load rasters, build patch metadata, subsample, and save artifacts."""
        x_stack, target, channel_stats_df, valid_mask = load_aligned_rasters(
            feature_raster_paths=self.config["paths"]["feature_raster_paths"],
            target_raster_path=self.config["paths"]["target_raster_path"],
            target_min=float(self.config["data"]["target_min"]),
            target_max=float(self.config["data"]["target_max"]),
            standardize_continuous_channels=bool(self.config["data"]["standardize_continuous_channels"]),
        )

        metadata_df = build_patch_metadata(
            target=target,
            valid_mask=valid_mask,
            patch_size=int(self.config["data"]["patch_size"]),
        )

        original_count = len(metadata_df)
        metadata_df = self._subsample_metadata(metadata_df)

        save_patch_metadata(metadata_df, self.config["paths"]["patch_metadata_path"])

        self.logger.info("Saved patch metadata to %s", self.config["paths"]["patch_metadata_path"])
        self.logger.info(
            "CNN dataset stats: original_samples=%d sampled_samples=%d target_min=%.6f target_max=%.6f target_mean=%.6f target_std=%.6f",
            original_count,
            len(metadata_df),
            metadata_df["target"].min(),
            metadata_df["target"].max(),
            metadata_df["target"].mean(),
            metadata_df["target"].std(),
        )

        return x_stack, target, metadata_df

    def create_splits(self, metadata_df: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
        """Create both random and spatial splits."""
        results = {}

        random_cfg = self.config["split"]["random"]
        random_splits = make_random_split(
            df=metadata_df,
            train_size=float(random_cfg["train_size"]),
            val_size=float(random_cfg["val_size"]),
            test_size=float(random_cfg["test_size"]),
            random_seed=int(self.config["project"]["random_seed"]),
        )
        save_splits(random_splits, self.config["paths"]["random_split_path"])
        results["random"] = random_splits
        self.logger.info("Saved random CNN splits to %s", self.config["paths"]["random_split_path"])

        spatial_cfg = self.config["split"]["spatial"]
        if bool(spatial_cfg["enabled"]):
            spatial_splits = make_spatial_block_split(
                df=metadata_df,
                n_row_blocks=int(spatial_cfg["n_row_blocks"]),
                n_col_blocks=int(spatial_cfg["n_col_blocks"]),
                train_blocks=float(spatial_cfg["train_blocks"]),
                val_blocks=float(spatial_cfg["val_blocks"]),
                test_blocks=float(spatial_cfg["test_blocks"]),
                random_seed=int(self.config["project"]["random_seed"]),
            )
            save_splits(spatial_splits, self.config["paths"]["spatial_split_path"])
            results["spatial"] = spatial_splits
            self.logger.info("Saved spatial CNN splits to %s", self.config["paths"]["spatial_split_path"])

        return results

    def _make_dataloaders(
        self,
        dataset: RasterPatchDataset,
        split_indices: dict[str, np.ndarray],
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        batch_size = int(self.config["train"]["batch_size"])
        num_workers = int(self.config["train"]["num_workers"])

        train_loader = DataLoader(
            Subset(dataset, split_indices["train_idx"]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            Subset(dataset, split_indices["val_idx"]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            Subset(dataset, split_indices["test_idx"]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return train_loader, val_loader, test_loader

    def _build_model(self) -> CNNBaselineRegressor:
        model_cfg = self.config["model"]
        return CNNBaselineRegressor(
            in_channels=int(model_cfg["in_channels"]),
            base_channels=int(model_cfg["base_channels"]),
            dropout=float(model_cfg["dropout"]),
        )

    def _train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: AdamW,
        criterion: nn.Module,
    ) -> float:
        model.train()
        total_loss = 0.0
        n_samples = 0

        for batch in loader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

        return total_loss / max(n_samples, 1)

    @torch.no_grad()
    def _evaluate_loss(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> float:
        model.eval()
        total_loss = 0.0
        n_samples = 0

        for batch in loader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)

            preds = model(x)
            loss = criterion(preds, y)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

        return total_loss / max(n_samples, 1)

    @torch.no_grad()
    def _predict(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        model.eval()

        y_true_all = []
        y_pred_all = []
        row_all = []
        col_all = []

        for batch in loader:
            x = batch["x"].to(self.device)
            preds = model(x).detach().cpu().numpy()

            y_true_all.append(batch["y"].numpy())
            y_pred_all.append(preds)
            row_all.append(batch["row_index"].numpy())
            col_all.append(batch["col_index"].numpy())

        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        row_idx = np.concatenate(row_all)
        col_idx = np.concatenate(col_all)

        return y_true, y_pred, row_idx, col_idx

    def _save_loss_curve(
        self,
        train_losses: list[float],
        val_losses: list[float],
        split_name: str,
    ) -> None:
        figures_dir = Path(self.config["outputs"]["figures_dir"])
        figures_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="train_loss")
        plt.plot(val_losses, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(f"CNN Loss Curve ({split_name.title()} Split)")
        plt.legend()
        plt.savefig(figures_dir / f"cnn_loss_curve_{split_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _save_pred_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        split_name: str,
    ) -> None:
        figures_dir = Path(self.config["outputs"]["figures_dir"])
        figures_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.2, s=8)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"CNN - Predicted vs True ({split_name.title()} Test)")
        plt.savefig(figures_dir / f"cnn_{split_name}_pred_vs_true.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _save_target_histogram(self, metadata_df: pd.DataFrame) -> None:
        figures_dir = Path(self.config["outputs"]["figures_dir"])
        figures_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 5))
        plt.hist(metadata_df["target"], bins=100)
        plt.title("CNN Target Distribution")
        plt.xlabel("Burn Probability")
        plt.ylabel("Frequency")
        plt.savefig(figures_dir / "cnn_target_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _save_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        row_idx: np.ndarray,
        col_idx: np.ndarray,
        split_name: str,
    ) -> None:
        pred_dir = Path(self.config["outputs"]["predictions_dir"])
        pred_dir.mkdir(parents=True, exist_ok=True)

        pred_df = pd.DataFrame(
            {
                "row_index": row_idx,
                "col_index": col_idx,
                "y_true": y_true,
                "y_pred": y_pred,
                "residual": y_true - y_pred,
            }
        )
        pred_df.to_csv(pred_dir / f"cnn_{split_name}_test_predictions.csv", index=False)

    def _save_split_comparison(self, results_df: pd.DataFrame) -> None:
        figures_dir = Path(self.config["outputs"]["figures_dir"])
        figures_dir.mkdir(parents=True, exist_ok=True)

        test_df = results_df[results_df["data_split"] == "test"].copy()

        plt.figure(figsize=(6, 5))
        plt.bar(test_df["split_type"], test_df["rmse"])
        plt.title("CNN Baseline Comparison Across Splits")
        plt.xlabel("Split Type")
        plt.ylabel("Test RMSE")
        plt.savefig(figures_dir / "cnn_split_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _train_on_split(
        self,
        dataset: RasterPatchDataset,
        split_indices: dict[str, np.ndarray],
        split_name: str,
    ) -> list[dict]:
        train_loader, val_loader, test_loader = self._make_dataloaders(dataset, split_indices)

        model = self._build_model().to(self.device)
        criterion = nn.MSELoss()
        optimizer = AdamW(
            model.parameters(),
            lr=float(self.config["train"]["learning_rate"]),
            weight_decay=float(self.config["train"]["weight_decay"]),
        )

        max_epochs = int(self.config["train"]["max_epochs"])
        patience = int(self.config["train"]["early_stopping_patience"])

        checkpoint_dir = Path(self.config["outputs"]["checkpoints_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"cnn_{split_name}_best.pt"

        best_val_loss = float("inf")
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(1, max_epochs + 1):
            train_loss = self._train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss = self._evaluate_loss(model, val_loader, criterion)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            self.logger.info(
                "[%s] epoch=%d train_loss=%.6f val_loss=%.6f",
                split_name,
                epoch,
                train_loss,
                val_loss,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.logger.info("[%s] Early stopping at epoch %d", split_name, epoch)
                break

        self._save_loss_curve(train_losses, val_losses, split_name)

        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        val_true, val_pred, _, _ = self._predict(model, val_loader)
        test_true, test_pred, test_rows, test_cols = self._predict(model, test_loader)

        val_metrics = compute_regression_metrics(val_true, val_pred)
        test_metrics = compute_regression_metrics(test_true, test_pred)

        self._save_pred_plot(test_true, test_pred, split_name)
        self._save_predictions(test_true, test_pred, test_rows, test_cols, split_name)

        self.logger.info("[%s] Validation metrics: %s", split_name, val_metrics)
        self.logger.info("[%s] Test metrics: %s", split_name, test_metrics)

        return [
            {"model": "cnn", "split_type": split_name, "data_split": "val", **val_metrics},
            {"model": "cnn", "split_type": split_name, "data_split": "test", **test_metrics},
        ]

    def run(self) -> pd.DataFrame:
        """Full CNN baseline pipeline."""
        x_stack, target, metadata_df = self.prepare_dataset()
        self._save_target_histogram(metadata_df)

        dataset = RasterPatchDataset(
            x_stack=x_stack,
            target=target,
            metadata=metadata_df,
            patch_size=int(self.config["data"]["patch_size"]),
        )

        split_map = self.create_splits(metadata_df)
        results = []

        for split_name, split_indices in split_map.items():
            self.logger.info(
                "[%s] train=%d val=%d test=%d",
                split_name,
                len(split_indices["train_idx"]),
                len(split_indices["val_idx"]),
                len(split_indices["test_idx"]),
            )
            results.extend(self._train_on_split(dataset, split_indices, split_name))

        results_df = pd.DataFrame(results)

        metrics_path = Path(self.config["outputs"]["metrics_table_path"])
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(metrics_path, index=False)

        self._save_split_comparison(results_df)

        self.logger.info("Saved CNN metrics to %s", metrics_path)
        return results_df