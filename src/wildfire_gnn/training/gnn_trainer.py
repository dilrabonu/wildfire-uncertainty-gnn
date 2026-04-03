from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wildfire_gnn.evaluation.calibration import (
    expected_calibration_error_regression,
    interval_coverage,
)
from wildfire_gnn.evaluation.metrics import regression_metrics
from wildfire_gnn.training.losses import GaussianNLLLossStable


@dataclass
class TrainerOutput:
    best_val_loss: float
    metrics: dict[str, Any]


class GNNTrainer:
    def __init__(self, model, config: dict, device: str = "cpu"):
        self.model = model.to(device)
        self.config = config
        self.device = device

        lr = config["training"]["learning_rate"]
        wd = config["training"]["weight_decay"]
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

        method = config["uncertainty"]["method"]
        if method == "gaussian_nll":
            self.criterion = GaussianNLLLossStable(
                min_variance=config["uncertainty"]["min_variance"]
            )
        else:
            self.criterion = torch.nn.MSELoss()

    def _compute_loss(self, pred_mean, pred_log_var, y):
        if isinstance(self.criterion, GaussianNLLLossStable):
            return self.criterion(pred_mean, pred_log_var, y)
        return self.criterion(pred_mean, y)

    def train(self, data, checkpoint_path: str | Path) -> TrainerOutput:
        data = data.to(self.device)
        best_val = float("inf")
        patience = 0
        max_patience = self.config["training"]["early_stopping_patience"]
        max_epochs = self.config["training"]["max_epochs"]
        grad_clip = self.config["training"]["gradient_clip_norm"]

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()

            pred_mean, pred_log_var = self.model(data.x, data.edge_index)
            train_loss = self._compute_loss(
                pred_mean[data.train_mask],
                None if pred_log_var is None else pred_log_var[data.train_mask],
                data.y[data.train_mask],
            )
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()

            val_loss = self.evaluate_loss(data, data.val_mask)

            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss.item():.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                patience += 1
                if patience >= max_patience:
                    print("Early stopping triggered.")
                    break

        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        metrics = self.evaluate_full(data)
        return TrainerOutput(best_val_loss=best_val, metrics=metrics)

    @torch.no_grad()
    def evaluate_loss(self, data, mask):
        self.model.eval()
        pred_mean, pred_log_var = self.model(data.x, data.edge_index)
        loss = self._compute_loss(
            pred_mean[mask],
            None if pred_log_var is None else pred_log_var[mask],
            data.y[mask],
        )
        return float(loss.item())

    @torch.no_grad()
    def predict(self, data):
        self.model.eval()
        pred_mean, pred_log_var = self.model(data.x, data.edge_index)

        if self.config["uncertainty"]["mc_dropout"]:
            self.model.train()
            samples = []
            mc_samples = self.config["uncertainty"]["mc_samples"]
            for _ in range(mc_samples):
                sample_mean, _ = self.model(data.x, data.edge_index)
                samples.append(sample_mean.unsqueeze(0))
            stacked = torch.cat(samples, dim=0)
            epistemic_std = stacked.std(dim=0)
            self.model.eval()
        else:
            epistemic_std = torch.zeros_like(pred_mean)

        if pred_log_var is not None:
            aleatoric_std = torch.exp(0.5 * pred_log_var)
        else:
            aleatoric_std = torch.zeros_like(pred_mean)

        total_std = torch.sqrt(epistemic_std**2 + aleatoric_std**2 + 1e-12)
        return pred_mean, total_std, aleatoric_std, epistemic_std

    @torch.no_grad()
    def evaluate_full(self, data):
        pred_mean, total_std, aleatoric_std, epistemic_std = self.predict(data)

        results = {}
        for split_name, mask in {
            "train": data.train_mask,
            "val": data.val_mask,
            "test": data.test_mask,
        }.items():
            y_true = data.y[mask].cpu().numpy().reshape(-1)
            y_pred = pred_mean[mask].cpu().numpy().reshape(-1)
            y_std = total_std[mask].cpu().numpy().reshape(-1)

            split_metrics = regression_metrics(y_true, y_pred)
            split_metrics["ece_reg"] = expected_calibration_error_regression(y_true, y_pred, y_std)
            split_metrics["coverage_95"] = interval_coverage(y_true, y_pred, y_std)

            results[split_name] = split_metrics

        return results