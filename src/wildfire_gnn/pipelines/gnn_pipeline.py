from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from wildfire_gnn.models.gnn_models import GraphSAGERegressor, ResidualGATRegressor
from wildfire_gnn.training.losses import (
    build_target_weights,
    gaussian_nll_loss,
    weighted_huber_loss,
)

from wildfire_gnn.training.losses import weighted_mse_loss
from wildfire_gnn.training.metrics import (
    regression_metrics,
    binwise_regression_metrics,
    regression_ece,
    reliability_table,
)


@dataclass
class TrainOutputs:
    best_model_path: str
    best_val_loss: float
    history: pd.DataFrame


class GNNPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config["training"].get("device", "cpu"))

        self.figures_dir = Path(config["paths"]["figures_dir"])
        self.tables_dir = Path(config["paths"]["metrics_table_path"]).parent
        self.checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
        self.logs_dir = Path(config["paths"]["logs_dir"])

        for p in [self.figures_dir, self.tables_dir, self.checkpoints_dir, self.logs_dir]:
            p.mkdir(parents=True, exist_ok=True)

    def build_model(self, in_dim: int, stage: str = "stage1"):
        model_cfg = self.config["model"]
        predict_variance = self.config["model"].get("predict_variance", False)

        if stage == "stage1":
            name = model_cfg["stage1_name"]
        else:
            name = model_cfg["stage2_name"]

        if name == "graphsage":
            model = GraphSAGERegressor(
                in_dim=in_dim,
                hidden_dim=model_cfg["hidden_dim"],
                num_layers=model_cfg["num_layers"],
                dropout=model_cfg["dropout"],
                predict_variance=predict_variance,
                min_variance=model_cfg.get("min_variance", 1e-6),
            )
        elif name == "residual_gat":
            model = ResidualGATRegressor(
                in_dim=in_dim,
                hidden_dim=model_cfg["hidden_dim"],
                num_layers=model_cfg["num_layers"],
                heads=model_cfg["gat_heads"],
                dropout=model_cfg["dropout"],
                attn_dropout=model_cfg["gat_dropout"],
                use_jk=model_cfg.get("use_jumping_knowledge", True),
                predict_variance=predict_variance,
                min_variance=model_cfg.get("min_variance", 1e-6),
            )
        else:
            raise ValueError(f"Unsupported model name: {name}")

        return model.to(self.device)

    def _compute_loss(self, out, y, mask):
        train_cfg = self.config["training"]
        loss_name = train_cfg["loss_name"]

        if isinstance(out, tuple):
            mean, var = out
        else:
            mean, var = out, None

        y_mask = y[mask]
        mean_mask = mean[mask]

        if loss_name == "weighted_huber":
            weights = build_target_weights(
                target=y_mask,
                bin_edges=train_cfg["target_bin_edges"],
                bin_weights=train_cfg["target_bin_weights"],
            ).to(y.device)
            return weighted_huber_loss(
                pred=mean_mask,
                target=y_mask,
                weights=weights,
                delta=train_cfg["huber_delta"],
            )

        if loss_name == "gaussian_nll":
            if var is None:
                raise ValueError("Gaussian NLL requires model.predict_variance = true")
            return gaussian_nll_loss(mean_mask, var[mask], y_mask)

        return torch.nn.functional.mse_loss(mean_mask, y_mask)

    def train(self, data: Data, stage: str = "stage1") -> TrainOutputs:
        data = data.to(self.device)
        model = self.build_model(in_dim=data.x.shape[1], stage=stage)

        train_cfg = self.config["training"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )

        best_val_loss = float("inf")
        best_model_path = self.checkpoints_dir / f"{stage}_{self.config['data']['split_type']}_best.pt"
        patience = 0
        history_rows = []

        for epoch in range(1, train_cfg["max_epochs"] + 1):
            model.train()
            optimizer.zero_grad()

            out = model(data.x, data.edge_index)
            train_loss = self._compute_loss(out, data.y, data.train_mask)

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip_norm"])
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out_val = model(data.x, data.edge_index)
                val_loss = self._compute_loss(out_val, data.y, data.val_mask)

            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss.item()),
                    "val_loss": float(val_loss.item()),
                }
            )

            if val_loss.item() < best_val_loss:
                best_val_loss = float(val_loss.item())
                patience = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience += 1

            if patience >= train_cfg["early_stopping_patience"]:
                break

        history = pd.DataFrame(history_rows)
        history.to_csv(self.tables_dir / "gnn_training_history.csv", index=False)

        return TrainOutputs(
            best_model_path=str(best_model_path),
            best_val_loss=best_val_loss,
            history=history,
        )

    def predict(self, data: Data, checkpoint_path: str, stage: str = "stage1"):
        data = data.to(self.device)
        model = self.build_model(in_dim=data.x.shape[1], stage=stage)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.eval()

        with torch.no_grad():
            out = model(data.x, data.edge_index)

        if isinstance(out, tuple):
            mean, var = out
            unc = torch.sqrt(var)
        else:
            mean = out
            unc = torch.zeros_like(mean)

        return (
            mean.detach().cpu().numpy(),
            unc.detach().cpu().numpy(),
        )

    def evaluate(self, data: Data, checkpoint_path: str, stage: str = "stage1") -> Dict[str, float]:
        preds, unc = self.predict(data, checkpoint_path, stage=stage)

        y_true = data.y.detach().cpu().numpy()
        test_mask = data.test_mask.detach().cpu().numpy().astype(bool)

        y_test = y_true[test_mask]
        pred_test = preds[test_mask]
        unc_test = unc[test_mask]

        overall = regression_metrics(y_test, pred_test)

        bins = [tuple(b) for b in self.config["evaluation"]["regression_bins"]]
        bin_df = binwise_regression_metrics(y_test, pred_test, bins=bins)
        bin_df.to_csv(self.config["paths"]["bin_metrics_table_path"], index=False)

        calibration = {}
        if np.any(unc_test > 0):
            ece = regression_ece(
                y_true=y_test,
                y_pred=pred_test,
                y_unc=unc_test,
                num_bins=self.config["evaluation"]["calibration_num_bins"],
            )
            calibration["regression_ece"] = ece

            rel_df = reliability_table(
                y_true=y_test,
                y_pred=pred_test,
                y_unc=unc_test,
                num_bins=self.config["evaluation"]["calibration_num_bins"],
            )
            rel_df.to_csv(self.tables_dir / "gnn_reliability_table.csv", index=False)

        metrics = {**overall, **calibration}
        pd.DataFrame([metrics]).to_csv(self.config["paths"]["metrics_table_path"], index=False)

        pred_df = pd.DataFrame(
            {
                "y_true": y_test.reshape(-1),
                "y_pred": pred_test.reshape(-1),
                "uncertainty": unc_test.reshape(-1),
            }
        )
        pred_df.to_csv(self.config["paths"]["predictions_table_path"], index=False)

        return metrics