from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from wildfire_gnn.models.gnn_models import GraphSAGERegressor, ResidualGATRegressor
from wildfire_gnn.training.losses import (
    build_target_weights,
    classification_loss,
    gaussian_nll_loss,
    weighted_huber_loss,
    weighted_mse_loss,
)
from wildfire_gnn.training.metrics import (
    binwise_regression_metrics,
    regression_ece,
    regression_metrics,
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

    def _target_output_max(self) -> float:
        target_max_raw = float(self.config["data"]["target_max_raw"])
        transform = self.config["data"].get("target_transform", "none")
        if transform == "log1p":
            return float(np.log1p(target_max_raw))
        return target_max_raw

    def _inverse_target_transform(self, arr: np.ndarray) -> np.ndarray:
        transform = self.config["data"].get("target_transform", "none")
        if transform == "log1p":
            return np.expm1(arr)
        return arr

    def _build_risk_bins(self, y_raw: torch.Tensor) -> torch.Tensor:
        edges = self.config["training"]["target_bin_edges"]
        boundaries = torch.tensor(edges[1:-1], device=y_raw.device, dtype=y_raw.dtype)
        return torch.bucketize(y_raw.view(-1), boundaries)

    def build_model(self, in_dim: int, stage: str = "stage1"):
        model_cfg = self.config["model"]

        predict_variance = model_cfg.get("predict_variance", False)
        hybrid_enabled = model_cfg.get("hybrid_enabled", False)
        num_risk_bins = model_cfg.get("num_risk_bins", 4)
        output_activation = model_cfg.get("output_activation", "none")
        output_max = self._target_output_max() if output_activation == "bounded_sigmoid" else None

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
                min_variance=model_cfg["min_variance"],
                output_activation=output_activation,
                output_max=output_max,
                hybrid_enabled=hybrid_enabled,
                num_risk_bins=num_risk_bins,
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
                min_variance=model_cfg["min_variance"],
                output_activation=output_activation,
                output_max=output_max,
                hybrid_enabled=hybrid_enabled,
                num_risk_bins=num_risk_bins,
            )
        else:
            raise ValueError(f"Unsupported model name: {name}")

        return model.to(self.device)

    def _compute_loss(self, out: dict, y: torch.Tensor, y_raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        train_cfg = self.config["training"]
        model_cfg = self.config["model"]
        loss_name = train_cfg["loss_name"]

        y_mask = y[mask]
        y_raw_mask = y_raw[mask]
        pred_mask = out["mean"][mask]

        weights = build_target_weights(
            target_raw=y_raw_mask,
            bin_edges=train_cfg["target_bin_edges"],
            bin_weights=train_cfg["target_bin_weights"],
        ).to(y.device)

        if loss_name == "weighted_huber":
            reg_loss = weighted_huber_loss(
                pred=pred_mask,
                target=y_mask,
                weights=weights,
                delta=train_cfg["huber_delta"],
            )
        elif loss_name == "weighted_mse":
            reg_loss = weighted_mse_loss(
                pred=pred_mask,
                target=y_mask,
                weights=weights,
            )
        elif loss_name == "gaussian_nll":
            if "var" not in out:
                raise ValueError("Gaussian NLL requires predict_variance=true")
            reg_loss = gaussian_nll_loss(
                mean=pred_mask,
                var=out["var"][mask],
                target=y_mask,
            )
        else:
            reg_loss = torch.nn.functional.mse_loss(pred_mask, y_mask)

        if not model_cfg.get("hybrid_enabled", False):
            return reg_loss

        y_cls = self._build_risk_bins(y_raw)
        class_weights = torch.tensor(
            train_cfg["cls_class_weights"],
            dtype=torch.float32,
            device=y.device,
        )
        cls_loss = classification_loss(
            logits=out["logits"][mask],
            target_cls=y_cls[mask],
            class_weights=class_weights,
        )

        total = reg_loss + model_cfg["cls_loss_weight"] * cls_loss
        return total

    def train(self, data: Data, stage: str = "stage1") -> TrainOutputs:
        required_masks = ["train_mask", "val_mask", "test_mask"]
        for mask_name in required_masks:
            if not hasattr(data, mask_name):
                raise ValueError(f"Data object is missing '{mask_name}'.")

        data = data.to(self.device)
        y_raw = data.y_raw if hasattr(data, "y_raw") else data.y.clone()

        model = self.build_model(in_dim=data.x.shape[1], stage=stage)

        train_cfg = self.config["training"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )

        best_val_loss = float("inf")
        checkpoint_name = self.config["paths"].get(
            "checkpoint_name",
            f"{stage}_{self.config['data']['split_type']}_best.pt"
        )
        best_model_path = self.checkpoints_dir / checkpoint_name

        patience = 0
        history_rows = []

        for epoch in range(1, train_cfg["max_epochs"] + 1):
            model.train()
            optimizer.zero_grad()

            out = model(data.x, data.edge_index)
            train_loss = self._compute_loss(out, data.y, y_raw, data.train_mask)

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip_norm"])
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out_val = model(data.x, data.edge_index)
                val_loss = self._compute_loss(out_val, data.y, y_raw, data.val_mask)

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
        history.to_csv(self.tables_dir / self.config["paths"]["training_history_name"], index=False)

        return TrainOutputs(
            best_model_path=str(best_model_path),
            best_val_loss=best_val_loss,
            history=history,
        )

    def predict(self, data: Data, checkpoint_path: str, stage: str = "stage1"):
        data = data.to(self.device)
        model = self.build_model(in_dim=data.x.shape[1], stage=stage)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=False))

        mc_enabled = self.config["uncertainty"].get("enable_mc_dropout", False)
        mc_samples = self.config["uncertainty"].get("mc_samples", 20)

        if not mc_enabled:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)

            mean = out["mean"]
            if "var" in out:
                unc = torch.sqrt(out["var"])
            else:
                unc = torch.zeros_like(mean)

            return mean.detach().cpu().numpy(), unc.detach().cpu().numpy()

        preds = []
        vars_ = []

        model.train()
        with torch.no_grad():
            for _ in range(mc_samples):
                out = model(data.x, data.edge_index)
                preds.append(out["mean"])
                if "var" in out:
                    vars_.append(out["var"])

        preds = torch.stack(preds, dim=0)
        pred_mean = preds.mean(dim=0)
        epistemic = preds.std(dim=0)

        if vars_:
            vars_ = torch.stack(vars_, dim=0)
            aleatoric = torch.sqrt(vars_.mean(dim=0))
            total_unc = torch.sqrt(epistemic ** 2 + aleatoric ** 2)
        else:
            total_unc = epistemic

        return pred_mean.cpu().numpy(), total_unc.cpu().numpy()

    def evaluate(self, data: Data, checkpoint_path: str, stage: str = "stage1") -> Dict[str, float]:
        preds, unc = self.predict(data, checkpoint_path, stage=stage)

        y_true = data.y.detach().cpu().numpy()
        y_true_raw = data.y_raw.detach().cpu().numpy() if hasattr(data, "y_raw") else self._inverse_target_transform(y_true)
        test_mask = data.test_mask.detach().cpu().numpy().astype(bool)

        y_test = y_true_raw[test_mask]
        pred_test = self._inverse_target_transform(preds[test_mask])
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
            rel_df.to_csv(self.tables_dir / self.config["paths"]["reliability_table_name"], index=False)

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