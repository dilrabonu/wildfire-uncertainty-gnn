from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
from torch_geometric.data import Data

from wildfire_gnn.features.graph_enhancements import prepare_graph_for_gnn
from wildfire_gnn.models.uncertainty_gnn import UncertaintyGNN
from wildfire_gnn.training.losses import hybrid_gaussian_nll


@dataclass
class TrainResult:
    best_val_loss: float
    best_epoch: int
    model_state: Dict[str, Any]
    transformer: Any
    enhanced_data: Data


def build_masks_from_split(data: Data, split_npz_path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    split = np.load(split_npz_path)

    if "train_idx" in split:
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
        test_idx = split["test_idx"]
    else:
        raise ValueError("Split file must contain train_idx, val_idx, test_idx")

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[torch.from_numpy(train_idx).long()] = True
    val_mask[torch.from_numpy(val_idx).long()] = True
    test_mask[torch.from_numpy(test_idx).long()] = True

    return train_mask, val_mask, test_mask


def train_uncertainty_gnn(data: Data, cfg: dict) -> TrainResult:
    data, transformer = prepare_graph_for_gnn(
        data,
        transform_name=cfg["target"]["transform"],
        target_max=cfg["target"]["target_max"],
    )

    train_mask, val_mask, test_mask = build_masks_from_split(data, cfg["paths"]["split_path"])
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    model = UncertaintyGNN(
        input_dim=data.x.shape[1],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        heads=cfg["model"]["heads"],
        dropout=cfg["model"]["dropout"],
        attn_dropout=cfg["model"]["attn_dropout"],
        edge_dim=cfg["model"]["edge_dim"],
        min_variance=cfg["uncertainty"]["min_variance"],
    )

    device = torch.device(cfg["training"]["device"])
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    patience = 0

    for epoch in range(cfg["training"]["max_epochs"]):
        model.train()
        optimizer.zero_grad()

        mean, var = model(data.x, data.edge_index, data.edge_attr)
        train_loss = hybrid_gaussian_nll(
            mean[data.train_mask],
            var[data.train_mask],
            data.y_transformed[data.train_mask],
            delta=cfg["loss"]["huber_delta"],
            power=cfg["loss"]["tail_weight_power"],
            max_weight=cfg["loss"]["max_tail_weight"],
            alpha=cfg["loss"]["hybrid_alpha"],
        )
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip_norm"])
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_mean, val_var = model(data.x, data.edge_index, data.edge_attr)
            val_loss = hybrid_gaussian_nll(
                val_mean[data.val_mask],
                val_var[data.val_mask],
                data.y_transformed[data.val_mask],
                delta=cfg["loss"]["huber_delta"],
                power=cfg["loss"]["tail_weight_power"],
                max_weight=cfg["loss"]["max_tail_weight"],
                alpha=cfg["loss"]["hybrid_alpha"],
            ).item()

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss: {train_loss.item():.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if patience >= cfg["training"]["early_stopping_patience"]:
            print("Early stopping triggered.")
            break

    data = data.cpu()

    return TrainResult(
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        model_state=best_state,
        transformer=transformer,
        enhanced_data=data,
    )