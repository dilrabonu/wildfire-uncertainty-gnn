from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from wildfire_gnn.models.uncertainty_gnn import UncertaintyGNN, mc_dropout_predict
from wildfire_gnn.training.metrics import regression_metrics, tail_rmse


def load_trained_model(cfg: dict, input_dim: int, checkpoint_path: str) -> UncertaintyGNN:
    model = UncertaintyGNN(
        input_dim=input_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        heads=cfg["model"]["heads"],
        dropout=cfg["model"]["dropout"],
        attn_dropout=cfg["model"]["attn_dropout"],
        edge_dim=cfg["model"]["edge_dim"],
        min_variance=cfg["uncertainty"]["min_variance"],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_with_uncertainty(model, data, transformer, mc_samples: int):
    mean_t, epistemic_std_t, aleatoric_std_t = mc_dropout_predict(
        model=model,
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        num_samples=mc_samples,
    )

    mean = transformer.inverse(mean_t).cpu().numpy().reshape(-1)
    epistemic_std = epistemic_std_t.cpu().numpy().reshape(-1)
    aleatoric_std = aleatoric_std_t.cpu().numpy().reshape(-1)
    total_std = np.sqrt(epistemic_std ** 2 + aleatoric_std ** 2)

    return mean, epistemic_std, aleatoric_std, total_std


def build_prediction_dataframe(data, y_pred, epistemic_std, aleatoric_std, total_std, mask, split_name: str):
    y_true = data.y.cpu().numpy().reshape(-1)
    pos = data.pos.cpu().numpy() if hasattr(data, "pos") and data.pos is not None else None
    idx = np.where(mask.cpu().numpy())[0]

    df = pd.DataFrame({
        "node_id": idx,
        "split": split_name,
        "y_true": y_true[idx],
        "y_pred": y_pred[idx],
        "epistemic_std": epistemic_std[idx],
        "aleatoric_std": aleatoric_std[idx],
        "total_std": total_std[idx],
    })

    if pos is not None:
        df["row_norm"] = pos[idx, 0]
        df["col_norm"] = pos[idx, 1]

    return df


def evaluate_split(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict:
    metrics = regression_metrics(y_true, y_pred)
    metrics["tail_rmse"] = tail_rmse(y_true, y_pred, threshold=threshold)
    return metrics