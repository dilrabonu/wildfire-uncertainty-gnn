from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from wildfire_gnn.models.gat_model import GATRegressor
from wildfire_gnn.models.gcn_model import GCNRegressor
from wildfire_gnn.training.gnn_trainer import GNNTrainer


def build_model(config: dict):
    model_name = config["model"]["name"].lower()
    common = dict(
        in_channels=config["model"]["in_channels"],
        hidden_channels=config["model"]["hidden_channels"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        use_batch_norm=config["model"]["use_batch_norm"],
        residual=config["model"]["residual"],
        uncertainty=config["uncertainty"]["enabled"],
    )

    if model_name == "gcn":
        return GCNRegressor(**common)
    if model_name == "gat":
        return GATRegressor(**common, heads=config["model"]["heads"])

    raise ValueError(f"Unsupported model: {model_name}")


def run_gnn_pipeline(config: dict) -> dict:
    graph_path = config["data"]["graph_data_path"]
    data = torch.load(graph_path)

    model = build_model(config)

    ckpt_dir = Path(config["outputs"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / f'{config["model"]["name"]}_{config["split"]["type"]}_best.pt'

    trainer = GNNTrainer(model=model, config=config, device=config["training"]["device"])
    output = trainer.train(data, checkpoint_path=checkpoint_path)

    metrics_rows = []
    for split_name, vals in output.metrics.items():
        row = {
            "model": config["model"]["name"],
            "split_type": config["split"]["type"],
            "data_split": split_name,
            **vals,
        }
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = Path(config["outputs"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if metrics_path.exists():
        old_df = pd.read_csv(metrics_path)
        metrics_df = pd.concat([old_df, metrics_df], ignore_index=True)

    metrics_df.to_csv(metrics_path, index=False)
    return output.metrics