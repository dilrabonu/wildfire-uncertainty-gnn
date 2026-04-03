from __future__ import annotations

from pathlib import Path
from typing import Any

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


def _load_graph_object(graph_path: str | Path) -> Any:
    """
    Load graph artifact safely for trusted local project files.

    PyTorch 2.6 changed torch.load default behavior to weights_only=True,
    which breaks loading PyG Data objects and other custom serialized objects.
    """
    graph_path = Path(graph_path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    return torch.load(graph_path, map_location="cpu", weights_only=False)


def _extract_graph_data(graph_obj: Any):
    """
    Support either:
    1) a raw PyG Data object
    2) a dict like {"data": ..., "node_df": ...}
    """
    if isinstance(graph_obj, dict):
        if "data" in graph_obj:
            return graph_obj["data"]
        raise KeyError(
            "Loaded graph object is a dict, but it does not contain key 'data'. "
            f"Available keys: {list(graph_obj.keys())}"
        )

    return graph_obj


def _validate_graph_masks(data) -> None:
    """
    Ensure the graph object already contains train/val/test masks.
    """
    required_masks = ["train_mask", "val_mask", "test_mask"]
    missing = [name for name in required_masks if not hasattr(data, name)]

    if missing:
        raise AttributeError(
            "Graph data is missing required masks: "
            f"{missing}. "
            "Please attach train/val/test masks before training, "
            "or update the pipeline to create/load graph splits."
        )


def run_gnn_pipeline(config: dict) -> dict:
    graph_path = config["data"]["graph_data_path"]

    graph_obj = _load_graph_object(graph_path)
    data = _extract_graph_data(graph_obj)
    _validate_graph_masks(data)

    model = build_model(config)

    ckpt_dir = Path(config["outputs"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / f'{config["model"]["name"]}_{config["split"]["type"]}_best.pt'

    trainer = GNNTrainer(
        model=model,
        config=config,
        device=config["training"]["device"],
    )
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