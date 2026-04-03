from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from wildfire_gnn.data.graph_splitters import (
    attach_masks_to_graph,
    load_splits,
    make_random_node_split,
    make_spatial_block_node_split,
    save_splits,
)
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
    graph_path = Path(graph_path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    return torch.load(graph_path, map_location="cpu", weights_only=False)


def _extract_graph_and_node_df(graph_obj: Any):
    """
    Supported graph artifact formats:
    1) raw PyG Data object with .pos available
    2) dict with keys like {"data": ..., "node_df": ...}
    """
    if isinstance(graph_obj, dict):
        if "data" not in graph_obj:
            raise KeyError(
                "Loaded graph object is a dict, but does not contain key 'data'. "
                f"Available keys: {list(graph_obj.keys())}"
            )

        data = graph_obj["data"]
        node_df = graph_obj.get("node_df", None)
        return data, node_df

    return graph_obj, None


def _build_node_df_from_graph(data) -> pd.DataFrame:
    """
    Build node metadata DataFrame from graph object.
    Priority:
    1) row_index / col_index attributes if present
    2) pos tensor if present
    """
    if hasattr(data, "row_index") and hasattr(data, "col_index"):
        row_index = data.row_index.cpu().numpy().reshape(-1)
        col_index = data.col_index.cpu().numpy().reshape(-1)
        return pd.DataFrame({
            "row_index": row_index,
            "col_index": col_index,
        })

    if hasattr(data, "pos") and data.pos is not None:
        pos = data.pos.cpu().numpy()
        if pos.shape[1] < 2:
            raise ValueError("data.pos exists but does not contain at least 2 columns.")
        return pd.DataFrame({
            "row_index": pos[:, 0].astype(int),
            "col_index": pos[:, 1].astype(int),
        })

    raise AttributeError(
        "Could not build node_df from graph. "
        "Expected either graph_obj['node_df'], or data.row_index/data.col_index, or data.pos."
    )


def _resolve_node_df(data, node_df: pd.DataFrame | None) -> pd.DataFrame:
    if node_df is not None:
        required_cols = {"row_index", "col_index"}
        if not required_cols.issubset(node_df.columns):
            raise ValueError(
                f"node_df is missing required columns {required_cols}. "
                f"Found columns: {list(node_df.columns)}"
            )
        return node_df.reset_index(drop=True).copy()

    return _build_node_df_from_graph(data)


def _get_split_path(config: dict) -> Path:
    split_type = config["split"]["type"].lower()
    return Path(f"data/processed/gnn_splits_{split_type}.npz")


def _create_or_load_splits(config: dict, node_df: pd.DataFrame) -> dict[str, Any]:
    split_cfg = config["split"]
    split_type = split_cfg["type"].lower()
    split_path = _get_split_path(config)

    if split_path.exists():
        return load_splits(split_path)

    if split_type == "random":
        splits = make_random_node_split(
            node_df=node_df,
            train_size=split_cfg["train_ratio"],
            val_size=split_cfg["val_ratio"],
            test_size=split_cfg["test_ratio"],
            random_seed=config["project"]["random_seed"],
        )
    elif split_type == "spatial":
        splits = make_spatial_block_node_split(
            node_df=node_df,
            n_row_blocks=split_cfg["n_row_blocks"],
            n_col_blocks=split_cfg["n_col_blocks"],
            train_blocks=split_cfg["train_ratio"],
            val_blocks=split_cfg["val_ratio"],
            test_blocks=split_cfg["test_ratio"],
            random_seed=config["project"]["random_seed"],
        )
    else:
        raise ValueError(f"Unsupported split type: {split_type}")

    save_splits(splits, split_path)
    return splits


def run_gnn_pipeline(config: dict) -> dict:
    graph_path = config["data"]["graph_data_path"]

    graph_obj = _load_graph_object(graph_path)
    data, node_df = _extract_graph_and_node_df(graph_obj)
    node_df = _resolve_node_df(data, node_df)

    splits = _create_or_load_splits(config, node_df)

    data = attach_masks_to_graph(
        data,
        splits["train_idx"],
        splits["val_idx"],
        splits["test_idx"],
    )

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