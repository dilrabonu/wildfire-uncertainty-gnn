from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from wildfire_gnn.data.graph_feature_recovery import add_recovery_features
from wildfire_gnn.data.graph_splitters import (
    attach_masks_to_graph,
    load_splits,
    make_random_node_split,
    make_spatial_block_node_split,
    save_splits,
)
from wildfire_gnn.data.graph_topology import rebuild_spatial_edges_from_pos
from wildfire_gnn.evaluation.graph_diagnostics import (
    make_binned_target_distribution,
    make_split_diagnostics,
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


def _extract_graph_data(graph_obj: Any):
    if isinstance(graph_obj, dict):
        if "data" in graph_obj:
            return graph_obj["data"]
        raise KeyError(
            "Loaded graph object is a dict, but it does not contain key 'data'. "
            f"Available keys: {list(graph_obj.keys())}"
        )
    return graph_obj


def _build_node_df_from_graph(data) -> pd.DataFrame:
    if hasattr(data, "pos") and data.pos is not None:
        pos = data.pos.cpu().numpy()
        return pd.DataFrame({
            "row_index": pos[:, 0].astype(int),
            "col_index": pos[:, 1].astype(int),
        })

    raise AttributeError("Graph data must contain pos to build node_df.")


def _get_split_path(config: dict) -> Path:
    split_type = config["split"]["type"].lower()
    return Path(f"data/processed/gnn_splits_{split_type}.npz")


def _create_or_load_splits(config: dict, node_df: pd.DataFrame) -> dict:
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


def _save_metrics(config: dict, output_metrics: dict) -> None:
    metrics_rows = []
    for split_name, vals in output_metrics.items():
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


def _save_predictions(config: dict, predictions: dict[str, pd.DataFrame]) -> None:
    if not config["evaluation"]["save_predictions"]:
        return

    out_dir = Path(config["outputs"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = config["model"]["name"]
    split_type = config["split"]["type"]

    for split_name, df in predictions.items():
        path = out_dir / f"{model_name}_{split_type}_{split_name}_predictions.csv"
        df.to_csv(path, index=False)


def _save_split_diagnostics(config: dict, data) -> None:
    if not config["evaluation"]["save_split_diagnostics"]:
        return

    split_stats = make_split_diagnostics(data)
    split_bins = make_binned_target_distribution(data)

    base_path = Path(config["outputs"]["split_diagnostics_path"])
    base_path.parent.mkdir(parents=True, exist_ok=True)

    split_stats.to_csv(base_path, index=False)

    bins_path = base_path.with_name(base_path.stem + "_bins.csv")
    split_bins.to_csv(bins_path, index=False)


def run_gnn_pipeline(config: dict) -> dict:
    graph_path = config["data"]["graph_data_path"]
    graph_obj = _load_graph_object(graph_path)
    data = _extract_graph_data(graph_obj)

    if config["topology"]["rebuild_from_pos"]:
        data = rebuild_spatial_edges_from_pos(
            data,
            connectivity=config["topology"]["connectivity"],
            use_distance_weights=config["topology"]["use_distance_weights"],
        )

    data = add_recovery_features(data, config)

    config["model"]["in_channels"] = int(data.x.shape[1])

    node_df = _build_node_df_from_graph(data)
    splits = _create_or_load_splits(config, node_df)

    data = attach_masks_to_graph(
        data,
        splits["train_idx"],
        splits["val_idx"],
        splits["test_idx"],
    )

    _save_split_diagnostics(config, data)

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

    _save_metrics(config, output.metrics)
    _save_predictions(config, output.predictions)

    return output.metrics