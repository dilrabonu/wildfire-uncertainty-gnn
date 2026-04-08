from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import yaml
import numpy as np
import pandas as pd
import torch

from wildfire_gnn.features.graph_enhancements import prepare_graph_for_gnn
from wildfire_gnn.pipelines.train_pipeline import build_masks_from_split
from wildfire_gnn.pipelines.eval_pipeline import load_trained_model, predict_with_uncertainty
from wildfire_gnn.evaluation.intervention import intervene_reduce_feature


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_yaml("configs/gnn_config.yaml")

    data = torch.load(cfg["paths"]["graph_data_path"], map_location="cpu", weights_only=False,)
    data, transformer = prepare_graph_for_gnn(
        data,
        transform_name=cfg["target"]["transform"],
        target_max=cfg["target"]["target_max"],
    )

    train_mask, val_mask, test_mask = build_masks_from_split(data, cfg["paths"]["split_path"])
    data.test_mask = test_mask

    checkpoint_path = os.path.join(
        cfg["outputs"]["checkpoints_dir"],
        cfg["outputs"]["checkpoint_name"],
    )

    model = load_trained_model(cfg, input_dim=data.x.shape[1], checkpoint_path=checkpoint_path)

    original_pred, ep_std, al_std, total_std = predict_with_uncertainty(
        model=model,
        data=data,
        transformer=transformer,
        mc_samples=cfg["uncertainty"]["mc_dropout_samples"],
    )

    test_indices = np.where(test_mask.cpu().numpy())[0]
    max_nodes = min(cfg["intervention"]["max_nodes_to_modify"], len(test_indices))
    selected_nodes = test_indices[:max_nodes]

    intervened_data = intervene_reduce_feature(
        data=data,
        feature_index=cfg["intervention"]["feature_index_for_fuel_like_intervention"],
        node_indices=selected_nodes,
        reduction_factor=cfg["intervention"]["reduction_factor"],
    )

    new_pred, new_ep_std, new_al_std, new_total_std = predict_with_uncertainty(
        model=model,
        data=intervened_data,
        transformer=transformer,
        mc_samples=cfg["uncertainty"]["mc_dropout_samples"],
    )

    df = pd.DataFrame({
        "node_id": test_indices,
        "y_true": data.y.cpu().numpy().reshape(-1)[test_indices],
        "y_pred_original": original_pred[test_indices],
        "y_pred_intervened": new_pred[test_indices],
        "effect": new_pred[test_indices] - original_pred[test_indices],
        "original_total_std": total_std[test_indices],
        "intervened_total_std": new_total_std[test_indices],
    })

    pred_path = os.path.join(
        cfg["outputs"]["tables_dir"],
        cfg["outputs"]["intervention_predictions_filename"],
    )
    df.to_csv(pred_path, index=False)

    metrics_df = pd.DataFrame([{
        "n_modified_nodes": len(selected_nodes),
        "reduction_factor": cfg["intervention"]["reduction_factor"],
        "mean_effect": df["effect"].mean(),
        "median_effect": df["effect"].median(),
        "std_effect": df["effect"].std(),
    }])

    metrics_path = os.path.join(
        cfg["outputs"]["tables_dir"],
        cfg["outputs"]["intervention_metrics_filename"],
    )
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Saved intervention predictions to: {pred_path}")
    print(f"Saved intervention metrics to: {metrics_path}")


if __name__ == "__main__":
    main()