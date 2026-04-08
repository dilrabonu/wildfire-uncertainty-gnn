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
from wildfire_gnn.pipelines.eval_pipeline import (
    load_trained_model,
    predict_with_uncertainty,
    build_prediction_dataframe,
    evaluate_split,
)


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
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    checkpoint_path = os.path.join(
        cfg["outputs"]["checkpoints_dir"],
        cfg["outputs"]["checkpoint_name"],
    )

    model = load_trained_model(cfg, input_dim=data.x.shape[1], checkpoint_path=checkpoint_path)

    y_pred, epistemic_std, aleatoric_std, total_std = predict_with_uncertainty(
        model=model,
        data=data,
        transformer=transformer,
        mc_samples=cfg["uncertainty"]["mc_dropout_samples"],
    )

    y_true = data.y.cpu().numpy().reshape(-1)
    threshold = cfg["evaluation"]["high_risk_threshold"]

    rows = []
    for split_name, mask in [("val", data.val_mask), ("test", data.test_mask)]:
        idx = mask.cpu().numpy()
        metrics = evaluate_split(y_true[idx], y_pred[idx], threshold=threshold)
        metrics["split"] = split_name
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(cfg["outputs"]["tables_dir"], cfg["outputs"]["metrics_filename"])
    metrics_df.to_csv(metrics_path, index=False)

    tail_rows = []
    for split_name, mask in [("val", data.val_mask), ("test", data.test_mask)]:
        idx = mask.cpu().numpy()
        tail_rows.append({
            "split": split_name,
            "high_risk_threshold": threshold,
            "tail_rmse": evaluate_split(y_true[idx], y_pred[idx], threshold)["tail_rmse"],
        })

    tail_df = pd.DataFrame(tail_rows)
    tail_path = os.path.join(cfg["outputs"]["tables_dir"], cfg["outputs"]["tail_metrics_filename"])
    tail_df.to_csv(tail_path, index=False)

    val_df = build_prediction_dataframe(
        data=data,
        y_pred=y_pred,
        epistemic_std=epistemic_std,
        aleatoric_std=aleatoric_std,
        total_std=total_std,
        mask=data.val_mask,
        split_name="val",
    )
    val_df.to_csv(
        os.path.join(cfg["outputs"]["tables_dir"], cfg["outputs"]["val_predictions_filename"]),
        index=False,
    )

    test_df = build_prediction_dataframe(
        data=data,
        y_pred=y_pred,
        epistemic_std=epistemic_std,
        aleatoric_std=aleatoric_std,
        total_std=total_std,
        mask=data.test_mask,
        split_name="test",
    )
    test_df.to_csv(
        os.path.join(cfg["outputs"]["tables_dir"], cfg["outputs"]["predictions_filename"]),
        index=False,
    )

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved tail metrics to: {tail_path}")


if __name__ == "__main__":
    main()