from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import yaml
import pandas as pd
import torch

from wildfire_gnn.features.graph_enhancements import prepare_graph_for_gnn
from wildfire_gnn.pipelines.train_pipeline import build_masks_from_split
from wildfire_gnn.pipelines.eval_pipeline import (
    load_trained_model,
    predict_with_uncertainty,
    build_prediction_dataframe,
)
from wildfire_gnn.evaluation.calibration import calibration_summary


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

    y_pred, epistemic_std, aleatoric_std, total_std = predict_with_uncertainty(
        model=model,
        data=data,
        transformer=transformer,
        mc_samples=cfg["uncertainty"]["mc_dropout_samples"],
    )

    test_df = build_prediction_dataframe(
        data=data,
        y_pred=y_pred,
        epistemic_std=epistemic_std,
        aleatoric_std=aleatoric_std,
        total_std=total_std,
        mask=test_mask,
        split_name="test",
    )

    uncertainty_path = os.path.join(
        cfg["outputs"]["tables_dir"],
        cfg["outputs"]["uncertainty_predictions_filename"],
    )
    test_df.to_csv(uncertainty_path, index=False)

    summary = calibration_summary(
        y_true=test_df["y_true"].values,
        mean=test_df["y_pred"].values,
        std=test_df["total_std"].values,
        n_bins=cfg["evaluation"]["calibration_bins"],
    )
    summary_df = pd.DataFrame([summary])

    calibration_path = os.path.join(
        cfg["outputs"]["tables_dir"],
        cfg["outputs"]["calibration_metrics_filename"],
    )
    summary_df.to_csv(calibration_path, index=False)

    print(f"Saved uncertainty predictions to: {uncertainty_path}")
    print(f"Saved calibration metrics to: {calibration_path}")


if __name__ == "__main__":
    main()