import os
import sys
import copy
import gc
import pandas as pd
import torch

sys.path.append("src")

from wildfire_gnn.utils.config import load_yaml_config
from wildfire_gnn.pipelines.gnn_pipeline import GNNPipeline
from wildfire_gnn.features.feature_engineering import (
    set_feature_names,
    get_feature_names,
    add_degree_feature,
    add_neighborhood_summary_features,
    add_two_hop_summary_features,
    add_feature_interactions,
    subset_graph_features,
)
from wildfire_gnn.data.graph_splitters import attach_masks_from_split_file


def main():
    config = load_yaml_config("configs/gnn_config_gat.yaml")
    stage = "stage2"

    data = torch.load(
        config["paths"]["graph_data_path"],
        map_location="cpu",
        weights_only=False
    )

    data = attach_masks_from_split_file(
        data,
        config["paths"]["spatial_split_path"]
    )

    data = set_feature_names(data, config["data"]["raw_feature_names"])
    data.y_raw = data.y.clone()

    if config["data"].get("target_transform", "none") == "log1p":
        data.y = torch.log1p(data.y)

    if config["feature_engineering"].get("add_degree_feature", False):
        data = add_degree_feature(data)

    if config["feature_engineering"].get("add_neighborhood_features", False):
        aggs = set(config["feature_engineering"].get("neighborhood_aggs", []))
        data = add_neighborhood_summary_features(
            data,
            add_mean=("mean" in aggs),
            add_std=("std" in aggs),
            add_max=("max" in aggs),
            add_min=("min" in aggs),
            add_residual=("residual" in aggs),
            add_contrast=("contrast" in aggs),
        )

    if config["feature_engineering"].get("add_two_hop_features", False):
        aggs2 = set(config["feature_engineering"].get("two_hop_aggs", []))
        data = add_two_hop_summary_features(
            data,
            add_mean=("mean" in aggs2),
            add_std=("std" in aggs2),
        )

    if config["feature_engineering"].get("add_feature_interactions", False):
        data = add_feature_interactions(
            data,
            config["feature_engineering"]["interaction_pairs"]
        )

    all_feature_names = get_feature_names(data)
    raw_names = config["data"]["raw_feature_names"]

    feature_sets = {
        "all_features": all_feature_names,
        "raw_only": [n for n in all_feature_names if n in raw_names],
        "raw_plus_degree": [n for n in all_feature_names if n in raw_names or n == "degree"],
        "raw_plus_mean_std": [
            n for n in all_feature_names
            if n in raw_names or n == "degree" or n.endswith("_nbr_mean") or n.endswith("_nbr_std")
        ],
        "full_engineered": all_feature_names,
        "no_coordinates": [n for n in all_feature_names if n not in ["row_norm", "col_norm"]],
        "no_fuel": [n for n in all_feature_names if "Fuel_Models.img" not in n and not n.startswith("fuel")],
        "coordinates_only": [n for n in all_feature_names if n in ["row_norm", "col_norm"]],
        "fuel_and_coordinates": [
            n for n in all_feature_names
            if n in ["Fuel_Models.img", "row_norm", "col_norm"] or "Fuel_Models.img" in n
        ],
    }

    ablation_rows = []

    for set_name, keep_names in feature_sets.items():
        print(f"Running ablation: {set_name}")

        data_sub = subset_graph_features(data, keep_names)

        config_sub = copy.deepcopy(config)
        config_sub["paths"]["metrics_table_path"] = f"reports/tables/ablation_{set_name}_metrics.csv"
        config_sub["paths"]["bin_metrics_table_path"] = f"reports/tables/ablation_{set_name}_bin_metrics.csv"
        config_sub["paths"]["predictions_table_path"] = f"reports/tables/ablation_{set_name}_predictions.csv"
        config_sub["paths"]["checkpoint_name"] = f"ablation_{set_name}_best.pt"
        config_sub["paths"]["training_history_name"] = f"ablation_{set_name}_training_history.csv"
        config_sub["paths"]["reliability_table_name"] = f"ablation_{set_name}_reliability.csv"

        pipe_sub = GNNPipeline(config_sub)

        out_sub = pipe_sub.train(data_sub, stage=stage)
        metrics_sub = pipe_sub.evaluate(
            data=data_sub,
            checkpoint_path=out_sub.best_model_path,
            stage=stage,
        )

        row = {"feature_set": set_name, "num_features": data_sub.x.shape[1], **metrics_sub}
        ablation_rows.append(row)

        pd.DataFrame(ablation_rows).to_csv("reports/tables/gat_feature_ablation_partial.csv", index=False)

        del data_sub, pipe_sub, out_sub, metrics_sub
        gc.collect()

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(config["paths"]["ablation_table_path"], index=False)
    print(ablation_df.sort_values("rmse"))


if __name__ == "__main__":
    main()