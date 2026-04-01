from __future__ import annotations

from wildfire_gnn.pipelines.baseline_pipeline import BaselinePipeline
from wildfire_gnn.utils.config import load_yaml_config
from wildfire_gnn.utils.seed import set_global_seed


def main() -> None:
    config = load_yaml_config("configs/baseline_config.yaml")
    seed = int(config["project"]["random_seed"])
    set_global_seed(seed)

    pipeline = BaselinePipeline(config)
    results_df = pipeline.train_and_evaluate()

    print("Phase 4A baselines completed successfully.")
    print(results_df.sort_values(["split_type", "data_split", "rmse"]))


if __name__ == "__main__":
    main()