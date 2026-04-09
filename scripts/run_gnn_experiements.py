import os
import sys

os.chdir("..")
sys.path.append("src")

import torch

from wildfire_gnn.pipelines.gnn_pipeline import GNNPipeline
from wildfire_gnn.utils.config import load_yaml_config


def main():
    config = load_yaml_config("configs/gnn_config.yaml")
    pipeline = GNNPipeline(config)

    graph_path = config["paths"]["graph_data_path"]
    data = torch.load(graph_path)

    print("Loaded graph:")
    print(data)

    train_outputs = pipeline.train(data, stage="stage1")
    print(f"Best model saved to: {train_outputs.best_model_path}")
    print(f"Best val loss: {train_outputs.best_val_loss:.6f}")

    metrics = pipeline.evaluate(
        data=data,
        checkpoint_path=train_outputs.best_model_path,
        stage="stage1",
    )
    print("Final test metrics:")
    print(metrics)


if __name__ == "__main__":
    main()