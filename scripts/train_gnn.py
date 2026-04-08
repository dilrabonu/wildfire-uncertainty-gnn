from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import yaml
import torch

from wildfire_gnn.pipelines.train_pipeline import train_uncertainty_gnn


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    cfg = load_yaml("configs/gnn_config.yaml")

    ensure_dir(cfg["outputs"]["checkpoints_dir"])
    ensure_dir(cfg["outputs"]["logs_dir"])
    ensure_dir(cfg["outputs"]["tables_dir"])
    ensure_dir(cfg["outputs"]["figures_dir"])

    data = torch.load(cfg["paths"]["graph_data_path"], map_location="cpu", weights_only=False,)
    result = train_uncertainty_gnn(data, cfg)

    checkpoint_path = os.path.join(
        cfg["outputs"]["checkpoints_dir"],
        cfg["outputs"]["checkpoint_name"],
    )

    torch.save(
        {
            "model_state_dict": result.model_state,
            "best_val_loss": result.best_val_loss,
            "best_epoch": result.best_epoch,
        },
        checkpoint_path,
    )

    print(f"Saved checkpoint to: {checkpoint_path}")
    print(f"Best epoch: {result.best_epoch}")
    print(f"Best val loss: {result.best_val_loss:.6f}")


if __name__ == "__main__":
    main()