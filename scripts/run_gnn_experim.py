from __future__ import annotations

import copy
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from wildfire_gnn.pipelines.gnn_pipeline import run_gnn_pipeline
from wildfire_gnn.utils.config import load_yaml_config


def main():
    config = load_yaml_config("configs/gnn_config.yaml")

    experiments = [
        {"model": "gcn", "layers": 2, "hidden": 64, "dropout": 0.1},
        {"model": "gcn", "layers": 3, "hidden": 64, "dropout": 0.1},
        {"model": "gcn", "layers": 3, "hidden": 128, "dropout": 0.1},
        {"model": "gat", "layers": 2, "hidden": 64, "dropout": 0.1},
        {"model": "gat", "layers": 3, "hidden": 64, "dropout": 0.1},
        {"model": "gat", "layers": 3, "hidden": 128, "dropout": 0.1},
    ]

    for exp in experiments:
        run_cfg = copy.deepcopy(config)
        run_cfg["model"]["name"] = exp["model"]
        run_cfg["model"]["num_layers"] = exp["layers"]
        run_cfg["model"]["hidden_channels"] = exp["hidden"]
        run_cfg["model"]["dropout"] = exp["dropout"]

        print("\n" + "=" * 80)
        print(
            f"Running {exp['model'].upper()} | "
            f"layers={exp['layers']} | hidden={exp['hidden']} | dropout={exp['dropout']}"
        )
        print("=" * 80)

        metrics = run_gnn_pipeline(run_cfg)
        print(metrics)


if __name__ == "__main__":
    main()