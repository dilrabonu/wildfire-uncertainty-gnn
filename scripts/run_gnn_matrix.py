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
    base_cfg = load_yaml_config("configs/gnn_config.yaml")

    experiments = [
        {
            "run_name": "recovery_gcn_conn4",
            "model": "gcn",
            "layers": 2,
            "hidden": 64,
            "dropout": 0.1,
            "connectivity": 4,
        },
        {
            "run_name": "recovery_gcn_conn8",
            "model": "gcn",
            "layers": 2,
            "hidden": 64,
            "dropout": 0.1,
            "connectivity": 8,
        },
        {
            "run_name": "recovery_gat_conn4",
            "model": "gat",
            "layers": 2,
            "hidden": 64,
            "dropout": 0.1,
            "connectivity": 4,
        },
        {
            "run_name": "recovery_gat_conn8",
            "model": "gat",
            "layers": 2,
            "hidden": 64,
            "dropout": 0.1,
            "connectivity": 8,
        },
        {
            "run_name": "recovery_gat_conn4_hidden128",
            "model": "gat",
            "layers": 2,
            "hidden": 128,
            "dropout": 0.1,
            "connectivity": 4,
        },
    ]

    for exp in experiments:
        cfg = copy.deepcopy(base_cfg)
        cfg["run"]["name"] = exp["run_name"]
        cfg["model"]["name"] = exp["model"]
        cfg["model"]["num_layers"] = exp["layers"]
        cfg["model"]["hidden_channels"] = exp["hidden"]
        cfg["model"]["dropout"] = exp["dropout"]
        cfg["topology"]["connectivity"] = exp["connectivity"]

        print("\n" + "=" * 90)
        print(exp["run_name"])
        print("=" * 90)

        metrics = run_gnn_pipeline(cfg)
        print(metrics)


if __name__ == "__main__":
    main()