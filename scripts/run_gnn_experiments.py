from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from wildfire_gnn.pipelines.gnn_pipeline import run_gnn_pipeline


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gnn_config.yaml")
    parser.add_argument("--model", type=str, default=None, choices=["gcn", "gat"])
    parser.add_argument("--split", type=str, default=None, choices=["random", "spatial"])
    args = parser.parse_args()

    config = load_yaml(args.config)

    if args.model is not None:
        config["model"]["name"] = args.model
    if args.split is not None:
        config["split"]["type"] = args.split

    metrics = run_gnn_pipeline(config)

    print("\nFinal Metrics")
    for split_name, split_metrics in metrics.items():
        print(f"\n[{split_name.upper()}]")
        for k, v in split_metrics.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()