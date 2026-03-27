from __future__ import annotations

from wildfire_gnn.data.graph_builder import WildfireGraphBuilder
from wildfire_gnn.data.loader import WildfireDatasetManager
from wildfire_gnn.utils.config import load_yaml_config
from wildfire_gnn.utils.logger import get_logger
from wildfire_gnn.utils.seed import set_global_seed


def main() -> None:
    config = load_yaml_config("configs/data_config.yaml")
    logger = get_logger("build_graphs", log_file="reports/logs/build_graphs.log")

    seed = int(config["project"]["random_seed"])
    set_global_seed(seed)
    logger.info("Global seed set to %d", seed)

    manager = WildfireDatasetManager(config)
    manager.validate_structure()

    builder = WildfireGraphBuilder(config=config, dataset_manager=manager)
    graph_data = builder.build_pyg_data()
    builder.save_pyg_data(graph_data)

    logger.info("Graph build complete.")
    logger.info("Number of valid nodes: %d", graph_data.num_valid_nodes)
    logger.info("Feature names: %s", graph_data.feature_names)
    logger.info("Edge count: %d", graph_data.edge_index.shape[1])


if __name__ == "__main__":
    main()