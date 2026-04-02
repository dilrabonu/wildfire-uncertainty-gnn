from __future__ import annotations

from pathlib import Path

from wildfire_gnn.data.loader import WildfireDatasetManager
from wildfire_gnn.data.preprocessing import (
    align_feature_stack_to_reference,
    read_single_band_raster,
    summarize_array,
)
from wildfire_gnn.data.simulation_parser import discover_metadata_files
from wildfire_gnn.utils.config import load_yaml_config
from wildfire_gnn.utils.logger import get_logger
from wildfire_gnn.utils.seed import set_global_seed


def main() -> None:
    config = load_yaml_config("configs/data_config.yaml")
    logger = get_logger("preprocess_data", log_file="reports/logs/preprocess_data.log")

    seed = int(config["project"]["random_seed"])
    set_global_seed(seed)
    logger.info("Global seed set to %d", seed)

    manager = WildfireDatasetManager(config)
    manager.validate_structure()

    raster_files = manager.list_raster_files()
    vector_files = manager.list_vector_files()
    metadata_files = discover_metadata_files(manager.paths.metadata_dir)

    logger.info("Raster files found: %d", len(raster_files))
    logger.info("Vector files found: %d", len(vector_files))
    logger.info("Metadata files found: %d", len(metadata_files))

    for raster_path in raster_files:
        array, meta = read_single_band_raster(raster_path)

        try:
            stats = summarize_array(array, nodata=meta.get("nodata"))
        except ValueError as exc:
            logger.warning("Could not summarize raster %s: %s", raster_path.name, exc)
            continue

        logger.info("Raster: %s", raster_path.name)
        logger.info(
            "Shape=%s | DType=%s | CRS=%s | Bounds=%s | NoData=%s | Stats=%s",
            array.shape,
            meta.get("dtype"),
            meta.get("crs"),
            meta.get("bounds"),
            meta.get("nodata"),
            stats,
        )

    try:
        gdb_layers = manager.list_gdb_layers()
        logger.info("GDB layers: %s", gdb_layers)
    except ImportError as exc:
        logger.warning("Skipping GDB layer listing: %s", exc)
        gdb_layers = []

    out_path = Path("reports/logs/dataset_inventory.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("DATASET INVENTORY\n")
        f.write("=================\n\n")

        f.write("Rasters:\n")
        for p in raster_files:
            f.write(f"- {p}\n")

        f.write("\nVectors:\n")
        for p in vector_files:
            f.write(f"- {p}\n")

        f.write("\nMetadata:\n")
        for p in metadata_files:
            f.write(f"- {p}\n")

        f.write("\nGDB Layers:\n")
        if gdb_layers:
            for layer in gdb_layers:
                f.write(f"- {layer}\n")
        else:
            f.write("- Not available yet (install fiona to inspect .gdb layers)\n")

    logger.info("Dataset inventory saved to %s", out_path)

   
    raw_raster_dir = manager.paths.raw_files_dir
    aligned_dir = Path("data/interim/aligned")

    logger.info("Starting raster alignment...")
    logger.info("Reference raster: Burn_Prob.img")
    logger.info("Raw raster dir: %s", raw_raster_dir)
    logger.info("Aligned output dir: %s", aligned_dir)

    saved_paths = align_feature_stack_to_reference(
        raw_dir=raw_raster_dir,
        output_dir=aligned_dir,
        reference_filename="Burn_Prob.img",
    )

    logger.info("Aligned rasters saved successfully:")
    for path in saved_paths:
        logger.info("  %s", path)

  
    logger.info("Verifying aligned rasters...")
    for aligned_path in saved_paths:
        array, meta = read_single_band_raster(aligned_path)

        try:
            stats = summarize_array(array, nodata=meta.get("nodata"))
        except ValueError as exc:
            logger.warning("Could not summarize aligned raster %s: %s", aligned_path.name, exc)
            continue

        logger.info("Aligned Raster: %s", aligned_path.name)
        logger.info(
            "Shape=%s | DType=%s | CRS=%s | Transform=%s | NoData=%s | Stats=%s",
            array.shape,
            meta.get("dtype"),
            meta.get("crs"),
            meta.get("transform"),
            meta.get("nodata"),
            stats,
        )

    logger.info("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()