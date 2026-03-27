from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import rasterio
from rasterio.io import DatasetReader

from wildfire_gnn.utils.logger import get_logger


@dataclass(frozen=True)
class DatasetPaths:
    raw_dir: Path
    raw_files_dir: Path
    gdb_dir: Path
    metadata_dir: Path
    styles_dir: Path


class WildfireDatasetManager:
    """Manage dataset paths, validation, and data loading."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger(__name__)

        paths_cfg = config["paths"]
        dataset_cfg = config["dataset"]

        raw_dir = Path(paths_cfg["raw_dir"])

        self.paths = DatasetPaths(
            raw_dir=raw_dir,
            raw_files_dir=raw_dir / dataset_cfg["raw_files_dirname"],
            gdb_dir=raw_dir / dataset_cfg["gdb_dirname"],
            metadata_dir=raw_dir / dataset_cfg["metadata_dirname"],
            styles_dir=raw_dir / dataset_cfg["styles_dirname"],
        )

    def validate_structure(self) -> None:
        """Validate that the expected dataset folders exist."""
        expected_paths = {
            "raw_dir": self.paths.raw_dir,
            "raw_files_dir": self.paths.raw_files_dir,
            "gdb_dir": self.paths.gdb_dir,
            "metadata_dir": self.paths.metadata_dir,
            "styles_dir": self.paths.styles_dir,
        }

        missing = [name for name, path in expected_paths.items() if not path.exists()]
        if missing:
            details = "\n".join(f"- {name}: {expected_paths[name]}" for name in missing)
            raise FileNotFoundError(f"Missing dataset paths:\n{details}")

        self.logger.info("Dataset structure validation passed.")

    def list_raster_files(self) -> list[Path]:
        """List configured raster files present in the raw files directory."""
        raster_names = self.config["layers"]["rasters"]
        found_files: list[Path] = []

        for name in raster_names:
            path = self.paths.raw_files_dir / name
            if path.exists():
                found_files.append(path)
            else:
                self.logger.warning("Raster file not found: %s", path)

        return found_files

    def list_vector_files(self) -> list[Path]:
        """List configured vector files present in the raw files directory."""
        vector_names = self.config["layers"]["vectors"]
        found_files: list[Path] = []

        for name in vector_names:
            path = self.paths.raw_files_dir / name
            if path.exists():
                found_files.append(path)
            else:
                self.logger.warning("Vector file not found: %s", path)

        return found_files

    def open_raster(self, raster_path: str | Path) -> DatasetReader:
        """Open a raster file with rasterio."""
        path = Path(raster_path)
        if not path.exists():
            raise FileNotFoundError(f"Raster not found: {path}")

        self.logger.info("Opening raster: %s", path)
        return rasterio.open(path)

    def load_vector(self, vector_path: str | Path) -> gpd.GeoDataFrame:
        """Load a vector dataset."""
        path = Path(vector_path)
        if not path.exists():
            raise FileNotFoundError(f"Vector file not found: {path}")

        self.logger.info("Loading vector: %s", path)
        return gpd.read_file(path)

    def list_gdb_layers(self) -> list[str]:
        """List layers inside the file geodatabase."""
        if not self.paths.gdb_dir.exists():
            raise FileNotFoundError(f"GDB directory not found: {self.paths.gdb_dir}")

        try:
            import fiona
        except ImportError as e:
            raise ImportError(
                "fiona is required to inspect geodatabase layers. "
                "Install it with: conda install -c conda-forge fiona -y"
            ) from e

        layers = fiona.listlayers(str(self.paths.gdb_dir))
        self.logger.info("Found %d GDB layers.", len(layers))
        return list(layers)

    def load_gdb_layer(self, layer_name: str) -> gpd.GeoDataFrame:
        """Load a specific layer from the file geodatabase."""
        self.logger.info("Loading GDB layer: %s", layer_name)
        return gpd.read_file(self.paths.gdb_dir, layer=layer_name)