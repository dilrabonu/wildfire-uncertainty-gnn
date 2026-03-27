from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from wildfire_gnn.data.preprocessing import build_valid_data_mask, read_single_band_raster
from wildfire_gnn.utils.logger import get_logger

try:
    import torch
    from torch_geometric.data import Data
except ImportError:  # pragma: no cover
    torch = None
    Data = None


@dataclass(frozen=True)
class ReferenceRaster:
    array: np.ndarray
    meta: dict[str, Any]
    transform: Any
    crs: Any
    width: int
    height: int
    nodata: float | int | None


class WildfireGraphBuilder:
    """Build graph-ready aligned raster stacks and PyG graph objects."""

    def __init__(self, config: dict[str, Any], dataset_manager: Any) -> None:
        self.config = config
        self.dataset_manager = dataset_manager
        self.logger = get_logger(__name__)

        self.target_name = config["layers"]["target"]
        self.continuous_features = config["layers"]["continuous_features"]
        self.categorical_features = config["layers"]["categorical_features"]

        self.connectivity = int(config["graph"]["connectivity"])
        self.include_coordinates = bool(config["graph"]["include_coordinates"])
        self.coordinate_mode = str(config["graph"]["coordinate_mode"])
        self.normalize_continuous_features = bool(
            config["graph"]["normalize_continuous_features"]
        )

        self.max_nodes = config["graph"].get("max_nodes")
        self.downsample_factor = int(config["graph"].get("downsample_factor", 1))

        self.output_graph_path = Path(config["output"]["graph_data_path"])
        self.aligned_stack_dir = Path(config["output"]["aligned_stack_dir"])

    def _get_reference_raster(self) -> ReferenceRaster:
        target_path = self.dataset_manager.paths.raw_files_dir / self.target_name
        array, meta = read_single_band_raster(target_path)

        if self.downsample_factor > 1:
            array = array[:: self.downsample_factor, :: self.downsample_factor]

            transform = meta["transform"] * rasterio.Affine.scale(
                self.downsample_factor, self.downsample_factor
            )

            meta = meta.copy()
            meta["transform"] = transform
            meta["height"], meta["width"] = array.shape

        return ReferenceRaster(
            array=array,
            meta=meta,
            transform=meta["transform"],
            crs=meta["crs"],
            width=meta["width"],
            height=meta["height"],
            nodata=meta["nodata"],
        )

    @staticmethod
    def _resampling_from_string(name: str) -> Resampling:
        mapping = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
        }
        if name not in mapping:
            raise ValueError(f"Unsupported resampling method: {name}")
        return mapping[name]

    def _align_raster_to_reference(
        self,
        source_path: str | Path,
        reference: ReferenceRaster,
        resampling: Resampling,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        source_path = Path(source_path)

        with rasterio.open(source_path) as src:
            src_array = src.read(1)

            dst_array = np.full(
                (reference.height, reference.width),
                fill_value=np.nan,
                dtype=np.float32,
            )

            reproject(
                source=src_array,
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=reference.transform,
                dst_crs=reference.crs,
                dst_nodata=np.nan,
                resampling=resampling,
            )

            dst_meta = {
                "source_name": source_path.name,
                "source_crs": src.crs,
                "reference_crs": reference.crs,
                "transform": reference.transform,
                "width": reference.width,
                "height": reference.height,
                "nodata": np.nan,
                "dtype": str(dst_array.dtype),
            }

        return dst_array, dst_meta

    def align_all_features(self) -> tuple[ReferenceRaster, dict[str, np.ndarray]]:
        reference = self._get_reference_raster()

        aligned_features: dict[str, np.ndarray] = {}
        self.aligned_stack_dir.mkdir(parents=True, exist_ok=True)

        continuous_resampling = self._resampling_from_string(
            self.config["alignment"]["continuous_resampling"]
        )
        categorical_resampling = self._resampling_from_string(
            self.config["alignment"]["categorical_resampling"]
        )

        for feature_name in self.continuous_features:
            feature_path = self.dataset_manager.paths.raw_files_dir / feature_name
            aligned_array, _ = self._align_raster_to_reference(
                source_path=feature_path,
                reference=reference,
                resampling=continuous_resampling,
            )
            aligned_features[feature_name] = aligned_array
            self.logger.info("Aligned continuous feature: %s", feature_name)

        for feature_name in self.categorical_features:
            feature_path = self.dataset_manager.paths.raw_files_dir / feature_name
            aligned_array, _ = self._align_raster_to_reference(
                source_path=feature_path,
                reference=reference,
                resampling=categorical_resampling,
            )
            aligned_features[feature_name] = aligned_array
            self.logger.info("Aligned categorical feature: %s", feature_name)

        return reference, aligned_features

    def build_valid_mask(
        self,
        reference: ReferenceRaster,
        aligned_features: dict[str, np.ndarray],
    ) -> np.ndarray:
        valid_mask = build_valid_data_mask(reference.array, nodata=reference.nodata)

        for name, arr in aligned_features.items():
            feature_mask = build_valid_data_mask(arr, nodata=np.nan)
            valid_mask &= feature_mask
            self.logger.info(
                "Valid pixels after feature %s: %d", name, int(valid_mask.sum())
            )

        return valid_mask

    @staticmethod
    def _clean_continuous_values(values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float32, copy=False)
        mask = np.isfinite(values)
        cleaned = values[mask]

        if cleaned.size == 0:
            raise ValueError("No valid finite values available for normalization.")

        return cleaned

    @staticmethod
    def _normalize_feature(values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float32, copy=False)

        finite_mask = np.isfinite(values)
        if not np.all(finite_mask):
            raise ValueError("Feature contains non-finite values before normalization.")

        values64 = values.astype(np.float64, copy=False)
        mean = np.mean(values64)
        std = np.std(values64)

        if not np.isfinite(mean) or not np.isfinite(std):
            raise ValueError("Normalization statistics are not finite.")

        if std < 1e-12:
            return (values64 - mean).astype(np.float32)

        normalized = (values64 - mean) / std
        return normalized.astype(np.float32)

    def build_node_features_and_target(
        self,
        reference: ReferenceRaster,
        aligned_features: dict[str, np.ndarray],
        valid_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        row_indices, col_indices = np.where(valid_mask)
        num_nodes = len(row_indices)

        if self.max_nodes is not None and num_nodes > self.max_nodes:
            self.logger.info(
                "Subsampling nodes from %d to %d for memory-safe prototype.",
                num_nodes,
                self.max_nodes,
            )
            rng = np.random.default_rng(seed=int(self.config["project"]["random_seed"]))
            chosen = rng.choice(num_nodes, size=int(self.max_nodes), replace=False)
            row_indices = row_indices[chosen]
            col_indices = col_indices[chosen]
            num_nodes = len(row_indices)

        self.logger.info("Number of valid nodes used: %d", num_nodes)

        feature_columns: list[np.ndarray] = []

        for feature_name in self.continuous_features:
            full_values = aligned_features[feature_name][row_indices, col_indices].astype(np.float32)

            if self.normalize_continuous_features:
                full_values = self._normalize_feature(full_values)

            feature_columns.append(full_values.reshape(-1, 1))
            self.logger.info(
                "Continuous feature %s | min=%.4f max=%.4f",
                feature_name,
                float(np.min(full_values)),
                float(np.max(full_values)),
            )

        for feature_name in self.categorical_features:
            values = aligned_features[feature_name][row_indices, col_indices].astype(np.float32)
            feature_columns.append(values.reshape(-1, 1))
            self.logger.info(
                "Categorical feature %s | min=%.4f max=%.4f",
                feature_name,
                float(np.min(values)),
                float(np.max(values)),
            )

        if self.include_coordinates:
            if self.coordinate_mode == "normalized_row_col":
                row_feat = row_indices.astype(np.float32) / max(reference.height - 1, 1)
                col_feat = col_indices.astype(np.float32) / max(reference.width - 1, 1)
                feature_columns.append(row_feat.reshape(-1, 1))
                feature_columns.append(col_feat.reshape(-1, 1))
            else:
                raise ValueError(f"Unsupported coordinate_mode: {self.coordinate_mode}")

        x = np.concatenate(feature_columns, axis=1).astype(np.float32)
        y = reference.array[row_indices, col_indices].astype(np.float32).reshape(-1, 1)
        pos = np.stack([row_indices, col_indices], axis=1).astype(np.int64)

        sampled_mask = np.zeros(valid_mask.shape, dtype=bool)
        sampled_mask[row_indices, col_indices] = True

        node_index_map = np.full(valid_mask.shape, fill_value=-1, dtype=np.int64)
        node_index_map[row_indices, col_indices] = np.arange(num_nodes, dtype=np.int64)

        return x, y, pos, node_index_map, sampled_mask

    def build_edge_index(
        self,
        active_mask: np.ndarray,
        node_index_map: np.ndarray,
    ) -> np.ndarray:
        if self.connectivity == 4:
            neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif self.connectivity == 8:
            neighbor_offsets = [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1),
            ]
        else:
            raise ValueError("Connectivity must be either 4 or 8.")

        rows, cols = np.where(active_mask)
        edges: list[tuple[int, int]] = []

        height, width = active_mask.shape

        for r, c in zip(rows, cols):
            src_idx = node_index_map[r, c]
            for dr, dc in neighbor_offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and active_mask[nr, nc]:
                    dst_idx = node_index_map[nr, nc]
                    edges.append((src_idx, dst_idx))

        edge_index = np.array(edges, dtype=np.int64).T
        self.logger.info("Built edge_index with %d directed edges.", edge_index.shape[1])
        return edge_index

    def build_pyg_data(self) -> Any:
        if torch is None or Data is None:
            raise ImportError(
                "torch and torch_geometric are required to build the PyG graph object."
            )

        reference, aligned_features = self.align_all_features()
        valid_mask = self.build_valid_mask(reference, aligned_features)

        x, y, pos, node_index_map, active_mask = self.build_node_features_and_target(
            reference=reference,
            aligned_features=aligned_features,
            valid_mask=valid_mask,
        )

        edge_index = self.build_edge_index(
            active_mask=active_mask,
            node_index_map=node_index_map,
        )

        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            pos=torch.tensor(pos, dtype=torch.long),
        )

        data.num_nodes_original_grid = int(reference.height * reference.width)
        data.num_valid_nodes_before_sampling = int(valid_mask.sum())
        data.num_valid_nodes = int(x.shape[0])
        data.reference_height = int(reference.height)
        data.reference_width = int(reference.width)
        data.target_name = self.target_name
        data.feature_names = self.continuous_features + self.categorical_features + (
            ["row_norm", "col_norm"] if self.include_coordinates else []
        )

        return data

    def save_pyg_data(self, data: Any) -> None:
        if torch is None:
            raise ImportError("torch is required to save graph data.")

        self.output_graph_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, self.output_graph_path)
        self.logger.info("Saved graph data to %s", self.output_graph_path)