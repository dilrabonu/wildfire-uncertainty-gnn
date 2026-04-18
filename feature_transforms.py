import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from sklearn.preprocessing import QuantileTransformer as SKQuantileTransformer
from sklearn.preprocessing import StandardScaler
import warnings
from typing import List, Tuple

warnings.filterwarnings("ignore")


class BaseTransformer:
    """sklearn-compatible base for all transforms in this file."""

    def __init__(self, name: str):
        self.name = name
        self.fitted = False

    def fit(self, df: pd.DataFrame, **kwargs) -> "BaseTransformer":
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
        raise NotImplementedError

    def fit_transform(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
        return self.fit(df, **kwargs).transform(df, **kwargs)


class LogTransformer(BaseTransformer):
    """
    Applies log1p to selected skewed continuous columns, then z-scores numeric features,
    while preserving categorical columns such as Fuel_Models.

    Why:
    - wildfire raster predictors like CFL / FSP / Ignition_Prob are right-skewed
    - Fuel_Models must remain categorical, not be standardized as a float
    """

    def __init__(
        self,
        log_cols: List[str] = None,
        scale_all: bool = True,
        exclude_from_scaling: List[str] = None,
    ):
        super().__init__("LogTransformer")
        self.log_cols = log_cols or [
            "CFL", "FSP_Index", "Struct_Exp_Index", "Ignition_Prob"
        ]
        self.scale_all = scale_all
        self.exclude_from_scaling = exclude_from_scaling or ["Fuel_Models"]
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame, **kwargs) -> "LogTransformer":
        df_work = self._apply_log(df)

        numeric_cols = df_work.select_dtypes(include=np.number).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in self.exclude_from_scaling]

        self.numeric_cols_ = numeric_cols
        self.non_scaled_cols_ = [c for c in self.exclude_from_scaling if c in df_work.columns]

        if self.scale_all and self.numeric_cols_:
            self.scaler.fit(df_work[self.numeric_cols_])

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
        assert self.fitted, "Call fit() first."

        df_work = self._apply_log(df)

        if self.scale_all and self.numeric_cols_:
            scaled = self.scaler.transform(df_work[self.numeric_cols_])
            df_out = pd.DataFrame(scaled, columns=self.numeric_cols_, index=df_work.index)

            preserved_cols = [c for c in df_work.columns if c not in self.numeric_cols_]
            for c in preserved_cols:
                df_out[c] = df_work[c].values
        else:
            df_out = df_work.copy()

        ordered_cols = [c for c in df_work.columns if c in df_out.columns]
        remaining_cols = [c for c in df_out.columns if c not in ordered_cols]
        df_out = df_out[ordered_cols + remaining_cols]

        return df_out, list(self.numeric_cols_)

    def _apply_log(self, df: pd.DataFrame) -> pd.DataFrame:
        df_work = df.copy()

        for col in self.log_cols:
            if col in df_work.columns:
                df_work[col] = np.log1p(np.clip(df_work[col].values, 0, None))

        if "Fuel_Models" in df_work.columns:
            df_work["Fuel_Models"] = df_work["Fuel_Models"].astype(str)

        return df_work


class FuelModelEncoder(BaseTransformer):
    """
    One-hot encodes the Fuel_Models categorical column.
    """

    def __init__(
        self,
        fuel_col: str = "Fuel_Models",
        max_categories: int = 20,
        drop_first: bool = False,
    ):
        super().__init__("FuelModelEncoder")
        self.fuel_col = fuel_col
        self.max_categories = max_categories
        self.drop_first = drop_first

    def fit(self, df: pd.DataFrame, **kwargs) -> "FuelModelEncoder":
        if self.fuel_col not in df.columns:
            self.fuel_col_present_ = False
            self.fitted = True
            return self

        self.fuel_col_present_ = True
        counts = df[self.fuel_col].astype(str).value_counts()
        self.top_categories_ = counts.index[:self.max_categories].tolist()
        self.dummy_cols_ = None
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
        assert self.fitted

        if not self.fuel_col_present_ or self.fuel_col not in df.columns:
            return df, []

        df_work = df.copy()
        df_work[self.fuel_col] = df_work[self.fuel_col].astype(str).apply(
            lambda x: x if x in self.top_categories_ else "other"
        )

        dummies = pd.get_dummies(
            df_work[self.fuel_col],
            prefix="fuel",
            drop_first=self.drop_first,
        )

        if self.dummy_cols_ is None:
            self.dummy_cols_ = list(dummies.columns)
        else:
            for col in self.dummy_cols_:
                if col not in dummies.columns:
                    dummies[col] = 0
            dummies = dummies[self.dummy_cols_]

        df_work = df_work.drop(columns=[self.fuel_col])
        df_out = pd.concat(
            [df_work.reset_index(drop=True), dummies.reset_index(drop=True)],
            axis=1
        )

        return df_out, self.dummy_cols_


class InteractionTerms(BaseTransformer):
    """
    Creates physically motivated interaction features.
    """

    PAIRS = [
        ("Ignition_Prob", "CFL", "inter_ignition_x_cfl"),
        ("Ignition_Prob", "Struct_Exp_Index", "inter_ignition_x_struct"),
        ("CFL", "FSP_Index", "inter_cfl_x_fsp"),
    ]

    def __init__(self, pairs: List[Tuple[str, str, str]] = None):
        super().__init__("InteractionTerms")
        self.pairs = pairs or self.PAIRS
        self.fitted = True

    def fit(self, df: pd.DataFrame, **kwargs) -> "InteractionTerms":
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
        df_work = df.copy()
        new_cols = []

        for col_a, col_b, name in self.pairs:
            if col_a in df_work.columns and col_b in df_work.columns:
                df_work[name] = df_work[col_a].values * df_work[col_b].values
                new_cols.append(name)

        return df_work, new_cols


class MultiScaleStats(BaseTransformer):
    """
    Computes mean/std neighborhood statistics over 3x3, 7x7, 15x15 windows.
    """

    def __init__(
        self,
        feature_cols: List[str] = None,
        window_sizes: List[int] = None,
        raster_shape: Tuple[int, int] = None,
    ):
        super().__init__("MultiScaleStats")
        self.feature_cols = feature_cols or ["Ignition_Prob", "CFL", "FSP_Index"]
        self.window_sizes = window_sizes or [3, 7, 15]
        self.raster_shape = raster_shape
        self.fitted = True

    def fit(self, df: pd.DataFrame, raster_shape: Tuple[int, int] = None, **kwargs):
        if raster_shape is not None:
            self.raster_shape = raster_shape
        self.fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        raster_shape: Tuple[int, int] = None,
        pixel_indices: np.ndarray = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, List[str]]:
        shape = raster_shape or self.raster_shape
        if shape is None:
            print("[MultiScaleStats] WARNING: no raster_shape provided — skipping.")
            return df, []

        df_work = df.copy()
        new_cols = []
        H, W = shape

        for col in self.feature_cols:
            if col not in df_work.columns:
                continue

            if pixel_indices is not None:
                raster = np.zeros(H * W, dtype=np.float32)
                raster[pixel_indices] = df_work[col].values
                raster = raster.reshape(H, W)
            else:
                n = len(df_work)
                pad = H * W - n
                vals = np.concatenate([df_work[col].values, np.zeros(pad)])
                raster = vals.reshape(H, W)

            for k in self.window_sizes:
                mean_raster = uniform_filter(raster, size=k, mode="reflect")
                mean2_raster = uniform_filter(raster**2, size=k, mode="reflect")
                std_raster = np.sqrt(np.clip(mean2_raster - mean_raster**2, 0, None))

                col_mean = f"{col}_mean{k}"
                col_std = f"{col}_std{k}"

                if pixel_indices is not None:
                    flat_mean = mean_raster.ravel()[pixel_indices]
                    flat_std = std_raster.ravel()[pixel_indices]
                else:
                    flat_mean = mean_raster.ravel()[:n]
                    flat_std = std_raster.ravel()[:n]

                df_work[col_mean] = flat_mean
                df_work[col_std] = flat_std
                new_cols.extend([col_mean, col_std])

        print(f"[MultiScaleStats] Added {len(new_cols)} neighborhood features.")
        return df_work, new_cols


class SpatialGradients(BaseTransformer):
    """
    Computes row/col gradients and gradient magnitude.
    """

    def __init__(
        self,
        gradient_cols: List[str] = None,
        raster_shape: Tuple[int, int] = None,
    ):
        super().__init__("SpatialGradients")
        self.gradient_cols = gradient_cols or ["Ignition_Prob", "CFL"]
        self.raster_shape = raster_shape
        self.fitted = True

    def fit(self, df: pd.DataFrame, raster_shape=None, **kwargs):
        if raster_shape:
            self.raster_shape = raster_shape
        self.fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        raster_shape: Tuple[int, int] = None,
        pixel_indices: np.ndarray = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, List[str]]:
        shape = raster_shape or self.raster_shape
        if shape is None:
            print("[SpatialGradients] WARNING: no raster_shape — skipping.")
            return df, []

        H, W = shape
        df_work = df.copy()
        new_cols = []

        for col in self.gradient_cols:
            if col not in df_work.columns:
                continue

            if pixel_indices is not None:
                raster = np.zeros(H * W, dtype=np.float32)
                raster[pixel_indices] = df_work[col].values
                raster = raster.reshape(H, W)
            else:
                n = len(df_work)
                pad = H * W - n
                vals = np.concatenate([df_work[col].values, np.zeros(pad)])
                raster = vals.reshape(H, W)

            grad_row, grad_col = np.gradient(raster)

            col_grow = f"{col}_grad_row"
            col_gcol = f"{col}_grad_col"
            col_gmag = f"{col}_grad_mag"

            if pixel_indices is not None:
                gr = grad_row.ravel()[pixel_indices]
                gc = grad_col.ravel()[pixel_indices]
            else:
                gr = grad_row.ravel()[:len(df_work)]
                gc = grad_col.ravel()[:len(df_work)]

            df_work[col_grow] = gr
            df_work[col_gcol] = gc
            df_work[col_gmag] = np.sqrt(gr**2 + gc**2)

            new_cols.extend([col_grow, col_gcol, col_gmag])

        print(f"[SpatialGradients] Added {len(new_cols)} gradient features.")
        return df_work, new_cols


class FuelConnectivityIndex(BaseTransformer):
    """
    Fraction of neighbors with the same fuel model in 3x3 neighborhood.
    """

    def __init__(self, fuel_col: str = "Fuel_Models", raster_shape=None):
        super().__init__("FuelConnectivityIndex")
        self.fuel_col = fuel_col
        self.raster_shape = raster_shape
        self.fitted = True

    def fit(self, df, raster_shape=None, **kwargs):
        if raster_shape:
            self.raster_shape = raster_shape
        self.fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        raster_shape=None,
        pixel_indices=None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, List[str]]:
        shape = raster_shape or self.raster_shape
        if shape is None or self.fuel_col not in df.columns:
            return df, []

        H, W = shape
        df_work = df.copy()

        fuel_numeric = pd.factorize(df_work[self.fuel_col].astype(str))[0].astype(np.float32)

        if pixel_indices is not None:
            fuel_flat = np.zeros(H * W, dtype=np.float32)
            fuel_flat[pixel_indices] = fuel_numeric
            fuel_grid = fuel_flat.reshape(H, W)
        else:
            n = len(df_work)
            pad = H * W - n
            vals = np.concatenate([fuel_numeric, np.zeros(pad)])
            fuel_grid = vals.reshape(H, W)

        same_count = np.zeros_like(fuel_grid, dtype=np.float32)
        total_neighbors = np.zeros_like(fuel_grid, dtype=np.float32)

        shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dr, dc in shifts:
            shifted = np.roll(np.roll(fuel_grid, dr, axis=0), dc, axis=1)
            same_count += (shifted == fuel_grid).astype(np.float32)
            total_neighbors += 1.0

        connectivity = same_count / total_neighbors
        col_name = "fuel_connectivity"

        if pixel_indices is not None:
            df_work[col_name] = connectivity.ravel()[pixel_indices]
        else:
            df_work[col_name] = connectivity.ravel()[:len(df_work)]

        print(f"[FuelConnectivity] Mean connectivity: {df_work[col_name].mean():.3f}")
        return df_work, [col_name]


class QuantileTargetTransformer:
    """
    Maps Burn_Prob to near-Gaussian form for regression.
    """

    def __init__(self, n_quantiles: int = 1000, output_distribution: str = "normal"):
        self.qt = SKQuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            random_state=42,
        )
        self.fitted = False

    def fit(self, y: np.ndarray) -> "QuantileTargetTransformer":
        self.qt.fit(y.reshape(-1, 1))
        self.fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return self.qt.transform(y.reshape(-1, 1)).ravel()

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        assert self.fitted
        return self.qt.inverse_transform(y_transformed.reshape(-1, 1)).ravel()


class FeatureTransformPipeline:
    """
    Chains transformers in order.
    """

    def __init__(self, steps: List[BaseTransformer]):
        self.steps = steps

    def fit_transform(
        self,
        df: pd.DataFrame,
        raster_shape: Tuple[int, int] = None,
        pixel_indices: np.ndarray = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        current_df = df.copy()
        all_new_cols: List[str] = []

        for step in self.steps:
            current_df, new_cols = step.fit_transform(
                current_df,
                raster_shape=raster_shape,
                pixel_indices=pixel_indices,
            )
            all_new_cols.extend(new_cols)
            print(f"  [{step.name}] → {len(current_df.columns)} total columns")

        return current_df, all_new_cols

    def transform(
        self,
        df: pd.DataFrame,
        raster_shape: Tuple[int, int] = None,
        pixel_indices: np.ndarray = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        current_df = df.copy()
        all_new_cols: List[str] = []

        for step in self.steps:
            current_df, new_cols = step.transform(
                current_df,
                raster_shape=raster_shape,
                pixel_indices=pixel_indices,
            )
            all_new_cols.extend(new_cols)

        return current_df, all_new_cols


def build_default_pipeline(raster_shape: Tuple[int, int] = None) -> FeatureTransformPipeline:
    """
    Returns the full default transformation pipeline.
    """
    return FeatureTransformPipeline([
        LogTransformer(
            log_cols=["CFL", "FSP_Index", "Struct_Exp_Index", "Ignition_Prob"],
            scale_all=True,
            exclude_from_scaling=["Fuel_Models"],
        ),
        FuelModelEncoder(
            fuel_col="Fuel_Models",
            max_categories=20,
            drop_first=False,
        ),
        InteractionTerms(),
        MultiScaleStats(
            feature_cols=["Ignition_Prob", "CFL", "FSP_Index"],
            window_sizes=[3, 7, 15],
            raster_shape=raster_shape,
        ),
        SpatialGradients(
            gradient_cols=["Ignition_Prob", "CFL"],
            raster_shape=raster_shape,
        ),
        FuelConnectivityIndex(
            fuel_col="Fuel_Models",
            raster_shape=raster_shape,
        ),
    ])