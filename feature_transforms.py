"""
Stateless + stateful feature transformations for the wildfire GNN pipeline.

Transformations implemented:
  1. LogTransformer       — log1p for right-skewed raster features
  2. FuelModelEncoder     — one-hot or embedding for Fuel_Models categorical
  3. InteractionTerms     — Ignition_Prob × CFL, Ignition_Prob × Struct_Exp_Index
  4. MultiScaleStats      — mean/std at 3×3, 7×7, 15×15 windows per feature
  5. SpatialGradients     — ∂f/∂row, ∂f/∂col for ignition prob and CFL
  6. FuelConnectivityIdx  — % same-fuel neighbors in 3×3 window
  7. QuantileTargetTransformer — map Burn_Prob to near-Gaussian for regression

Design:
  - All transformers follow sklearn fit/transform interface
  - Stateless transforms (no fit needed) set fitted=True immediately
  - Pipeline-composable via FeatureTransformPipeline

Usage:
    from feature_transforms import FeatureTransformPipeline, build_default_pipeline
    pipeline = build_default_pipeline()
    X_train_transformed, feature_names = pipeline.fit_transform(df_train, raster_grid)
    X_test_transformed, _ = pipeline.transform(df_test, raster_grid)
"""

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter, generic_filter
from sklearn.preprocessing import QuantileTransformer as SKQuantileTransformer
from sklearn.preprocessing import StandardScaler
import warnings
from typing import List, Tuple, Optional

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
    Applies log1p to right-skewed columns, then z-scores all features.

    Why: z-scoring raw skewed data compresses all variation into a spike.
    log1p restores Gaussian-like distribution before standardization.

    Columns to transform (configurable via log_cols):
        CFL, FSP_Index, Struct_Exp_Index, Ignition_Prob
    """

    def __init__(
        self,
        log_cols: List[str] = None,
        scale_all: bool = True,
    ):
        super().__init__("LogTransformer")
        self.log_cols = log_cols or [
            "CFL", "FSP_Index", "Struct_Exp_Index", "Ignition_Prob"
        ]
        self.scale_all = scale_all
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame, **kwargs) -> "LogTransformer":
        df_work = self._apply_log(df)
        numeric_cols = df_work.select_dtypes(include=np.number).columns.tolist()
        self.numeric_cols_ = numeric_cols
        if self.scale_all:
            self.scaler.fit(df_work[numeric_cols])
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
        assert self.fitted, "Call fit() first."
        df_work = self._apply_log(df)
        if self.scale_all:
            scaled = self.scaler.transform(df_work[self.numeric_cols_])
            df_out = pd.DataFrame(scaled, columns=self.numeric_cols_, index=df_work.index)
            # Carry over non-numeric columns
            non_num = [c for c in df_work.columns if c not in self.numeric_cols_]
            for c in non_num:
                df_out[c] = df_work[c].values
        else:
            df_out = df_work
        return df_out, list(self.numeric_cols_)

    def _apply_log(self, df: pd.DataFrame) -> pd.DataFrame:
        df_work = df.copy()
        for col in self.log_cols:
            if col in df_work.columns:
                # Guard: ensure non-negative before log1p
                df_work[col] = np.log1p(np.clip(df_work[col].values, 0, None))
        return df_work


class FuelModelEncoder(BaseTransformer):
    """
    One-hot encodes the Fuel_Models categorical column.

    Why: treating fuel model as a continuous ordinal destroys signal.
    One-hot lets the model learn independent coefficients per fuel type.

    Parameters
    ----------
    fuel_col : str — column name for fuel model codes
    max_categories : int — cap rare categories into an "other" bin
    drop_first : bool — drop first dummy to avoid multicollinearity
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

        # Count frequencies, keep top-k, rest → "other"
        counts = df[self.fuel_col].value_counts()
        self.top_categories_ = counts.index[:self.max_categories].tolist()
        self.dummy_cols_ = None  # will be set in first transform
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
        assert self.fitted
        if not self.fuel_col_present_ or self.fuel_col not in df.columns:
            return df, []

        df_work = df.copy()
        # Recode rare categories
        df_work[self.fuel_col] = df_work[self.fuel_col].apply(
            lambda x: x if x in self.top_categories_ else "other"
        ).astype(str)

        dummies = pd.get_dummies(
            df_work[self.fuel_col],
            prefix=f"fuel",
            drop_first=self.drop_first,
        )

        # Align columns to training dummies (fill missing test-time categories)
        if self.dummy_cols_ is None:
            self.dummy_cols_ = list(dummies.columns)
        else:
            for col in self.dummy_cols_:
                if col not in dummies.columns:
                    dummies[col] = 0
            dummies = dummies[self.dummy_cols_]

        df_work = df_work.drop(columns=[self.fuel_col])
        df_out = pd.concat([df_work.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

        return df_out, self.dummy_cols_

class InteractionTerms(BaseTransformer):
    """
    Creates physically-motivated interaction features.

    Pairs (after log1p transformation is applied upstream):
      - Ignition_Prob × CFL                  → joint ignition × fuel load risk
      - Ignition_Prob × Struct_Exp_Index      → ignition × structural exposure
      - CFL × FSP_Index                       → fuel × spread potential
    """

    PAIRS = [
        ("Ignition_Prob", "CFL",              "inter_ignition_x_cfl"),
        ("Ignition_Prob", "Struct_Exp_Index",  "inter_ignition_x_struct"),
        ("CFL",           "FSP_Index",          "inter_cfl_x_fsp"),
    ]

    def __init__(self, pairs: List[Tuple[str, str, str]] = None):
        super().__init__("InteractionTerms")
        self.pairs = pairs or self.PAIRS
        self.fitted = True  # stateless

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
    For each feature column, computes mean and std over 3×3, 7×7, 15×15
    windows in the original raster grid.

    Why: this is what CNN implicitly learns. Made explicit for GNN nodes
    so single-pixel features gain neighborhood context.

    Parameters
    ----------
    feature_cols : columns to compute stats for
    window_sizes : list of odd integers (kernel sizes)
    raster_shape : (nrows, ncols) of the spatial raster grid
    """

    def __init__(
        self,
        feature_cols: List[str] = None,
        window_sizes: List[int] = None,
        raster_shape: Tuple[int, int] = None,
    ):
        super().__init__("MultiScaleStats")
        self.feature_cols = feature_cols or [
            "Ignition_Prob", "CFL", "FSP_Index"
        ]
        self.window_sizes = window_sizes or [3, 7, 15]
        self.raster_shape = raster_shape
        self.fitted = True  # stateless given raster_shape

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
        """
        Parameters
        ----------
        df : DataFrame with feature columns
        raster_shape : (H, W) grid shape
        pixel_indices : array of shape (N,) with flat pixel index for each row in df
                        If None, assumes df rows correspond to raster pixels in row-major order.
        """
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

            # Reconstruct the raster from node values
            if pixel_indices is not None:
                raster = np.zeros(H * W, dtype=np.float32)
                raster[pixel_indices] = df_work[col].values
                raster = raster.reshape(H, W)
            else:
                # Assume df rows correspond to raster pixels in order
                n = len(df_work)
                pad = H * W - n
                vals = np.concatenate([df_work[col].values, np.zeros(pad)])
                raster = vals.reshape(H, W)

            for k in self.window_sizes:
                # Mean via uniform filter
                mean_raster = uniform_filter(raster, size=k, mode="reflect")
                # Std via E[X²] - E[X]²
                mean2_raster = uniform_filter(raster**2, size=k, mode="reflect")
                std_raster = np.sqrt(np.clip(mean2_raster - mean_raster**2, 0, None))

                col_mean = f"{col}_mean{k}"
                col_std  = f"{col}_std{k}"

                if pixel_indices is not None:
                    flat_mean = mean_raster.ravel()[pixel_indices]
                    flat_std  = std_raster.ravel()[pixel_indices]
                else:
                    flat_mean = mean_raster.ravel()[:n]
                    flat_std  = std_raster.ravel()[:n]

                df_work[col_mean] = flat_mean
                df_work[col_std]  = flat_std
                new_cols.extend([col_mean, col_std])

        print(f"[MultiScaleStats] Added {len(new_cols)} neighborhood features.")
        return df_work, new_cols

class SpatialGradients(BaseTransformer):
    """
    Computes directional gradients ∂f/∂row and ∂f/∂col in the raster grid.

    Physical meaning:
    - ∂(Ignition_Prob)/∂row: north-south risk gradient → fire spread direction
    - ∂(CFL)/∂col: east-west fuel gradient → fuel continuity

    These capture fire spread directionality that neither the raw value
    nor neighborhood stats encode.
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
                df_work[col_grow] = grad_row.ravel()[pixel_indices]
                df_work[col_gcol] = grad_col.ravel()[pixel_indices]
                df_work[col_gmag] = np.sqrt(
                    grad_row.ravel()[pixel_indices]**2
                    + grad_col.ravel()[pixel_indices]**2
                )
            else:
                n = len(df_work)
                df_work[col_grow] = grad_row.ravel()[:n]
                df_work[col_gcol] = grad_col.ravel()[:n]
                df_work[col_gmag] = np.sqrt(
                    grad_row.ravel()[:n]**2
                    + grad_col.ravel()[:n]**2
                )

            new_cols.extend([col_grow, col_gcol, col_gmag])

        print(f"[SpatialGradients] Added {len(new_cols)} gradient features.")
        return df_work, new_cols

class FuelConnectivityIndex(BaseTransformer):
    """
    For each pixel, fraction of 8-neighbors sharing the same fuel model.

    High connectivity → high fire spread potential along uniform fuel beds.
    A fuel model of 9 surrounded by fuel model 9 neighbors = connectivity 1.0.

    Requires raster_shape and pixel_indices (same as MultiScaleStats).
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

        if pixel_indices is not None:
            fuel_flat = np.zeros(H * W, dtype=np.float32)
            fuel_flat[pixel_indices] = df_work[self.fuel_col].values
            fuel_grid = fuel_flat.reshape(H, W)
        else:
            n = len(df_work)
            pad = H * W - n
            vals = np.concatenate([df_work[self.fuel_col].values, np.zeros(pad)])
            fuel_grid = vals.reshape(H, W)

        # For each pixel, count same-fuel neighbors using shifted grids
        same_count = np.zeros_like(fuel_grid, dtype=np.float32)
        total_neighbors = np.zeros_like(fuel_grid, dtype=np.float32)

        shifts = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        for dr, dc in shifts:
            shifted = np.roll(np.roll(fuel_grid, dr, axis=0), dc, axis=1)
            same_count    += (shifted == fuel_grid).astype(np.float32)
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
    Maps Burn_Prob (heavily right-skewed) to near-Gaussian for regression.

    Usage:
        qt = QuantileTargetTransformer()
        y_train_transformed = qt.fit_transform(y_train)
        y_pred_original = qt.inverse_transform(y_pred_transformed)
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
    Chains multiple BaseTransformer objects in order.
    Passes raster_shape and pixel_indices through to each step.

    Example:
        pipeline = FeatureTransformPipeline([
            LogTransformer(),
            FuelModelEncoder(),
            InteractionTerms(),
            MultiScaleStats(raster_shape=(500, 600)),
            SpatialGradients(raster_shape=(500, 600)),
            FuelConnectivityIndex(raster_shape=(500, 600)),
        ])
        df_out, feature_names = pipeline.fit_transform(df_train, raster_shape=(500,600))
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
    Call this in prepare_features.py.
    """
    return FeatureTransformPipeline([
        LogTransformer(
            log_cols=["CFL", "FSP_Index", "Struct_Exp_Index", "Ignition_Prob"],
            scale_all=True,
        ),
        FuelModelEncoder(fuel_col="Fuel_Models", max_categories=20),
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
        FuelConnectivityIndex(raster_shape=raster_shape),
    ])

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N = 200
    H, W = 20, 10
    pixel_idx = rng.choice(H * W, size=N, replace=False)

    df_test = pd.DataFrame({
        "Ignition_Prob":   rng.exponential(0.03, N),
        "CFL":             rng.exponential(2.0,  N),
        "FSP_Index":       rng.exponential(1.5,  N),
        "Struct_Exp_Index":rng.exponential(0.5,  N),
        "Fuel_Models":     rng.integers(1, 15, N).astype(str),
        "row":             rng.integers(0, H, N),
        "col":             rng.integers(0, W, N),
    })
    y_test = rng.exponential(0.05, N)

    pipeline = build_default_pipeline(raster_shape=(H, W))
    df_out, new_cols = pipeline.fit_transform(df_test, raster_shape=(H, W), pixel_indices=pixel_idx)

    qt = QuantileTargetTransformer()
    y_transformed = qt.fit_transform(y_test)
    y_back = qt.inverse_transform(y_transformed)

    print(f"\nInput shape:  {df_test.shape}")
    print(f"Output shape: {df_out.shape}")
    print(f"New columns ({len(new_cols)}): {new_cols[:5]} ...")
    print(f"Target transform — original mean: {y_test.mean():.4f}, "
          f"reconstructed mean: {y_back.mean():.4f} (should match)")
    assert np.allclose(y_test, y_back, atol=1e-4), "Quantile inverse mismatch!"
    print("\nAll assertions passed. feature_transforms.py is ready.")