from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from wildfire_gnn.evaluation.metrics import compute_regression_metrics
from wildfire_gnn.features.baseline_dataset import (
    load_graph_as_dataframe,
    save_dataframe,
    validate_baseline_dataframe,
)
from wildfire_gnn.features.splitters import (
    make_random_split,
    make_spatial_block_split,
    save_splits,
)
from wildfire_gnn.models.baselines import (
    XGBRegressor,
    build_mlp,
    build_random_forest,
    build_xgboost,
)
from wildfire_gnn.utils.logger import get_logger


class BaselinePipeline:
    """Train and evaluate baseline regression models."""

    def __init__(self, config: dict) -> None:
        self.config = config
        log_file = Path(config["outputs"]["logs_dir"]) / "run_baselines.log"
        self.logger = get_logger("baseline_pipeline", log_file=str(log_file))

    def prepare_dataset(self) -> pd.DataFrame:
        """Load graph data, validate, and export baseline dataframe."""
        graph_path = self.config["paths"]["graph_data_path"]
        csv_path = self.config["paths"]["baseline_dataset_path"]

        clip_cfg = self.config["data"]["clip_target"]

        df = load_graph_as_dataframe(
            graph_path=graph_path,
            clip_target=bool(clip_cfg["enabled"]),
            target_min=float(clip_cfg["min_value"]),
            target_max=float(clip_cfg["max_value"]),
            drop_constant_features=bool(self.config["data"]["drop_constant_features"]),
        )
        save_dataframe(df, csv_path)

        stats = validate_baseline_dataframe(df)
        self.logger.info("Saved baseline dataset to %s", csv_path)
        self.logger.info("Baseline dataset stats: %s", stats)

        return df

    def create_splits(self, df: pd.DataFrame) -> dict[str, dict]:
        """Create both random and spatial splits."""
        split_outputs = {}

        random_cfg = self.config["split"]["random"]
        random_splits = make_random_split(
            df=df,
            train_size=float(random_cfg["train_size"]),
            val_size=float(random_cfg["val_size"]),
            test_size=float(random_cfg["test_size"]),
            random_seed=int(self.config["project"]["random_seed"]),
        )
        save_splits(random_splits, self.config["paths"]["random_split_path"])
        split_outputs["random"] = random_splits

        self.logger.info("Saved random splits to %s", self.config["paths"]["random_split_path"])

        spatial_cfg = self.config["split"]["spatial"]
        if bool(spatial_cfg["enabled"]):
            spatial_splits = make_spatial_block_split(
                df=df,
                n_row_blocks=int(spatial_cfg["n_row_blocks"]),
                n_col_blocks=int(spatial_cfg["n_col_blocks"]),
                train_blocks=float(spatial_cfg["train_blocks"]),
                val_blocks=float(spatial_cfg["val_blocks"]),
                test_blocks=float(spatial_cfg["test_blocks"]),
                random_seed=int(self.config["project"]["random_seed"]),
            )
            save_splits(spatial_splits, self.config["paths"]["spatial_split_path"])
            split_outputs["spatial"] = spatial_splits
            self.logger.info("Saved spatial splits to %s", self.config["paths"]["spatial_split_path"])

        return split_outputs

    def build_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics table for features."""
        exclude_cols = {"target"}
        rows = []

        for col in df.columns:
            if col in exclude_cols:
                continue
            rows.append(
                {
                    "feature": col,
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "n_unique": df[col].nunique(),
                }
            )

        summary_df = pd.DataFrame(rows)
        output_path = Path(self.config["outputs"]["feature_summary_table_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_path, index=False)

        self.logger.info("Saved feature summary to %s", output_path)
        return summary_df

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        target_col = self.config["data"]["target_column"]
        coord_cols = set(self.config["data"]["coordinate_columns"])

        feature_cols = [c for c in df.columns if c != target_col]

        if bool(self.config["data"]["drop_coordinate_features_for_training"]):
            feature_cols = [c for c in feature_cols if c not in coord_cols]

        return feature_cols

    def _make_xy(self, df: pd.DataFrame, indices) -> tuple[pd.DataFrame, pd.Series]:
        feature_cols = self._get_feature_columns(df)
        x = df.iloc[indices][feature_cols].copy()
        y = df.iloc[indices][self.config["data"]["target_column"]].copy()
        return x, y

    def _save_target_histogram(self, df: pd.DataFrame) -> None:
        figures_dir = Path(self.config["outputs"]["figures_dir"])
        figures_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 5))
        plt.hist(df["target"], bins=100)
        plt.title("Baseline Target Distribution")
        plt.xlabel("Burn Probability")
        plt.ylabel("Frequency")
        plt.savefig(figures_dir / "baseline_target_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _save_pred_plot(self, y_true, y_pred, model_name: str) -> None:
        figures_dir = Path(self.config["outputs"]["figures_dir"])
        figures_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.2, s=8)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"{model_name.upper()} - Predicted vs True")
        plt.savefig(figures_dir / f"{model_name.lower()}_pred_vs_true.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _save_feature_importance(self, model, feature_names: list[str], model_name: str) -> None:
        figures_dir = Path(self.config["outputs"]["figures_dir"])
        figures_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            return

        imp_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False).head(20)

        plt.figure(figsize=(8, 6))
        plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
        plt.title(f"{model_name.upper()} Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.savefig(figures_dir / f"{model_name.lower()}_feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _save_predictions(
        self,
        df: pd.DataFrame,
        indices,
        y_true,
        y_pred,
        model_name: str,
    ) -> None:
        pred_dir = Path(self.config["outputs"]["predictions_dir"])
        pred_dir.mkdir(parents=True, exist_ok=True)

        pred_df = df.iloc[indices][["row_index", "col_index"]].copy().reset_index(drop=True)
        pred_df["y_true"] = y_true
        pred_df["y_pred"] = y_pred
        pred_df["residual"] = pred_df["y_true"] - pred_df["y_pred"]

        pred_df.to_csv(pred_dir / f"{model_name.lower()}_test_predictions.csv", index=False)

    def _log_split_stats(self, df: pd.DataFrame, splits: dict[str, pd.Index], split_name: str) -> None:
        for part_name, idx in splits.items():
            subset = df.iloc[idx]
            self.logger.info(
                "[%s] %s size=%d target_mean=%.6f target_std=%.6f",
                split_name,
                part_name,
                len(subset),
                subset["target"].mean(),
                subset["target"].std(),
            )

    def _build_models(self) -> dict[str, object]:
        seed = int(self.config["project"]["random_seed"])

        models = {
            "rf": build_random_forest(self.config["models"]["random_forest"], seed),
            "mlp": build_mlp(self.config["models"]["mlp"], seed),
        }

        xgb_cfg = self.config["models"]["xgboost"]
        if bool(xgb_cfg["enabled"]):
            try:
                models["xgb"] = build_xgboost(xgb_cfg, seed)
            except ImportError as exc:
                self.logger.warning("Skipping XGBoost: %s", exc)

        return models

    def train_and_evaluate(self) -> pd.DataFrame:
        """Run all baseline models on all configured splits."""
        df = self.prepare_dataset()
        self.build_feature_summary(df)
        self._save_target_histogram(df)

        all_splits = self.create_splits(df)
        all_models = self._build_models()
        results = []

        for split_name, splits in all_splits.items():
            self._log_split_stats(df, splits, split_name)

            train_x, train_y = self._make_xy(df, splits["train_idx"])
            val_x, val_y = self._make_xy(df, splits["val_idx"])
            test_x, test_y = self._make_xy(df, splits["test_idx"])

            for model_name, model in all_models.items():
                self.logger.info("Training model=%s split=%s", model_name, split_name)
                model.fit(train_x, train_y)

                val_pred = model.predict(val_x)
                test_pred = model.predict(test_x)

                val_metrics = compute_regression_metrics(val_y.values, val_pred)
                test_metrics = compute_regression_metrics(test_y.values, test_pred)

                results.append(
                    {
                        "model": model_name,
                        "split_type": split_name,
                        "data_split": "val",
                        **val_metrics,
                    }
                )
                results.append(
                    {
                        "model": model_name,
                        "split_type": split_name,
                        "data_split": "test",
                        **test_metrics,
                    }
                )

                if split_name == "random":
                    self._save_pred_plot(test_y.values, test_pred, model_name)
                    self._save_predictions(df, splits["test_idx"], test_y.values, test_pred, model_name)

                    if model_name in {"rf", "xgb"}:
                        underlying_model = model
                        self._save_feature_importance(
                            underlying_model,
                            feature_names=list(train_x.columns),
                            model_name=model_name,
                        )

                self.logger.info("Completed model=%s split=%s", model_name, split_name)
                self.logger.info("Validation metrics: %s", val_metrics)
                self.logger.info("Test metrics: %s", test_metrics)

        results_df = pd.DataFrame(results)

        metrics_path = Path(self.config["outputs"]["metrics_table_path"])
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(metrics_path, index=False)
        self.logger.info("Saved baseline metrics to %s", metrics_path)

        self._save_comparison_plots(results_df)

        return results_df

    def _save_comparison_plots(self, results_df: pd.DataFrame) -> None:
        figures_dir = Path(self.config["outputs"]["figures_dir"])
        figures_dir.mkdir(parents=True, exist_ok=True)

        test_df = results_df[results_df["data_split"] == "test"].copy()

        for split_type in test_df["split_type"].unique():
            subset = test_df[test_df["split_type"] == split_type]

            plt.figure(figsize=(8, 5))
            plt.bar(subset["model"], subset["rmse"])
            plt.title(f"Baseline Model Comparison ({split_type.title()} Test RMSE)")
            plt.xlabel("Model")
            plt.ylabel("RMSE")
            plt.savefig(
                figures_dir / f"baseline_model_comparison_{split_type}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()