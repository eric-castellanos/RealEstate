"""Shared utilities for model training scripts."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

INPUT_PATH = "data/processed/model_features.parquet"
TARGET_COLUMN = "log_price"
TEST_SIZE = float(os.getenv("MODEL_TEST_SIZE", "0.2"))
RANDOM_SEED = int(os.getenv("MODEL_RANDOM_SEED", "42"))
MODELS_DIR = "models"
PRICE_MODEL_EXPERIMENTS_DIR = "models/price_model_experiments"
MLFLOW_ARTIFACT_ROOT = "price_models"


def project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not locate project root (missing pyproject.toml in parent dirs)")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_output_dir(model_name: str) -> Path:
    return ensure_dir(project_root() / PRICE_MODEL_EXPERIMENTS_DIR / model_name)


def setup_mlflow() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "price model")
    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.set_experiment(experiment_name)
    except MlflowException as exc:
        # Recover from soft-deleted experiment names in local dev tracking DB.
        if "deleted experiment" not in str(exc).lower():
            raise
        client = MlflowClient(tracking_uri=tracking_uri)
        deleted_exp = client.get_experiment_by_name(experiment_name)
        if deleted_exp and deleted_exp.lifecycle_stage == "deleted":
            LOGGER.warning(
                "Experiment '%s' is deleted; restoring it before continuing",
                experiment_name,
            )
            client.restore_experiment(deleted_exp.experiment_id)
            mlflow.set_experiment(experiment_name)
        else:
            raise
    LOGGER.info("Using MLflow tracking URI: %s (experiment: %s)", tracking_uri, experiment_name)


def load_train_test_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    root = project_root()
    input_path = root / INPUT_PATH
    if not input_path.exists():
        raise FileNotFoundError(f"Input data not found: {input_path}")

    data = pd.read_parquet(input_path)
    LOGGER.info("Loaded data with shape: %s rows x %s cols", len(data), len(data.columns))

    numeric_data = data.select_dtypes(include=["number", "bool"]).copy()
    if TARGET_COLUMN not in numeric_data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from numeric dataset")

    y = pd.to_numeric(numeric_data[TARGET_COLUMN], errors="coerce")
    X = numeric_data.drop(columns=[TARGET_COLUMN]).copy()
    X = X.astype(float)

    valid_mask = y.notna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]
    feature_names = X.columns.tolist()
    LOGGER.info("Using %s numeric features", len(feature_names))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )
    LOGGER.info("Train/Test split complete: %s train rows, %s test rows", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test, feature_names


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def evaluate_train_test_regression(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
) -> dict[str, float]:
    train_metrics = evaluate_regression(y_train_true, y_train_pred)
    test_metrics = evaluate_regression(y_test_true, y_test_pred)
    return {
        "train_rmse": train_metrics["rmse"],
        "train_mae": train_metrics["mae"],
        "train_r2": train_metrics["r2"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
    }


def print_metrics(model_name: str, metrics: dict[str, float]) -> None:
    print(f"\n=== {model_name} Metrics ===")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE : {metrics['mae']:.6f}")
    print(f"R2  : {metrics['r2']:.6f}")


def print_train_test_metrics(model_name: str, metrics: dict[str, float]) -> None:
    print(f"\n=== {model_name} Metrics ===")
    print("Train:")
    print(f"  RMSE: {metrics['train_rmse']:.6f}")
    print(f"  MAE : {metrics['train_mae']:.6f}")
    print(f"  R2  : {metrics['train_r2']:.6f}")
    print("Test:")
    print(f"  RMSE: {metrics['test_rmse']:.6f}")
    print(f"  MAE : {metrics['test_mae']:.6f}")
    print(f"  R2  : {metrics['test_r2']:.6f}")
