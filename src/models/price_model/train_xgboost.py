"""Train and evaluate an XGBoost regressor."""

from __future__ import annotations

import os

import matplotlib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
from mlflow.models import infer_signature
from matplotlib import pyplot as plt
from xgboost import XGBRegressor

from src.models.price_model.common import (
    LOGGER,
    MLFLOW_ARTIFACT_ROOT,
    RANDOM_SEED,
    evaluate_train_test_regression,
    load_train_test_data,
    model_output_dir,
    print_train_test_metrics,
    setup_mlflow,
)

matplotlib.use("Agg")

SHAP_SAMPLE_SIZE = int(os.getenv("SHAP_SAMPLE_SIZE", "5000"))


def create_feature_importance_artifacts(
    model: XGBRegressor,
    feature_names: list[str],
    out_dir: str,
) -> tuple[str, str]:
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_gain": model.feature_importances_,
        }
    ).sort_values("importance_gain", ascending=False)
    importance_csv = os.path.join(out_dir, "feature_importance.csv")
    importance.to_csv(importance_csv, index=False)

    top_k = importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_k["feature"], top_k["importance_gain"])
    ax.set_title("XGBoost Feature Importance (Top 20)")
    ax.set_xlabel("Importance (gain)")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    importance_plot = os.path.join(out_dir, "feature_importance_top20.png")
    fig.savefig(importance_plot, dpi=150)
    plt.close(fig)
    return importance_csv, importance_plot


def create_shap_artifacts(
    model: XGBRegressor,
    X_reference: pd.DataFrame,
    out_dir: str,
) -> tuple[str, str]:
    sample_size = min(SHAP_SAMPLE_SIZE, len(X_reference))
    X_shap = X_reference.sample(n=sample_size, random_state=RANDOM_SEED)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    mean_abs_shap = pd.DataFrame(
        {
            "feature": X_shap.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_csv = os.path.join(out_dir, "shap_mean_abs.csv")
    mean_abs_shap.to_csv(shap_csv, index=False)

    shap_plot = os.path.join(out_dir, "shap_summary.png")
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.tight_layout()
    plt.savefig(shap_plot, dpi=150, bbox_inches="tight")
    plt.close()
    return shap_csv, shap_plot


def run() -> None:
    setup_mlflow()
    X_train, X_test, y_train, y_test, feature_names = load_train_test_data()

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    metrics = evaluate_train_test_regression(
        y_train_true=y_train.values,
        y_train_pred=train_preds,
        y_test_true=y_test.values,
        y_test_pred=test_preds,
    )
    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_train.head(5)

    out_dir = model_output_dir("xgboost")
    model_path = out_dir / "xgboost_model.json"
    model.save_model(model_path)
    LOGGER.info("Saved model artifact to %s", model_path)
    explain_dir = ensure_dir(out_dir / "xgboost_explainability")

    importance_csv, importance_plot = create_feature_importance_artifacts(
        model=model,
        feature_names=feature_names,
        out_dir=str(explain_dir),
    )
    shap_csv, shap_plot = create_shap_artifacts(
        model=model,
        X_reference=X_train,
        out_dir=str(explain_dir),
    )

    with mlflow.start_run(run_name="xgboost_regressor"):
        mlflow.log_param("model_type", "XGBRegressor")
        mlflow.log_param("num_features", len(feature_names))
        mlflow.log_param("n_estimators", 400)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("shap_sample_size", min(SHAP_SAMPLE_SIZE, len(X_train)))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(
            str(model_path),
            artifact_path=f"{MLFLOW_ARTIFACT_ROOT}/xgboost/files",
        )
        mlflow.log_artifact(
            importance_csv,
            artifact_path=f"{MLFLOW_ARTIFACT_ROOT}/xgboost/explainability",
        )
        mlflow.log_artifact(
            importance_plot,
            artifact_path=f"{MLFLOW_ARTIFACT_ROOT}/xgboost/explainability",
        )
        mlflow.log_artifact(
            shap_csv,
            artifact_path=f"{MLFLOW_ARTIFACT_ROOT}/xgboost/explainability",
        )
        mlflow.log_artifact(
            shap_plot,
            artifact_path=f"{MLFLOW_ARTIFACT_ROOT}/xgboost/explainability",
        )
        mlflow.xgboost.log_model(
            model,
            artifact_path=f"{MLFLOW_ARTIFACT_ROOT}/xgboost/model",
            signature=signature,
            input_example=input_example,
        )

    print_train_test_metrics("XGBoost", metrics)


if __name__ == "__main__":
    run()
