"""Train and evaluate a Linear Regression baseline model."""

from __future__ import annotations

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.price_model.common import (
    LOGGER,
    MLFLOW_ARTIFACT_ROOT,
    evaluate_train_test_regression,
    load_train_test_data,
    model_output_dir,
    print_train_test_metrics,
    setup_mlflow,
)


def run() -> None:
    setup_mlflow()
    X_train, X_test, y_train, y_test, feature_names = load_train_test_data()

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
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

    out_dir = model_output_dir("linear_regression")
    model_path = out_dir / "linear_regression.joblib"
    joblib.dump({"model": model, "features": feature_names}, model_path)
    LOGGER.info("Saved model artifact to %s", model_path)

    with mlflow.start_run(run_name="linear_regression"):
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("num_features", len(feature_names))
        mlflow.log_param("standardized_features", True)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(
            str(model_path),
            artifact_path=f"{MLFLOW_ARTIFACT_ROOT}/linear_regression/files",
        )
        mlflow.sklearn.log_model(
            model,
            artifact_path=f"{MLFLOW_ARTIFACT_ROOT}/linear_regression/model",
            signature=signature,
            input_example=input_example,
        )

    print_train_test_metrics("Linear Regression", metrics)


if __name__ == "__main__":
    run()
