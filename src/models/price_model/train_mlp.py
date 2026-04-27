"""Train and evaluate a simple PyTorch MLP regressor."""

from __future__ import annotations

import os

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from mlflow.models import infer_signature
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

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

BATCH_SIZE = int(os.getenv("MLP_BATCH_SIZE", "256"))
LEARNING_RATE = float(os.getenv("MLP_LR", "0.001"))
EPOCHS = int(os.getenv("MLP_EPOCHS", "30"))
HIDDEN_DIM_1 = int(os.getenv("MLP_HIDDEN_DIM_1", "128"))
HIDDEN_DIM_2 = int(os.getenv("MLP_HIDDEN_DIM_2", "64"))


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run() -> None:
    setup_mlflow()
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    X_train, X_test, y_train, y_test, feature_names = load_train_test_data()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MLPRegressor(input_dim=X_train_tensor.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch_X)
        avg_loss = epoch_loss / len(train_dataset)
        LOGGER.info("Epoch %s/%s - train_mse: %.6f", epoch + 1, EPOCHS, avg_loss)

    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_tensor).squeeze(1).cpu().numpy()
        train_preds_for_signature = model(X_train_tensor).squeeze(1).cpu().numpy()
    metrics = evaluate_train_test_regression(
        y_train_true=y_train.values,
        y_train_pred=train_preds_for_signature,
        y_test_true=y_test.values,
        y_test_pred=test_preds,
    )
    signature = infer_signature(X_train_scaled, train_preds_for_signature)
    input_example = X_train_scaled[:5]

    out_dir = model_output_dir("mlp")
    weights_path = out_dir / "mlp_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": X_train_tensor.shape[1],
            "feature_names": feature_names,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        },
        weights_path,
    )
    LOGGER.info("Saved model artifact to %s", weights_path)

    with mlflow.start_run(run_name="pytorch_mlp"):
        mlflow.log_param("model_type", "PyTorchMLP")
        mlflow.log_param("num_features", len(feature_names))
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("hidden_dim_1", HIDDEN_DIM_1)
        mlflow.log_param("hidden_dim_2", HIDDEN_DIM_2)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(
            str(weights_path),
            artifact_path=f"{MLFLOW_ARTIFACT_ROOT}/mlp/files",
        )
        mlflow.pytorch.log_model(
            model,
            artifact_path=f"{MLFLOW_ARTIFACT_ROOT}/mlp/model",
            signature=signature,
            input_example=input_example,
        )

    print_train_test_metrics("PyTorch MLP", metrics)


if __name__ == "__main__":
    run()
