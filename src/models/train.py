"""Pipeline de treinamento LSTM com MLflow tracking padronizado.

Implementa:
- Treino com early stopping e gradient clipping
- MLflow tracking: params, metrics, artifacts, tags obrigatórias
- Model Registry: registro automático com metadata
- Champion-challenger: comparação antes de promover modelo
"""

import hashlib
import json
import logging
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.feature_engineering import create_sequences, load_config, split_data
from src.models.lstm_model import LSTMPredictor

logger = logging.getLogger(__name__)


# --- Métricas ---


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula métricas de regressão.

    Args:
        y_true: Valores reais.
        y_pred: Valores preditos.

    Returns:
        Dicionário com MAE, RMSE, MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # MAPE com proteção contra divisão por zero
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    return {"mae": mae, "rmse": rmse, "mape": mape}


# --- Early Stopping ---


class EarlyStopping:
    """Early stopping para evitar overfitting.

    Args:
        patience: Épocas sem melhoria antes de parar.
        min_delta: Melhoria mínima para considerar progresso.
    """

    def __init__(self, patience: int = 15, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Verifica se deve parar o treinamento.

        Args:
            val_loss: Loss de validação da época atual.

        Returns:
            True se deve parar.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


# --- Git SHA ---


def get_git_sha() -> str:
    """Retorna o SHA do commit atual."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()[:8]
    except Exception:
        return "unknown"


# --- Training Loop ---


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_value: float = 1.0,
) -> float:
    """Treina uma época.

    Args:
        model: Modelo LSTM.
        dataloader: DataLoader de treino.
        criterion: Função de perda.
        optimizer: Otimizador.
        device: Device (cpu/cuda).
        clip_value: Valor máximo para gradient clipping.

    Returns:
        Loss média da época.
    """
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch).squeeze(-1)
        loss = criterion(output, y_batch)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()
        total_loss += loss.item() * len(X_batch)

    return total_loss / len(dataloader.dataset)


def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Avalia uma época.

    Args:
        model: Modelo LSTM.
        dataloader: DataLoader de validação/teste.
        criterion: Função de perda.
        device: Device.

    Returns:
        Loss média.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch).squeeze(-1)
            loss = criterion(output, y_batch)
            total_loss += loss.item() * len(X_batch)

    return total_loss / len(dataloader.dataset)


# --- Main Training Pipeline ---


def train_and_log(config_path: str = "configs/model_config.yaml") -> str:
    """Pipeline completa de treinamento com MLflow tracking.

    Args:
        config_path: Caminho da configuração YAML.

    Returns:
        run_id do MLflow.
    """
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Carregar dados ---
    df = pd.read_parquet("data/processed/petr4_features.parquet")

    feature_cols = [
        "Close",
        "Volume",
        "sma_20",
        "sma_50",
        "ema_12",
        "ema_26",
        "rsi_14",
        "macd",
        "macd_signal",
        "bollinger_upper",
        "bollinger_lower",
        "volume_sma_20",
        "daily_return",
        "log_return",
    ]
    # Filtrar colunas que existem
    feature_cols = [c for c in feature_cols if c in df.columns]

    data = df[feature_cols].values

    # Escalar dados
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Criar sequências
    seq_len = config["features"]["sequence_length"]
    horizon = config["features"]["prediction_horizon"]
    X, y = create_sequences(data_scaled, seq_len, horizon, target_idx=0)

    # Split temporal
    splits = split_data(X, y, config["data"]["train_split"], config["data"]["validation_split"])

    # DataLoaders
    batch_size = config["training"]["batch_size"]
    train_ds = TensorDataset(
        torch.FloatTensor(splits["train"][0]), torch.FloatTensor(splits["train"][1])
    )
    val_ds = TensorDataset(
        torch.FloatTensor(splits["val"][0]), torch.FloatTensor(splits["val"][1])
    )
    test_ds = TensorDataset(
        torch.FloatTensor(splits["test"][0]), torch.FloatTensor(splits["test"][1])
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # --- Modelo ---
    model_cfg = config["model"]
    model = LSTMPredictor(
        input_size=len(feature_cols),
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        bidirectional=model_cfg["bidirectional"],
        output_size=model_cfg["output_size"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config["training"]["scheduler_patience"],
        factor=config["training"]["scheduler_factor"],
    )
    early_stopping = EarlyStopping(patience=config["training"]["early_stopping_patience"])

    # --- MLflow Tracking ---
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=f"lstm-{config['ticker']}") as run:
        # Tags obrigatórias (Schema do GAP 05)
        tags = config["mlflow"]["tags"].copy()
        tags["git_sha"] = get_git_sha()
        tags["training_data_version"] = hashlib.md5(
            open("data/processed/petr4_features.parquet", "rb").read()[:4096]
        ).hexdigest()[:8]
        tags["fairness_checked"] = "false"
        for k, v in tags.items():
            mlflow.set_tag(k, str(v))

        # Log de parâmetros
        mlflow.log_params(
            {
                "ticker": config["ticker"],
                "sequence_length": seq_len,
                "prediction_horizon": horizon,
                "n_features": len(feature_cols),
                "n_samples_train": len(splits["train"][0]),
                "n_samples_val": len(splits["val"][0]),
                "n_samples_test": len(splits["test"][0]),
                "hidden_size": model_cfg["hidden_size"],
                "num_layers": model_cfg["num_layers"],
                "dropout": model_cfg["dropout"],
                "bidirectional": model_cfg["bidirectional"],
                "batch_size": batch_size,
                "learning_rate": config["training"]["learning_rate"],
                "weight_decay": config["training"]["weight_decay"],
                "epochs_max": config["training"]["epochs"],
                "early_stopping_patience": config["training"]["early_stopping_patience"],
                "gradient_clip_value": config["training"]["gradient_clip_value"],
                "feature_columns": ",".join(feature_cols),
            }
        )

        # --- Training Loop ---
        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(config["training"]["epochs"]):
            train_loss = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                config["training"]["gradient_clip_value"],
            )
            val_loss = evaluate_epoch(model, val_loader, criterion, device)

            scheduler.step(val_loss)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()

            if epoch % 10 == 0:
                logger.info(
                    "Epoch %d/%d — train_loss=%.6f, val_loss=%.6f",
                    epoch,
                    config["training"]["epochs"],
                    train_loss,
                    val_loss,
                )

            if early_stopping(val_loss):
                logger.info("Early stopping na época %d", epoch)
                mlflow.log_metric("early_stop_epoch", epoch)
                break

        # --- Avaliar no test set ---
        model.load_state_dict(best_model_state)
        model.eval()

        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch).squeeze(-1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y_batch.numpy())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)

        metrics = compute_metrics(y_true, y_pred)
        metrics["best_val_loss"] = best_val_loss
        mlflow.log_metrics(metrics)

        logger.info(
            "Test metrics: MAE=%.6f, RMSE=%.6f, MAPE=%.2f%%",
            metrics["mae"],
            metrics["rmse"],
            metrics["mape"],
        )

        # --- Salvar artefatos ---
        # Modelo
        model_path = "models/lstm_petr4_best.pt"
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_model_state,
                "model_config": model_cfg,
                "feature_columns": feature_cols,
                "scaler_params": {
                    "min_": scaler.min_.tolist(),
                    "scale_": scaler.scale_.tolist(),
                    "data_min_": scaler.data_min_.tolist(),
                    "data_max_": scaler.data_max_.tolist(),
                },
                "sequence_length": seq_len,
                "prediction_horizon": horizon,
            },
            model_path,
        )

        mlflow.log_artifact(model_path)
        mlflow.pytorch.log_model(model, "model")

        # Métricas em JSON
        metrics_path = "metrics/train_metrics.json"
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(metrics_path)

        # Scaler
        scaler_path = "models/scaler.joblib"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)

        # Registrar no Model Registry
        model_uri = f"runs:/{run.info.run_id}/model"
        mv = mlflow.register_model(model_uri, config["mlflow"]["model_name"])
        logger.info("Modelo registrado: %s v%s", mv.name, mv.version)

        return run.info.run_id


# --- Champion-Challenger ---


def champion_challenger(
    challenger_run_id: str,
    model_name: str = "lstm-petr4",
    metric: str = "rmse",
    min_improvement: float = 0.005,
) -> bool:
    """Compara challenger com champion atual no Model Registry.

    Args:
        challenger_run_id: Run ID do modelo challenger.
        model_name: Nome do modelo no Registry.
        metric: Métrica de comparação (menor = melhor para rmse/mae).
        min_improvement: Melhoria mínima para promover.

    Returns:
        True se challenger deve ser promovido.
    """
    client = mlflow.tracking.MlflowClient()

    # Buscar champion (última versão em Production)
    try:
        champion_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not champion_versions:
            logger.info("Nenhum champion em produção. Promovendo challenger.")
            return True

        champion = champion_versions[0]
        champion_run = client.get_run(champion.run_id)
        champion_metric = champion_run.data.metrics.get(metric)
    except Exception:
        logger.info("Sem champion anterior. Promovendo challenger.")
        return True

    # Buscar challenger
    challenger_run = client.get_run(challenger_run_id)
    challenger_metric = challenger_run.data.metrics.get(metric)

    if champion_metric is None or challenger_metric is None:
        logger.warning("Métrica '%s' não encontrada. Abortando comparação.", metric)
        return False

    improvement = (champion_metric - challenger_metric) / champion_metric
    logger.info(
        "Champion %s=%.6f vs Challenger %s=%.6f (improvement=%.4f%%)",
        metric,
        champion_metric,
        metric,
        challenger_metric,
        improvement * 100,
    )

    return improvement >= min_improvement


def main() -> None:
    """Entry point para treinamento."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    run_id = train_and_log()
    logger.info("Treinamento concluído. Run ID: %s", run_id)


if __name__ == "__main__":
    main()
