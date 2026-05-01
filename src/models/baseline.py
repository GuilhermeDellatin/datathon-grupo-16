"""Baselines para benchmark do LSTM: Ridge (sklearn) e MLP (PyTorch).

Executa ambos os baselines com a MESMA pipeline de dados do LSTM
(features escaladas, split temporal, sequências achatadas) para que a
comparação seja justa. Loga cada baseline como uma run no MLflow no
mesmo experimento e escreve `metrics/baseline_metrics.json`.

Métricas reportadas (em escala original de preço, R$):
- MAE, RMSE, MAPE
- target_sigma, sigma_threshold_0_5, sigma_coverage_0_5
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.feature_engineering import create_sequences, load_config, split_data
from src.models.train import (
    _ensure_mlflow_tracking_uri,
    compute_metrics,
    compute_sigma_coverage,
    get_git_sha,
)

logger = logging.getLogger(__name__)


# --- Modelo MLP ---


class MLPRegressor(nn.Module):
    """MLP feed-forward para regressão univariada.

    Recebe features achatadas (sequence_length * n_features) e prevê um
    único valor (Close escalado). Usado como baseline contra o LSTM.

    Args:
        input_size: Dimensão da entrada achatada.
        hidden_sizes: Lista com tamanho das camadas ocultas.
        dropout: Dropout entre camadas (0 desabilita).
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_sizes = hidden_sizes or [128, 64]

        layers: list[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        logger.info(
            "MLPRegressor: input=%d, hidden=%s, dropout=%.2f, params=%d",
            input_size,
            hidden_sizes,
            dropout,
            sum(p.numel() for p in self.parameters()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor (batch_size, input_size).

        Returns:
            Predição (batch_size,).
        """
        return torch.as_tensor(self.net(x).squeeze(-1))


# --- Pipeline de dados compartilhado ---


def prepare_baseline_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    sequence_length: int,
    prediction_horizon: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, Any]:
    """Prepara dados achatados para baselines tabulares.

    Usa o mesmo encadeamento do LSTM (scale → sequences → split) e
    achata cada janela em um vetor único (seq_len * n_features) para
    consumo por Ridge e MLP.

    Args:
        df: DataFrame com features já calculadas.
        feature_cols: Colunas a usar como features (Close em primeiro lugar).
        sequence_length: Tamanho da janela de input.
        prediction_horizon: Passos à frente para previsão.
        train_ratio: Fração de treino.
        val_ratio: Fração de validação.

    Returns:
        Dicionário com splits achatados, scaler e parâmetros para
        desnormalização da coluna Close.
    """
    data = df[feature_cols].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = create_sequences(
        data_scaled, sequence_length, prediction_horizon, target_idx=0
    )
    splits = split_data(X, y, train_ratio, val_ratio)

    flat_splits = {
        name: (X_split.reshape(X_split.shape[0], -1), y_split)
        for name, (X_split, y_split) in splits.items()
    }

    close_min = float(scaler.data_min_[0])
    close_max = float(scaler.data_max_[0])
    return {
        "splits": flat_splits,
        "scaler": scaler,
        "close_min": close_min,
        "close_max": close_max,
        "input_size": sequence_length * len(feature_cols),
    }


def to_price_scale(
    y_scaled: np.ndarray, close_min: float, close_max: float
) -> np.ndarray:
    """Inverte o MinMax na coluna Close.

    Args:
        y_scaled: Valores escalados [0, 1].
        close_min: Mínimo da coluna Close no fit do scaler.
        close_max: Máximo da coluna Close no fit do scaler.

    Returns:
        Valores em escala original (R$).
    """
    return y_scaled * (close_max - close_min) + close_min


def evaluate_predictions(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    close_min: float,
    close_max: float,
) -> dict[str, float]:
    """Calcula métricas em escala original de preço.

    Args:
        y_true_scaled: Targets escalados.
        y_pred_scaled: Predições escaladas.
        close_min: Mínimo da Close (do scaler).
        close_max: Máximo da Close (do scaler).

    Returns:
        Dicionário com MAE, RMSE, MAPE e sigma_coverage_0_5.
    """
    y_true = to_price_scale(y_true_scaled, close_min, close_max)
    y_pred = to_price_scale(y_pred_scaled, close_min, close_max)

    metrics = compute_metrics(y_true, y_pred)
    sigma = compute_sigma_coverage(y_true, y_pred, threshold_sigma=0.5)
    metrics["target_sigma"] = sigma["target_sigma"]
    metrics["sigma_threshold_0_5"] = sigma["sigma_threshold"]
    metrics["sigma_coverage_0_5"] = sigma["sigma_coverage"]
    return metrics


# --- Treino dos baselines ---


def train_ridge(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    close_min: float,
    close_max: float,
    alpha: float = 1.0,
) -> tuple[Ridge, dict[str, float]]:
    """Treina Ridge regression e avalia no test set.

    Args:
        splits: Dict com (X, y) achatados para train/val/test.
        close_min: Mínimo da Close no scaler.
        close_max: Máximo da Close no scaler.
        alpha: Regularização L2.

    Returns:
        Tupla (modelo treinado, métricas em escala original).
    """
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    # Ridge usa train+val combinados (não tem early stopping)
    X_fit = np.concatenate([X_train, X_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)

    model = Ridge(alpha=alpha)
    model.fit(X_fit, y_fit)

    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred, close_min, close_max)
    logger.info(
        "Ridge: MAE=%.4f, RMSE=%.4f, MAPE=%.2f%%, sigma_coverage=%.2f%%",
        metrics["mae"],
        metrics["rmse"],
        metrics["mape"],
        metrics["sigma_coverage_0_5"] * 100,
    )
    return model, metrics


def train_mlp_baseline(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    close_min: float,
    close_max: float,
    input_size: int,
    hidden_sizes: list[int],
    dropout: float,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    device: torch.device | None = None,
) -> tuple[MLPRegressor, dict[str, float]]:
    """Treina MLP em PyTorch e avalia no test set.

    Args:
        splits: Dict com (X, y) achatados para train/val/test.
        close_min: Mínimo da Close no scaler.
        close_max: Máximo da Close no scaler.
        input_size: Dimensão da entrada (seq_len * n_features).
        hidden_sizes: Camadas ocultas.
        dropout: Dropout entre camadas.
        epochs: Épocas de treino.
        learning_rate: Taxa de aprendizado.
        batch_size: Tamanho do batch.
        weight_decay: Regularização L2.
        device: Device PyTorch (default cuda se disponível).

    Returns:
        Tupla (modelo treinado, métricas em escala original).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=batch_size,
        shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=batch_size,
        shuffle=False,
    )

    model = MLPRegressor(
        input_size=input_size, hidden_sizes=hidden_sizes, dropout=dropout
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    best_val = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item() * len(X_batch)
        val_loss = val_loss / max(len(val_loader.dataset), 1)  # type: ignore[arg-type]

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            logger.info("MLP epoch %d/%d val_loss=%.6f", epoch, epochs, val_loss)

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()

    metrics = evaluate_predictions(y_test, y_pred, close_min, close_max)
    metrics["best_val_loss"] = best_val
    logger.info(
        "MLP: MAE=%.4f, RMSE=%.4f, MAPE=%.2f%%, sigma_coverage=%.2f%%",
        metrics["mae"],
        metrics["rmse"],
        metrics["mape"],
        metrics["sigma_coverage_0_5"] * 100,
    )
    return model, metrics


# --- Orquestrador ---


CANDIDATE_FEATURES = [
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


def _set_mlflow_tags(config: dict, model_name: str, framework: str) -> None:
    """Aplica as tags obrigatórias da run, sobrescrevendo model_name/framework."""
    tags = config["mlflow"]["tags"].copy()
    tags["git_sha"] = get_git_sha()
    tags["fairness_checked"] = "false"
    tags["model_name"] = model_name
    tags["framework"] = framework
    tags["model_role"] = "baseline"
    for k, v in tags.items():
        mlflow.set_tag(k, str(v))


def run_baselines(
    config_path: str = "configs/model_config.yaml",
    features_path: str = "data/processed/petr4_features.parquet",
    output_path: str = "metrics/baseline_metrics.json",
    ridge_model_path: str = "models/ridge_petr4.joblib",
    mlp_model_path: str = "models/mlp_petr4.pt",
) -> dict[str, Any]:
    """Executa Ridge e MLP, loga no MLflow e escreve JSON consolidado.

    Args:
        config_path: YAML com hiperparâmetros (lê seção `baseline`).
        features_path: Parquet com features do pipeline.
        output_path: Arquivo de métricas consolidadas.
        ridge_model_path: Saída do modelo Ridge.
        mlp_model_path: Saída do modelo MLP.

    Returns:
        Dict com métricas dos dois baselines.
    """
    config = load_config(config_path)

    df = pd.read_parquet(features_path)
    feature_cols = [c for c in CANDIDATE_FEATURES if c in df.columns]

    seq_len = config["features"]["sequence_length"]
    horizon = config["features"]["prediction_horizon"]
    train_ratio = config["data"]["train_split"]
    val_ratio = config["data"]["validation_split"]

    prepared = prepare_baseline_data(
        df=df,
        feature_cols=feature_cols,
        sequence_length=seq_len,
        prediction_horizon=horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    baseline_cfg = config.get("baseline", {})
    ridge_cfg = baseline_cfg.get("ridge", {"alpha": 1.0})
    mlp_cfg = baseline_cfg.get(
        "mlp",
        {
            "hidden_sizes": [128, 64],
            "dropout": 0.1,
            "epochs": 50,
            "learning_rate": 0.001,
            "batch_size": 32,
            "weight_decay": 0.0001,
        },
    )

    _ensure_mlflow_tracking_uri()
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # --- Ridge ---
    with mlflow.start_run(run_name="baseline-ridge"):
        _set_mlflow_tags(config, model_name="ridge-petr4", framework="sklearn")
        mlflow.log_params(
            {
                "model": "ridge",
                "alpha": ridge_cfg["alpha"],
                "input_size": prepared["input_size"],
                "feature_columns": ",".join(feature_cols),
                "sequence_length": seq_len,
                "prediction_horizon": horizon,
            }
        )
        ridge_model, ridge_metrics = train_ridge(
            splits=prepared["splits"],
            close_min=prepared["close_min"],
            close_max=prepared["close_max"],
            alpha=float(ridge_cfg["alpha"]),
        )
        mlflow.log_metrics(ridge_metrics)
        Path(ridge_model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(ridge_model, ridge_model_path)
        mlflow.log_artifact(ridge_model_path)

    # --- MLP ---
    with mlflow.start_run(run_name="baseline-mlp"):
        _set_mlflow_tags(config, model_name="mlp-petr4", framework="pytorch")
        mlflow.log_params(
            {
                "model": "mlp",
                "input_size": prepared["input_size"],
                "hidden_sizes": str(mlp_cfg["hidden_sizes"]),
                "dropout": mlp_cfg["dropout"],
                "epochs": mlp_cfg["epochs"],
                "learning_rate": mlp_cfg["learning_rate"],
                "batch_size": mlp_cfg["batch_size"],
                "weight_decay": mlp_cfg["weight_decay"],
                "feature_columns": ",".join(feature_cols),
                "sequence_length": seq_len,
                "prediction_horizon": horizon,
            }
        )
        mlp_model, mlp_metrics = train_mlp_baseline(
            splits=prepared["splits"],
            close_min=prepared["close_min"],
            close_max=prepared["close_max"],
            input_size=prepared["input_size"],
            hidden_sizes=list(mlp_cfg["hidden_sizes"]),
            dropout=float(mlp_cfg["dropout"]),
            epochs=int(mlp_cfg["epochs"]),
            learning_rate=float(mlp_cfg["learning_rate"]),
            batch_size=int(mlp_cfg["batch_size"]),
            weight_decay=float(mlp_cfg["weight_decay"]),
        )
        mlflow.log_metrics(mlp_metrics)
        Path(mlp_model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": mlp_model.state_dict(),
                "input_size": prepared["input_size"],
                "hidden_sizes": list(mlp_cfg["hidden_sizes"]),
                "dropout": float(mlp_cfg["dropout"]),
                "feature_columns": feature_cols,
                "sequence_length": seq_len,
            },
            mlp_model_path,
        )
        mlflow.log_artifact(mlp_model_path)

    # --- Consolidar ---
    consolidated: dict[str, Any] = {
        "ridge": ridge_metrics,
        "mlp": mlp_metrics,
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "sequence_length": seq_len,
        "prediction_horizon": horizon,
    }

    # Comparação opcional com LSTM se já houver treino
    lstm_path = Path("metrics/train_metrics.json")
    if lstm_path.exists():
        with open(lstm_path) as f:
            lstm_metrics = json.load(f)
        consolidated["lstm"] = lstm_metrics
        consolidated["comparison"] = _build_comparison(
            ridge_metrics, mlp_metrics, lstm_metrics
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(consolidated, f, indent=2, ensure_ascii=False)
    logger.info("Métricas dos baselines salvas em %s", output_path)

    return consolidated


def _build_comparison(
    ridge: dict[str, float], mlp: dict[str, float], lstm: dict[str, float]
) -> dict[str, Any]:
    """Sumariza qual modelo lidera em cada métrica.

    Args:
        ridge: Métricas Ridge.
        mlp: Métricas MLP.
        lstm: Métricas LSTM.

    Returns:
        Dicionário com vencedor por métrica (menor é melhor para erro,
        maior é melhor para sigma_coverage).
    """
    candidates = {"ridge": ridge, "mlp": mlp, "lstm": lstm}
    out: dict[str, Any] = {}
    for metric in ("mae", "rmse", "mape"):
        scores = {
            name: m.get(metric)
            for name, m in candidates.items()
            if m.get(metric) is not None
        }
        if scores:
            best = min(scores, key=lambda k: scores[k])  # type: ignore[arg-type]
            out[f"best_{metric}"] = {"model": best, "value": scores[best]}
    if any("sigma_coverage_0_5" in m for m in candidates.values()):
        scores = {
            name: m.get("sigma_coverage_0_5")
            for name, m in candidates.items()
            if m.get("sigma_coverage_0_5") is not None
        }
        if scores:
            best = max(scores, key=lambda k: scores[k])  # type: ignore[arg-type]
            out["best_sigma_coverage_0_5"] = {"model": best, "value": scores[best]}
    return out


def main() -> None:
    """Entry point CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    run_baselines()


if __name__ == "__main__":
    main()
