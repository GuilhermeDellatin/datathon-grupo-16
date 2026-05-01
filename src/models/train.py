"""Pipeline de treinamento LSTM com MLflow tracking padronizado.

Implementa:
- Treino com early stopping e gradient clipping
- MLflow tracking: params, metrics, artifacts, tags obrigatórias
- Model Registry: registro automático com metadata
- Champion-challenger: comparação antes de promover modelo
"""

import copy
import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

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


def _ensure_mlflow_tracking_uri() -> None:
    """Garante um tracking URI válido para o MLflow.

    Quando ``MLFLOW_TRACKING_URI`` não está setada, o MLflow escolhe
    heuristicamente um backend SQLite no CWD (``sqlite:///mlflow.db``),
    o que falha quando o arquivo não existe ou foi criado por uma
    versão incompatível de SQLite. Para execução local sem servidor,
    forçamos ``file:./mlruns`` (filesystem store) explicitamente.
    """
    if not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("file:./mlruns")


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


def compute_fairness_by_volatility(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_quantiles: int = 3,
    fairness_tolerance: float = 0.10,
) -> dict[str, float]:
    """Avalia fairness do modelo entre regimes de volatilidade.

    Divide o teste em ``n_quantiles`` faixas pela volatilidade local da
    série real (calculada em janela móvel) e compara o ``sigma_coverage_0_5``
    em cada faixa contra a média global. O modelo é considerado "fair"
    quando o gap entre o melhor e o pior regime fica abaixo de
    ``fairness_tolerance``.

    Slice declarado: regime de volatilidade (proxy para "condições de
    mercado calmas vs voláteis"). É a única dimensão de fairness aplicável
    a um modelo single-asset; quando passarmos para multi-ticker, faixas
    por ativo serão adicionadas.

    Args:
        y_true: Valores reais em escala original (R$).
        y_pred: Predições em escala original.
        n_quantiles: Número de faixas (3 = baixa/média/alta).
        fairness_tolerance: Gap máximo aceitável (0.10 = 10 p.p.).

    Returns:
        Dict com ``coverage_by_regime``, ``gap`` e flag ``fairness_ok``.
    """
    n = len(y_true)
    if n < n_quantiles * 5:
        return {
            "coverage_by_regime": {},
            "gap": float("nan"),
            "fairness_ok": False,
            "reason": "amostra_insuficiente",
        }

    # Volatilidade local: rolling std de retornos absolutos
    returns = np.abs(np.diff(y_true, prepend=y_true[0]))
    window = max(5, n // (n_quantiles * 4))
    rolling = pd.Series(returns).rolling(window=window, min_periods=1).mean().values

    quantile_edges = np.quantile(rolling, np.linspace(0, 1, n_quantiles + 1))
    sigma_global = float(np.std(y_true, ddof=0))
    threshold = 0.5 * sigma_global
    errors = np.abs(y_pred - y_true)

    coverage_by_regime: dict[str, float] = {}
    for i in range(n_quantiles):
        lo, hi = quantile_edges[i], quantile_edges[i + 1]
        if i == n_quantiles - 1:
            mask = (rolling >= lo) & (rolling <= hi)
        else:
            mask = (rolling >= lo) & (rolling < hi)
        if not mask.any():
            continue
        coverage_by_regime[f"q{i + 1}"] = float(np.mean(errors[mask] <= threshold))

    if not coverage_by_regime:
        return {
            "coverage_by_regime": {},
            "gap": float("nan"),
            "fairness_ok": False,
            "reason": "sem_amostras_em_quantil",
        }

    values = list(coverage_by_regime.values())
    gap = float(max(values) - min(values))
    return {
        "coverage_by_regime": coverage_by_regime,
        "gap": gap,
        "fairness_ok": gap <= fairness_tolerance,
    }


def compute_sigma_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_sigma: float = 0.5,
) -> dict[str, float]:
    """Calcula a métrica de negócio de cobertura por desvios-padrão.

    Métrica do Datathon: percentual de predições com erro absoluto
    <= threshold_sigma * sigma(y_true). Os arrays devem estar na mesma
    escala (preferencialmente preço original em R$) para que o sigma
    represente a volatilidade real do ativo.

    Args:
        y_true: Valores reais.
        y_pred: Valores preditos.
        threshold_sigma: Tolerância em desvios-padrão (default 0.5).

    Returns:
        Dicionário com sigma observado, threshold absoluto e coverage [0, 1].
    """
    sigma = float(np.std(y_true, ddof=0))
    threshold = threshold_sigma * sigma
    errors = np.abs(y_pred - y_true)
    coverage = float(np.mean(errors <= threshold))
    return {
        "target_sigma": sigma,
        "sigma_threshold": threshold,
        "sigma_coverage": coverage,
    }


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

    return float(total_loss / len(dataloader.dataset))  # type: ignore[arg-type,operator]


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

    return float(total_loss / len(dataloader.dataset))  # type: ignore[arg-type,operator]


# --- Main Training Pipeline ---


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Mescla recursivamente `overrides` em `base` sem mutar a entrada.

    Valores em `overrides` prevalecem; sub-dicionários são mesclados
    recursivamente; outros tipos são substituídos.

    Args:
        base: Dicionário base (ex.: config carregado do YAML).
        overrides: Sobrescritas a aplicar.

    Returns:
        Novo dicionário mesclado.
    """
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def train_and_log(
    config_path: str = "configs/model_config.yaml",
    overrides: dict[str, Any] | None = None,
) -> str:
    """Pipeline completa de treinamento com MLflow tracking.

    Args:
        config_path: Caminho da configuração YAML.
        overrides: Dict com sobrescritas opcionais aplicadas após carregar o
            YAML (ex.: ``{"training": {"epochs": 10}}``). Útil para retreinos
            disparados via API com hiperparâmetros customizados.

    Returns:
        run_id do MLflow.
    """
    config = load_config(config_path)
    if overrides:
        config = _deep_merge(config, overrides)
        logger.info("Config sobrescrito com overrides: %s", list(overrides.keys()))
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
    _ensure_mlflow_tracking_uri()
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=f"lstm-{config['ticker']}") as run:
        # Tags obrigatórias (Schema do GAP 05)
        tags = config["mlflow"]["tags"].copy()
        tags["git_sha"] = get_git_sha()
        with open("data/processed/petr4_features.parquet", "rb") as _features_file:
            tags["training_data_version"] = hashlib.md5(
                _features_file.read(4096), usedforsecurity=False
            ).hexdigest()[:8]
        # `fairness_checked` é setado de verdade após avaliação por regime de
        # volatilidade no test set (ver compute_fairness_by_volatility).
        tags["fairness_checked"] = "pending"
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
        assert best_model_state is not None, "Nenhum checkpoint válido encontrado"
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

        # --- Métricas em escala original de preço (R$) ---
        # target_idx=0 == coluna Close. Reverter o MinMaxScaler apenas dessa
        # coluna usando data_min_/data_max_ correspondentes.
        close_min = float(scaler.data_min_[0])
        close_max = float(scaler.data_max_[0])
        close_range = close_max - close_min
        y_true_price = y_true * close_range + close_min
        y_pred_price = y_pred * close_range + close_min

        price_metrics = compute_metrics(y_true_price, y_pred_price)
        metrics["mae_price"] = price_metrics["mae"]
        metrics["rmse_price"] = price_metrics["rmse"]
        metrics["mape_price"] = price_metrics["mape"]

        # Métrica de negócio (Datathon): cobertura dentro de 0.5 sigma.
        sigma_metrics = compute_sigma_coverage(
            y_true_price, y_pred_price, threshold_sigma=0.5
        )
        metrics["target_sigma"] = sigma_metrics["target_sigma"]
        metrics["sigma_threshold_0_5"] = sigma_metrics["sigma_threshold"]
        metrics["sigma_coverage_0_5"] = sigma_metrics["sigma_coverage"]

        gate_target = 0.70
        sigma_gate_passed = sigma_metrics["sigma_coverage"] >= gate_target
        mlflow.set_tag("sigma_gate_70pct_passed", str(sigma_gate_passed).lower())

        # --- Fairness por regime de volatilidade ---
        fairness = compute_fairness_by_volatility(
            y_true_price, y_pred_price, n_quantiles=3, fairness_tolerance=0.10
        )
        for regime, cov in fairness.get("coverage_by_regime", {}).items():
            metrics[f"sigma_coverage_0_5_{regime}"] = cov
        if not np.isnan(fairness.get("gap", float("nan"))):
            metrics["fairness_gap"] = fairness["gap"]
        mlflow.set_tag("fairness_checked", str(fairness.get("fairness_ok", False)).lower())
        mlflow.set_tag(
            "fairness_method", "volatility_quantiles_3_tolerance_0.10"
        )
        if "reason" in fairness:
            mlflow.set_tag("fairness_skip_reason", str(fairness["reason"]))

        mlflow.log_metrics(metrics)

        logger.info(
            "Test metrics (norm): MAE=%.6f, RMSE=%.6f, MAPE=%.2f%%",
            metrics["mae"],
            metrics["rmse"],
            metrics["mape"],
        )
        logger.info(
            "Test metrics (R$): MAE=%.4f, RMSE=%.4f, MAPE=%.2f%%",
            metrics["mae_price"],
            metrics["rmse_price"],
            metrics["mape_price"],
        )
        logger.info(
            "Métrica de negócio: sigma=%.4f, threshold(0.5σ)=%.4f, "
            "coverage=%.2f%% (gate %s, alvo %.0f%%)",
            sigma_metrics["target_sigma"],
            sigma_metrics["sigma_threshold"],
            sigma_metrics["sigma_coverage"] * 100,
            "PASSOU" if sigma_gate_passed else "FALHOU",
            gate_target * 100,
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
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(metrics_path)

        # Scaler
        scaler_path = "models/scaler.joblib"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)

        # Registrar no Model Registry
        model_uri = f"runs:/{run.info.run_id}/model"
        mv = mlflow.register_model(model_uri, config["mlflow"]["model_name"])
        logger.info("Modelo registrado: %s v%s", mv.name, mv.version)

        # --- Champion-challenger gate ---
        # Promove para Production se (a) RMSE melhora >= min_improvement
        # vs champion atual, E (b) o quality gate de sigma_coverage_0_5
        # passa (>= 0.70).
        try:
            should_promote = champion_challenger(
                challenger_run_id=run.info.run_id,
                model_name=config["mlflow"]["model_name"],
                metric="rmse",
                min_improvement=0.005,
            )
            sigma_ok = bool(sigma_gate_passed)

            client = mlflow.tracking.MlflowClient()
            if should_promote and sigma_ok:
                client.transition_model_version_stage(
                    name=mv.name,
                    version=mv.version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                mlflow.set_tag("promotion_status", "promoted_to_production")
                logger.info(
                    "Modelo %s v%s PROMOVIDO para Production (champion+gate ok).",
                    mv.name,
                    mv.version,
                )
            else:
                stage = "Staging"
                client.transition_model_version_stage(
                    name=mv.name,
                    version=mv.version,
                    stage=stage,
                    archive_existing_versions=False,
                )
                reason = []
                if not should_promote:
                    reason.append("rmse_no_improvement")
                if not sigma_ok:
                    reason.append("sigma_gate_failed")
                mlflow.set_tag(
                    "promotion_status",
                    f"staging:{','.join(reason) or 'manual_review'}",
                )
                logger.info(
                    "Modelo %s v%s mantido em Staging (motivo: %s).",
                    mv.name,
                    mv.version,
                    ",".join(reason) or "default",
                )
        except Exception as exc:
            logger.warning(
                "Falha no champion-challenger / promoção: %s", exc, exc_info=True
            )
            mlflow.set_tag("promotion_status", "error")

        return str(run.info.run_id)


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

    return bool(improvement >= min_improvement)


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
