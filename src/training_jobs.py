"""Bridge entre o endpoint POST /training/jobs e os stages do DVC.

A DAG ``train_lstm_stock`` persiste o ``dag_run.conf`` em
``.tmp/training_jobs/<job_id>/config.yaml`` e expõe o caminho na variável de
ambiente ``TRAINING_CONFIG_PATH``. Os módulos do pipeline (collector, train)
leem essa configuração através das funções deste arquivo para aplicar os
hiperparâmetros do request sem alterar ``configs/model_config.yaml``.

Formato esperado do YAML (mesmo do request validado pela API):

```yaml
job_id: train_20260502_petr4_abc123
ticker: PETR4.SA
period: 5y
model_config:
  sequence_length: 60
  hidden_size: 64
  num_layers: 2
  dropout: 0.2
training_config:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
```
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

TRAINING_CONFIG_PATH_ENV = "TRAINING_CONFIG_PATH"
TRAINING_TICKER_ENV = "TRAINING_TICKER"
TRAINING_PERIOD_ENV = "TRAINING_PERIOD"
TRAINING_JOB_ID_ENV = "TRAINING_JOB_ID"


def load_training_job_conf(path: str | None = None) -> dict[str, Any] | None:
    """Carrega a configuração do job persistida pela DAG.

    Args:
        path: Caminho explícito para o YAML. Quando ``None``, lê
            ``TRAINING_CONFIG_PATH`` do ambiente.

    Returns:
        Dict com a configuração do job ou ``None`` quando não há job ativo
        (caso típico de execução manual fora do Airflow).
    """
    resolved = path or os.getenv(TRAINING_CONFIG_PATH_ENV)
    if not resolved:
        return None

    config_path = Path(resolved)
    if not config_path.is_file():
        logger.warning(
            "TRAINING_CONFIG_PATH=%s aponta para arquivo inexistente; ignorando.",
            resolved,
        )
        return None

    with config_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        logger.warning(
            "TRAINING_CONFIG_PATH=%s não contém dict YAML; ignorando.", resolved
        )
        return None

    return data


def conf_to_train_overrides(conf: dict[str, Any]) -> dict[str, Any]:
    """Converte o ``conf`` do job no formato esperado por ``train_and_log``.

    Mapeamento:
        - ``ticker``                       → ``ticker`` e ``mlflow.tags.ticker``
        - ``model_config.sequence_length`` → ``features.sequence_length``
        - ``model_config.hidden_size``     → ``model.hidden_size``
        - ``model_config.num_layers``      → ``model.num_layers``
        - ``model_config.dropout``         → ``model.dropout``
        - ``training_config.epochs``       → ``training.epochs``
        - ``training_config.batch_size``   → ``training.batch_size``
        - ``training_config.learning_rate``→ ``training.learning_rate``

    Campos ausentes ou ``None`` no ``conf`` são ignorados (deixa o YAML
    canônico definir o default). ``period`` não vira override do treino,
    pois afeta apenas a coleta.

    Args:
        conf: Dict carregado de ``TRAINING_CONFIG_PATH``.

    Returns:
        Dict pronto para ``train_and_log(overrides=...)``. Vazio quando o
        ``conf`` não contém campos relevantes.
    """
    overrides: dict[str, Any] = {}

    ticker = conf.get("ticker")
    if ticker:
        overrides["ticker"] = ticker
        overrides.setdefault("mlflow", {}).setdefault("tags", {})["ticker"] = ticker

    model_cfg = conf.get("model_config") or {}
    if "sequence_length" in model_cfg and model_cfg["sequence_length"] is not None:
        overrides.setdefault("features", {})["sequence_length"] = model_cfg[
            "sequence_length"
        ]
    model_overrides: dict[str, Any] = {}
    for key in ("hidden_size", "num_layers", "dropout"):
        if key in model_cfg and model_cfg[key] is not None:
            model_overrides[key] = model_cfg[key]
    if model_overrides:
        overrides.setdefault("model", {}).update(model_overrides)

    training_cfg = conf.get("training_config") or {}
    training_overrides: dict[str, Any] = {}
    for key in ("epochs", "batch_size", "learning_rate"):
        if key in training_cfg and training_cfg[key] is not None:
            training_overrides[key] = training_cfg[key]
    if training_overrides:
        overrides.setdefault("training", {}).update(training_overrides)

    return overrides


def resolve_collection_params(
    fallback_ticker: str,
    fallback_start: str | None = None,
    fallback_end: str | None = None,
) -> tuple[str, str | None, str | None, str | None]:
    """Decide ticker/start/end/period para o stage de coleta.

    A precedência é:
        1. ``TRAINING_TICKER`` / ``TRAINING_PERIOD`` (job ativo).
        2. ``TRAINING_CONFIG_PATH`` carregado (ticker e period).
        3. Defaults vindos do YAML (``fallback_*``).

    Quando ``period`` é definido pelo job, ``start``/``end`` são zerados — o
    yfinance ignora o range absoluto nesse modo.

    Args:
        fallback_ticker: Ticker default (geralmente ``config["ticker"]``).
        fallback_start: ``start_date`` default do YAML.
        fallback_end: ``end_date`` default do YAML.

    Returns:
        Tuple ``(ticker, start_date, end_date, period)``.
    """
    env_ticker = os.getenv(TRAINING_TICKER_ENV)
    env_period = os.getenv(TRAINING_PERIOD_ENV)

    ticker = env_ticker or fallback_ticker
    period = env_period or None

    if not env_ticker or not env_period:
        conf = load_training_job_conf()
        if conf:
            ticker = ticker if env_ticker else conf.get("ticker", ticker)
            period = period if env_period else conf.get("period", period)

    if period:
        return ticker, None, None, period
    return ticker, fallback_start, fallback_end, None
