"""Endpoints de treinamento.

- ``POST /train`` (legado) executa retreino em background no próprio processo.
- ``POST /training/jobs`` delega o disparo ao Airflow, que orquestra o pipeline
  reprodutível executado pelo DVC. Esse é o fluxo recomendado.
"""

import logging
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, status

from src.api import dependencies
from src.api.schemas.train import (
    TrainingJobRequest,
    TrainingJobResponse,
    TrainRequest,
    TrainResponse,
)
from src.api.services.airflow_client import (
    AirflowClient,
    AirflowClientError,
    get_airflow_client,
    get_training_dag_id,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["train"])


def _refresh_data_for_ticker(ticker: str, period: str | None) -> None:
    """Re-coleta e processa features para um ticker específico.

    Sobrescreve `data/raw/petr4_raw.parquet` e
    `data/processed/petr4_features.parquet` com dados do ticker informado.
    Os caminhos são mantidos para alinhar com o resto do pipeline.

    Args:
        ticker: Símbolo a coletar.
        period: Janela relativa (yfinance). Se None, usa start/end do YAML.
    """
    from src.data.collector import collect_stock_data, load_config, save_raw_data
    from src.data.feature_engineering import compute_features

    cfg = load_config()
    df = collect_stock_data(
        ticker=ticker,
        start_date=cfg["data"]["start_date"] if not period else None,
        end_date=cfg["data"]["end_date"] if not period else None,
        period=period,
    )
    save_raw_data(df)

    df_features = compute_features(df)
    output_path = "data/processed/petr4_features.parquet"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(output_path)
    logger.info(
        "Pipeline de dados atualizado para ticker=%s, period=%s (%d linhas)",
        ticker,
        period or "config",
        len(df_features),
    )


def _build_training_overrides(req: TrainRequest | None) -> dict:
    """Converte um TrainRequest em dict de overrides para `train_and_log`.

    Args:
        req: Request opcional do endpoint.

    Returns:
        Dict de overrides (vazio quando o request não exige mudanças).
    """
    overrides: dict = {}
    if req is None:
        return overrides

    if req.num_epochs is not None:
        overrides.setdefault("training", {})["epochs"] = req.num_epochs

    if req.tickers:
        ticker = req.tickers[0]
        overrides["ticker"] = ticker
        overrides.setdefault("mlflow", {}).setdefault("tags", {})["ticker"] = ticker

    return overrides


def _run_training_task(req: TrainRequest | None = None) -> None:
    """Executa o pipeline de treinamento em background.

    Quando `req` traz `tickers`/`period`, re-roda coleta e feature
    engineering antes do treino. `num_epochs` é aplicado como override
    de config diretamente em `train_and_log`.

    Args:
        req: Parâmetros opcionais do retreino (TrainRequest).
    """
    logger.info("Iniciando pipeline de treinamento em background")
    try:
        if req and (req.tickers or req.period):
            ticker = (
                req.tickers[0] if req.tickers else "PETR4.SA"
            )
            _refresh_data_for_ticker(ticker, req.period)

        from src.models.train import train_and_log

        overrides = _build_training_overrides(req)
        run_id = train_and_log(overrides=overrides if overrides else None)
        logger.info("Treinamento concluído com run_id=%s", run_id)

        try:
            from src.models.predict import StockPredictor

            dependencies._predictor = StockPredictor()
            logger.info("Modelo recarregado com sucesso após treinamento")
        except Exception:
            logger.error(
                "Falha ao recarregar modelo após treinamento", exc_info=True
            )
    except Exception:
        logger.error("Falha no pipeline de treinamento", exc_info=True)


@router.post(
    "/train",
    response_model=TrainResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_training(
    background_tasks: BackgroundTasks,
    request: Annotated[TrainRequest | None, Body()] = None,
) -> TrainResponse:
    """Dispara o pipeline de treinamento em background.

    O corpo da requisição é opcional. Quando fornecido, aceita os
    campos `tickers`, `period` e `num_epochs` (todos opcionais) para
    customizar o retreino. Sem corpo, usa os defaults do YAML.

    Args:
        background_tasks: Gerenciador de tasks em background do FastAPI.
        request: Parâmetros opcionais do retreino.

    Returns:
        Status inicial do processamento.
    """
    background_tasks.add_task(_run_training_task, request)
    detail_parts = []
    if request:
        if request.tickers:
            detail_parts.append(f"tickers={request.tickers}")
        if request.period:
            detail_parts.append(f"period={request.period}")
        if request.num_epochs is not None:
            detail_parts.append(f"num_epochs={request.num_epochs}")
    suffix = f" ({', '.join(detail_parts)})" if detail_parts else ""
    return TrainResponse(
        message=f"Pipeline de treinamento iniciado em background{suffix}.",
        status="processing",
    )


def _slug_for_ticker(ticker: str) -> str:
    """Converte um ticker em slug seguro para uso em ids do Airflow."""
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in ticker).strip("_")
    return slug or "ticker"


def _generate_job_id(ticker: str) -> str:
    """Gera um ``job_id`` estável e legível.

    Formato: ``train_{YYYYMMDD}_{ticker_slug}_{random_suffix}``. O sufixo
    aleatório garante unicidade quando vários jobs são criados no mesmo dia
    para o mesmo ticker.
    """
    today = datetime.now(UTC).strftime("%Y%m%d")
    suffix = secrets.token_hex(3)
    return f"train_{today}_{_slug_for_ticker(ticker)}_{suffix}"


def _build_dag_run_conf(
    request: TrainingJobRequest,
    job_id: str,
) -> dict:
    """Serializa o request validado no formato do ``dag_run.conf`` do Airflow.

    Usa ``by_alias=True`` para preservar o nome ``model_config`` no payload.
    """
    payload = request.model_dump(by_alias=True, mode="json")
    payload["job_id"] = job_id
    return payload


@router.post(
    "/training/jobs",
    response_model=TrainingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_training_job(
    request: TrainingJobRequest,
    airflow: Annotated[AirflowClient, Depends(get_airflow_client)],
) -> TrainingJobResponse:
    """Cria um job de treinamento e dispara a DAG no Airflow.

    Fluxo:
        1. Pydantic valida o payload (status 422 em caso de schema inválido).
        2. ``ticker`` é normalizado e o ``job_id`` é gerado.
        3. O ``conf`` é montado a partir do payload validado.
        4. A DAG ``train_lstm_stock`` é disparada via REST API do Airflow,
           usando ``job_id`` também como ``dag_run_id``.
        5. Em sucesso, a API responde ``202 Accepted`` sem aguardar o treino.

    Args:
        request: Payload do treino (ticker, period, model_config, training_config).
        airflow: Cliente Airflow injetado via FastAPI Depends.

    Raises:
        HTTPException: ``409`` quando já existe job em andamento para o mesmo
            ``dag_run_id``; ``502``/``503``/``504`` para falhas de comunicação
            com o Airflow.
    """
    job_id = _generate_job_id(request.ticker)
    dag_id = get_training_dag_id()
    conf = _build_dag_run_conf(request, job_id)

    logger.info(
        "Recebido job de treinamento: ticker=%s period=%s job_id=%s dag_id=%s",
        request.ticker,
        request.period,
        job_id,
        dag_id,
    )

    try:
        airflow.trigger_dag_run(dag_id=dag_id, dag_run_id=job_id, conf=conf)
    except AirflowClientError as exc:
        if exc.status_code == 409:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc),
        ) from exc

    return TrainingJobResponse(
        job_id=job_id,
        status="queued",
        ticker=request.ticker,
        airflow_dag_id=dag_id,
        airflow_dag_run_id=job_id,
    )
