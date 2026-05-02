"""Endpoint /train — disparo de retreinamento em background."""

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Body, status

from src.api import dependencies
from src.api.schemas.train import TrainRequest, TrainResponse

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
