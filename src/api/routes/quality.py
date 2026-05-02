"""Endpoint /evaluate_quality — quality gate sobre as métricas mais recentes."""

import logging
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from src.api.schemas.quality import QualityRequest, QualityResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["quality"])


@router.post("/evaluate_quality", response_model=QualityResponse)
async def evaluate_quality(
    request: Annotated[QualityRequest | None, Body()] = None,
) -> QualityResponse:
    """Executa o quality gate sobre as métricas de treino mais recentes.

    Lê `metrics/train_metrics.json` (ou caminho fornecido) e verifica se
    `sigma_coverage_0_5` >= threshold. O threshold pode vir do corpo do
    request ou do `configs/monitoring_config.yaml`. A resposta indica se
    o gate passou e contém o valor observado.

    Args:
        request: Parâmetros opcionais (`metrics_path`, `threshold`).

    Returns:
        Resultado do gate.

    Raises:
        HTTPException: 404 se o arquivo de métricas não existir, 500 se o
            avaliador falhar inesperadamente.
    """
    from src.monitoring.quality_gates import (
        DEFAULT_CONFIG_PATH,
        DEFAULT_METRICS_PATH,
        DEFAULT_SIGMA_COVERAGE_KEY,
        DEFAULT_SIGMA_COVERAGE_THRESHOLD,
        check_sigma_coverage_gate,
        load_gate_config,
        load_metrics,
    )

    if request is None:
        request = QualityRequest()
    metrics_path = request.metrics_path or DEFAULT_METRICS_PATH
    try:
        metrics_data = load_metrics(metrics_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Erro ao carregar métricas: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    gate_cfg = load_gate_config(DEFAULT_CONFIG_PATH).get("sigma_coverage", {})
    metric_key = str(gate_cfg.get("metric_key", DEFAULT_SIGMA_COVERAGE_KEY))
    threshold = (
        request.threshold
        if request.threshold is not None
        else float(gate_cfg.get("min_coverage", DEFAULT_SIGMA_COVERAGE_THRESHOLD))
    )

    passed, message = check_sigma_coverage_gate(
        metrics_data, threshold=threshold, metric_key=metric_key
    )
    raw = metrics_data.get(metric_key)
    observed: float | None
    try:
        observed = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        observed = None

    return QualityResponse(
        passed=passed,
        gate=metric_key,
        threshold=threshold,
        observed=observed,
        message=message,
    )
