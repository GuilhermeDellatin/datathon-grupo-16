"""Probes Kubernetes (liveness, readiness, startup), health legado e métricas Prometheus."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api import dependencies
from src.api.schemas.health import (
    HealthResponse,
    LivenessResponse,
    ReadinessResponse,
    StartupResponse,
)

router = APIRouter(tags=["health"])

API_VERSION = "0.1.0"


@router.get("/", response_model=LivenessResponse)
async def liveness() -> LivenessResponse:
    """Liveness probe: o processo está vivo?

    Sempre retorna 200 quando a API está respondendo. Não inspeciona
    dependências (modelo, agente) — orquestradores devem usar `/ready`
    para isso. Adequado como `livenessProbe` em Kubernetes.
    """
    return LivenessResponse(version=API_VERSION)


@router.get("/ready", response_model=ReadinessResponse)
async def readiness(
    predictor=Depends(dependencies.get_predictor),
    agent=Depends(dependencies.get_agent),
) -> ReadinessResponse:
    """Readiness probe: a API consegue servir tráfego?

    Retorna 200 apenas se modelo LSTM e agente ReAct estão carregados.
    Caso contrário, retorna 503 para que orquestradores parem de rotear
    requisições para esta instância até a recuperação.
    """
    ready = predictor is not None and agent is not None
    response = ReadinessResponse(
        ready=ready,
        model_loaded=predictor is not None,
        agent_ready=agent is not None,
    )
    if not ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.model_dump(),
        )
    return response


@router.get("/startup", response_model=StartupResponse)
async def startup_probe(
    predictor=Depends(dependencies.get_predictor),
    agent=Depends(dependencies.get_agent),
    started: bool = Depends(dependencies.get_startup_complete),
) -> StartupResponse:
    """Startup probe: a fase de boot terminou?

    Retorna 200 após o `lifespan` completar a inicialização (carregamento
    de modelo e agente atendidos com sucesso ou falha registrada). Retorna
    503 enquanto a inicialização ainda está em curso.
    """
    response = StartupResponse(
        started=started,
        model_loaded=predictor is not None,
        agent_ready=agent is not None,
    )
    if not started:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.model_dump(),
        )
    return response


@router.get("/health", response_model=HealthResponse)
async def health_check(
    predictor=Depends(dependencies.get_predictor),
    agent=Depends(dependencies.get_agent),
) -> HealthResponse:
    """Health check legado da API (mantido para compatibilidade)."""
    return HealthResponse(
        status="healthy" if predictor else "degraded",
        model_loaded=predictor is not None,
        agent_ready=agent is not None,
        version=API_VERSION,
    )


@router.get("/metrics")
async def metrics() -> Response:
    """Endpoint Prometheus para scraping de métricas."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
