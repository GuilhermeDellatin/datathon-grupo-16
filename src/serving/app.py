"""API FastAPI para serving do modelo LSTM e agente ReAct.

Endpoints:
- POST /predict — Predição de preço via LSTM
- POST /agent — Query ao agente ReAct
- POST /train — Retreinamento em background
- POST /infer — Inferência raw (machine-to-machine)
- GET /health — Health check
- GET /metrics — Métricas Prometheus
"""

import logging
import time
from contextlib import asynccontextmanager

import numpy as np
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src.monitoring.metrics import (
    AGENT_REQUESTS,
    PREDICTION_LATENCY,
    PREDICTION_REQUESTS,
)

load_dotenv()
logger = logging.getLogger(__name__)


# --- Pydantic Models ---


class PredictionRequest(BaseModel):
    """Request para predição de preço."""

    ticker: str = Field(default="PETR4.SA", description="Símbolo da ação")
    horizon_days: int = Field(
        default=5, ge=1, le=30, description="Horizonte de predição em dias"
    )


class PredictionResponse(BaseModel):
    """Response da predição."""

    ticker: str
    current_price: float
    predicted_price: float
    variation_percent: float
    horizon_days: int
    model_version: str
    disclaimer: str = "Esta predição NÃO constitui recomendação de investimento."


class AgentRequest(BaseModel):
    """Request para o agente ReAct."""

    question: str = Field(
        ..., min_length=3, max_length=4096, description="Pergunta em linguagem natural"
    )


class AgentResponse(BaseModel):
    """Response do agente."""

    answer: str
    tools_used: list[str]
    success: bool


class HealthResponse(BaseModel):
    """Response do health check."""

    status: str
    model_loaded: bool
    agent_ready: bool
    version: str


class TrainResponse(BaseModel):
    """Response do disparo de treinamento."""

    message: str
    status: str


class InferRequest(BaseModel):
    """Request de inferência raw com features já escaladas."""

    features: list[list[float]] = Field(
        ...,
        description="Array 2D de features já escaladas (sequence_length, n_features)",
    )


class InferResponse(BaseModel):
    """Response da inferência raw."""

    predicted_scaled: float


# --- App Lifecycle ---

_predictor = None
_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega modelo e agente no startup."""
    global _predictor, _agent

    logger.info("Inicializando API...")

    # Carregar modelo LSTM
    try:
        from src.models.predict import StockPredictor

        _predictor = StockPredictor()
        logger.info("Modelo LSTM carregado com sucesso")
    except Exception as e:
        logger.error("Falha ao carregar modelo: %s", e)
        _predictor = None

    # Criar agente ReAct
    try:
        from src.agent.react_agent import create_stock_agent

        _agent = create_stock_agent(verbose=False)
        logger.info("Agente ReAct criado com sucesso")
    except Exception as e:
        logger.error("Falha ao criar agente: %s", e)
        _agent = None

    yield

    logger.info("Shutting down API...")


# --- FastAPI App ---

app = FastAPI(
    title="Datathon LSTM Stock Predictor",
    description=(
        "API para predição de preços de ações (PETR4.SA) com LSTM e agente ReAct"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check da API."""
    return HealthResponse(
        status="healthy" if _predictor else "degraded",
        model_loaded=_predictor is not None,
        agent_ready=_agent is not None,
        version="0.1.0",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Predição de preço de fechamento via modelo LSTM."""
    start_time = time.time()

    if _predictor is None:
        PREDICTION_REQUESTS.labels(ticker=request.ticker, status="error").inc()
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    try:
        import pandas as pd
        import yfinance as yf

        from src.data.feature_engineering import compute_features

        # Buscar dados recentes
        df = yf.download(request.ticker, period="6mo", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            raise HTTPException(
                status_code=404, detail=f"Sem dados para {request.ticker}"
            )

        df_features = compute_features(df)
        result = _predictor.predict_from_dataframe(df_features)

        current_price = float(df["Close"].iloc[-1])
        predicted = result["predicted_close"]
        variation = ((predicted - current_price) / current_price) * 100

        PREDICTION_LATENCY.observe(time.time() - start_time)
        PREDICTION_REQUESTS.labels(ticker=request.ticker, status="success").inc()

        return PredictionResponse(
            ticker=request.ticker,
            current_price=current_price,
            predicted_price=predicted,
            variation_percent=round(variation, 4),
            horizon_days=request.horizon_days,
            model_version="v1",
        )

    except HTTPException:
        raise
    except Exception as e:
        PREDICTION_REQUESTS.labels(ticker=request.ticker, status="error").inc()
        logger.error("Erro na predição: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/agent", response_model=AgentResponse)
async def agent_query(request: AgentRequest) -> AgentResponse:
    """Query ao agente ReAct."""
    if _agent is None:
        AGENT_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Agente não disponível")

    try:
        # Guardrails de input
        from src.security.guardrails import InputGuardrail

        guardrail = InputGuardrail()
        is_valid, reason = guardrail.validate(request.question)
        if not is_valid:
            AGENT_REQUESTS.labels(status="blocked").inc()
            raise HTTPException(status_code=400, detail=reason)

        # Executar agente
        from src.agent.react_agent import query_agent

        result = query_agent(_agent, request.question)

        # Guardrails de output
        from src.security.guardrails import OutputGuardrail

        output_guard = OutputGuardrail()
        sanitized_answer = output_guard.sanitize(result["answer"])

        tools_used = [step["tool"] for step in result.get("intermediate_steps", [])]

        AGENT_REQUESTS.labels(status="success").inc()

        return AgentResponse(
            answer=sanitized_answer,
            tools_used=tools_used,
            success=result["success"],
        )

    except HTTPException:
        raise
    except Exception as e:
        AGENT_REQUESTS.labels(status="error").inc()
        logger.error("Erro no agente: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


def _run_training_task() -> None:
    """Executa o pipeline de treinamento em background.

    Returns:
        None.
    """
    global _predictor

    logger.info("Iniciando pipeline de treinamento em background")
    try:
        from src.models.train import train_and_log

        run_id = train_and_log()
        logger.info("Treinamento concluído com run_id=%s", run_id)

        try:
            from src.models.predict import StockPredictor

            new_predictor = StockPredictor()
            _predictor = new_predictor
            logger.info("Modelo recarregado com sucesso após treinamento")
        except Exception:
            logger.error(
                "Falha ao recarregar modelo após treinamento", exc_info=True
            )
    except Exception:
        logger.error("Falha no pipeline de treinamento", exc_info=True)


@app.post(
    "/train",
    response_model=TrainResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_training(
    background_tasks: BackgroundTasks,
) -> TrainResponse:
    """Dispara o pipeline de treinamento em background.

    Args:
        background_tasks: Gerenciador de tasks em background do FastAPI.

    Returns:
        Status inicial do processamento.
    """
    background_tasks.add_task(_run_training_task)
    return TrainResponse(
        message="Pipeline de treinamento iniciado em background.",
        status="processing",
    )


@app.post("/infer", response_model=InferResponse)
async def infer_raw(request: InferRequest) -> InferResponse:
    """Executa inferência raw com features já escaladas.

    Args:
        request: Payload com matriz 2D de features escaladas.

    Returns:
        Valor predito na escala do modelo.

    Raises:
        HTTPException: Quando o modelo não está carregado, o payload tem shape
            inválido ou a inferência falha.
    """
    start_time = time.time()
    try:
        if _predictor is None:
            PREDICTION_REQUESTS.labels(ticker="RAW", status="error").inc()
            raise HTTPException(
                status_code=503, detail="Modelo não carregado"
            )

        data = np.array(request.features, dtype=np.float32)

        expected_shape = (
            _predictor.sequence_length,
            len(_predictor.feature_columns),
        )
        if (
            data.ndim != 2
            or data.shape[0] != expected_shape[0]
            or data.shape[1] != expected_shape[1]
            or not np.isfinite(data).all()
        ):
            PREDICTION_REQUESTS.labels(ticker="RAW", status="error").inc()
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Shape inválido: esperado {expected_shape}, "
                    f"recebido {data.shape}"
                ),
            )

        prediction = _predictor.predict(data)

        PREDICTION_REQUESTS.labels(ticker="RAW", status="success").inc()
        return InferResponse(predicted_scaled=float(prediction))

    except HTTPException:
        raise
    except Exception as exc:
        PREDICTION_REQUESTS.labels(ticker="RAW", status="error").inc()
        logger.error("Erro na inferência raw: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        PREDICTION_LATENCY.observe(time.time() - start_time)


def _run_training_task() -> None:
    """Executa o pipeline de treinamento em background.

    Returns:
        None.
    """
    global _predictor

    logger.info("Iniciando pipeline de treinamento em background")
    try:
        from src.models.train import train_and_log

        run_id = train_and_log()
        logger.info("Treinamento concluído com run_id=%s", run_id)

        try:
            from src.models.predict import StockPredictor

            new_predictor = StockPredictor()
            _predictor = new_predictor
            logger.info("Modelo recarregado com sucesso após treinamento")
        except Exception:
            logger.error(
                "Falha ao recarregar modelo após treinamento", exc_info=True
            )
    except Exception:
        logger.error("Falha no pipeline de treinamento", exc_info=True)


@app.post(
    "/train",
    response_model=TrainResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_training(
    background_tasks: BackgroundTasks,
) -> TrainResponse:
    """Dispara o pipeline de treinamento em background.

    Args:
        background_tasks: Gerenciador de tasks em background do FastAPI.

    Returns:
        Status inicial do processamento.
    """
    background_tasks.add_task(_run_training_task)
    return TrainResponse(
        message="Pipeline de treinamento iniciado em background.",
        status="processing",
    )


@app.post("/infer", response_model=InferResponse)
async def infer_raw(request: InferRequest) -> InferResponse:
    """Executa inferência raw com features já escaladas.

    Args:
        request: Payload com matriz 2D de features escaladas.

    Returns:
        Valor predito na escala do modelo.

    Raises:
        HTTPException: Quando o modelo não está carregado, o payload tem shape
            inválido ou a inferência falha.
    """
    start_time = time.time()
    try:
        if _predictor is None:
            PREDICTION_REQUESTS.labels(ticker="RAW", status="error").inc()
            raise HTTPException(
                status_code=503, detail="Modelo não carregado"
            )

        data = np.array(request.features, dtype=np.float32)

        expected_shape = (
            _predictor.sequence_length,
            len(_predictor.feature_columns),
        )
        if (
            data.ndim != 2
            or data.shape[0] != expected_shape[0]
            or data.shape[1] != expected_shape[1]
            or not np.isfinite(data).all()
        ):
            PREDICTION_REQUESTS.labels(ticker="RAW", status="error").inc()
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Shape inválido: esperado {expected_shape}, "
                    f"recebido {data.shape}"
                ),
            )

        prediction = _predictor.predict(data)

        PREDICTION_REQUESTS.labels(ticker="RAW", status="success").inc()
        return InferResponse(predicted_scaled=float(prediction))

    except HTTPException:
        raise
    except Exception as exc:
        PREDICTION_REQUESTS.labels(ticker="RAW", status="error").inc()
        logger.error("Erro na inferência raw: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        PREDICTION_LATENCY.observe(time.time() - start_time)


@app.get("/metrics")
async def metrics():
    """Endpoint Prometheus para scraping de métricas."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
