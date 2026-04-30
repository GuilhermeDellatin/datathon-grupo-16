"""API FastAPI para serving do modelo LSTM e agente ReAct.

Endpoints:
- GET / — Liveness probe (Kubernetes pattern)
- GET /ready — Readiness probe (modelo e agente carregados?)
- GET /startup — Startup probe (lifespan completou?)
- GET /health — Health check legado (mantido por compatibilidade)
- GET /metrics — Métricas Prometheus
- POST /predict — Predição de preço via LSTM
- POST /infer — Inferência raw (machine-to-machine)
- POST /agent — Query ao agente ReAct
- POST /train — Retreinamento em background
- POST /evaluate_quality — Quality gate sobre as métricas mais recentes
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Body, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import AliasChoices, BaseModel, Field

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
    """Request para o agente ReAct.

    Aceita o campo `question` ou seu alias `query` no payload, para casar
    com clientes que seguem o exemplo do README (`{"query": "..."}`) e com
    clientes legados (`{"question": "..."}`).
    """

    question: str = Field(
        ...,
        min_length=3,
        max_length=4096,
        description="Pergunta em linguagem natural (aceita 'question' ou 'query').",
        validation_alias=AliasChoices("question", "query"),
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


class LivenessResponse(BaseModel):
    """Response do liveness probe (GET /)."""

    status: str = "ok"
    service: str = "datathon-lstm-stocks"
    version: str


class ReadinessResponse(BaseModel):
    """Response do readiness probe (GET /ready)."""

    ready: bool
    model_loaded: bool
    agent_ready: bool


class StartupResponse(BaseModel):
    """Response do startup probe (GET /startup)."""

    started: bool
    model_loaded: bool
    agent_ready: bool


class QualityRequest(BaseModel):
    """Request opcional para o quality gate."""

    metrics_path: str | None = Field(
        default=None,
        description="Caminho do JSON de métricas. Default: configs/monitoring_config.yaml",
    )
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Cobertura mínima de sigma_coverage_0_5. Default: YAML.",
    )


class QualityResponse(BaseModel):
    """Resultado do quality gate."""

    passed: bool
    gate: str
    threshold: float
    observed: float | None
    message: str


class TrainRequest(BaseModel):
    """Request opcional para o disparo de treinamento.

    Todos os campos são opcionais. Quando ausentes, o pipeline usa os
    defaults de `configs/model_config.yaml`. Quando presentes:
    - `tickers`: o primeiro elemento é usado como ticker do retreino;
      a coleta de dados é re-executada para esse ticker.
    - `period`: passado ao yfinance como janela relativa (ex.: "2y").
    - `num_epochs`: sobrescreve `training.epochs` no config.
    """

    tickers: list[str] | None = Field(
        default=None,
        max_length=5,
        description="Lista de tickers (apenas o primeiro é usado).",
    )
    period: str | None = Field(
        default=None,
        max_length=10,
        description="Janela relativa do yfinance (ex.: '1y', '2y', 'max').",
    )
    num_epochs: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="Sobrescreve o número de épocas de treinamento.",
    )


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
_startup_complete = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega modelo e agente no startup."""
    global _predictor, _agent, _startup_complete

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

    _startup_complete = True
    logger.info("Startup concluído.")

    yield

    _startup_complete = False
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


_API_VERSION = "0.1.0"


@app.get("/", response_model=LivenessResponse)
async def liveness() -> LivenessResponse:
    """Liveness probe: o processo está vivo?

    Sempre retorna 200 quando a API está respondendo. Não inspeciona
    dependências (modelo, agente) — orquestradores devem usar `/ready`
    para isso. Adequado como `livenessProbe` em Kubernetes.
    """
    return LivenessResponse(version=_API_VERSION)


@app.get("/ready", response_model=ReadinessResponse)
async def readiness() -> ReadinessResponse:
    """Readiness probe: a API consegue servir tráfego?

    Retorna 200 apenas se modelo LSTM e agente ReAct estão carregados.
    Caso contrário, retorna 503 para que orquestradores parem de rotear
    requisições para esta instância até a recuperação.
    """
    ready = _predictor is not None and _agent is not None
    response = ReadinessResponse(
        ready=ready,
        model_loaded=_predictor is not None,
        agent_ready=_agent is not None,
    )
    if not ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.model_dump(),
        )
    return response


@app.get("/startup", response_model=StartupResponse)
async def startup_probe() -> StartupResponse:
    """Startup probe: a fase de boot terminou?

    Retorna 200 após o `lifespan` completar a inicialização (carregamento
    de modelo e agente atendidos com sucesso ou falha registrada). Retorna
    503 enquanto a inicialização ainda está em curso.
    """
    response = StartupResponse(
        started=_startup_complete,
        model_loaded=_predictor is not None,
        agent_ready=_agent is not None,
    )
    if not _startup_complete:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.model_dump(),
        )
    return response


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check legado da API (mantido para compatibilidade)."""
    return HealthResponse(
        status="healthy" if _predictor else "degraded",
        model_loaded=_predictor is not None,
        agent_ready=_agent is not None,
        version=_API_VERSION,
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
    global _predictor

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


@app.get("/metrics")
async def metrics():
    """Endpoint Prometheus para scraping de métricas."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/evaluate_quality", response_model=QualityResponse)
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
