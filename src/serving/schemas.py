"""Contratos Pydantic da API FastAPI.

Centraliza todos os modelos de request e response usados em
`src/serving/app.py`. Manter os schemas em um módulo separado:

- Facilita reuso (clients tipados, OpenAPI generation, mocks).
- Reduz acoplamento de regras de validação ao roteamento.
- Permite testar contratos isoladamente sem subir a app.
"""

from pydantic import AliasChoices, BaseModel, Field

# --- Predição de preço ---


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


# --- Inferência raw ---


class InferRequest(BaseModel):
    """Request de inferência raw com features já escaladas."""

    features: list[list[float]] = Field(
        ...,
        description="Array 2D de features já escaladas (sequence_length, n_features)",
    )


class InferResponse(BaseModel):
    """Response da inferência raw."""

    predicted_scaled: float


# --- Agente ReAct ---


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


# --- Saúde / probes ---


class HealthResponse(BaseModel):
    """Response do health check legado."""

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


# --- Treinamento ---


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


# --- Quality gate ---


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
