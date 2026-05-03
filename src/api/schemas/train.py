"""Contratos Pydantic dos endpoints /train e /training/jobs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ALLOWED_PERIODS: tuple[str, ...] = (
    "1mo",
    "3mo",
    "6mo",
    "1y",
    "2y",
    "5y",
    "10y",
    "max",
)


class TrainRequest(BaseModel):
    """Request opcional para o endpoint legado /train.

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
    """Response do endpoint legado /train."""

    message: str
    status: str


class TrainingModelConfig(BaseModel):
    """Hiperparâmetros do modelo LSTM passados em /training/jobs."""

    sequence_length: int = Field(
        ...,
        ge=5,
        le=252,
        description="Tamanho da janela de entrada do LSTM (em dias úteis).",
    )
    hidden_size: int = Field(
        ...,
        ge=8,
        le=512,
        description="Tamanho do estado oculto do LSTM.",
    )
    num_layers: int = Field(
        ...,
        ge=1,
        le=5,
        description="Número de camadas empilhadas do LSTM.",
    )
    dropout: float = Field(
        ...,
        ge=0.0,
        le=0.8,
        description="Taxa de dropout entre camadas do LSTM.",
    )


class TrainingRunConfig(BaseModel):
    """Hiperparâmetros do loop de treinamento passados em /training/jobs."""

    epochs: int = Field(
        ...,
        ge=1,
        le=1000,
        description="Número de épocas de treinamento.",
    )
    batch_size: int = Field(
        ...,
        ge=1,
        le=512,
        description="Tamanho do batch.",
    )
    learning_rate: float = Field(
        ...,
        gt=0.0,
        le=1.0,
        description="Taxa de aprendizado do otimizador.",
    )


class TrainingJobRequest(BaseModel):
    """Request do endpoint POST /training/jobs.

    O campo JSON `model_config` é mapeado para o atributo Python `architecture`
    para evitar colisão com o atributo reservado `model_config` do Pydantic v2.
    """

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Símbolo do ativo no yfinance (ex.: 'PETR4.SA').",
    )
    period: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Janela relativa do yfinance.",
    )
    architecture: TrainingModelConfig = Field(
        ...,
        alias="model_config",
        description="Hiperparâmetros do modelo LSTM.",
    )
    training_config: TrainingRunConfig = Field(
        ...,
        description="Hiperparâmetros do loop de treinamento.",
    )

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        """Normaliza ticker para uppercase e remove espaços."""
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("ticker não pode ser vazio")
        return cleaned

    @field_validator("period")
    @classmethod
    def _validate_period(cls, value: str) -> str:
        """Garante que `period` é um valor aceito pelo yfinance."""
        cleaned = value.strip().lower()
        if cleaned not in ALLOWED_PERIODS:
            raise ValueError(
                f"period inválido: '{value}'. Valores aceitos: {list(ALLOWED_PERIODS)}"
            )
        return cleaned


class TrainingJobResponse(BaseModel):
    """Response do endpoint POST /training/jobs."""

    job_id: str = Field(..., description="Identificador estável do job.")
    status: Literal["queued"] = Field(
        default="queued",
        description="Status inicial do job (sempre 'queued' nesta resposta).",
    )
    ticker: str = Field(..., description="Ticker normalizado.")
    airflow_dag_id: str = Field(..., description="DAG do Airflow responsável.")
    airflow_dag_run_id: str = Field(
        ..., description="run_id do Airflow correlacionado ao job."
    )
