"""Contratos Pydantic do endpoint /train (disparo de retreinamento)."""

from pydantic import BaseModel, Field


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
