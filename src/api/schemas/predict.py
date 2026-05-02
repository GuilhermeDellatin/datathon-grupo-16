"""Contratos Pydantic do endpoint /predict."""

from pydantic import BaseModel, Field


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
