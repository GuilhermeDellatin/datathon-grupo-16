"""Endpoint /predict — predição de preço de fechamento via LSTM."""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_predictor
from src.api.schemas.predict import PredictionRequest, PredictionResponse
from src.monitoring.metrics import (
    PREDICTION_LATENCY,
    PREDICTION_REQUESTS,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    predictor=Depends(get_predictor),
) -> PredictionResponse:
    """Predição de preço de fechamento via modelo LSTM."""
    start_time = time.time()

    if predictor is None:
        PREDICTION_REQUESTS.labels(ticker=request.ticker, status="error").inc()
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    try:
        import pandas as pd
        import yfinance as yf

        from src.data.feature_engineering import compute_features

        df = yf.download(request.ticker, period="6mo", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            raise HTTPException(
                status_code=404, detail=f"Sem dados para {request.ticker}"
            )

        df_features = compute_features(df)
        result = predictor.predict_from_dataframe(df_features)

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
