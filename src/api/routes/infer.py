"""Endpoint /infer — inferência raw com features já escaladas (machine-to-machine)."""

import logging
import time

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_predictor
from src.api.schemas.infer import InferRequest, InferResponse
from src.monitoring.metrics import (
    PREDICTION_LATENCY,
    PREDICTION_REQUESTS,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["infer"])


@router.post("/infer", response_model=InferResponse)
async def infer_raw(
    request: InferRequest,
    predictor=Depends(get_predictor),
) -> InferResponse:
    """Executa inferência raw com features já escaladas.

    Args:
        request: Payload com matriz 2D de features escaladas.
        predictor: StockPredictor injetado via lifespan.

    Returns:
        Valor predito na escala do modelo.

    Raises:
        HTTPException: Quando o modelo não está carregado, o payload tem shape
            inválido ou a inferência falha.
    """
    start_time = time.time()
    try:
        if predictor is None:
            PREDICTION_REQUESTS.labels(ticker="RAW", status="error").inc()
            raise HTTPException(
                status_code=503, detail="Modelo não carregado"
            )

        data = np.array(request.features, dtype=np.float32)

        expected_shape = (
            predictor.sequence_length,
            len(predictor.feature_columns),
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

        prediction = predictor.predict(data)

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
