"""Contratos Pydantic da API FastAPI.

Os schemas são divididos por domínio (predict, infer, agent, health, train,
quality) para reduzir acoplamento entre rotas e facilitar reuso (clients
tipados, OpenAPI generation, mocks). Re-exportados aqui para manter um
ponto único de import.
"""

from src.api.schemas.agent import AgentRequest, AgentResponse
from src.api.schemas.health import (
    HealthResponse,
    LivenessResponse,
    ReadinessResponse,
    StartupResponse,
)
from src.api.schemas.infer import InferRequest, InferResponse
from src.api.schemas.predict import PredictionRequest, PredictionResponse
from src.api.schemas.quality import QualityRequest, QualityResponse
from src.api.schemas.train import TrainRequest, TrainResponse

__all__ = [
    "AgentRequest",
    "AgentResponse",
    "HealthResponse",
    "InferRequest",
    "InferResponse",
    "LivenessResponse",
    "PredictionRequest",
    "PredictionResponse",
    "QualityRequest",
    "QualityResponse",
    "ReadinessResponse",
    "StartupResponse",
    "TrainRequest",
    "TrainResponse",
]
