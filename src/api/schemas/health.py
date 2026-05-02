"""Contratos Pydantic dos probes (liveness, readiness, startup, health)."""

from pydantic import BaseModel


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
