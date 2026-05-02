"""Contratos Pydantic do endpoint /infer (inferência raw machine-to-machine)."""

from pydantic import BaseModel, Field


class InferRequest(BaseModel):
    """Request de inferência raw com features já escaladas."""

    features: list[list[float]] = Field(
        ...,
        description="Array 2D de features já escaladas (sequence_length, n_features)",
    )


class InferResponse(BaseModel):
    """Response da inferência raw."""

    predicted_scaled: float
