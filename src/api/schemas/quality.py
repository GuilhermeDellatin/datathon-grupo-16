"""Contratos Pydantic do endpoint /evaluate_quality (quality gate)."""

from pydantic import BaseModel, Field


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
