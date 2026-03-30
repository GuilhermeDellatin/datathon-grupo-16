"""Métricas Prometheus customizadas para o Datathon.

Métricas operacionais:
- prediction_latency_seconds: Tempo de inferência
- prediction_requests_total: Total de requisições de predição
- agent_requests_total: Total de requisições ao agente
- rag_retrieval_latency_seconds: Tempo de retrieval do RAG
- model_drift_psi: PSI score atual

Métricas de negócio:
- prediction_error_absolute: Erro absoluto da última predição verificada
- prediction_direction_accuracy: Acurácia direcional (subiu/desceu)
"""

import logging

from prometheus_client import Counter, Gauge, Histogram, Info

logger = logging.getLogger(__name__)

# --- Métricas Operacionais ---

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Tempo de inferência do modelo LSTM",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total de requisições de predição",
    labelnames=["ticker", "status"],
)

AGENT_REQUESTS = Counter(
    "agent_requests_total",
    "Total de requisições ao agente ReAct",
    labelnames=["status"],
)

RAG_RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Tempo de retrieval do RAG",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# --- Métricas de Drift ---

MODEL_DRIFT_PSI = Gauge(
    "model_drift_psi",
    "Population Stability Index — drift score atual",
    labelnames=["feature"],
)

DRIFT_ALERT = Counter(
    "drift_alerts_total",
    "Total de alertas de drift disparados",
    labelnames=["severity"],
)

# --- Métricas de Negócio ---

PREDICTION_ERROR = Gauge(
    "prediction_error_absolute",
    "Erro absoluto da última predição verificada",
)

PREDICTION_DIRECTION_ACCURACY = Gauge(
    "prediction_direction_accuracy",
    "Acurácia direcional acumulada",
)

# --- Model Info ---

MODEL_INFO = Info(
    "model",
    "Informações do modelo em produção",
)
