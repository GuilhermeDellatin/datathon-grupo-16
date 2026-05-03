"""FastAPI factory + lifespan + middleware.

Endpoints registrados (via `src.api.routes`):
- GET  / — Liveness probe (Kubernetes pattern)
- GET  /ready — Readiness probe (modelo e agente carregados?)
- GET  /startup — Startup probe (lifespan completou?)
- GET  /health — Health check legado (mantido por compatibilidade)
- GET  /metrics — Métricas Prometheus
- POST /predict — Predição de preço via LSTM
- POST /infer — Inferência raw (machine-to-machine)
- POST /agent — Query ao agente ReAct
- POST /train — Retreinamento em background (legado; mantido por compatibilidade)
- POST /training/jobs — Cria um job de retreino e delega ao Airflow + DVC
- POST /evaluate_quality — Quality gate sobre as métricas mais recentes
"""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import dependencies
from src.api.routes import agent, health, infer, predict, quality, train

load_dotenv()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega modelo e agente no startup e libera ao encerrar."""
    logger.info("Inicializando API...")

    try:
        from src.models.predict import StockPredictor

        dependencies._predictor = StockPredictor()
        logger.info("Modelo LSTM carregado com sucesso")
    except Exception as e:
        logger.error("Falha ao carregar modelo: %s", e)
        dependencies._predictor = None

    try:
        from src.agent.react_agent import create_stock_agent

        dependencies._agent = create_stock_agent(verbose=False)
        logger.info("Agente ReAct criado com sucesso")
    except Exception as e:
        logger.error("Falha ao criar agente: %s", e)
        dependencies._agent = None

    dependencies._startup_complete = True
    logger.info("Startup concluído.")

    yield

    dependencies._startup_complete = False
    logger.info("Shutting down API...")


def create_app() -> FastAPI:
    """Constrói a instância do FastAPI com middleware e routers registrados."""
    app = FastAPI(
        title="Datathon LSTM Stock Predictor",
        description=(
            "API para predição de preços de ações (PETR4.SA) com LSTM e agente ReAct"
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(infer.router)
    app.include_router(agent.router)
    app.include_router(train.router)
    app.include_router(quality.router)

    return app


app = create_app()
