"""Estado compartilhado e dependencies do FastAPI.

Mantém os artefatos carregados no startup (modelo LSTM, agente ReAct e flag
de boot) como atributos de módulo. Rotas consomem via `Depends(get_*)` para
desacoplar a lógica de roteamento da inicialização.

Os atributos privados (`_predictor`, `_agent`, `_startup_complete`) são
escritos pelo lifespan em `src.api.app` e podem ser substituídos por
`monkeypatch` em testes (ex.:
`monkeypatch.setattr("src.api.dependencies._predictor", FakePredictor())`).
"""

from __future__ import annotations

from typing import Any

_predictor: Any | None = None
_agent: Any | None = None
_startup_complete: bool = False


def get_predictor() -> Any | None:
    """Retorna o StockPredictor carregado no startup, ou None se indisponível."""
    return _predictor


def get_agent() -> Any | None:
    """Retorna o agente ReAct criado no startup, ou None se indisponível."""
    return _agent


def get_startup_complete() -> bool:
    """Indica se a fase de boot do lifespan terminou."""
    return _startup_complete
