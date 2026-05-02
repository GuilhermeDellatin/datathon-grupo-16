"""Endpoint /agent — query ao agente ReAct com guardrails de input/output."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_agent
from src.api.schemas.agent import AgentRequest, AgentResponse
from src.monitoring.metrics import AGENT_REQUESTS

logger = logging.getLogger(__name__)
router = APIRouter(tags=["agent"])


@router.post("/agent", response_model=AgentResponse)
async def agent_query(
    request: AgentRequest,
    agent=Depends(get_agent),
) -> AgentResponse:
    """Query ao agente ReAct."""
    if agent is None:
        AGENT_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Agente não disponível")

    try:
        from src.security.guardrails import InputGuardrail

        guardrail = InputGuardrail()
        is_valid, reason = guardrail.validate(request.question)
        if not is_valid:
            AGENT_REQUESTS.labels(status="blocked").inc()
            raise HTTPException(status_code=400, detail=reason)

        from src.agent.react_agent import query_agent

        result = query_agent(agent, request.question)

        from src.security.guardrails import OutputGuardrail

        output_guard = OutputGuardrail()
        sanitized_answer = output_guard.sanitize(result["answer"])

        tools_used = [step["tool"] for step in result.get("intermediate_steps", [])]

        AGENT_REQUESTS.labels(status="success").inc()

        return AgentResponse(
            answer=sanitized_answer,
            tools_used=tools_used,
            success=result["success"],
        )

    except HTTPException:
        raise
    except Exception as e:
        AGENT_REQUESTS.labels(status="error").inc()
        logger.error("Erro no agente: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
