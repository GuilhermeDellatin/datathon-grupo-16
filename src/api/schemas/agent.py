"""Contratos Pydantic do endpoint /agent (agente ReAct)."""

from pydantic import AliasChoices, BaseModel, Field


class AgentRequest(BaseModel):
    """Request para o agente ReAct.

    Aceita o campo `question` ou seu alias `query` no payload, para casar
    com clientes que seguem o exemplo do README (`{"query": "..."}`) e com
    clientes legados (`{"question": "..."}`).
    """

    question: str = Field(
        ...,
        min_length=3,
        max_length=4096,
        description="Pergunta em linguagem natural (aceita 'question' ou 'query').",
        validation_alias=AliasChoices("question", "query"),
    )


class AgentResponse(BaseModel):
    """Response do agente."""

    answer: str
    tools_used: list[str]
    success: bool
