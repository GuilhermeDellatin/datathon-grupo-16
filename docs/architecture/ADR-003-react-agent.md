# ADR-003 — Agente ReAct com 4 Tools sobre LangChain

- **Status**: Aceito
- **Data**: 2026-04-27
- **Decisores**: Grupo 16 (Datathon Fase 05 — Pós Tech MLET/FIAP)

## Contexto

Além da predição numérica (LSTM, ADR-001), o sistema precisa expor uma
interface conversacional capaz de:

1. Responder perguntas em **português brasileiro** sobre PETR4.SA.
2. Combinar predição do LSTM, dados de mercado em tempo real e contexto
   indexado (RAG sobre documentação financeira) em uma resposta coerente.
3. Operar com **disclaimers obrigatórios** ("não constitui recomendação de
   investimento") e postura neutra.
4. Resistir a ataques OWASP LLM (prompt injection, PII leakage, etc.).
5. Ser auditável (logs do raciocínio, das tools chamadas e das observações).

## Alternativas Avaliadas

| Padrão | Prós | Contras | Decisão |
|---|---|---|---|
| **ReAct** (Yao et al., 2023) | Pensamento + ação alternados; trilhas de raciocínio inspecionáveis | Mais tokens por interação | ✅ Adotado |
| Function calling direto | Latência menor; menos tokens | Sem traços de raciocínio; menor explicabilidade | ❌ |
| LangGraph (state machine) | Bom para fluxos complexos | Overkill para 4 tools sequenciais | ❌ |
| Orquestrador customizado | Controle total | Reinventa primitiva já madura na LangChain | ❌ |

## Decisão

Adotamos o agente **ReAct** instanciado via
`langchain_classic.agents.create_react_agent()` com `AgentExecutor`, e expomos
**4 tools** customizadas. O LLM é OpenAI `gpt-4o-mini` (ADR-002).

### Tools

Definidas em `src/agent/tools.py::ALL_TOOLS`:

| Tool | Função | Backend |
|---|---|---|
| `predict_stock_price` | Predição LSTM + variação esperada | `src/models/predict.py::StockPredictor` |
| `fetch_market_data` | Preço atual, OHLCV, RSI, SMA | `yfinance` |
| `search_financial_docs` | Busca semântica em docs | RAG FAISS + MiniLM |
| `compare_model_versions` | Histórico de versões e métricas | MLflow Model Registry |

Justificativas:

- 4 tools cobrem o universo da pergunta esperada (predição numérica, dado
  bruto, conhecimento de documentos, governança de modelo) sem tornar o
  espaço de ação grande demais para o ReAct decidir.
- Cada tool retorna **string formatada** já estruturada para o LLM
  sintetizar.
- Falhas são capturadas e devolvem mensagem em vez de propagar exceção
  (`try/except` interno em cada tool).

### Configuração do AgentExecutor

```python
AgentExecutor(
    agent=...,
    tools=ALL_TOOLS,
    verbose=False,
    max_iterations=10,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)
```

- `max_iterations=10` evita loops sem convergência.
- `handle_parsing_errors=True` recupera de falhas do parser ReAct.
- `return_intermediate_steps=True` permite registrar quais tools foram
  usadas em cada chamada (telemetria via Prometheus
  `AGENT_REQUESTS{status=...}`).

### Prompt do sistema

`src/agent/react_agent.py::SYSTEM_PROMPT` impõe:

1. Disclaimer obrigatório em qualquer predição.
2. Proibição de recomendações explícitas de compra/venda.
3. Resposta em português brasileiro.
4. Uso obrigatório de `search_financial_docs` quando o usuário fala em
   "dividendos", "políticas", "estratégia" ou "governança" — bloqueia
   alucinações sobre conteúdo factual.

### Guardrails (OWASP Top 10 LLM)

- `InputGuardrail` (`src/security/guardrails.py`): 13 padrões regex de
  prompt injection + detecção de encoding attacks (`\xNN`, `\uNNNN`,
  entidades HTML) + limite de 4096 chars (anti-context-stuffing).
- `OutputGuardrail`: anonimização via Presidio + `BrazilianPIIDetector`
  (CPF/CNPJ com dígitos verificadores) + enforcement de disclaimers.

### Endpoint

- `POST /agent` aceita o campo `question` ou `query` (alias Pydantic via
  `AliasChoices`, plano #6).
- `min_length=3, max_length=4096`.
- Retorna `answer`, `tools_used` (telemetria) e `success`.

## Consequências

### Positivas

- Trilhas de raciocínio (`intermediate_steps`) permitem auditoria e
  depuração — útil para a banca e para debugging em produção.
- 4 tools dão ao agente capacidade real de responder perguntas factuais
  (RAG), preditivas (LSTM) e operacionais (versões do modelo).
- Guardrails desacoplados do prompt — possível trocar/atualizar sem
  re-treinar/re-prompar.
- Compatível com OpenAI ou outro LLM (basta trocar a fábrica em
  `create_stock_agent`).

### Negativas / Trade-offs

- ReAct gasta mais tokens que function calling direto (mais turnos por
  pergunta) — mitigável com prompt caching da OpenAI.
- Mais latência (P50) por causa dos múltiplos passos.
- Dependência de `OPENAI_API_KEY` (herdada do ADR-002).

### Mitigações

1. `temperature=0.0` reduz não-determinismo entre chamadas.
2. Telemetria Prometheus (`AGENT_REQUESTS{status="success|error|blocked"}`)
   detecta degradação rápida em produção.
3. Avaliação contínua via RAGAS + LLM-as-judge (ADR-004) sobre golden set.
4. Guardrails são testados em `tests/test_guardrails.py` e
   `docs/RED_TEAM_REPORT.md` documenta 7 cenários adversariais.

## Alternativas Rejeitadas

- **Function calling direto**: rejeitado por perda de explicabilidade
  (sem `Thought`/`Observation` visíveis), o que prejudica auditoria
  exigida pela rubrica de governança.
- **LangGraph**: rejeitado por adicionar complexidade de state machine
  desnecessária para 4 tools relativamente independentes.
- **Mais tools** (ex.: news_sentiment, options_chain): rejeitado para
  manter o espaço de ação pequeno e o agente convergir em 2–3 passos.

## Referências

- Yao, S. et al. (2023). _ReAct: Synergizing Reasoning and Acting in
  Language Models_. ICLR 2023.
- OWASP Top 10 for LLM Applications (2025) — LLM01, LLM02, LLM06, LLM07,
  LLM09, LLM10. Mapeamento em `docs/OWASP_MAPPING.md`.
- `src/agent/react_agent.py`, `src/agent/tools.py`,
  `src/agent/rag_pipeline.py`.
- `src/security/guardrails.py`, `src/security/pii_detection.py`.
- `tests/test_agent.py`, `tests/test_react_agent.py`,
  `tests/test_tools.py`, `tests/test_guardrails.py`.
- ADR-002 — Escolha do LLM (OpenAI vs Qwen local).
- ADR-004 — Estratégia de Avaliação.