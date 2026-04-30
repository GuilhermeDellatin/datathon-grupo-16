# ADR-002 — Escolha do LLM: OpenAI gpt-4o-mini vs Qwen 2.5-0.5B Local

- **Status**: Aceito
- **Data**: 2026-04-27
- **Decisores**: Grupo 16 (Datathon Fase 05 — Pós Tech MLET/FIAP)
- **Substitui**: rascunho anterior que indicava Qwen 2.5-0.5B-Instruct + bitsandbytes INT4

## Contexto

O agente ReAct deste projeto precisa de um LLM para:

1. Raciocinar sobre perguntas em português brasileiro relativas à PETR4.SA.
2. Selecionar e invocar 4 tools (`predict_stock_price`, `fetch_market_data`,
   `search_financial_docs`, `compare_model_versions`).
3. Sintetizar respostas com disclaimers obrigatórios e estilo neutro.

Duas alternativas foram consideradas:

| Critério | Qwen 2.5-0.5B + bitsandbytes INT4 | OpenAI gpt-4o-mini (via LangChain) |
|---|---|---|
| Latência (P50) | Alta dependência da GPU local | Baixa e estável (API) |
| Hardware exigido | NVIDIA GPU CUDA 12.x, ~300MB VRAM | Sem GPU local |
| Determinismo | Bom com `temperature=0`, mas ainda variável | Bom com `temperature=0` |
| Qualidade ReAct | Limitada para 0.5B params | Adequada para o escopo |
| Custo operacional | Apenas energia/HW | Pago por token (controlável com cache) |
| Privacidade | Roda 100% on-prem | Tráfego para OpenAI |
| Reprodutibilidade em CI | Difícil (precisa GPU) | Possível com mock + smoke real opcional |
| Conformidade LGPD | Maior controle local | Necessita acordo de processamento |

## Decisão

Adotaremos **OpenAI `gpt-4o-mini`** (via `langchain_openai.ChatOpenAI`) como
LLM oficial do agente para a Fase 05 do Datathon. **FAISS** permanece como
vector store do RAG, em vez de ChromaDB.

Justificativa principal:

- **Qualidade do ReAct**: o `gpt-4o-mini` tem desempenho consistentemente
  superior em raciocínio multi-passo e seleção de tools que um modelo de 0.5B
  parâmetros, o que reduz drasticamente o risco de respostas incoerentes na
  apresentação à banca.
- **Reprodutibilidade**: a CI atual roda em GitHub Actions sem GPU; manter
  Qwen quantizado exigiria runners self-hosted ou alterações estruturais no
  pipeline.
- **Tempo de implementação**: a integração com OpenAI já está concluída e
  testada (`src/agent/react_agent.py`, guardrails, evaluation com RAGAS e
  LLM-as-judge). Migrar agora atrasaria entregas mais críticas (métrica de
  negócio, baselines, endpoints faltantes).
- **FAISS vs ChromaDB**: o índice FAISS atual está estável, integrado ao
  Dockerfile via `data/rag_index/`, e cobre a necessidade do projeto. A troca
  por ChromaDB exigiria nova ingestão, novo schema e re-validação dos
  testes RAG sem benefício funcional perceptível.

## Consequências

### Positivas

- Qualidade superior das respostas do agente.
- Pipeline simples (sem GPU obrigatória; CI executa sem mocks pesados de LLM).
- Stack alinhado ao código já em produção e à suíte de testes existente.
- FAISS permite indexação local e empacotamento direto na imagem Docker.

### Negativas / Trade-offs

- Dependência de serviço externo (`OPENAI_API_KEY` obrigatório).
- Custo por requisição em produção (mitigável com prompt caching e batching).
- Tráfego de dados sai do ambiente local — necessário registrar isso no plano
  LGPD (`docs/LGPD_PLAN.md`) como transferência internacional de dados.
- Não atende à intenção original do `README_TO_ANALYSIS.md` de demonstrar
  inferência local quantizada.

### Mitigações

1. `InputGuardrail` e `OutputGuardrail` continuam ativos
   (`src/security/guardrails.py`) para conter prompt injection e remover PII
   antes do tráfego sair/voltar do OpenAI.
2. `BrazilianPIIDetector` (`src/security/pii_detection.py`) anonimiza CPF/CNPJ
   no output.
3. Métricas de latência e taxa de erro do agente são monitoradas via
   `AGENT_REQUESTS` no Prometheus (`src/monitoring/metrics.py`).
4. Caso a banca avalie negativamente a dependência externa, o ADR pode ser
   revisitado com migração para Qwen ou outro LLM local em uma fase
   posterior.

## Alternativas rejeitadas

- **Qwen 2.5-0.5B + bitsandbytes INT4**: rejeitada por exigir GPU em CI e por
  qualidade insuficiente em ReAct multi-tool no idioma português.
- **Llama 3.1-8B em servidor dedicado**: rejeitada por custo de
  infraestrutura incompatível com o cronograma do datathon.
- **ChromaDB no lugar de FAISS**: rejeitada por não trazer ganho relevante e
  exigir reescrita do `rag_pipeline.py` e dos testes RAG.

## Referências

- `src/agent/react_agent.py` — uso de `ChatOpenAI(model="gpt-4o-mini")`.
- `src/agent/rag_pipeline.py` — uso de
  `langchain_community.vectorstores.FAISS`.
- `evaluation/ragas_eval.py`, `evaluation/llm_judge.py` — avaliação que
  depende do mesmo modelo.
- OWASP Top 10 for LLM Applications (2025) — riscos de LLM remoto.
- LGPD (Lei 13.709/2018), Art. 33 — transferência internacional de dados.
