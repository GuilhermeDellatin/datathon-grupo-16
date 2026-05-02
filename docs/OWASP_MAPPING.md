# OWASP Top 10 for LLM Applications — Mapeamento

Referência: OWASP Top 10 for LLM Applications (2025)

---

## LLM01: Prompt Injection

- **Descrição**: Atacante manipula input para alterar comportamento do LLM, fazendo-o ignorar instruções do sistema ou executar ações não autorizadas.
- **Risco no projeto**: Alto — o agente ReAct aceita input livre do usuário via endpoint `/agent`.
- **Mitigação implementada**:
  - `InputGuardrail` com 13 regex patterns para detecção de prompt injection direto e indireto.
  - Validação de tamanho máximo (4096 chars) para prevenir context stuffing.
  - Detecção de encoding attacks (hex, unicode, HTML entities).
- **Código**: `src/security/guardrails.py::InputGuardrail`
- **Teste**: `tests/test_guardrails.py::test_prompt_injection_blocked`

---

## LLM02: Insecure Output Handling

- **Descrição**: Output do LLM é usado sem validação, podendo conter código malicioso, PII ou informações sensíveis.
- **Risco no projeto**: Médio — respostas do agente são exibidas diretamente ao usuário.
- **Mitigação implementada**:
  - `OutputGuardrail` com Presidio para detecção e anonimização de PII (PERSON, EMAIL, PHONE, CREDIT_CARD, IBAN).
  - `BrazilianPIIDetector` para PII específica do Brasil (CPF, CNPJ, telefone BR) com validação de dígitos verificadores.
  - Enforcing de disclaimers obrigatórios em respostas com predições financeiras.
- **Código**: `src/security/guardrails.py::OutputGuardrail`, `src/security/pii_detection.py::BrazilianPIIDetector`
- **Teste**: `tests/test_guardrails.py::test_pii_detection`

---

## LLM04: Model Denial of Service

- **Descrição**: Atacante consome recursos do LLM/sistema com inputs longos, requisições recorrentes ou loops de raciocínio, causando degradação de performance, indisponibilidade ou estouro de custo (em provedores cobrados por token, como OpenAI).
- **Risco no projeto**: Médio — a API expõe `/agent`, `/predict`, `/infer` e `/train` publicamente; cada chamada ao agente consome tokens da OpenAI (custo direto), e o pipeline de treinamento é caro em CPU/GPU.
- **Mitigação implementada**:
  - **Limite de tamanho de input**: `InputGuardrail.max_length = 4096` chars no `/agent` (anti-context-stuffing) e `min_length = 3` para descartar payloads vazios ou irrelevantes — `src/security/guardrails.py::InputGuardrail.validate()`.
  - **Limite de iterações do agente**: `AgentExecutor(max_iterations=10)` evita loops infinitos do ReAct mesmo com prompt injection parcial — `src/agent/react_agent.py::create_stock_agent()`.
  - **Validação Pydantic nos endpoints**:
    - `/predict`: `horizon_days: int = Field(ge=1, le=30)` em `PredictionRequest`.
    - `/train`: `num_epochs: int | None = Field(ge=1, le=1000)`, `tickers: list[str] | None = Field(max_length=5)`, `period: str | None = Field(max_length=10)` em `TrainRequest`.
    - `/infer`: shape exato `(sequence_length, n_features)` validado via `np.isfinite` antes de chegar ao modelo.
    - `/evaluate_quality`: `threshold: float | None = Field(ge=0.0, le=1.0)` em `QualityRequest`.
  - **Background tasks** para operações longas: `POST /train` retorna `202 ACCEPTED` imediatamente e roda o treinamento via `BackgroundTasks`, sem bloquear o event loop nem segurar conexões — `src/serving/app.py::trigger_training()`.
  - **Cobertura de probes Kubernetes**: `GET /ready` e `GET /startup` permitem que orquestradores parem de rotear tráfego para uma instância sob estresse antes da indisponibilidade total — `src/serving/app.py`.
  - **Telemetria**: `AGENT_REQUESTS{status="error|blocked"}` e `PREDICTION_LATENCY` no Prometheus permitem detectar tempestades de requisições em tempo real e alertar via Grafana.
- **Mitigações futuras (não implementadas)**: rate limiting por IP via middleware (ex.: `slowapi`), quotas por API key, circuit breaker em chamadas à OpenAI.
- **Código**: `src/security/guardrails.py::InputGuardrail`, `src/agent/react_agent.py`, `src/serving/app.py` (modelos `PredictionRequest`, `TrainRequest`, `InferRequest`, `QualityRequest`).
- **Teste**: `tests/test_guardrails.py::TestInputGuardrail::test_max_length_enforced`, `tests/test_api.py::TestTrainEndpoint::test_train_endpoint_validates_num_epochs`, `tests/test_api.py::TestTrainEndpoint::test_train_endpoint_validates_tickers_max_length`, `tests/test_api.py::TestEvaluateQualityEndpoint::test_invalid_threshold_validation`.

---

## LLM06: Sensitive Information Disclosure

- **Descrição**: LLM pode revelar informações sensíveis presentes nos dados de treinamento, configurações ou prompts do sistema.
- **Risco no projeto**: Médio — modelo treinado com dados de mercado públicos, mas agente tem acesso a configurações internas.
- **Mitigação implementada**:
  - Secrets gerenciados via `.env` + `python-dotenv`, nunca hardcoded.
  - `.gitignore` configurado para excluir `.env`, dados brutos e artefatos de modelo.
  - Presidio detecta e anonimiza PII no output antes de retornar ao usuário.
  - Prompt do agente instruído a nunca revelar configurações internas.
- **Código**: `src/security/guardrails.py::OutputGuardrail.sanitize()`
- **Teste**: `tests/test_guardrails.py::test_output_sanitization`

---

## LLM07: Insecure Plugin Design

- **Descrição**: Tools/plugins do LLM podem ser explorados se não tiverem validação adequada de input/output.
- **Risco no projeto**: Médio — 4 tools customizadas acessam yfinance, modelo LSTM, MLflow e FAISS.
- **Mitigação implementada**:
  - Cada tool tem tratamento de exceção individual (try/except com logging).
  - Tool `predict_stock_price` sempre inclui disclaimer de não-recomendação.
  - Tool `search_financial_docs` limita output a 500 chars por documento.
  - Tool `compare_model_versions` limita a 5 versões mais recentes.
  - Nenhuma tool executa código arbitrário — todas têm escopo fixo.
- **Código**: `src/agent/tools.py`
- **Teste**: `tests/test_agent.py::test_tools_error_handling`

---

## LLM09: Overreliance

- **Descrição**: Usuários confiam excessivamente nas respostas do LLM sem verificação, especialmente em contextos financeiros.
- **Risco no projeto**: Alto — predições de preço podem ser interpretadas como recomendação de investimento.
- **Mitigação implementada**:
  - Disclaimer obrigatório em todas as predições: "Esta predição NÃO constitui recomendação de investimento."
  - `OutputGuardrail.validate_disclaimers()` verifica e adiciona disclaimers automaticamente.
  - Prompt do agente instrui a sempre incluir avisos de risco.
  - Model Card documenta limitações e casos de uso não recomendados.
  - Métricas de erro (MAE, RMSE, MAPE) publicamente disponíveis no MLflow.
- **Código**: `src/security/guardrails.py::OutputGuardrail.validate_disclaimers()`
- **Teste**: `tests/test_guardrails.py::test_disclaimer_enforcement`

---

## LLM10: Model Theft

- **Descrição**: Extração do modelo através de queries repetidas ou acesso não autorizado aos artefatos.
- **Risco no projeto**: Baixo — modelo acadêmico, mas boas práticas aplicadas.
- **Mitigação implementada**:
  - Artefatos de modelo (`.pt`, `.joblib`) excluídos do Git via `.gitignore`.
  - Rate limiting implícito via infraestrutura (pode ser adicionado via middleware).
  - Model Registry no MLflow com controle de versão e metadata.
  - Docker container não expõe artefatos de modelo diretamente.
- **Código**: `.gitignore`, `docker-compose.yml`, `src/serving/Dockerfile`
