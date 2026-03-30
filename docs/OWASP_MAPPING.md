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
