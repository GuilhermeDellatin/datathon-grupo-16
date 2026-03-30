# System Card — Datathon LSTM Stock Analysis (PETR4.SA)

Grupo 16 — Datathon Fase 05, Pós Tech MLET/FIAP

---

## Visão Geral do Sistema

Sistema integrado de análise e predição de ações da Petrobras (PETR4.SA) que combina:

1. **Modelo LSTM** (PyTorch) para predição de preço de fechamento
2. **Agente conversacional ReAct** (LangChain + OpenAI gpt-4o-mini) com 4 tools customizadas
3. **Pipeline RAG** (sentence-transformers + FAISS) para busca em documentação financeira
4. **API RESTful** (FastAPI) com guardrails de segurança e observabilidade

O sistema demonstra maturidade MLOps Nível 2 (Microsoft MLOps Maturity Model) com experiment tracking, model registry, CI/CD, monitoramento e governança.

## Arquitetura

```
[Usuário] → [FastAPI API]
                ├── POST /predict → [LSTM Model] → Predição + Disclaimer
                ├── POST /agent  → [InputGuardrail] → [ReAct Agent] → [OutputGuardrail]
                │                                          ├── Tool: predict_stock_price → [LSTM]
                │                                          ├── Tool: fetch_market_data → [yfinance]
                │                                          ├── Tool: search_financial_docs → [RAG/FAISS]
                │                                          └── Tool: compare_model_versions → [MLflow]
                ├── GET /health  → Health Check
                └── GET /metrics → [Prometheus] → [Grafana]
```

### Fluxo de Dados

```
yfinance → data/raw/ → feature_engineering → data/processed/ → train (MLflow)
                                                                    ↓
                                                              models/ (checkpoint .pt)
                                                                    ↓
                                                          FastAPI /predict endpoint
```

## Componentes

### 1. Pipeline de Dados (`src/data/`)

- **collector.py**: Download de dados OHLCV via yfinance para PETR4.SA
- **feature_engineering.py**: Cálculo de 14 indicadores técnicos (SMA, EMA, RSI, MACD, Bollinger, etc.)
- **Validação**: Schemas Pandera para integridade dos dados
- **Versionamento**: DVC para dados raw e processed

### 2. Modelo LSTM (`src/models/`)

- **lstm_model.py**: Arquitetura LSTMPredictor (PyTorch nn.Module)
- **train.py**: Pipeline de treinamento com MLflow tracking, early stopping, gradient clipping
- **predict.py**: Inferência isolada com StockPredictor (carrega checkpoint + scaler)
- **Model Card**: [docs/MODEL_CARD.md](MODEL_CARD.md)

### 3. Agente ReAct (`src/agent/`)

- **LLM**: OpenAI gpt-4o-mini (temperatura 0.0, determinístico)
- **Framework**: LangChain com AgentExecutor
- **Tools customizadas**:
  - `predict_stock_price`: Executa predição via modelo LSTM
  - `fetch_market_data`: Busca dados recentes de mercado via yfinance
  - `search_financial_docs`: Busca documentos financeiros via RAG/FAISS
  - `compare_model_versions`: Compara versões do modelo no MLflow Registry
- **Max iterations**: 10

### 4. RAG Pipeline (`src/agent/rag_pipeline.py`)

- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS (local, sem servidor)
- **Chunk size**: 1000 caracteres, overlap 200
- **Documentos**: Relatórios financeiros, documentação do modelo, glossário (em `data/rag_documents/`)
- **Avaliação**: RAGAS com 4 métricas sobre golden set de 25 pares

### 5. API (`src/serving/`)

- **Framework**: FastAPI com lifespan context manager
- **Endpoints**: `/predict`, `/agent`, `/health`, `/metrics`
- **Containerização**: Docker + docker-compose (API + MLflow + Prometheus + Grafana)
- **CI/CD**: GitHub Actions (lint → test → build)

### 6. Monitoramento (`src/monitoring/`)

- **Prometheus**: Métricas customizadas (latência, requests, drift PSI, erro de predição)
- **Grafana**: Dashboard com 6 painéis de observabilidade
- **Evidently**: Drift detection com PSI por feature
- **MLflow**: Experiment tracking e model registry

## Segurança

### Guardrails Implementados (`src/security/`)

- **InputGuardrail**: 13 padrões regex para detecção de prompt injection, limite de tamanho (4096 chars), detecção de encoding attacks (hex, unicode, HTML entities)
- **OutputGuardrail**: Remoção de PII via Presidio (PERSON, EMAIL, PHONE, CREDIT_CARD, IBAN), disclaimers automáticos em predições
- **BrazilianPIIDetector**: Detecção de CPF (com validação de dígitos verificadores), CNPJ e telefone BR

### OWASP Top 10 for LLMs

Mapeamento de 6 ameaças OWASP com mitigações implementadas:
- LLM01: Prompt Injection → InputGuardrail
- LLM02: Insecure Output Handling → OutputGuardrail + Presidio
- LLM06: Sensitive Information Disclosure → PII detection + secrets management
- LLM07: Insecure Plugin Design → Tools com escopo fixo e error handling
- LLM09: Overreliance → Disclaimers obrigatórios
- LLM10: Model Theft → Artefatos excluídos do Git

Detalhes: [docs/OWASP_MAPPING.md](OWASP_MAPPING.md)

### Red Teaming

7 cenários adversariais testados, todos PASS:
- Prompt injection direto, context stuffing, PII leakage, jailbreak (DAN), hallucination, encoding bypass, tool abuse

Detalhes: [docs/RED_TEAM_REPORT.md](RED_TEAM_REPORT.md)

## Privacidade (LGPD)

- **Base legal**: Legítimo interesse (Art. 7, IX) + dados públicos de mercado
- **Dados pessoais**: Sistema não persiste queries do usuário; output sanitizado com Presidio + BrazilianPIIDetector
- **Direitos dos titulares**: Documentados conforme Art. 18
- **Incidentes**: Plano de resposta conforme Art. 48
- **DPO**: Líder do Grupo 16

Detalhes: [docs/LGPD_PLAN.md](LGPD_PLAN.md)

## Avaliação

### Modelo LSTM
- MAE, RMSE, MAPE logados no MLflow
- Champion-challenger: promoção se RMSE melhora >= 0.5%

### Pipeline RAG (RAGAS)
- 4 métricas: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- Golden set: 25 pares pergunta-resposta em 6 categorias

### Agente (LLM-as-Judge)
- 5 critérios (nota 1-5): Correção Técnica, Relevância, Clareza, Utilidade para Investidor, Disclaimers de Risco
- A/B testing de prompts (3 variantes)

## Limitações do Sistema

1. **Dependência de APIs externas**: yfinance e OpenAI — falhas podem degradar funcionalidade
2. **Predições baseadas em padrões históricos**: Não capturam eventos futuros imprevistos
3. **Agente ReAct limitado**: Máximo de 10 iterações — queries muito complexas podem ser truncadas
4. **RAG limitado**: Busca apenas documentos indexados localmente, sem web search
5. **Ativo único**: Modelo treinado apenas para PETR4.SA
6. **Latência**: Primeira predição pode ser lenta (cold start do modelo)

## Equipe e Responsabilidades

| Papel | Responsável |
|-------|-------------|
| ML Engineer | Grupo 16 |
| Data Engineer | Grupo 16 |
| MLOps | Grupo 16 |
| DPO (LGPD) | Líder do Grupo 16 |

## Referências

- YAO, S. et al. ReAct: Synergizing Reasoning and Acting in Language Models. In: ICLR, 2023.
- ES, S. et al. RAGAS: Automated Evaluation of Retrieval Augmented Generation. 2024.
- MITCHELL, M. et al. Model Cards for Model Reporting. In: FAT*, 2019.
- SCULLEY, D. et al. Hidden Technical Debt in Machine Learning Systems. In: NeurIPS, 2015.
- OWASP. Top 10 for LLM Applications. 2025.
- BRASIL. Lei 13.709/2018 (Lei Geral de Proteção de Dados — LGPD).
- MICROSOFT. MLOps Maturity Model. 2026.
