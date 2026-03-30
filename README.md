# LSTM Stock Price Predictor + ReAct Agent — PETR4.SA

[![CI](https://github.com/grupo-16/datathon-lstm-stocks/actions/workflows/ci.yml/badge.svg)](https://github.com/grupo-16/datathon-lstm-stocks/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Datathon Fase 05 — Pós Tech MLET/FIAP — Grupo 16

---

## Sobre o Projeto

Sistema de análise e predição de preço de ações da **Petrobras (PETR4.SA)** que combina um modelo **LSTM (PyTorch)** para predição de séries temporais com um **agente conversacional ReAct** (LangChain + OpenAI) equipado com 4 tools customizadas e um **pipeline RAG** (FAISS + sentence-transformers) sobre documentação financeira.

O projeto demonstra maturidade **MLOps Nível 2** (Microsoft MLOps Maturity Model) com experiment tracking (MLflow), monitoramento (Prometheus + Grafana), drift detection (Evidently), guardrails de segurança (OWASP Top 10 for LLMs), conformidade LGPD e documentação completa (Model Card + System Card).

**Disclaimer**: Este projeto é exclusivamente educacional. As predições geradas NÃO constituem recomendação de investimento.

---

## Arquitetura

```
[Usuário] → [FastAPI]
                ├── POST /predict  → [LSTM Model] → Predição + Disclaimer
                ├── POST /agent   → [Guardrails] → [ReAct Agent]
                │                                      ├── predict_stock_price  → [LSTM]
                │                                      ├── fetch_market_data    → [yfinance]
                │                                      ├── search_financial_docs → [FAISS/RAG]
                │                                      └── compare_model_versions → [MLflow]
                ├── GET  /health  → Health Check
                └── GET  /metrics → [Prometheus] → [Grafana]
```

---

## Quick Start

```bash
# 1. Clonar e entrar no repositório
git clone https://github.com/grupo-16/datathon-lstm-stocks.git
cd datathon-lstm-stocks

# 2. Criar ambiente virtual e instalar dependências
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
make install

# 3. Configurar variáveis de ambiente
cp .env.example .env
# Editar .env com suas API keys (OPENAI_API_KEY obrigatória para o agente)

# 4. Executar pipeline de dados + treinamento
make train

# 5. Subir API
make serve
# ou com Docker: make docker-up
```

---

## Endpoints da API

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/health` | Health check do sistema |
| `POST` | `/predict` | Predição de preço via modelo LSTM |
| `POST` | `/agent` | Query ao agente conversacional ReAct |
| `GET` | `/metrics` | Métricas Prometheus |

### Exemplos de Uso

```bash
# Health check
curl http://localhost:8000/health

# Predição de preço
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "PETR4.SA", "days": 60}'

# Query ao agente
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "Qual a previsão de preço da PETR4 para os próximos 5 dias?"}'

# Métricas Prometheus
curl http://localhost:8000/metrics
```

---

## Estrutura do Repositório

```
datathon-lstm-stocks/
├── src/
│   ├── data/                    # Pipeline de dados (collector, features)
│   ├── models/                  # LSTM (modelo, treino, inferência)
│   ├── agent/                   # ReAct agent + RAG + tools
│   ├── serving/                 # FastAPI + Dockerfile
│   ├── monitoring/              # Prometheus metrics + drift detection
│   └── security/                # Guardrails + PII detection
├── tests/                       # Suíte pytest (≥60% coverage)
├── evaluation/                  # RAGAS, LLM-as-judge, A/B testing
├── configs/                     # YAML configs (modelo, monitoring, Prometheus)
├── data/
│   ├── raw/                     # Dados OHLCV (DVC-tracked)
│   ├── processed/               # Features processadas (DVC-tracked)
│   ├── golden_set/              # Golden set para RAGAS (25 pares)
│   └── rag_documents/           # Documentos para RAG
├── docs/                        # Model Card, System Card, OWASP, LGPD
├── docker-compose.yml           # API + MLflow + Prometheus + Grafana
├── dvc.yaml                     # Pipeline: collect → features → train → evaluate
├── Makefile                     # Atalhos: make train, serve, test, etc.
└── pyproject.toml               # Dependências e config
```

---

## Tech Stack

| Componente | Tecnologia | Versão |
|-----------|------------|--------|
| Linguagem | Python | 3.11+ |
| Deep Learning | PyTorch | 2.2+ |
| Agente | LangChain + OpenAI (gpt-4o-mini) | 1.x |
| Vector Store | FAISS | - |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | - |
| API | FastAPI + Uvicorn | 0.100+ |
| Experiment Tracking | MLflow | 2.x |
| Monitoramento | Prometheus + Grafana | - |
| Drift Detection | Evidently | 0.7+ |
| PII Detection | Presidio (Microsoft) | - |
| Dados Financeiros | yfinance | - |
| Indicadores Técnicos | ta (Technical Analysis) | - |
| Validação | Pandera | - |
| CI/CD | GitHub Actions | - |
| Containerização | Docker + docker-compose | - |
| Versionamento de Dados | DVC | - |
| Linter | ruff | - |

---

## Avaliação

### Modelo LSTM

| Métrica | Descrição |
|---------|-----------|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| MAPE | Mean Absolute Percentage Error |

> Métricas logadas automaticamente no MLflow a cada run de treinamento.

### Pipeline RAG (RAGAS)

4 métricas avaliadas sobre golden set de 25 pares:
- **Faithfulness**: Fidelidade da resposta ao contexto recuperado
- **Answer Relevancy**: Relevância da resposta à pergunta
- **Context Precision**: Precisão dos chunks recuperados
- **Context Recall**: Recall dos chunks relevantes

### Agente (LLM-as-Judge)

5 critérios avaliados por GPT-4o-mini (nota 1-5):
- Correção Técnica, Relevância, Clareza, Utilidade para Investidor, Disclaimers de Risco

---

## Monitoramento e Drift

- **Prometheus** coleta métricas operacionais (latência, requests, erros)
- **Grafana** exibe dashboard com 6 painéis
- **Evidently** calcula PSI por feature para detectar data drift
- **Thresholds**: PSI < 0.1 OK | PSI 0.1-0.2 WARNING | PSI > 0.2 RETRAIN

---

## Segurança e Governança

- **Guardrails**: Input (13 padrões anti-injection) + Output (PII removal via Presidio)
- **OWASP**: Mapeamento de 6 ameaças do Top 10 for LLMs — [docs/OWASP_MAPPING.md](docs/OWASP_MAPPING.md)
- **Red Team**: 7 cenários adversariais, 7/7 PASS — [docs/RED_TEAM_REPORT.md](docs/RED_TEAM_REPORT.md)
- **LGPD**: Plano de conformidade completo — [docs/LGPD_PLAN.md](docs/LGPD_PLAN.md)
- **Model Card**: [docs/MODEL_CARD.md](docs/MODEL_CARD.md)
- **System Card**: [docs/SYSTEM_CARD.md](docs/SYSTEM_CARD.md)

---

## Comandos Úteis

```bash
make install      # Instalar dependências
make train        # Pipeline: coleta → features → treinamento
make serve        # Subir API local (uvicorn)
make test         # Rodar testes (pytest, ≥60% coverage)
make lint         # Linter (ruff)
make evaluate     # Avaliação RAGAS + LLM-as-judge
make docker-up    # Subir stack completa (Docker)
make docker-down  # Derrubar stack Docker
make clean        # Limpar artefatos
```

---

## Equipe

**Grupo 16** — Pós Tech MLET/FIAP — Datathon Fase 05

---

## Licença

Este projeto é de uso acadêmico no contexto do Datathon FIAP Fase 05.

---

## Referências

- YAO, S. et al. ReAct: Synergizing Reasoning and Acting in Language Models. ICLR, 2023.
- ES, S. et al. RAGAS: Automated Evaluation of Retrieval Augmented Generation. 2024.
- MITCHELL, M. et al. Model Cards for Model Reporting. FAT*, 2019.
- SCULLEY, D. et al. Hidden Technical Debt in Machine Learning Systems. NeurIPS, 2015.
- BRECK, E. et al. The ML Test Score: A Rubric for ML Production Readiness. IEEE BigData, 2017.
- OWASP. Top 10 for LLM Applications. 2025.
- BRASIL. Lei 13.709/2018 (LGPD).
- MICROSOFT. MLOps Maturity Model. 2026.
