# Datathon LSTM Stock Predictor — PETR4.SA

Sistema de predição de preços da Petrobras (PETR4.SA) usando rede LSTM em PyTorch, agente conversacional ReAct (LangChain + OpenAI), pipeline RAG sobre documentação financeira, servido via FastAPI, rastreado com MLflow, orquestrado com Airflow + DVC e containerizado com Docker Compose.

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Agent-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.x-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Airflow](https://img.shields.io/badge/Airflow-2.x-017CEE?style=for-the-badge&logo=apacheairflow&logoColor=white)](https://airflow.apache.org/)
[![DVC](https://img.shields.io/badge/DVC-Data-13ADC7?style=for-the-badge&logo=dvc&logoColor=white)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org/)

</div>

---

## Sumário

- [Visão Geral](#visão-geral)
- [Features](#features)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Arquitetura](#arquitetura)
- [Quick Start](#quick-start)
  - [Docker (recomendado)](#docker-recomendado)
  - [Imagem pré-buildada (staging)](#imagem-pré-buildada-staging)
  - [Instalação Local](#instalação-local)
- [API Endpoints](#api-endpoints)
- [Configuração](#configuração)
- [Avaliação](#avaliação)
- [Monitoramento e Drift](#monitoramento-e-drift)
- [Segurança e Governança](#segurança-e-governança)
- [Desenvolvimento](#desenvolvimento)
- [Autores](#autores)

> **Disclaimer:** projeto educacional. As predições NÃO constituem recomendação de investimento.

---

## Visão Geral

Pipeline completo de Machine Learning + LLM para análise de PETR4.SA:

1. **Coleta** — dados OHLCV via `yfinance`, versionados com DVC
2. **Feature engineering** — indicadores técnicos validados com `pandera`
3. **Treinamento** — LSTM (PyTorch) com early stopping, rastreado no MLflow
4. **Agente ReAct** — LangChain + `gpt-4o-mini` com 4 tools customizadas
5. **RAG** — sentence-transformers + FAISS sobre documentação financeira
6. **Serviço** — API REST (FastAPI) com guardrails de input/output
7. **Orquestração** — Airflow dispara retreino sob demanda; DVC garante reprodutibilidade
8. **Observabilidade** — Prometheus + Grafana + Evidently (drift PSI)
9. **Governança** — Model Card, System Card, OWASP Top 10 for LLMs, plano LGPD

O projeto demonstra maturidade **MLOps Nível 2** (Microsoft MLOps Maturity Model).

---

## Features

- **Predição LSTM** — preço de fechamento futuro com horizonte configurável
- **Agente conversacional** — ReAct com tools `predict_stock_price`, `fetch_market_data`, `search_financial_docs`, `compare_model_versions`
- **Pipeline RAG** — recuperação semântica sobre documentos financeiros (FAISS)
- **Treinamento via API** — `POST /training/jobs` dispara DAG do Airflow (resposta `202 Accepted`)
- **Reprodutibilidade** — pipeline DVC com stages `collect → features → train → evaluate`
- **Rastreamento MLflow** — params, metrics, artifacts e tags obrigatórias por run
- **Quality gate** — endpoint `/evaluate_quality` valida métricas antes da promoção
- **Guardrails** — anti prompt-injection na entrada e remoção de PII (Presidio) na saída
- **Drift detection** — PSI por feature com thresholds WARNING/RETRAIN
- **Avaliação RAG** — RAGAS (4 métricas) sobre golden set + LLM-as-judge (5 critérios)
- **Probes Kubernetes-ready** — `/`, `/ready`, `/startup` separados
- **Docker ready** — stack completa (API + MLflow + Airflow + Prometheus + Grafana) em um comando

---

## Estrutura do Projeto

```
datathon-grupo-16/
├── src/
│   ├── api/                    # FastAPI factory, rotas, schemas
│   │   ├── app.py
│   │   ├── routes/             # health, predict, infer, agent, train, quality
│   │   ├── schemas/            # Pydantic models
│   │   ├── services/           # Airflow client, etc.
│   │   └── Dockerfile
│   ├── agent/                  # ReAct agent + RAG pipeline + tools
│   ├── data/                   # collector (yfinance), feature engineering
│   ├── features/               # feature store / cache
│   ├── models/                 # LSTM (lstm_model, train, predict, baseline)
│   ├── monitoring/             # Prometheus metrics + drift (Evidently/PSI)
│   └── security/               # guardrails + PII detection (Presidio)
├── airflow/dags/               # DAG train_lstm_stock (orquestra DVC)
├── configs/                    # model_config.yaml, monitoring_config.yaml, prometheus.yml
├── data/
│   ├── raw/                    # OHLCV bruto (DVC-tracked)
│   ├── processed/              # Features (DVC-tracked)
│   ├── golden_set/             # ≥20 pares para RAGAS
│   ├── rag_documents/          # Documentos para indexação
│   └── rag_index/              # Índice FAISS (DVC-tracked)
├── evaluation/                 # ragas_eval.py, llm_judge.py, ab_test_prompts.py
├── docs/                       # MODEL_CARD, SYSTEM_CARD, OWASP_MAPPING, LGPD_PLAN, RED_TEAM_REPORT
├── tests/                      # pytest (≥60% coverage)
├── scripts/                    # index_documents.py, utilitários CLI
├── metrics/                    # baseline / train / RAGAS / LLM-judge metrics (JSON)
├── docker-compose.yml          # API + MLflow + Airflow + Prometheus + Grafana
├── dvc.yaml                    # collect → features → train → baseline → evaluate
├── Makefile
└── pyproject.toml
```

---

## Arquitetura

A stack roda em Docker Compose: a **API FastAPI** (porta 8000) atende requisições, delega retreino ao **Airflow** (porta 8080), que executa o pipeline **DVC** e registra runs no **MLflow** (porta 5000). Prometheus (9090) coleta métricas expostas em `/metrics` e Grafana (3000) renderiza dashboards.

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                            Docker Compose                              │
  │                                                                        │
  │   ┌─────────────┐  REST   ┌───────────────────────────────────────┐    │
  │   │   Cliente   │────────▶│            FastAPI :8000              │    │
  │   └─────────────┘         │  /predict /infer /agent /train        │    │
  │                           │  /training/jobs /evaluate_quality     │    │
  │                           │  /health /ready /startup /metrics     │    │
  │                           └───────┬───────────────┬───────────────┘    │
  │                                   │               │                    │
  │                  ┌────────────────┘               │                    │
  │                  │                                │ trigger DAG        │
  │                  ▼                                ▼                    │
  │          ┌──────────────┐                ┌─────────────────┐           │
  │          │  Guardrails  │                │  Airflow :8080  │           │
  │          │ in/out (PII) │                │ train_lstm_stock│           │
  │          └──────┬───────┘                └────────┬────────┘           │
  │                 │                                 │ dvc repro          │
  │     ┌───────────┴──────────┐                      ▼                    │
  │     ▼                      ▼                ┌──────────┐               │
  │ ┌──────────┐        ┌──────────────┐        │   DVC    │               │
  │ │  LSTM    │        │ ReAct Agent  │        │ pipeline │               │
  │ │ PyTorch  │        │ (LangChain)  │        └─────┬────┘               │
  │ │ /predict │        │  4 tools:    │              │                    │
  │ └────┬─────┘        │  predict /   │              ▼                    │
  │      │              │  fetch_data /│       ┌──────────────┐            │
  │      │              │  search_rag /│       │   MLflow     │            │
  │      │              │  compare_mv  │       │   :5000      │            │
  │      │              └───────┬──────┘       │ runs/metrics │            │
  │      │                      │              │  artifacts   │            │
  │      │                      ▼              └──────────────┘            │
  │      │              ┌──────────────┐                                   │
  │      │              │ RAG (FAISS + │                                   │
  │      │              │ embeddings)  │                                   │
  │      │              └──────────────┘                                   │
  │      │                                                                 │
  │      ▼                                                                 │
  │ ┌──────────────┐    cache    ┌──────────────┐                          │
  │ │ FeatureStore │◀────────────│   yfinance   │                          │
  │ │  (parquet)   │             │  (mercado)   │                          │
  │ └──────────────┘             └──────────────┘                          │
  │                                                                        │
  │ ┌─────────────┐  scrape  ┌─────────────┐    ┌─────────────┐            │
  │ │ Prometheus  │◀─────────│  /metrics   │    │   Grafana   │            │
  │ │   :9090     │          │  (FastAPI)  │    │    :3000    │            │
  │ └──────┬──────┘          └─────────────┘    └─────────────┘            │
  │        └─────────────────────────────────────────────▲                 │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘
```

### Fluxo MLflow (Orchestrator Pattern)

Apenas os pontos de entrada (`scripts/train.py` e a rota `/training/jobs`) abrem runs MLflow. O `ModelTrainer` é passivo: loga métricas só quando há run ativa, mantendo o tracking opcional. Tags obrigatórias por run: `model_name`, `model_version`, `model_type`, `training_data_version`, `git_sha`, `framework`, `phase`, `ticker`, `risk_level`, `fairness_checked`, `owner`.

---

## Quick Start

### Docker (recomendado)

A forma mais simples. Requer apenas [Docker Desktop](https://www.docker.com/products/docker-desktop/).

```bash
git clone https://github.com/GuilhermeDellatin/datathon-grupo-16.git
cd datathon-grupo-16

cp .env.example .env
# Editar .env e preencher OPENAI_API_KEY

# Sobe a stack completa (API + MLflow + Airflow + Prometheus + Grafana)
docker compose up -d --build
```

| Serviço | URL |
|---------|-----|
| API (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Airflow UI | http://localhost:8080 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

#### Treinar e prever via API

```bash
# 1. Disparar treinamento via Airflow (retorna 202 + job_id)
curl -X POST http://localhost:8000/training/jobs \
  -H "Content-Type: application/json" \
  -d '{"ticker": "PETR4.SA", "period": "5y", "training_config": {"epochs": 50}}'

# 2. Predição com LSTM treinado
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "PETR4.SA", "days": 5}'

# 3. Pergunta ao agente ReAct
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "Qual a previsão da PETR4 para os próximos 5 dias?"}'
```

> **Atenção:** os endpoints `/predict` e `/infer` exigem um checkpoint LSTM em `models/`. Sem ele, respondem `503` (modo degradado).

---

### Imagem pré-buildada (staging)

Cada merge na `master` publica a imagem da API no GitHub Container Registry com as tags `:staging` (último build verde) e `:<git-sha>` (imutável, rastreável).

```bash
# Imagem pública — não exige docker login
docker pull ghcr.io/guilhermedellatin/datathon-grupo-16:staging

# Modo standalone (apenas API; sem MLflow/Airflow/Prometheus)
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=<sua-key> \
  ghcr.io/guilhermedellatin/datathon-grupo-16:staging
```

A imagem não embarca o checkpoint LSTM. Para experiência completa, use `docker compose up` no repositório clonado (a stack monta `./models` como volume).

---

### Instalação Local

Para desenvolvimento sem Docker. Requer Python 3.11+.

```bash
git clone https://github.com/GuilhermeDellatin/datathon-grupo-16.git
cd datathon-grupo-16

python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\Activate.ps1       # Windows PowerShell

make install
cp .env.example .env                # Preencher OPENAI_API_KEY

# Pipeline de dados + treino reprodutíveis via DVC
dvc repro

# Indexar documentos para o RAG
python scripts/index_documents.py

# Subir API local
make serve
```

---

## API Endpoints

A API expõe rotas de predição, agente, retreino e observabilidade. Treinamentos são disparados de forma assíncrona (`202 Accepted`). A documentação interativa (Swagger UI) está em `http://localhost:8000/docs`.

### Predição & Inferência

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/predict` | Predição de preço com LSTM (formato amigável) |
| POST | `/infer` | Inferência raw (machine-to-machine, payload pré-processado) |

### Agente ReAct

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/agent` | Query em linguagem natural ao agente ReAct + RAG |

### Treinamento

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/training/jobs` | Cria job e dispara DAG do Airflow (recomendado) |

### Observabilidade & Quality

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/` | Liveness probe |
| GET | `/ready` | Readiness probe (modelo e agente carregados) |
| GET | `/startup` | Startup probe (lifespan completou) |
| GET | `/health` | Health check legado |
| GET | `/metrics` | Métricas Prometheus |
| POST | `/evaluate_quality` | Quality gate sobre métricas mais recentes |

---

## Configuração

Crie um `.env` na raiz a partir de `.env.example`:

```env
# OpenAI (obrigatório para o agente)
OPENAI_API_KEY=sk-...

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=datathon-petr4

# API
API_HOST=0.0.0.0
API_PORT=8000

# Modelo
TICKER=PETR4.SA
DATA_START_DATE=2018-01-01
DATA_END_DATE=2025-12-31
SEQUENCE_LENGTH=60
PREDICTION_HORIZON=5

# Monitoramento
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
DRIFT_PSI_WARNING=0.1
DRIFT_PSI_RETRAIN=0.2

# Segurança
MAX_INPUT_LENGTH=4096
PII_LANGUAGE=pt

# Airflow REST API (usado por POST /training/jobs)
AIRFLOW_BASE_URL=http://airflow:8080
AIRFLOW_USERNAME=admin
AIRFLOW_PASSWORD=admin
AIRFLOW_TRAINING_DAG_ID=train_lstm_stock
```

---

## Avaliação

### Modelo LSTM

| Métrica | Descrição |
|---------|-----------|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| MAPE | Mean Absolute Percentage Error |

> Métricas logadas automaticamente no MLflow a cada run. Champion-challenger: promove se RMSE melhora ≥0.5%.

### Pipeline RAG (RAGAS)

4 métricas sobre golden set de ≥20 pares:

- **Faithfulness** — fidelidade da resposta ao contexto recuperado
- **Answer Relevancy** — relevância da resposta à pergunta
- **Context Precision** — precisão dos chunks recuperados
- **Context Recall** — recall dos chunks relevantes

### Agente (LLM-as-Judge)

5 critérios avaliados por `gpt-4o-mini` (nota 1-5): Correção Técnica, Relevância, Clareza, Utilidade para Investidor, Disclaimers de Risco.

---

## Monitoramento e Drift

- **Prometheus** coleta métricas operacionais (latência, requests, erros) em `/metrics`
- **Grafana** exibe dashboard com painéis de SLO + drift
- **Evidently** calcula PSI por feature (referência vs janela atual)
- **Thresholds:** PSI < 0.1 OK · 0.1-0.2 WARNING · > 0.2 RETRAIN

---

## Segurança e Governança

- **Guardrails** — input (13 padrões anti-injection) + output (PII removal via Presidio com `pt_core_news_sm`)
- **OWASP Top 10 for LLMs** — mapeamento em [docs/OWASP_MAPPING.md](docs/OWASP_MAPPING.md)
- **Red Team** — 7 cenários adversariais em [docs/RED_TEAM_REPORT.md](docs/RED_TEAM_REPORT.md)
- **LGPD** — plano de conformidade em [docs/LGPD_PLAN.md](docs/LGPD_PLAN.md)
- **Model Card** — [docs/MODEL_CARD.md](docs/MODEL_CARD.md)
- **System Card** — [docs/SYSTEM_CARD.md](docs/SYSTEM_CARD.md)

---

## Desenvolvimento

```bash
make install      # Instala dependências (-e .[dev])
make train        # Pipeline DVC: collect → features → train
make serve        # Sobe API local (uvicorn)
make test         # pytest com --cov-fail-under=60
make lint         # ruff check
make evaluate     # RAGAS + LLM-as-judge
make docker-up    # Stack completa
make docker-down  # Derruba a stack
make clean        # Limpa artefatos
```

Convenções obrigatórias: type hints + docstrings Google em todas as funções, logging estruturado (sem `print`), line length 100, ruff configurado em `pyproject.toml`, secrets via `.env` (nunca hardcoded), DVC para dados (nunca commitar dados reais), split temporal em séries (nunca shuffle).

---

## Autores

Desenvolvido pelo **Grupo 16** — Pós Tech MLET/FIAP — Datathon Fase 05.

| Nome | RM |
|------|----|
| Guilherme Fernandes Dellatin | RM365508 |
| Iana Alexandre Neri | RM360484 |
| Beatriz Rosa Carneiro Gomes | RM365967 |
| Cristine Scheibler | RM365433 |
| João Lucas Oliveira Hilario | RM366185 |

> Projeto educacional. Distribuído sob licença MIT.
