# ADR-004 — Estratégia de Avaliação Multi-Camada

- **Status**: Aceito
- **Data**: 2026-04-27
- **Decisores**: Grupo 16 (Datathon Fase 05 — Pós Tech MLET/FIAP)

## Contexto

O sistema combina dois subsistemas com características de avaliação
distintas:

1. **LSTM (regressão sobre série temporal)** — métricas numéricas
   tradicionais + métrica de negócio.
2. **Pipeline RAG + Agente ReAct (NLP)** — não há "valor verdadeiro" único;
   é preciso medir fidelidade ao contexto, relevância da resposta,
   utilidade para o investidor e presença de disclaimers.

A rubrica do Datathon exige avaliação automatizada e reprodutível, com
quality gate antes de promover modelos.

## Decisão

Adotamos uma **estratégia em quatro camadas**, todas com saída versionada
em `metrics/*.json` e logging no MLflow:

### 1. Métricas do LSTM (regressão)

| Métrica | Implementação | Onde |
|---|---|---|
| MAE, RMSE, MAPE | `compute_metrics()` em escala normalizada | `src/models/train.py` |
| `target_sigma`, `sigma_threshold_0_5`, `sigma_coverage_0_5` | `compute_sigma_coverage()` em escala original (R$) | `src/models/train.py` |

A **métrica de negócio** principal é `sigma_coverage_0_5` (≥ 70 % das
predições com erro absoluto ≤ 0.5 σ do preço observado). Foi
explicitamente projetada para casar com o critério da rubrica.

### 2. Quality Gate

`src/monitoring/quality_gates.py` lê
`metrics/train_metrics.json` e bloqueia (exit 1) se
`sigma_coverage_0_5 < threshold` (default 0.70 via
`configs/monitoring_config.yaml`).

Pode ser disparado de 3 formas:

- CLI: `python -m src.monitoring.quality_gates` (`make quality-gate`).
- Endpoint: `POST /evaluate_quality` (FastAPI, plano #5) com override
  opcional de `metrics_path` e `threshold` no body.
- CI / DVC: como step adicional após `make train`.

### 3. Avaliação do Pipeline RAG (RAGAS)

`evaluation/ragas_eval.py` calcula sobre o golden set de **25 pares**
(`data/golden_set/golden_set.json`, mínimo 20 exigido pela rubrica) as **4
métricas obrigatórias**:

- `faithfulness` — resposta fiel aos contextos recuperados.
- `answer_relevancy` — resposta relevante para a pergunta.
- `context_precision` — contextos recuperados são precisos.
- `context_recall` — contextos cobrem a resposta esperada.

Saída em `metrics/ragas_metrics.json`, logada no experimento MLflow
`datathon-petr4-evaluation`.

### 4. LLM-as-Judge (avaliação de respostas do agente)

`evaluation/llm_judge.py` usa o mesmo LLM (`gpt-4o-mini`) como avaliador
com **5 critérios** (extensão dos 3 mínimos da rubrica), nota 1–5:

1. **Correção Técnica** — factualmente correto?
2. **Relevância** — aborda diretamente a pergunta?
3. **Clareza** — bem estruturado e legível?
4. **Utilidade para Investidor** — auxilia tomada de decisão? *(extensão
   específica do domínio financeiro)*
5. **Disclaimers de Risco** — quando aplicável, inclui aviso de não-
   recomendação? *(extensão específica de governança financeira)*

A pontuação média final (`avg_overall`) é derivada determinísticamente
como média dos 5 critérios por item, ignorando o `overall_score`
auto-relatado pelo juiz (que pode vir fora da escala). Saída em
`metrics/llm_judge_metrics.json`.

### 5. Drift detection

`src/monitoring/drift.py` calcula PSI por feature comparando uma janela de
referência (~252 dias úteis) contra a janela corrente (30 dias). Lê
`configs/monitoring_config.yaml`:

- `psi_warning_threshold = 0.1` → alerta.
- `psi_retrain_threshold = 0.2` → trigger de retreinamento.

Métricas expostas via Prometheus (`model_drift_psi` gauge) e dashboard
Grafana.

## Consequências

### Positivas

- **Múltiplas dimensões cobertas**: precisão numérica, fidelidade RAG,
  qualidade conversacional, drift e gate de negócio.
- **Reprodutível**: cada avaliação grava em JSON + MLflow.
- **Bloqueia regressões**: quality gate impede promoção de modelo abaixo
  do alvo de 0.5 σ.
- **Transparente**: critérios e justificativas do juiz são gravados em
  `metrics/llm_judge_metrics.json`.

### Negativas / Trade-offs

- RAGAS e LLM-as-Judge dependem de OpenAI API (custo + variabilidade) —
  herdado do ADR-002.
- LLM-as-Judge é não-determinístico — mitigado com `temperature=0` e
  pontuação derivada determinísticamente.
- 5 critérios em vez dos 3 mínimos da rubrica — decisão deliberada para
  capturar Utilidade ao Investidor e Disclaimers, ambos críticos no
  domínio financeiro/regulatório. Justificada formalmente neste ADR.

### Mitigações

1. Avaliações são **opcionais em CI** (rodam fora do pipeline `quality`
   por causa de custo OpenAI). Recomendação: rodar manualmente antes de
   tags de release.
2. Golden set é versionado em `data/golden_set/golden_set.json` para
   permitir comparação histórica.
3. Quality gate cobre o piso de qualidade do LSTM independente das
   avaliações de NLP.
4. Métricas em escala original (R$) garantem que o `sigma_coverage_0_5`
   reflete volatilidade real do ativo, não a escala 0–1 do MinMax.

## Alternativas Rejeitadas

- **Apenas MAE/RMSE/MAPE**: rejeitado por não capturar a métrica de
  negócio do datathon (≥ 70 % dentro de 0.5 σ).
- **Avaliação manual humana**: rejeitada por não escalar com o golden set
  e por introduzir viés inter-anotador não controlado.
- **Apenas RAGAS sem LLM-as-judge**: rejeitado porque RAGAS mede o
  pipeline RAG, não a *resposta final do agente* (que combina RAG +
  raciocínio + tools).
- **3 critérios estritos no LLM-judge**: tecnicamente atende a rubrica,
  mas perde a lente regulatória (Disclaimer) e de negócio (Utilidade).

## Referências

- Es, S. et al. (2024). _RAGAS: Automated Evaluation of Retrieval
  Augmented Generation_.
- Mitchell, M. et al. (2019). _Model Cards for Model Reporting_. FAT*.
- Breck, E. et al. (2017). _The ML Test Score_. IEEE BigData.
- `src/models/train.py::compute_metrics`, `compute_sigma_coverage`.
- `src/monitoring/quality_gates.py`,
  `tests/test_quality_gate.py`.
- `evaluation/ragas_eval.py`, `evaluation/llm_judge.py`.
- `src/monitoring/drift.py`, `configs/monitoring_config.yaml`.
- `data/golden_set/golden_set.json` (25 pares).
- ADR-001 — LSTM PyTorch.
- ADR-003 — Agente ReAct.
