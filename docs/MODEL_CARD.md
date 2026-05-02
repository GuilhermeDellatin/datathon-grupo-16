# Model Card — LSTM Stock Price Predictor (PETR4.SA)

Referência: Mitchell, M. et al. (2019). Model Cards for Model Reporting. FAT*.

---

## Informações do Modelo

| Campo | Valor |
|-------|-------|
| **Nome** | lstm-petr4 |
| **Versão** | 1.0.0 |
| **Tipo** | Regressão (série temporal) |
| **Framework** | PyTorch 2.2+ |
| **Ticker** | PETR4.SA (Petrobras S.A.) |
| **Owner** | Grupo 16 — Datathon Fase 05, Pós Tech MLET/FIAP |
| **Última atualização do card** | 2026-04-29 |
| **Última run de treino registrada** | run em `metrics/train_metrics.json` |
| **Git SHA (último commit conhecido deste card)** | `d20a1fc` |
| **MLflow Experiment** | `datathon-petr4` |
| **MLflow Registered Model** | `lstm-petr4` |

> Os campos `Data de Treinamento`, `Git SHA exato da run` e métricas
> abaixo são atualizados automaticamente no MLflow a cada execução de
> `make train` (tags `git_sha`, `training_data_version`, `phase`,
> `ticker`, `framework`, `owner`, `risk_level`, `fairness_checked`,
> `model_type`, `model_name`, `model_version`).

## Descrição

Modelo LSTM (Long Short-Term Memory) para predição do preço de fechamento da ação PETR4.SA (Petrobras S.A.). O modelo utiliza dados históricos OHLCV e indicadores técnicos calculados como features de entrada, produzindo uma predição de preço para t+5 dias úteis à frente.

O desenvolvimento segue práticas de MLOps Nível 2 (Microsoft MLOps Maturity Model), com experiment tracking via MLflow, versionamento de dados via DVC, e monitoramento contínuo via Prometheus/Grafana.

## Intended Use

- **Uso pretendido**: Ferramenta educacional e de análise para estudo de predição de séries temporais financeiras no contexto do Datathon FIAP Fase 05.
- **Usuários pretendidos**: Estudantes, pesquisadores e analistas em contexto acadêmico.
- **Uso fora do escopo**: NÃO deve ser usado como base para decisões reais de investimento.

## Dados de Treinamento

- **Fonte**: Yahoo Finance via biblioteca `yfinance`
- **Período coletado**: 2018-01-01 a 2025-12-31 (configurável em
  `configs/model_config.yaml`)
- **Período efetivo após drop de NaN dos indicadores**: 2018-03-14 a
  2025-12-30
- **Volume**: 1.939 registros (dias úteis da B3) na última snapshot
- **Validação de schema**: `pandera` aplicado em
  `validate_raw_data()` (RAW_SCHEMA: Open/High/Low/Close > 0,
  Volume ≥ 0) e `validate_feature_data()` (FEATURE_SCHEMA: tipos +
  range RSI ∈ [0, 100]) — `src/data/feature_engineering.py`.
- **Features de entrada (14 colunas usadas pelo LSTM)**:
  - OHLCV: Close, Volume *(Open/High/Low presentes mas não usados como input)*
  - Médias móveis: `sma_20`, `sma_50`, `ema_12`, `ema_26`
  - Osciladores: `rsi_14`
  - Tendência: `macd`, `macd_signal`
  - Volatilidade: `bollinger_upper`, `bollinger_lower`
  - Volume: `volume_sma_20`
  - Retornos: `daily_return`, `log_return`
- **Target**: Preço de fechamento (Close) em t+5 dias úteis
- **Split**: 80% treino / 10% validação / 10% teste (temporal, sem shuffle)
- **Pré-processamento**: `MinMaxScaler` aplicado a todas as features.
  Parâmetros (`data_min_`, `data_max_`, `scale_`, `min_`) salvos no
  checkpoint `models/lstm_petr4_best.pt` para inferência reprodutível.
- **Janela**: 60 timesteps (`sequence_length`) → predição em t+5
  (`prediction_horizon`).

## Arquitetura

```
Input (batch, 60, N_features)
    │
    ▼
LSTM Layer 1 (128 hidden, dropout=0.2)
    │
    ▼
LSTM Layer 2 (128 hidden, dropout=0.2)
    │
    ▼
Linear (128 → 1)
    │
    ▼
Output (batch, 1) → preço normalizado
```

- **Tipo**: LSTM multi-camada (bidirecional opcional via config)
- **Input size**: 14 features
- **Hidden size**: 128 unidades
- **Num layers**: 2
- **Bidirectional**: false (default)
- **Dropout**: 0.2 (entre camadas LSTM e antes da Linear)
- **Parâmetros treináveis**: ~205.953 (input=14, hidden=128, layers=2)
- **Loss function**: `MSELoss`
- **Optimizer**: `Adam(lr=1e-3, weight_decay=1e-4)`
- **Scheduler**: `ReduceLROnPlateau(patience=10, factor=0.5)`
- **Early stopping**: patience=15 sobre `val_loss`
- **Gradient clipping**: `max_norm=1.0`
- **Batch size**: 32
- **Épocas máximas**: 100 (geralmente para antes via early stopping)

## Métricas de Performance

A última run registrada em `metrics/train_metrics.json` (escala
normalizada [0, 1] sobre o test set):

| Métrica | Valor |
|---------|-------|
| MAE | 0.0285 |
| RMSE | 0.0371 |
| MAPE (%) | 3.36% |
| Best val_loss | 0.00118 |

Métricas em **escala original (R$)** e **métrica de negócio do Datathon**
são logadas pelo treino atual a cada run (não estavam no snapshot
abaixo, pois foram introduzidas no plano #1 do `GAP_ANALYSIS.md`):

- `mae_price`, `rmse_price`, `mape_price` — MAE/RMSE/MAPE em escala R$.
- `target_sigma` — desvio-padrão observado da Close (R$).
- `sigma_threshold_0_5` — `0.5 * target_sigma`.
- `sigma_coverage_0_5` — fração de predições com erro absoluto ≤ 0.5σ
  (alvo: ≥ 0.70 — ver Quality Gate).

### Métrica de Negócio (Datathon Fase 05)

- **Definição**: ≥ 70 % das predições com erro absoluto dentro de 0.5
  desvios-padrão do preço observado (Close em R$).
- **Implementação**: `compute_sigma_coverage()` em
  `src/models/train.py` (escala original, ddof=0).
- **Quality gate**: `src/monitoring/quality_gates.py` falha o pipeline
  (exit 1) se `sigma_coverage_0_5 < 0.70`. Disponível também via
  `POST /evaluate_quality` e `make quality-gate`.
- **Slice de fairness**: `compute_fairness_by_volatility()` divide o
  test set em 3 quantis de volatilidade local e checa se a cobertura é
  estável (gap ≤ 10 p.p.) entre regimes calmos vs voláteis.

### Champion-Challenger

Implementado em `src/models/train.py::champion_challenger()` e chamado
no fim de `train_and_log()`. Promove a versão do modelo para
**Production** apenas se:

1. RMSE melhora ≥ 0.5 % em relação ao champion atual em Production.
2. `sigma_coverage_0_5 ≥ 0.70` (quality gate).

Caso contrário, transita para **Staging** com tag
`promotion_status="staging:<motivo>"`.

## Limitações

1. **Horizonte limitado**: Predições para mais de 5 dias úteis têm acurácia significativamente menor.
2. **Eventos extremos**: O modelo não captura eventos black swan (crises, guerras, decisões políticas súbitas).
3. **Viés temporal**: Performance pode degradar significativamente em regimes de mercado diferentes dos vistos no treinamento.
4. **Ação única**: Treinado apenas para PETR4.SA — não generalizável para outros ativos sem re-treinamento.
5. **Dados de mercado**: Depende de dados do Yahoo Finance, que podem ter atrasos ou imprecisões.
6. **Estacionariedade**: Séries financeiras são não-estacionárias; o modelo assume que padrões passados se repetem.

## Considerações Éticas

- **Risco financeiro**: Predições de mercado são inerentemente incertas. Nenhuma predição deve ser tratada como certeza.
- **Viés de dados**: Dados históricos refletem condições de mercado passadas que podem não se repetir.
- **Transparência**: Todas as métricas e limitações são documentadas abertamente neste Model Card.
- **Disclaimer obrigatório**: Toda predição gerada pelo sistema inclui aviso de que não constitui recomendação de investimento (enforced via `OutputGuardrail`).

## Fairness

- **Impacto diferencial**: Por se tratar de predição de ativo financeiro
  (não de indivíduos), não há risco de discriminação individual direta.
- **Risco indireto**: Investidores com menor sofisticação financeira
  podem confiar excessivamente no modelo (OWASP LLM09 — Overreliance).
- **Avaliação implementada**: `compute_fairness_by_volatility()` em
  `src/models/train.py` divide o test set em 3 quantis de volatilidade
  local (rolling std de retornos) e calcula `sigma_coverage_0_5` para
  cada regime. O modelo é considerado **fair** se o gap entre o melhor
  e o pior regime fica ≤ 10 p.p. (`fairness_tolerance=0.10`).
- **Tags MLflow**: cada run registra `fairness_checked=true|false`
  (resultado da avaliação) e `fairness_method=volatility_quantiles_3_tolerance_0.10`.
  Quando há amostra insuficiente para o slice, registra-se
  `fairness_skip_reason`.
- **Mitigação operacional**: disclaimers obrigatórios via
  `OutputGuardrail.validate_disclaimers()`, limitações explícitas neste
  card e prompt do agente bloqueando recomendações explícitas.
- **Pendência futura**: quando o produto suportar múltiplos tickers,
  acrescentar slice por ativo ao `compute_fairness_by_volatility`.

## Monitoramento

- **Drift detection**: PSI (Population Stability Index) calculado para
  features de entrada via Evidently
  (`src/monitoring/drift.py::calculate_psi`).
- **Thresholds** (configuráveis em `configs/monitoring_config.yaml`):
  - PSI < 0.1 → OK (sem ação).
  - PSI 0.1–0.2 → WARNING (alerta + investigação).
  - PSI > 0.2 → CRITICAL (trigger de retraining recomendado).
- **Quality gate**: `POST /evaluate_quality` valida
  `sigma_coverage_0_5 ≥ 0.70` antes de promover modelos.
- **Dashboard**: Grafana (`configs/grafana/dashboard.json`) com painéis
  de latência, drift, erros de predição e telemetria do agente.
- **Métricas Prometheus** (`src/monitoring/metrics.py`):
  `prediction_latency_seconds`, `prediction_requests_total`,
  `model_drift_psi`, `agent_requests_total`, `rag_retrieval_latency_seconds`.
- **Retraining**: pipeline DVC (`dvc repro`) com estágios `collect_data
  → feature_engineering → train → baseline → evaluate`.
- **Endpoints de saúde**: `GET /` (liveness), `GET /ready`,
  `GET /startup`, `GET /health` — todos em `src/serving/app.py`.

## Referências

- MITCHELL, M. et al. Model Cards for Model Reporting. In: FAT*, 2019.
- SCULLEY, D. et al. Hidden Technical Debt in Machine Learning Systems. In: NeurIPS, 2015.
- BRECK, E. et al. The ML Test Score: A Rubric for ML Production Readiness. In: IEEE BigData, 2017.
- HOCHREITER, S.; SCHMIDHUBER, J. Long Short-Term Memory. Neural Computation, v. 9, n. 8, 1997.
