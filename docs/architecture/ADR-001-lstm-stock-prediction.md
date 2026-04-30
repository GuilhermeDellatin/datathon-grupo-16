# ADR-001 — LSTM em PyTorch para Previsão de Preço PETR4.SA

- **Status**: Aceito
- **Data**: 2026-04-27
- **Decisores**: Grupo 16 (Datathon Fase 05 — Pós Tech MLET/FIAP)

## Contexto

O projeto precisa prever o preço de fechamento (Close) da PETR4.SA com
tolerância de **0.5 desvios-padrão** sobre o preço observado, atendendo a
métrica de negócio de **≥ 70 %** de cobertura. Restrições adicionais:

1. A Fase 4 da Pós Tech exige PyTorch como framework de deep learning.
2. Há séries temporais com dependência sequencial e correlação parcial entre
   indicadores técnicos (RSI, MACD, Bollinger, médias móveis).
3. O treinamento precisa ser reprodutível (MLflow tracking, split temporal,
   tags obrigatórias) e comparável contra baselines tabulares.
4. Inferência precisa ser barata e isolada do treinamento (módulos distintos).

## Alternativas Avaliadas

| Modelo | Prós | Contras | Decisão |
|---|---|---|---|
| **LSTM** (PyTorch) | Capta dependência sequencial; bom em séries financeiras; alinhado à rubrica | Requer mais dados que ARIMA; maior custo de tuning | ✅ Adotado |
| ARIMA / SARIMA | Bem estabelecido em séries univariadas | Frágil com não-estacionariedade; não usa indicadores técnicos | ❌ |
| Prophet | Boa para tendência+sazonalidade | Pouca flexibilidade para múltiplos features técnicos | ❌ |
| Transformer (encoder-only) | Capta dependências longas | Overkill para séries diárias; precisa de mais dados; treinamento caro | ❌ |
| XGBoost / LightGBM | Forte em tabular; rápido | Perde estrutura temporal sem flatten | ⚠️ Coberto pelo MLP baseline (ADR-001 anexa baselines) |
| Ridge linear (baseline) | Simples, regularização L2 | Linearidade demais para preços | ⚠️ Mantido só como baseline (`src/models/baseline.py`) |
| MLP (PyTorch) | Não-linear, flexível | Não capta ordem sem janela explícita | ⚠️ Mantido só como baseline |

## Decisão

Adotamos **LSTM multi-camada em PyTorch** como modelo principal de produção,
com baselines tabulares (Ridge + MLP) executados pelo mesmo pipeline para
comparação justa.

### Arquitetura

- `nn.LSTM` com `hidden_size=128`, `num_layers=2`, `dropout=0.2`,
  `bidirectional=False`, `batch_first=True`.
- Camada `Linear(hidden_size, 1)` no último timestep.
- Dropout adicional antes da saída.
- Definida em `src/models/lstm_model.py::LSTMPredictor`.

### Pipeline de dados

- Coleta via `yfinance` (`src/data/collector.py`).
- 14 indicadores técnicos calculados em `src/data/feature_engineering.py`
  (SMA, EMA, RSI, MACD, Bollinger, retornos, etc.).
- Schema Pandera declarado (`RAW_SCHEMA`, `FEATURE_SCHEMA`).
- `MinMaxScaler` aplicado a todas as features; parâmetros salvos no
  checkpoint para inferência reprodutível.
- Sequências de tamanho `60` com horizonte `5` via `create_sequences()`.
- **Split temporal estrito** (`train=80% / val=10% / test=10%`,
  `shuffle=False`).

### Treinamento

- `MSELoss`, `Adam` com `weight_decay=1e-4`, `lr=1e-3`.
- `ReduceLROnPlateau` (patience 10, factor 0.5).
- **Early stopping** (patience 15) e **gradient clipping** (`max_norm=1.0`).
- Logs MLflow obrigatórios (`model_name`, `model_version`, `git_sha`,
  `training_data_version`, `framework`, `phase`, `ticker`, `risk_level`,
  `fairness_checked`, `model_type`, `owner`).

### Métricas

- Padrão de regressão: MAE, RMSE, MAPE.
- **Métrica de negócio (Datathon)**: `sigma_coverage_0_5` em escala
  original de preço (R$). Alvo ≥ 0.70.
- Quality gate via `src/monitoring/quality_gates.py` (ADR-004).

### Checkpoint

- `models/lstm_petr4_best.pt` contém: `model_state_dict`, `model_config`,
  `feature_columns`, `scaler_params`, `sequence_length`,
  `prediction_horizon`. Permite recriar o predictor sem o YAML.

### Comparação com baselines

- `src/models/baseline.py` treina **Ridge** e **MLP em PyTorch** com a
  mesma pipeline (achatando a sequência), reportando as mesmas métricas em
  escala original. `_build_comparison()` consolida o vencedor de cada
  métrica em `metrics/baseline_metrics.json`.

## Consequências

### Positivas

- Aderente ao requisito de PyTorch + LSTM da rubrica.
- Captura dependências temporais sem feature engineering manual de lags.
- Pipeline reprodutível (DVC + MLflow + scaler no checkpoint).
- Comparações Ridge/MLP/LSTM no mesmo pipeline reduzem viés metodológico.

### Negativas / Trade-offs

- LSTM precisa de mais dados que ARIMA — mitigado pelo período 2018→atual.
- Tuning de hiperparâmetros é custoso — mitigado por defaults razoáveis e
  early stopping.
- Não estende facilmente a múltiplos tickers sem retreino — `POST /train`
  agora aceita `tickers` para suportar isso (plano #7).

### Mitigações

1. Champion-challenger via MLflow Registry (`champion_challenger()` em
   `src/models/train.py`) deve ser usado em produção para evitar regressão
   por retreino.
2. PSI/drift com Evidently (`src/monitoring/drift.py`) acompanha mudança
   de regime que pode exigir retreino.
3. Quality gate `sigma_coverage_0_5 >= 0.70` bloqueia promoção de modelos
   piores (ADR-004).

## Referências

- `src/models/lstm_model.py`, `src/models/train.py`, `src/models/predict.py`
- `src/models/baseline.py`, `tests/test_baseline.py`
- `configs/model_config.yaml`
- Hochreiter & Schmidhuber (1997) — Long Short-Term Memory.
- Sculley et al. (2015) — Hidden Technical Debt in ML Systems. NeurIPS.
- Microsoft (2026) — MLOps Maturity Model.
