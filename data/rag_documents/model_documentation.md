# Documentação do Modelo LSTM — PETR4.SA

## Visão Geral

Modelo de Deep Learning baseado em LSTM (Long Short-Term Memory) para predição do preço de fechamento da Petrobras (PETR4.SA). Desenvolvido como parte do Datathon Fase 05 da Pós Tech MLET/FIAP.

## Arquitetura

- **Tipo**: LSTM multi-camada com dropout
- **Framework**: PyTorch
- **Input**: Sequências de 60 dias (trading days) com 14 features
- **Output**: Preço de fechamento previsto para horizonte de 5 dias
- **Camadas LSTM**: 2 camadas empilhadas
- **Hidden size**: 128 neurônios
- **Dropout**: 0.2 (entre camadas LSTM e antes da camada linear)
- **Camada de saída**: Linear (hidden_size → 1)

## Features de Entrada

O modelo utiliza 14 features derivadas dos dados OHLCV:

1. **Close** — Preço de fechamento
2. **Volume** — Volume de negociação
3. **sma_20** — Média Móvel Simples de 20 dias
4. **sma_50** — Média Móvel Simples de 50 dias
5. **ema_12** — Média Móvel Exponencial de 12 dias
6. **ema_26** — Média Móvel Exponencial de 26 dias
7. **rsi_14** — Índice de Força Relativa (14 dias)
8. **macd** — Moving Average Convergence Divergence
9. **macd_signal** — Linha de sinal do MACD
10. **bollinger_upper** — Banda superior de Bollinger
11. **bollinger_lower** — Banda inferior de Bollinger
12. **volume_sma_20** — Média móvel de volume (20 dias)
13. **daily_return** — Retorno diário percentual
14. **log_return** — Log-retorno diário

## Pré-processamento

- **Scaling**: MinMaxScaler aplicado a todas as features (range 0-1).
- **Sequências**: Janela deslizante de 60 timesteps.
- **Split temporal**: 80% treino / 10% validação / 10% teste — sem shuffle.
- **NaN handling**: Primeiras ~50 linhas removidas (warmup dos indicadores técnicos).

## Treinamento

- **Otimizador**: Adam (lr=0.001, weight_decay=0.0001)
- **Loss**: MSE (Mean Squared Error)
- **Scheduler**: ReduceLROnPlateau (patience=10, factor=0.5)
- **Early Stopping**: Patience de 15 épocas
- **Gradient Clipping**: Norma máxima = 1.0
- **Batch size**: 32
- **Épocas máximas**: 100

## Métricas

As seguintes métricas são logadas no MLflow para cada run de treinamento:
- **MAE** (Mean Absolute Error) — erro absoluto médio
- **RMSE** (Root Mean Squared Error) — penaliza erros maiores
- **MAPE** (Mean Absolute Percentage Error) — erro percentual

## Limitações

1. **Não é recomendação de investimento**: O modelo é um exercício acadêmico.
2. **Dados históricos não garantem performance futura**: Mercados são influenciados por eventos imprevisíveis.
3. **Viés temporal**: Modelo treinado com dados de um período específico; performance pode degradar em regimes de mercado diferentes.
4. **Sem incorporação de sentimento**: Não utiliza notícias, redes sociais ou análise de sentimento.
5. **Single-stock**: Treinado apenas para PETR4.SA, não generalizável para outras ações.

## Governança

- Versionamento via MLflow Model Registry
- Tags obrigatórias em todas as runs (owner, risk_level, git_sha, etc.)
- Champion-challenger para promoção de modelos
- Drift detection via PSI (Population Stability Index)
