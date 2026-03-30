# Model Card — LSTM Stock Price Predictor (PETR4.SA)

Referência: Mitchell, M. et al. (2019). Model Cards for Model Reporting. FAT*.

---

## Informações do Modelo

| Campo | Valor |
|-------|-------|
| **Nome** | LSTM-PETR4 |
| **Versão** | 1.0.0 |
| **Tipo** | Regressão (série temporal) |
| **Framework** | PyTorch 2.2+ |
| **Ticker** | PETR4.SA (Petrobras S.A.) |
| **Owner** | Grupo 16 — Datathon Fase 05, Pós Tech MLET/FIAP |
| **Data de Treinamento** | [DATA] |
| **Git SHA** | [SHA] |
| **MLflow Experiment** | lstm-petr4-training |

## Descrição

Modelo LSTM (Long Short-Term Memory) para predição do preço de fechamento da ação PETR4.SA (Petrobras S.A.). O modelo utiliza dados históricos OHLCV e indicadores técnicos calculados como features de entrada, produzindo uma predição de preço para t+5 dias úteis à frente.

O desenvolvimento segue práticas de MLOps Nível 2 (Microsoft MLOps Maturity Model), com experiment tracking via MLflow, versionamento de dados via DVC, e monitoramento contínuo via Prometheus/Grafana.

## Intended Use

- **Uso pretendido**: Ferramenta educacional e de análise para estudo de predição de séries temporais financeiras no contexto do Datathon FIAP Fase 05.
- **Usuários pretendidos**: Estudantes, pesquisadores e analistas em contexto acadêmico.
- **Uso fora do escopo**: NÃO deve ser usado como base para decisões reais de investimento.

## Dados de Treinamento

- **Fonte**: Yahoo Finance via biblioteca yfinance
- **Período**: 2018-01-01 a 2025-12-31
- **Volume**: ~[N] registros (dias úteis da B3)
- **Features de entrada**:
  - OHLCV: Open, High, Low, Close, Volume
  - Médias móveis: SMA(20), SMA(50), EMA(12), EMA(26)
  - Osciladores: RSI(14)
  - Tendência: MACD, MACD Signal
  - Volatilidade: Bollinger Upper, Bollinger Lower
  - Volume: Volume SMA(20)
  - Retornos: Daily Return, Log Return
- **Target**: Preço de fechamento (Close) em t+5
- **Split**: 80% treino / 10% validação / 10% teste (temporal, sem shuffle)
- **Pré-processamento**: MinMaxScaler (parâmetros salvos no checkpoint), janela deslizante de 60 timesteps

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

- **Tipo**: LSTM multi-layer bidirecional opcional
- **Hidden size**: 128 unidades
- **Num layers**: 2
- **Dropout**: 0.2
- **Parâmetros treináveis**: ~[N]
- **Loss function**: MSELoss
- **Optimizer**: Adam (lr=0.001)
- **Early stopping**: Patience 10 épocas, monitorando val_loss

## Métricas de Performance

| Métrica | Treino | Validação | Teste |
|---------|--------|-----------|-------|
| MAE | [valor] | [valor] | [valor] |
| RMSE | [valor] | [valor] | [valor] |
| MAPE (%) | [valor] | [valor] | [valor] |

> Métricas atualizadas automaticamente no MLflow a cada run de treinamento.

### Champion-Challenger

- Modelo promovido a "Production" no MLflow Model Registry quando RMSE melhora >= 0.5% em relação ao champion atual.

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

- **Impacto diferencial**: Por se tratar de predição de ativo financeiro (não de indivíduos), não há risco de discriminação individual direta.
- **Risco indireto**: Investidores com menor sofisticação financeira podem confiar excessivamente no modelo.
- **Mitigação**: Disclaimers obrigatórios, limitações explícitas, e guardrails de output que reforçam avisos de risco.
- **Auditoria**: Tag `fairness_checked` registrada em todas as runs do MLflow.

## Monitoramento

- **Drift detection**: PSI (Population Stability Index) calculado para features de entrada via Evidently
- **Thresholds**:
  - PSI < 0.1 → OK (sem ação)
  - PSI 0.1–0.2 → WARNING (alerta + investigação)
  - PSI > 0.2 → CRITICAL (trigger de retraining automático)
- **Dashboard**: Grafana com painéis de latência, drift, erro de predição e uso do agente
- **Métricas Prometheus**: Latência de predição, taxa de requests, PSI por feature
- **Retraining**: Automático via pipeline DVC quando drift crítico detectado

## Referências

- MITCHELL, M. et al. Model Cards for Model Reporting. In: FAT*, 2019.
- SCULLEY, D. et al. Hidden Technical Debt in Machine Learning Systems. In: NeurIPS, 2015.
- BRECK, E. et al. The ML Test Score: A Rubric for ML Production Readiness. In: IEEE BigData, 2017.
- HOCHREITER, S.; SCHMIDHUBER, J. Long Short-Term Memory. Neural Computation, v. 9, n. 8, 1997.
