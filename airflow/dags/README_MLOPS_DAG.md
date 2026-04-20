# DAG MLOps Completa - PETR4 LSTM Pipeline

## 📋 Visão Geral

DAG Airflow completa de **MLOps em Produção** para predição de preços de ações PETR4 usando LSTM. Implementa o **ciclo completo** de ML com validações, monitoramento e quality gates.

## 🏗️ Arquitetura da DAG

```
start
  ↓
collect_data (Yahoo Finance)
  ↓
validate_data (Pandera + Evidently)
  ↓
feature_engineering (Features técnicas)
  ↓
drift_detection (Data + Concept Drift)
  ↓
train_model (LSTM com Early Stopping)
  ↓
validate_model (Performance)
  ↓
evaluate_quality (RAGAS + RAG)
  ↓
ab_test (Champion-Challenger)
  ↓
quality_check (Quality Gates)
  ↓
register_model (MLflow Registry)
  ↓
prepare_deployment (Preparação)
  ↓
generate_report (Relatório Final)
  ↓
success_notification / failure_notification
  ↓
end
```

## 📊 Etapas Detalhadas

### 1. **Coleta de Dados** (`collect_data`)
- **O quê**: Coleta histórico de PETR4 do Yahoo Finance
- **Como**: `src.data.collector`
- **Saída**: Dataset bruto em `data/`
- **Frequência**: Diário às 2 AM (configurable)

### 2. **Validação de Dados** (`validate_data`)
- **Métricas**: Completeness, Uniqueness, Validity, Consistency
- **Limiar**: Mínimo 85% de qualidade
- **Ferramentas**: Pandera (schema validation)
- **Falha**: Interrompe pipeline se qualidade < 85%

### 3. **Feature Engineering** (`feature_engineering`)
- **Features técnicas**: SMA, EMA, RSI, MACD, etc.
- **Features estatísticas**: Returns, Volatility
- **Normalização**: MinMax Scaler
- **Saída**: Dataset preparado para treino

### 4. **Detecção de Drift** (`drift_detection`)
- **Data Drift**: Mudanças na distribuição dos dados
- **Concept Drift**: Mudanças no padrão de predição
- **Ferramentas**: Evidently
- **Limiar**: 15% de drift
- **Ação**: Log e continua (não bloqueia)

### 5. **Treinamento** (`train_model`)
- **Modelo**: LSTM (Long Short-Term Memory)
- **Features**: 6 features técnicas
- **Sequência**: 60 timesteps
- **MLflow**: Log automático de params e métricas
- **Early Stopping**: Previne overfitting

### 6. **Validação de Performance** (`validate_model`)
- **Métricas**: Accuracy, F1, Precision, Recall, MAE, RMSE
- **Limiares**:
  - Accuracy ≥ 60%
  - F1 Score ≥ 55%
- **Falha**: Interrompe se abaixo dos limiares

### 7. **Avaliação Qualitativa** (`evaluate_quality`)
- **RAGAS**: Framework de avaliação automática
- **RAG Pipeline**: Validação com documentos contextuais
- **Métricas**: Relevância, Coerência, Utilidade

### 8. **Teste A/B** (`ab_test`)
- **Tipo**: Champion-Challenger
- **Comparação**: Novo modelo vs. modelo em produção
- **Teste Estatístico**: Z-test com p-value
- **Significância**: α = 0.05
- **Recomendação**: Substituir ou manter

### 9. **Quality Gates** (`quality_check`)
- **Gates Verificados**:
  - ✓ Accuracy ≥ 60%
  - ✓ F1 Score ≥ 55%
  - ✓ Teste A/B significante
  - ✓ Sem data drift crítico
- **Bloqueio**: Impede deploy se falhar

### 10. **Registro de Modelo** (`register_model`)
- **Registro**: MLflow Model Registry
- **Versionamento**: Automático
- **Metadados**: Tags, descrição, parâmetros
- **Stages**: Staging → Production

### 11. **Preparação de Deploy** (`prepare_deployment`)
- **Artifacts**: Scaler, config files
- **Validação**: Modelo importável
- **Documentação**: Model card gerado

### 12. **Relatório Final** (`generate_report`)
- **Conteúdo**: Todas as métricas compiladas
- **Formato**: JSON estruturado
- **Localização**: `metrics/mlops_report_*.json`
- **Timestamp**: `YYYYMMDD_HHMMSS`

### 13. **Notificações** (`success_notification` / `failure_notification`)
- **Sucesso**: Pipeline concluído com êxito
- **Falha**: Motivo da falha registrado
- **Extensível**: Integração com Slack, Email, etc.

## ⚙️ Configurações

### Variáveis de Ambiente

```bash
export PROJECT_ROOT=/root/datathon-grupo-16
export API_IMAGE=datathon-lstm-stocks-api:latest
export OPENAI_API_KEY=sk-...
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_NAME=datathon-petr4
```

### Limiares de Qualidade (em `quality_gates.py`)

```python
MIN_ACCURACY = 0.60
MIN_F1_SCORE = 0.55
MAX_DRIFT_THRESHOLD = 0.15
MIN_DATA_QUALITY = 0.85
```

## 📦 Dependências de Módulos

| Módulo | Arquivo | Função |
|--------|---------|--------|
| Coleta | `src/data/collector.py` | Yahoo Finance |
| Features | `src/data/feature_engineering.py` | Feature eng |
| Treino | `src/models/train.py` | LSTM training |
| Avaliação | `evaluation/ragas_eval.py` | RAGAS eval |
| Drift | `src/monitoring/drift_detector.py` | Drift detection |
| A/B Test | `src/monitoring/ab_test.py` | Champion-Challenger |
| Registry | `src/monitoring/model_registry.py` | MLflow |

## 🚀 Como Executar

### 1. Trigger Manual no Airflow UI
```
http://localhost:8080
DAG: petr4_mlops_pipeline
Clique em "Trigger DAG"
```

### 2. Via CLI Airflow
```bash
airflow dags trigger petr4_mlops_pipeline
```

### 3. Via API Airflow
```bash
curl -X POST \
  http://localhost:8080/api/v1/dags/petr4_mlops_pipeline/dagRuns \
  -H 'Content-Type: application/json' \
  -d '{"conf":{}}'
```

### 4. Scheduled (Automático)
- **Schedule**: Diário às 2:00 AM UTC
- **Configuração**: `schedule_interval="0 2 * * *"`

## 📊 Monitoramento e Logs

### Acessar Logs
```bash
# UI do Airflow
http://localhost:8080/dags/petr4_mlops_pipeline

# Logs via CLI
airflow tasks logs petr4_mlops_pipeline collect_data [execution_date]

# Logs Docker
docker logs [container_id]
```

### MLflow Tracking
```bash
# UI MLflow
http://localhost:5000

# Experiment: datathon-petr4
# Ver métricas, parâmetros, artifacts
```

### Relatórios Gerados
```
metrics/mlops_report_20260101_020500.json
metrics/drift_reports/drift_report_20260101_020500.html
```

## 🔧 Scripts de Suporte

### Quality Gates
```python
from src.monitoring.quality_gates import QualityGateValidator, create_default_gates

gates = create_default_gates()
validator = QualityGateValidator(gates)
results = validator.validate_all(model_metrics)
```

### Detecção de Drift
```python
from src.monitoring.drift_detector import DriftDetector

detector = DriftDetector()
data_drift = detector.calculate_data_drift(reference_data, current_data)
concept_drift = detector.calculate_concept_drift(y_true_hist, y_pred_hist, ...)
```

### A/B Test
```python
from src.monitoring.ab_test import ABTestAnalyzer, ModelMetrics

analyzer = ABTestAnalyzer()
results = analyzer.compare_models(challenger, champion)
```

### MLflow Registry
```python
from src.monitoring.model_registry import MLflowModelRegistry

registry = MLflowModelRegistry(tracking_uri, experiment_name)
registry.register_model(model_uri, "petr4-lstm")
registry.promote_model("petr4-lstm", version=2, stage="Production")
```

## 📈 Métricas Rastreadas

### Modelo
- Accuracy, F1, Precision, Recall
- MAE, RMSE, MSE
- Training time

### Dados
- Completeness, Uniqueness, Validity
- Data drift score, Concept drift score

### A/B Test
- Accuracy improvement %
- F1 improvement %
- P-value
- Recommendation

## ⚠️ Falhas e Tratamento

| Cenário | Ação |
|---------|------|
| Dados abaixo de qualidade | ❌ Falha, não treina |
| Model performance baixa | ❌ Falha, não registra |
| Drift significativo | ⚠️ Log, continua com cuidado |
| A/B test não significante | ⚠️ Log, pode continuar |
| Registro falha | ❌ Falha, manual review |

## 🔐 Segurança e Compliance

- ✓ LGPD: Dados anonimizados
- ✓ OWASP: Validação de entrada
- ✓ Audit: Todos os runs registrados
- ✓ Versioning: Rastreamento completo
- ✓ Segregation: Staging vs. Production

## 📝 Extensões Possíveis

1. **Notificações**: Slack, Email, PagerDuty
2. **Rollback automático**: Se modelo falha em produção
3. **A/A Testing**: Validar estabilidade
4. **Canary Deployment**: Deploy gradual
5. **Multi-Armed Bandit**: Exploração-Exploração
6. **Feature Store**: Gerenciamento centralizado
7. **Data Catalog**: Lineage tracking
8. **Custom Metrics**: Métricas de negócio

## 📚 Documentação Relacionada

- [MODEL_CARD.md](../docs/MODEL_CARD.md) - Card do modelo
- [SYSTEM_CARD.md](../docs/SYSTEM_CARD.md) - Card do sistema
- [OWASP_MAPPING.md](../docs/OWASP_MAPPING.md) - Segurança
- [LGPD_PLAN.md](../docs/LGPD_PLAN.md) - Compliance

## 🆘 Troubleshooting

### Problema: DAG não inicia
**Solução**: Verificar variáveis de ambiente e permissões Docker

### Problema: Falha na coleta de dados
**Solução**: Verificar conexão com Yahoo Finance, VPN

### Problema: Modelo com performance baixa
**Solução**: Verificar dados, revisar features, ajustar hiperparâmetros

### Problema: Drift detectado
**Solução**: Investigar mercado, atualizar modelo, aumentar frequência de retraining

## 📞 Contato

Grupo 16 - Datathon
