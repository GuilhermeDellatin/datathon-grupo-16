# Guia de Troubleshooting - DAG MLOps PETR4

## Problema: DAG não aparece no Airflow

### Possíveis Causas:
1. DAG não foi carregada pelo Airflow
2. Erro de sintaxe no arquivo DAG
3. Caminho incorreto do DAG

### Solução:
```bash
# 1. Verificar se o arquivo existe
ls -la airflow/dags/petr4_ml_pipeline.py

# 2. Validar sintaxe Python
python -m py_compile airflow/dags/petr4_ml_pipeline.py

# 3. Verificar logs do Airflow
docker logs -f airflow-scheduler

# 4. Forçar recarregamento
docker restart airflow-scheduler
docker restart airflow-webserver

# 5. Acessar UI
http://localhost:8080
```

---

## Problema: DAG dispara mas falha na tarefa `collect_data`

### Possíveis Causas:
1. Conexão com Yahoo Finance falhou
2. Variável `PROJECT_ROOT` não configurada
3. Imagem Docker não existe

### Solução:
```bash
# 1. Verificar variáveis de ambiente
echo $PROJECT_ROOT
echo $API_IMAGE
echo $MLFLOW_TRACKING_URI

# 2. Validar imagem Docker
docker images | grep datathon

# 3. Reconectar rede Docker
docker network ls
docker inspect datathon_default

# 4. Testar coleta manualmente
docker run -it \
  -v $PROJECT_ROOT:/app \
  --network host \
  $API_IMAGE \
  python -m src.data.collector

# 5. Verificar logs do container
docker logs [container_id]
```

---

## Problema: Falha na validação de dados (limiar mínimo não atingido)

### Possíveis Causas:
1. Dados de Yahoo Finance incompletos
2. Muitos valores faltando (NaN)
3. Dados contêm valores inválidos

### Solução:
```bash
# 1. Verificar dados coletados
python -c "
import pandas as pd
df = pd.read_csv('data/petr4_raw.csv')
print('Shape:', df.shape)
print('Missing:', df.isnull().sum())
print('Data types:', df.dtypes)
"

# 2. Aumentar período de coleta
# Editar airflow/dags/petr4_ml_pipeline.py
# Mudança em configs/mlops_config.yaml:
#   data_collection:
#     period: "10y"  # Aumentar de 5y

# 3. Reduzir limiar de qualidade temporariamente
#   data_validation:
#     overall_min_quality: 0.75  # De 0.85

# 4. Verificar dados manualmente
python scripts/data_quality_check.py
```

---

## Problema: Modelo tem performance baixa (F1 < 55%)

### Possíveis Causas:
1. Features não adequadas
2. Parâmetros LSTM ruins
3. Dados contêm muito ruído
4. Classe desbalanceada

### Solução:
```bash
# 1. Investigar features
python -c "
from src.data.feature_engineering import load_and_process
X, y = load_and_process()
print('Feature correlations:')
print(X.corr()['target'].sort_values())
"

# 2. Ajustar hiperparâmetros (em configs/mlops_config.yaml)
#   training:
#     optimization:
#       learning_rate: 0.0001  # Reduzir
#       batch_size: 16         # Reduzir
#       epochs: 200            # Aumentar

# 3. Aumentar early_stopping_patience
#   training:
#     optimization:
#       early_stopping_patience: 20

# 4. Verificar desbalanceamento
python -c "
import numpy as np
y = np.load('data/y_train.npy')
unique, counts = np.unique(y, return_counts=True)
print('Class distribution:', dict(zip(unique, counts)))
"

# 5. Adicionar class weighting
# Editar src/models/train.py para usar class_weight
```

---

## Problema: Drift detectado (acima de 15%)

### Possíveis Causas:
1. Mercado mudou (exemplo: crise, rally, volatilidade alta)
2. Mudança estrutural nos dados
3. Possível data leakage

### Solução:
```bash
# 1. Investigar dados atuais
python -c "
from src.monitoring.drift_detector import DriftDetector
import pandas as pd

detector = DriftDetector()
current_data = pd.read_csv('data/petr4_current.csv')
reference_data = pd.read_csv('data/petr4_reference.csv')

drift = detector.calculate_data_drift(reference_data, current_data)
print('Drift Score:', drift['drift_score'])

# Gerar relatório
report_path = detector.generate_drift_report(reference_data, current_data)
print('Report:', report_path)
"

# 2. Aumentar threshold temporariamente
# Em configs/mlops_config.yaml:
#   drift_detection:
#     data_drift:
#       threshold: 0.25  # De 0.15

# 3. Verificar se é mudança legítima do mercado
# Análise manual dos últimos dias

# 4. Retraining mais frequente
# Em airflow/dags/petr4_ml_pipeline.py:
#   schedule_interval="0 1 * * *"  # Diário às 1 AM (ao invés de 2 AM)

# 5. Criar novo modelo baseline
python -c "
from src.models.train import train_lstm
model = train_lstm(retrain=True)
print('Novo baseline criado')
"
```

---

## Problema: Teste A/B diz que challenger não é significante

### Possíveis Causas:
1. Amostra muito pequena (< 100)
2. Melhoria é muito pequena (< 2%)
3. Variância alta nos dados

### Solução:
```bash
# 1. Aumentar tamanho da amostra
# Em configs/mlops_config.yaml:
#   ab_test:
#     min_sample_size: 200

# 2. Reduzir threshold de melhoria
#   ab_test:
#     min_improvement_pct: 0.01  # De 0.02

# 3. Verificar se models.json tem dados suficientes
python -c "
import json
with open('metrics/ab_test_results.json') as f:
    data = json.load(f)
    print('Sample size:', data.get('sample_size'))
    print('Accuracy improvement:', data.get('accuracy_improvement_pct'))
"

# 4. Esperar mais dados se recém-lançado
# Executar A/B test novamente após 1 semana
```

---

## Problema: Quality Gate falhou, deploy bloqueado

### Possíveis Causas:
1. Modelo falhou em um ou mais gates
2. A/B test não passou
3. Drift crítico

### Solução:
```bash
# 1. Verificar qual gate falhou
python airflow/utils/dag_manager.py --report

# 2. Investigar gate específico
# Logs em metrics/mlops_report_*.json

# 3. Opções:
#    a. Ajustar features e retrainer
#    b. Aumentar threshold (com cuidado)
#    c. Debug manual

# 4. Exemplo: Aumentar threshold de F1
# Em src/monitoring/quality_gates.py:
#   QualityGate(
#       name="F1 Score",
#       metric="f1_score",
#       threshold=0.50,  # De 0.55
#       comparison="greater_than",
#   )

# 5. Reexecutar pipeline após ajustes
airflow dags trigger petr4_mlops_pipeline
```

---

## Problema: MLflow não encontra modelo registrado

### Possíveis Causas:
1. MLflow não está rodando
2. Modelo não foi registrado corretamente
3. Projeto não está em PYTHONPATH

### Solução:
```bash
# 1. Verificar se MLflow está online
curl http://localhost:5000

# 2. Verificar registros no MLflow
python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
client = mlflow.tracking.MlflowClient()
registered_models = client.search_registered_models()
for model in registered_models:
    print(f'{model.name} - {len(model.latest_versions)} versions')
"

# 3. Adicionar PYTHONPATH
export PYTHONPATH=\"$PYTHONPATH:$PROJECT_ROOT\"

# 4. Registrar modelo manualmente
python -c "
from src.monitoring.model_registry import MLflowModelRegistry
registry = MLflowModelRegistry('http://localhost:5000', 'datathon-petr4')
result = registry.register_model(
    model_uri='runs:/abc123/model',
    model_name='petr4-lstm'
)
print(result)
"

# 5. Verificar logs do MLflow
docker logs -f mlflow
```

---

## Problema: Memoria insuficiente durante treinamento

### Possíveis Causas:
1. Batch size muito grande
2. Sequence length muito longo
3. Hardware limitado

### Solução:
```bash
# 1. Reduzir batch size
# Em configs/mlops_config.yaml:
#   training:
#     optimization:
#       batch_size: 16  # De 32

# 2. Reduzir sequence length
#   feature_engineering:
#     sequence_length: 30  # De 60

# 3. Reduzir dimensão LSTM
#   training:
#     model:
#       hidden_size: 32  # De 50
#       num_layers: 1    # De 2

# 4. Usar GPU se disponível
#   training:
#     device: "cuda"

# 5. Monitorar memória
watch -n 1 'nvidia-smi'  # Se tiver GPU
# ou
watch -n 1 'free -h'     # CPU memory
```

---

## Problema: Docker container sai com erro

### Possíveis Causas:
1. Out of memory
2. Permissão de arquivo
3. Erro em script Python

### Solução:
```bash
# 1. Ver logs detalhados
docker logs [container_id] | tail -100

# 2. Rodar container interativamente
docker run -it \
  -v $PROJECT_ROOT:/app \
  --network host \
  $API_IMAGE \
  /bin/bash

# 3. Verificar permissões
ls -la $PROJECT_ROOT/data/
chmod -R 755 $PROJECT_ROOT/data/

# 4. Aumentar limite de memória do container
# Em docker-compose.yml:
#   services:
#     api:
#       mem_limit: 4g

# 5. Re-buildar imagem
docker-compose build --no-cache
```

---

## Problema: Pipeline muito lento

### Possíveis Causas:
1. Coleta de dados demora muito
2. Feature engineering custoso
3. Treinamento longo
4. Infraestrutura limitada

### Solução:
```bash
# 1. Perfilar cada tarefa
# Adicionar timing logs em cada etapa

# 2. Paralelizar tarefas
# Modificar DAG para executar tarefas em paralelo

# 3. Cache de dados
# Cachear dados coletados se não forem atualizados

# 4. Sample dos dados
# Em configs/mlops_config.yaml:
#   feature_engineering:
#     sample_rate: 0.8  # Usar 80% dos dados

# 5. Aumentar recursos de computação
# Mais CPUs, mais memória, GPU
```

---

## Problema: Notificação não funciona

### Possíveis Causas:
1. Slack/Email não configurado
2. Credentials inválidas
3. Função não implementada

### Solução:
```bash
# 1. Adicionar implementação de notificação
# Em airflow/dags/petr4_ml_pipeline.py:
#   success_notification e failure_notification

# 2. Configurar Slack
export SLACK_WEBHOOK_URL="https://hooks.slack.com/..."

# 3. Testar notificação manualmente
python -c "
import requests
WEBHOOK_URL = 'https://hooks.slack.com/...'
requests.post(WEBHOOK_URL, json={'text': 'Test message'})
"

# 4. Verificar permissões do container
# Container precisa acessar webhook URL
```

---

## Checklist para Produção

- [ ] Todas as variáveis de ambiente definidas
- [ ] Dados validados com qualidade > 85%
- [ ] Modelo com F1 > 55%
- [ ] A/B test passou (p < 0.05)
- [ ] Quality gates todos verdes
- [ ] Relatórios sendo gerados corretamente
- [ ] MLflow tracking funcionando
- [ ] Notificações configuradas
- [ ] Logs sendo armazenados corretamente
- [ ] Backup de dados configurado
- [ ] Monitoramento em tempo real setup
- [ ] Plano de rollback preparado

---

## Contato e Suporte

Para problemas persistentes:
1. Verificar logs completos: `docker logs -f [service]`
2. Consultar documentação: `airflow/dags/README_MLOPS_DAG.md`
3. Executar testes: `pytest tests/test_mlops_dag.py -v`
4. Contato: grupo16@datathon.com
