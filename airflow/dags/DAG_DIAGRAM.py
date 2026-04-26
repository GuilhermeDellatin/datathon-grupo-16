"""
Visualização da DAG MLOps em Texto
"""

import tempfile

# ASCII Art da DAG
dag_diagram = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                    DAG MLOps PETR4 LSTM PIPELINE                              ║
║                        (petr4_mlops_pipeline)                                  ║
╚════════════════════════════════════════════════════════════════════════════════╝

                                    START
                                     │
                                     ▼
                            ┌─────────────────┐
                            │ collect_data    │  (Yahoo Finance)
                            │ [Docker]        │
                            │ 5-10 min        │
                            └─────────────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │ validate_data   │  (Pandera)
                            │ [Python]        │  MIN_QUALITY >= 85%
                            │ <1 min          │
                            └─────────────────┘
                                     │
                                     ▼
                       ┌─────────────────────────────┐
                       │ feature_engineering        │  (SMA, EMA, RSI, etc)
                       │ [Docker]                    │  60 timesteps
                       │ 5-10 min                    │
                       └─────────────────────────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │ drift_detection │  (Evidently)
                            │ [Python]        │  Data + Concept
                            │ <1 min          │  Threshold: 15%
                            └─────────────────┘
                                     │
                                     ▼
                       ┌─────────────────────────────┐
                       │ train_model                 │  (LSTM)
                       │ [Docker]                    │  50 neurons, 2 layers
                       │ 10-20 min                   │  MLflow tracking
                       └─────────────────────────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │ validate_model  │  (Performance)
                            │ [Python]        │  Acc >= 60%, F1 >= 55%
                            │ <1 min          │
                            └─────────────────┘
                                     │
                                     ▼
                       ┌─────────────────────────────┐
                       │ evaluate_quality            │  (RAGAS + RAG)
                       │ [Docker]                    │  Avaliação qualitativa
                       │ 10-15 min                   │
                       └─────────────────────────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │ ab_test         │  (Champion-Challenger)
                            │ [Python]        │  Z-test, α = 0.05
                            │ <1 min          │
                            └─────────────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │ quality_check   │  (5 Gates)
                            │ [Python]        │  ✓ Todas as métricas
                            │ <1 min          │
                            └─────────────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │ register_model  │  (MLflow Registry)
                            │ [Docker]        │  Staging → Production
                            │ <1 min          │
                            └─────────────────┘
                                     │
                                     ▼
                       ┌─────────────────────────────┐
                       │ prepare_deployment          │  (Artifacts)
                       │ [Docker]                    │  Model card, config
                       │ 1-2 min                     │
                       └─────────────────────────────┘
                                     │
                                     ▼
                       ┌─────────────────────────────┐
                       │ generate_report             │  (JSON)
                       │ [Python]                    │  Resumo completo
                       │ <1 min                      │
                       └─────────────────────────────┘
                                     │
                  ┌──────────────────┴──────────────────┐
                  ▼                                     ▼
         ┌─────────────────────┐          ┌─────────────────────────┐
         │ success_notification│          │ failure_notification    │
         │ [Python]            │          │ [Python]                │
         │ Trigger: all_success│          │ Trigger: one_failed     │
         └─────────────────────┘          └─────────────────────────┘
                  │                                     │
                  └──────────────────┬──────────────────┘
                                     ▼
                                    END


╔════════════════════════════════════════════════════════════════════════════════╗
║                            LEGENDA                                             ║
╠════════════════════════════════════════════════════════════════════════════════╣
║  [Docker]   = Tarefa executada em container Docker                             ║
║  [Python]   = Tarefa Python pura (sem container)                               ║
║  MIN_QUALITY= Limiar mínimo de qualidade                                       ║
║  Acc/F1     = Accuracy / F1 Score                                              ║
║  α          = Nível de significância estatística                               ║
╚════════════════════════════════════════════════════════════════════════════════╝


FLUXO DE DADOS:
═══════════════════════════════════════════════════════════════════════════════

    Yahoo Finance
         │
         ▼
    [Raw Data] ──────────────────┐
         │                       │
         ▼                       │
    validate_data         Validação de Qualidade
         │                (Completeness, Uniqueness, etc)
         │◄──────────────────────┘
         │
         ▼
    [Clean Data]
         │
         ▼
    feature_engineering
         │
         ▼
    [Engineered Features]
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
    drift_detection    [Drift Report]
         │
         ▼
    train_model ◄──────────────────── [MLflow]
         │
         ▼
    [Trained Model] ──────────┐
         │                    │
         ├──────────────────┐ │
         │                  │ │
         ▼                  ▼ ▼
    validate_model  evaluate_quality
         │                  │
         │                  ▼
         │          [Qualitative Scores]
         │                  │
         ▼                  │
    [Model Metrics] ◄───────┘
         │
         ├─────────────────┬────────────┐
         │                 │            │
         ▼                 ▼            ▼
    ab_test      quality_check    register_model
         │                 │            │
         ▼                 ▼            ▼
    [AB Results] ◄──────┤       [MLflow Registry]
                        │
                    (Decision Point)
                        │
                        ├─ PASS ──┐
                        │         │
                        ├─ FAIL ──┤
                        │         │
                        └─────────┤
                                  ▼
                          prepare_deployment
                                  │
                                  ▼
                          [Deployment Ready]
                                  │
                                  ▼
                          generate_report
                                  │
                                  ▼
                          [mlops_report.json]
                                  │
                                  ├─ Sucesso ──┐
                                  ├─ Falha ────┤
                                  │            │
                                  └────────────┤
                                               ▼
                                        notifications


QUALITY GATES (Checkpoints):
═══════════════════════════════════════════════════════════════════════════════

   ┌─ validate_data ────── Data Quality >= 85% ─────┐
   │                                                 │
   │                                            ✓ PASS / ✗ FAIL
   │
   ├─ validate_model ───── Accuracy >= 60% ─────────┐
   │                       F1 >= 55% ────────────── │
   │                       Precision >= 65% ────── │
   │                       MAE <= 0.10 ──────────── │
   │                       RMSE <= 0.15 ──────────┤
   │                                            ✓ PASS / ✗ FAIL
   │
   ├─ drift_detection ───── Data Drift <= 15% ──────┐
   │                        Concept Drift <= 15% ──│
   │                                            ⚠ WARN / ✓ OK
   │
   ├─ ab_test ──────────── Stat. Significant ──────┐
   │                       p-value < 0.05 ────────┤
   │                                            ✓ YES / ✗ NO
   │
   └─ quality_check ────── Todos os gates ─────────┐
                           passam? ──────────────┤
                                            ✓ DEPLOY / ✗ BLOCK


TABELA DE TIMES E RECURSOS:
═══════════════════════════════════════════════════════════════════════════════

Tarefa                  Type    Tempo   Recurso     Docker
─────────────────────────────────────────────────────────────────────────────
collect_data            Docker  5-10m   2CPU 1GB    SIM
validate_data           Python  <1m     0.5CPU 256MB   NÃO
feature_engineering     Docker  5-10m   2CPU 1GB    SIM
drift_detection         Python  <1m     0.5CPU 256MB   NÃO
train_model             Docker  10-20m  4CPU 2GB    SIM
validate_model          Python  <1m     0.5CPU 256MB   NÃO
evaluate_quality        Docker  10-15m  2CPU 1GB    SIM
ab_test                 Python  <1m     0.5CPU 256MB   NÃO
quality_check           Python  <1m     0.5CPU 256MB   NÃO
register_model          Docker  <1m     0.5CPU 256MB   SIM
prepare_deployment      Docker  1-2m    1CPU 512MB  SIM
generate_report         Python  <1m     0.5CPU 256MB   NÃO
success_notification    Python  <1m     0.5CPU 256MB   NÃO
failure_notification    Python  <1m     0.5CPU 256MB   NÃO
─────────────────────────────────────────────────────────────────────────────
TOTAL                           ~80m    ~20CPU 10GB


ESTADO DA DAG DURANTE EXECUÇÃO:
═══════════════════════════════════════════════════════════════════════════════

T+0m:    [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 10% collect_data
T+10m:   [██████████░░░░░░░░░░░░░░░░░░░░░░░░░] 25% train_model
T+30m:   [████████████████░░░░░░░░░░░░░░░░░░░] 50% evaluate_quality
T+50m:   [████████████████████████░░░░░░░░░░░] 75% quality_check
T+70m:   [████████████████████████████░░░░░░░] 90% generate_report
T+80m:   [████████████████████████████████████] 100% ✓ SUCESSO


TRIGGERANDO A DAG:
═══════════════════════════════════════════════════════════════════════════════

Option 1: Airflow UI
  1. Navegar para: http://localhost:8080
  2. Encontrar: "petr4_mlops_pipeline"
  3. Clicar em: "Trigger DAG"

Option 2: CLI
  $ airflow dags trigger petr4_mlops_pipeline

Option 3: Python
  $ python airflow/utils/dag_manager.py --trigger

Option 4: API
  $ curl -X POST http://localhost:8080/api/v1/dags/petr4_mlops_pipeline/dagRuns

Option 5: Automático
  * Schedule: 0 2 * * * (Diariamente às 2 AM UTC)
  * Configurável em: configs/mlops_config.yaml


VENDO RESULTADOS:
═══════════════════════════════════════════════════════════════════════════════

1. Airflow UI Graph
   http://localhost:8080/dags/petr4_mlops_pipeline/graph

2. MLflow UI
   http://localhost:5000/experiments

3. Relatório JSON
   metrics/mlops_report_YYYYMMDD_HHMMSS.json

4. CLI
   python airflow/utils/dag_manager.py --report

5. Logs
   docker logs -f airflow-scheduler
   docker logs -f airflow-webserver
"""

# Imprimir diagrama
if __name__ == "__main__":
    print(dag_diagram)

    # Salvar em arquivo
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(dag_diagram)
        temp_path = f.name  # Use este caminho se precisar referenciar o arquivo depois

    print("\n✓ Diagrama salvo em: /tmp/dag_diagram.txt")
