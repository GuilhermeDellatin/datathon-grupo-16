#!/bin/bash
# Comandos Rápidos para DAG MLOps

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║             COMANDOS RÁPIDOS - DAG MLOps PETR4                            ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# INICIALIZAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

echo "📦 INICIALIZAÇÃO"
echo "─────────────────────────────────────────────────────────────────────────"
echo ""
echo "1. Setup automático completo:"
echo "   bash airflow/scripts/quickstart.sh"
echo ""
echo "2. Iniciar serviços Docker:"
echo "   docker-compose up -d"
echo ""
echo "3. Verificar status dos serviços:"
echo "   docker-compose ps"
echo ""
echo "4. Parar serviços:"
echo "   docker-compose down"
echo ""
echo "5. Ver logs em tempo real:"
echo "   docker-compose logs -f airflow-scheduler"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# DAG
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "🚀 DAG OPERAÇÕES"
echo "─────────────────────────────────────────────────────────────────────────"
echo ""
echo "1. Disparar DAG:"
echo "   python airflow/utils/dag_manager.py --trigger"
echo "   # ou"
echo "   airflow dags trigger petr4_mlops_pipeline"
echo ""
echo "2. Verificar status:"
echo "   python airflow/utils/dag_manager.py --status"
echo ""
echo "3. Ver últimos 10 runs:"
echo "   airflow dags list-runs -d petr4_mlops_pipeline | head -20"
echo ""
echo "4. Limpar logs antigos:"
echo "   airflow db clean --skip-archive"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# MONITORAMENTO
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "📊 MONITORAMENTO"
echo "─────────────────────────────────────────────────────────────────────────"
echo ""
echo "1. Airflow UI:"
echo "   http://localhost:8080"
echo ""
echo "2. MLflow UI:"
echo "   http://localhost:5000"
echo ""
echo "3. Relatório Pipeline:"
echo "   python airflow/utils/dag_manager.py --report"
echo ""
echo "4. Ver diagrama da DAG:"
echo "   python airflow/dags/DAG_DIAGRAM.py"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# TESTES
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "🧪 TESTES"
echo "─────────────────────────────────────────────────────────────────────────"
echo ""
echo "1. Rodar todos os testes:"
echo "   pytest tests/test_mlops_dag.py -v"
echo ""
echo "2. Rodar teste específico:"
echo "   pytest tests/test_mlops_dag.py::TestQualityGates::test_quality_gates_pass -v"
echo ""
echo "3. Rodar com coverage:"
echo "   pytest tests/test_mlops_dag.py --cov=src --cov-report=html"
echo ""
echo "4. Validar DAG syntax:"
echo "   python -m py_compile airflow/dags/petr4_ml_pipeline.py"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# DADOS E MODELOS
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "💾 DADOS E MODELOS"
echo "─────────────────────────────────────────────────────────────────────────"
echo ""
echo "1. Listar arquivos de dados:"
echo "   ls -lh data/"
echo ""
echo "2. Verificar tamanho dos dados:"
echo "   du -sh data/"
echo ""
echo "3. Listar modelos:"
echo "   ls -lh models/"
echo ""
echo "4. Ver histórico de versões:"
echo "   python -c \"import json; print(json.dumps(json.load(open('models/versions.json')), indent=2))\""
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# LOGS
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "📝 LOGS"
echo "─────────────────────────────────────────────────────────────────────────"
echo ""
echo "1. Ver logs da DAG:"
echo "   docker logs -f airflow-scheduler"
echo ""
echo "2. Ver logs do Webserver:"
echo "   docker logs -f airflow-webserver"
echo ""
echo "3. Ver logs de uma tarefa específica:"
echo "   airflow tasks logs petr4_mlops_pipeline collect_data 2024-01-01"
echo ""
echo "4. Logs em arquivo:"
echo "   ls -la logs/dag_id=petr4_mlops_pipeline/"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "⚙️  CONFIGURAÇÃO"
echo "─────────────────────────────────────────────────────────────────────────"
echo ""
echo "1. Ver arquivo de configuração:"
echo "   cat configs/mlops_config.yaml"
echo ""
echo "2. Validar configuração:"
echo "   python airflow/utils/dag_manager.py --validate"
echo ""
echo "3. Exibir configuração carregada:"
echo "   python airflow/utils/dag_manager.py --config"
echo ""
echo "4. Variáveis de ambiente:"
echo "   export PROJECT_ROOT=/root/datathon-grupo-16"
echo "   export MLFLOW_TRACKING_URI=http://localhost:5000"
echo "   export OPENAI_API_KEY=sk-..."
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# TROUBLESHOOTING
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "🆘 TROUBLESHOOTING"
echo "─────────────────────────────────────────────────────────────────────────"
echo ""
echo "1. Verificar conectividade:"
echo "   curl http://localhost:8080/health"
echo "   curl http://localhost:5000"
echo ""
echo "2. Sincronizar banco de dados:"
echo "   airflow db init"
echo ""
echo "3. Criar usuário Airflow:"
echo "   airflow users create --username admin --password admin --firstname Admin --lastname Admin --role Admin --email admin@example.com"
echo ""
echo "4. Ver help de comandos:"
echo "   airflow dags -h"
echo "   airflow tasks -h"
echo ""
echo "5. Reiniciar containers:"
echo "   docker-compose restart"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# DESENVOLVIMENTO
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "🛠️  DESENVOLVIMENTO"
echo "─────────────────────────────────────────────────────────────────────────"
echo ""
echo "1. Instalar dependências:"
echo "   pip install -r requirements.txt"
echo "   pip install -r airflow/requirements.txt"
echo ""
echo "2. Formatar código:"
echo "   black airflow/dags/petr4_ml_pipeline.py src/monitoring/"
echo ""
echo "3. Linting:"
echo "   flake8 airflow/dags/petr4_ml_pipeline.py"
echo "   pylint airflow/dags/petr4_ml_pipeline.py"
echo ""
echo "4. Verificar tipos:"
echo "   mypy airflow/dags/petr4_ml_pipeline.py"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# BACKUP E LIMPEZA
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "🔄 BACKUP E LIMPEZA"
echo "─────────────────────────────────────────────────────────────────────────"
echo ""
echo "1. Fazer backup dos dados:"
echo "   tar -czf data_backup_$(date +%Y%m%d).tar.gz data/"
echo ""
echo "2. Fazer backup dos modelos:"
echo "   tar -czf models_backup_$(date +%Y%m%d).tar.gz models/"
echo ""
echo "3. Limpar logs antigos:"
echo "   find logs/ -mtime +7 -delete"
echo ""
echo "4. Limpar relatórios antigos:"
echo "   find metrics/ -mtime +30 -delete"
echo ""
echo "5. Limpar tudo e reiniciar:"
echo "   docker-compose down -v"
echo "   rm -rf logs/ metrics/ models/ data/"
echo "   docker-compose up -d"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# AVANÇADO
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "🔬 AVANÇADO"
echo "─────────────────────────────────────────────────────────────────────────"
echo ""
echo "1. Executar DAG em modo debug:"
echo "   airflow dags test petr4_mlops_pipeline 2024-01-01"
echo ""
echo "2. Executar tarefa específica:"
echo "   airflow tasks test petr4_mlops_pipeline validate_model 2024-01-01"
echo ""
echo "3. Ver grafo de dependências:"
echo "   airflow dags show petr4_mlops_pipeline"
echo ""
echo "4. Trigger com configuração customizada:"
echo "   airflow dags trigger petr4_mlops_pipeline -c '{\"param1\": \"value1\"}'"
echo ""
echo "5. Pausar/Retomar DAG:"
echo "   airflow dags pause petr4_mlops_pipeline"
echo "   airflow dags unpause petr4_mlops_pipeline"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENTAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "📖 DOCUMENTAÇÃO"
echo "─────────────────────────────────────────────────────────────────────────"
echo ""
echo "1. Guia da DAG:"
echo "   cat airflow/dags/README_MLOPS_DAG.md"
echo ""
echo "2. Troubleshooting:"
echo "   cat TROUBLESHOOTING_MLOPS.md"
echo ""
echo "3. Resumo:"
echo "   cat MLOPS_DAG_SUMMARY.md"
echo ""
echo "4. Diagrama:"
echo "   python airflow/dags/DAG_DIAGRAM.py"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# ATALHOS
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "⚡ ATALHOS ÚTEIS"
echo "─────────────────────────────────────────────────────────────────────────"
echo ""
echo "# Alias no ~/.bashrc"
echo ""
echo "alias dag-ui='echo \"http://localhost:8080\"'"
echo "alias mlflow-ui='echo \"http://localhost:5000\"'"
echo "alias dag-trigger='python airflow/utils/dag_manager.py --trigger'"
echo "alias dag-status='python airflow/utils/dag_manager.py --status'"
echo "alias dag-report='python airflow/utils/dag_manager.py --report'"
echo "alias dag-logs='docker-compose logs -f airflow-scheduler'"
echo "alias dag-test='pytest tests/test_mlops_dag.py -v'"
echo ""

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                    💡 Dica: Criar um arquivo .env                          ║"
echo "║                                                                            ║"
echo "║  PROJECT_ROOT=/root/datathon-grupo-16                                     ║"
echo "║  MLFLOW_TRACKING_URI=http://localhost:5000                                ║"
echo "║  OPENAI_API_KEY=sk-...                                                    ║"
echo "║                                                                            ║"
echo "║  source .env para carregar variáveis automaticamente                       ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""
