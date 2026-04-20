#!/bin/bash
# Quick Start para DAG MLOps PETR4
# Uso: bash airflow/scripts/quickstart.sh

set -e

echo "================================================"
echo "QUICKSTART - DAG MLOps PETR4"
echo "================================================"
echo ""

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função de log
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# 1. Validar ambiente
log_info "1. Validando ambiente..."
if ! command -v docker &> /dev/null; then
    log_error "Docker não instalado"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    log_error "Docker Compose não instalado"
    exit 1
fi

log_success "Docker e Docker Compose encontrados"

# 2. Verificar variáveis de ambiente
log_info "2. Verificando variáveis de ambiente..."

if [ -z "$PROJECT_ROOT" ]; then
    export PROJECT_ROOT="/root/datathon-grupo-16"
    log_warning "PROJECT_ROOT não definido, usando: $PROJECT_ROOT"
else
    log_success "PROJECT_ROOT: $PROJECT_ROOT"
fi

if [ -z "$MLFLOW_TRACKING_URI" ]; then
    export MLFLOW_TRACKING_URI="http://localhost:5000"
    log_warning "MLFLOW_TRACKING_URI não definido, usando: $MLFLOW_TRACKING_URI"
else
    log_success "MLFLOW_TRACKING_URI: $MLFLOW_TRACKING_URI"
fi

if [ -z "$OPENAI_API_KEY" ]; then
    log_warning "⚠ OPENAI_API_KEY não definida (opcional para coleta de dados)"
fi

# 3. Iniciar serviços
log_info "3. Iniciando serviços..."

log_info "Iniciando Docker Compose..."
docker-compose -f docker-compose.yml up -d

log_success "Serviços iniciados"

# 4. Aguardar serviços ficarem prontos
log_info "4. Aguardando serviços ficarem prontos..."
sleep 10

log_info "Verificando Airflow..."
MAX_ATTEMPTS=30
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        log_success "Airflow está pronto"
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
        log_error "Timeout aguardando Airflow"
        exit 1
    fi
    echo -n "."
    sleep 2
done

log_info "Verificando MLflow..."
MAX_ATTEMPTS=30
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:5000 > /dev/null 2>&1; then
        log_success "MLflow está pronto"
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
        log_warning "MLflow pode estar demorando"
        break
    fi
    echo -n "."
    sleep 2
done

echo ""

# 5. Verificar DAG
log_info "5. Verificando DAG..."
AIRFLOW_HOME=${AIRFLOW_HOME:-.}

# Aguardar DAG ser carregada
sleep 5

log_info "DAG: petr4_mlops_pipeline"
log_info "Schedule: 0 2 * * * (Diário às 2 AM UTC)"

# 6. Mostrar URLs de acesso
log_info "6. URLs de Acesso"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Airflow UI${NC}         → http://localhost:8080"
echo -e "${GREEN}MLflow UI${NC}         → http://localhost:5000"
echo -e "${GREEN}Grafana${NC}           → http://localhost:3000 (opcional)"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

# 7. Mostrar próximos passos
log_info "7. Próximos Passos"
echo ""
echo -e "${YELLOW}1. Acessar Airflow UI:${NC}"
echo "   http://localhost:8080"
echo ""
echo -e "${YELLOW}2. Disparar DAG manualmente:${NC}"
echo "   Clique em 'petr4_mlops_pipeline' → 'Trigger DAG'"
echo ""
echo -e "${YELLOW}3. Ou via CLI:${NC}"
echo "   airflow dags trigger petr4_mlops_pipeline"
echo ""
echo -e "${YELLOW}4. Ou via Python:${NC}"
echo "   python airflow/utils/dag_manager.py --trigger"
echo ""
echo -e "${YELLOW}5. Ver status:${NC}"
echo "   python airflow/utils/dag_manager.py --status"
echo ""
echo -e "${YELLOW}6. Ver relatório:${NC}"
echo "   python airflow/utils/dag_manager.py --report"
echo ""
echo -e "${YELLOW}7. Verificar logs:${NC}"
echo "   docker logs -f airflow-webserver"
echo ""

# 8. Modo debug (opcional)
if [ "$1" == "--debug" ]; then
    log_info "Modo DEBUG ativado"
    
    log_info "Verificando estrutura de pastas..."
    echo "DATA:"
    ls -la $PROJECT_ROOT/data/ || echo "  (não existe)"
    echo ""
    echo "MODELS:"
    ls -la $PROJECT_ROOT/models/ || echo "  (não existe)"
    echo ""
    echo "METRICS:"
    ls -la $PROJECT_ROOT/metrics/ || echo "  (não existe)"
fi

log_success "QuickStart concluído!"
echo ""
echo -e "${GREEN}DAG MLOps pronta para ser executada!${NC}"
echo ""
