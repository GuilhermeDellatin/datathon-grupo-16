#!/usr/bin/env python
"""
Script de utilitários para gerenciamento da DAG MLOps PETR4.

Uso:
    python airflow/utils/dag_manager.py --trigger
    python airflow/utils/dag_manager.py --status
    python airflow/utils/dag_manager.py --logs <task_id>
    python airflow/utils/dag_manager.py --report <run_id>
"""

import json
import logging
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import yaml

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MLopsDagManager:
    """Gerencia DAG de MLOps."""

    def __init__(self, config_path: str = "configs/mlops_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Carrega configuração YAML."""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Erro ao carregar config: {e}")
            return {}

    def trigger_dag(self, conf: dict = None) -> dict:
        """Dispara DAG no Airflow."""
        logger.info("Disparando DAG petr4_mlops_pipeline...")

        from airflow.api.client.local_client import Client

        try:
            client = Client(None, None)
            run_id = client.trigger_dag(
                dag_id="petr4_mlops_pipeline",
                conf=conf or {},
            )

            logger.info(f"✓ DAG disparada com run_id: {run_id}")

            return {
                "status": "success",
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Erro ao disparar DAG: {e}")
            return {"status": "failed", "error": str(e)}

    def get_dag_status(self, run_id: str = None) -> dict:
        """Obtém status da DAG."""
        from airflow.models import DagRun

        logger.info("Obtendo status da DAG...")

        try:
            if run_id:
                run = DagRun.find(dag_id="petr4_mlops_pipeline", run_id=run_id)
            else:
                # Última execução
                run = DagRun.find(dag_id="petr4_mlops_pipeline", limit=1)

            if not run:
                return {"status": "not_found"}

            run = run[0]

            return {
                "run_id": run.run_id,
                "state": run.state,
                "start_date": run.start_date.isoformat() if run.start_date else None,
                "end_date": run.end_date.isoformat() if run.end_date else None,
                "execution_date": run.execution_date.isoformat() if run.execution_date else None,
            }
        except Exception as e:
            logger.error(f"Erro ao obter status: {e}")
            return {"status": "failed", "error": str(e)}

    def get_task_logs(self, run_id: str, task_id: str) -> str:
        """Obtém logs de uma tarefa."""

        logger.info(f"Obtendo logs da tarefa {task_id}...")

        try:
            # Implementação simplificada
            logger.info(f"Logs para {task_id} (não implementado)")
            return ""
        except Exception as e:
            logger.error(f"Erro ao obter logs: {e}")
            return ""

    def get_pipeline_report(self, run_id: str = None) -> dict:
        """Obtém relatório do pipeline."""
        logger.info("Obtendo relatório do pipeline...")

        # Buscar últimos relatórios
        metrics_dir = Path("metrics")
        reports = sorted(metrics_dir.glob("mlops_report_*.json"))

        if not reports:
            logger.warning("Nenhum relatório encontrado")
            return {"status": "no_reports"}

        try:
            # Carregar relatório mais recente
            latest_report_path = reports[-1]

            with open(latest_report_path) as f:
                report = json.load(f)

            logger.info(f"✓ Relatório carregado: {latest_report_path}")

            return {
                "status": "success",
                "report_path": str(latest_report_path),
                "report": report,
            }
        except Exception as e:
            logger.error(f"Erro ao carregar relatório: {e}")
            return {"status": "failed", "error": str(e)}

    def display_report_summary(self, report: dict):
        """Exibe resumo do relatório."""
        logger.info("=" * 70)
        logger.info("RESUMO DO PIPELINE MLOPS")
        logger.info("=" * 70)

        if "pipeline_run_date" in report:
            logger.info(f"Data: {report['pipeline_run_date']}")

        if "data_quality" in report:
            dq = report["data_quality"]
            logger.info(f"Qualidade dos Dados: {dq.get('overall_quality', 0):.1%}")

        if "drift_metrics" in report:
            drift = report["drift_metrics"]
            logger.info(f"Data Drift: {drift.get('data_drift_score', 0):.1%}")
            logger.info(f"Concept Drift: {drift.get('concept_drift_score', 0):.1%}")

        if "model_metrics" in report:
            mm = report["model_metrics"]
            logger.info(f"Accuracy: {mm.get('accuracy', 0):.1%}")
            logger.info(f"F1 Score: {mm.get('f1_score', 0):.1%}")
            logger.info(f"MAE: {mm.get('mae', 0):.4f}")

        if "ab_test_results" in report:
            ab = report["ab_test_results"]
            logger.info(f"A/B Test Recomendação: {ab.get('recommendation', 'N/A')}")

        if "quality_gates" in report:
            qg = report["quality_gates"]
            status = "✓ PASSOU" if qg.get("all_gates_passed") else "✗ FALHOU"
            logger.info(f"Quality Gates: {status}")

        logger.info("=" * 70)

    def print_config(self):
        """Exibe configuração carregada."""
        logger.info("=" * 70)
        logger.info("CONFIGURAÇÃO MLOPS")
        logger.info("=" * 70)

        logger.info(json.dumps(self.config, indent=2))

        logger.info("=" * 70)

    def validate_config(self) -> bool:
        """Valida configuração."""
        logger.info("Validando configuração...")

        required_keys = ["mlops"]

        for key in required_keys:
            if key not in self.config:
                logger.error(f"Chave obrigatória faltando: {key}")
                return False

        logger.info("✓ Configuração válida")
        return True


def main():
    """Main."""
    parser = ArgumentParser(description="Gerenciador de DAG MLOps")

    parser.add_argument("--trigger", action="store_true", help="Dispara a DAG")
    parser.add_argument("--status", action="store_true", help="Obtém status")
    parser.add_argument("--logs", type=str, help="Obtém logs da tarefa")
    parser.add_argument("--report", action="store_true", help="Mostra relatório")
    parser.add_argument("--config", action="store_true", help="Mostra configuração")
    parser.add_argument("--validate", action="store_true", help="Valida configuração")
    parser.add_argument("--run-id", type=str, help="Run ID específico")

    args = parser.parse_args()

    manager = MLopsDagManager()

    if args.validate:
        manager.validate_config()
        return

    if args.config:
        manager.print_config()
        return

    if args.trigger:
        result = manager.trigger_dag()
        logger.info(json.dumps(result, indent=2))
        return

    if args.status:
        status = manager.get_dag_status(args.run_id)
        logger.info(json.dumps(status, indent=2))
        return

    if args.logs:
        if not args.run_id:
            logger.error("--run-id é obrigatório com --logs")
            sys.exit(1)
        logs = manager.get_task_logs(args.run_id, args.logs)
        logger.info(logs)
        return

    if args.report:
        result = manager.get_pipeline_report(args.run_id)
        if result.get("status") == "success":
            manager.display_report_summary(result["report"])
            logger.info(f"\nRelatório completo: {result['report_path']}")
        else:
            logger.error(f"Erro: {result.get('error')}")
        return

    # Default: mostrar ajuda
    parser.print_help()


if __name__ == "__main__":
    main()
