"""DAG `train_lstm_stock` — disparada pelo endpoint POST /training/jobs.

Lê o payload validado em ``dag_run.conf`` (job_id, ticker, period, model_config,
training_config), persiste a configuração do job em
``.tmp/training_jobs/<job_id>/config.yaml`` e executa o pipeline reprodutível
via DVC (``dvc pull && dvc repro && dvc push``).

A configuração específica do job é exposta às tasks via variáveis de ambiente
(``TRAINING_JOB_ID``, ``TRAINING_TICKER`` etc.), conforme a estratégia descrita
na spec 12.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess  # nosec B404
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from airflow.exceptions import AirflowException
from airflow.operators.python import PythonOperator

from airflow import DAG  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/app")
TMP_DIR = Path(PROJECT_ROOT) / ".tmp" / "training_jobs"

REQUIRED_CONF_KEYS = (
    "job_id",
    "ticker",
    "period",
    "model_config",
    "training_config",
)


def _parse_conf(context: dict[str, Any]) -> dict[str, Any]:
    """Lê e valida `dag_run.conf` da execução atual."""
    dag_run = context.get("dag_run")
    conf: dict[str, Any] = {}
    if dag_run is not None and getattr(dag_run, "conf", None):
        conf = dict(dag_run.conf)

    missing = [key for key in REQUIRED_CONF_KEYS if key not in conf]
    if missing:
        raise AirflowException(
            "dag_run.conf incompleto: faltam "
            f"{missing}. Recebido: {sorted(conf)}. "
            "Esta DAG é disparada pelo endpoint POST /training/jobs da API "
            "(que envia o conf esperado). Triggers manuais pelo UI precisam "
            "informar `Configuration JSON` com ticker, period, model_config e "
            "training_config."
        )
    return conf


def _job_dir(job_id: str) -> Path:
    job_dir = TMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def _write_job_config(conf: dict[str, Any]) -> str:
    """Persiste o `conf` em YAML para que o stage de treino possa lê-lo."""
    job_dir = _job_dir(conf["job_id"])
    config_path = job_dir / "config.yaml"
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(conf, fh, sort_keys=False, allow_unicode=True)
    logger.info("Config do job persistida em %s", config_path)
    return str(config_path)


def _job_env(conf: dict[str, Any], config_path: str) -> dict[str, str]:
    """Monta o env passado para os subprocessos DVC."""
    env = os.environ.copy()
    env.update(
        {
            "TRAINING_JOB_ID": str(conf["job_id"]),
            "TRAINING_TICKER": str(conf["ticker"]),
            "TRAINING_PERIOD": str(conf["period"]),
            "TRAINING_CONFIG_PATH": config_path,
        }
    )
    return env


def _run_subprocess(cmd: list[str], env: dict[str, str], step: str) -> None:
    """Executa um comando externo com `cwd=PROJECT_ROOT` e propaga falhas."""
    logger.info("[%s] executando: %s", step, " ".join(cmd))
    try:
        subprocess.run(  # nosec B603
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
        )
    except FileNotFoundError as exc:
        raise AirflowException(
            f"[{step}] binário não encontrado: {cmd[0]}. Verifique se DVC está instalado "
            "na imagem do Airflow."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise AirflowException(
            f"[{step}] subprocesso falhou (exit={exc.returncode})."
        ) from exc


def prepare_job(**context: Any) -> str:
    """Valida o `dag_run.conf` e persiste a configuração do job em disco.

    Returns:
        Caminho do arquivo YAML com a configuração do job (XCom default).
    """
    conf = _parse_conf(context)
    config_path = _write_job_config(conf)

    ti = context["task_instance"]
    ti.xcom_push(key="job_id", value=conf["job_id"])
    ti.xcom_push(key="config_path", value=config_path)
    return config_path


def dvc_pull(**context: Any) -> None:
    """Sincroniza outputs/cache do DVC remote local antes do pipeline."""
    conf = _parse_conf(context)
    config_path = context["task_instance"].xcom_pull(
        task_ids="prepare_job", key="config_path"
    )
    env = _job_env(conf, config_path)
    _run_subprocess(["dvc", "pull"], env=env, step="dvc_pull")


def dvc_repro(**context: Any) -> None:
    """Executa o pipeline reprodutível (`dvc repro`)."""
    conf = _parse_conf(context)
    config_path = context["task_instance"].xcom_pull(
        task_ids="prepare_job", key="config_path"
    )
    env = _job_env(conf, config_path)
    _run_subprocess(["dvc", "repro"], env=env, step="dvc_repro")


def dvc_push(**context: Any) -> None:
    """Publica os outputs no remote local após o repro."""
    conf = _parse_conf(context)
    config_path = context["task_instance"].xcom_pull(
        task_ids="prepare_job", key="config_path"
    )
    env = _job_env(conf, config_path)
    _run_subprocess(["dvc", "push"], env=env, step="dvc_push")


def finalize_job(**context: Any) -> None:
    """Registra um manifest simples com o resultado do job."""
    conf = _parse_conf(context)
    job_dir = _job_dir(conf["job_id"])
    manifest = {
        "job_id": conf["job_id"],
        "ticker": conf["ticker"],
        "period": conf["period"],
        "finished_at": datetime.now(UTC).isoformat(),
    }
    manifest_path = job_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Manifest do job em %s", manifest_path)


default_args = {
    "owner": "mlops-team",
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="train_lstm_stock",
    description=(
        "Treino do LSTM (PETR4 e variações) disparado pelo endpoint "
        "POST /training/jobs da API."
    ),
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["datathon", "training", "dvc"],
) as dag:
    prepare = PythonOperator(
        task_id="prepare_job",
        python_callable=prepare_job,
    )

    pull = PythonOperator(
        task_id="dvc_pull",
        python_callable=dvc_pull,
    )

    repro = PythonOperator(
        task_id="dvc_repro",
        python_callable=dvc_repro,
    )

    push = PythonOperator(
        task_id="dvc_push",
        python_callable=dvc_push,
    )

    finalize = PythonOperator(
        task_id="finalize_job",
        python_callable=finalize_job,
    )

    prepare >> pull >> repro >> push >> finalize
