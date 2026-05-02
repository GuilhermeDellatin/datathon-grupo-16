from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from airflow import DAG
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.models.baseoperator import chain
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

logger = logging.getLogger(__name__)

# Defaults for containers started by docker compose in this repository. Override
# API_BASE_URL to http://host.docker.internal:8000 when Airflow is outside Compose.
API_BASE_URL = os.environ.get("API_BASE_URL", "http://api:8000").rstrip("/")
API_TIMEOUT_SECONDS = int(os.environ.get("API_TIMEOUT_SECONDS", "30"))
API_TICKER = os.environ.get("API_TICKER", "PETR4.SA")
API_HORIZON_DAYS = int(os.environ.get("API_HORIZON_DAYS", "5"))
API_AGENT_QUESTION = os.environ.get(
    "API_AGENT_QUESTION",
    "Qual o contexto atual para PETR4 considerando o modelo e os dados disponiveis?",
)
API_INFER_SEQUENCE_LENGTH = int(os.environ.get("API_INFER_SEQUENCE_LENGTH", "60"))
API_INFER_FEATURE_COUNT = int(os.environ.get("API_INFER_FEATURE_COUNT", "14"))
TRAIN_STATUS_POLL_SECONDS = int(os.environ.get("TRAIN_STATUS_POLL_SECONDS", "30"))
TRAIN_STATUS_TIMEOUT_SECONDS = int(os.environ.get("TRAIN_STATUS_TIMEOUT_SECONDS", "7200"))

METRICS_DIR = os.environ.get("METRICS_DIR", "/app/metrics")


def _decode_body(raw_body: str, content_type: str) -> Any:
    if not raw_body:
        return None

    if "json" in content_type.lower():
        try:
            return json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise AirflowException(f"Resposta JSON invalida da API: {raw_body[:500]}") from exc

    try:
        return json.loads(raw_body)
    except json.JSONDecodeError:
        return raw_body


def call_api(
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    expected_statuses: set[int] | None = None,
) -> dict[str, Any]:
    expected_statuses = expected_statuses or {200}
    url = f"{API_BASE_URL}/{path.lstrip('/')}"
    headers = {"Accept": "application/json"}
    data = None

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=data, headers=headers, method=method)
    logger.info("Calling API endpoint: %s %s", method, url)

    try:
        with urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:  # nosec B310
            raw_body = response.read().decode("utf-8", errors="replace")
            status_code = response.getcode()
            content_type = response.headers.get("content-type", "")
    except HTTPError as exc:
        raw_body = exc.read().decode("utf-8", errors="replace")
        content_type = exc.headers.get("content-type", "")
        if exc.code not in expected_statuses:
            raise AirflowException(
                f"API returned HTTP {exc.code} for {method} {path}: {raw_body[:1000]}"
            ) from exc

        return {
            "status_code": exc.code,
            "body": _decode_body(raw_body, content_type),
            "content_type": content_type,
        }
    except (TimeoutError, URLError) as exc:
        raise AirflowException(f"Could not call API at {url}: {exc}") from exc

    if status_code not in expected_statuses:
        raise AirflowException(
            f"API returned HTTP {status_code} for {method} {path}: {raw_body[:1000]}"
        )

    return {
        "status_code": status_code,
        "body": _decode_body(raw_body, content_type),
        "content_type": content_type,
    }


def check_api_health() -> dict[str, Any]:
    response = call_api("/health")
    body = response["body"]
    if not isinstance(body, dict):
        raise AirflowException(f"Resposta inesperada de /health: {body!r}")

    result = {"status_code": response["status_code"], **body}
    logger.info("API health: %s", result)
    return result


def trigger_api_training() -> dict[str, Any]:
    response = call_api("/train", method="POST", expected_statuses={202})
    result = {"status_code": response["status_code"], "body": response["body"]}
    logger.info("Training endpoint accepted request: %s", result)
    return result


def wait_for_training_completion() -> dict[str, Any]:
    deadline = time.monotonic() + TRAIN_STATUS_TIMEOUT_SECONDS
    last_status: dict[str, Any] | None = None

    while time.monotonic() <= deadline:
        response = call_api("/train/status")
        body = response["body"]
        if not isinstance(body, dict):
            raise AirflowException(f"Resposta inesperada de /train/status: {body!r}")

        last_status = {"status_code": response["status_code"], **body}
        training_state = str(body.get("status", "unknown"))
        logger.info("Training status: %s", last_status)

        if training_state == "completed":
            if not body.get("model_loaded"):
                raise AirflowException(
                    "Training completed, but API still reports model_loaded=False."
                )
            return last_status

        if training_state == "failed":
            raise AirflowException(f"Training failed: {body.get('error')}")

        if training_state == "idle":
            raise AirflowException("Training status is idle after POST /train.")

        time.sleep(TRAIN_STATUS_POLL_SECONDS)

    raise AirflowException(
        "Timed out waiting for training completion after "
        f"{TRAIN_STATUS_TIMEOUT_SECONDS} seconds. Last status: {last_status}"
    )


def _require_capability(capability: str, endpoint: str) -> dict[str, Any]:
    health = check_api_health()
    if not health.get(capability):
        raise AirflowSkipException(
            f"API reported {capability}=False; skipping {endpoint}. "
            "Check API startup logs or generate the required model artifacts."
        )
    return health


def call_prediction_endpoint() -> dict[str, Any]:
    _require_capability("model_loaded", "/predict")
    payload = {"ticker": API_TICKER, "horizon_days": API_HORIZON_DAYS}
    response = call_api("/predict", method="POST", payload=payload)
    result = {"request": payload, "status_code": response["status_code"], "body": response["body"]}
    logger.info("Prediction response: %s", result)
    return result


def call_agent_endpoint() -> dict[str, Any]:
    _require_capability("agent_ready", "/agent")
    payload = {"question": API_AGENT_QUESTION}
    response = call_api("/agent", method="POST", payload=payload)
    body = response["body"]
    if isinstance(body, dict) and "answer" in body:
        body = {**body, "answer": str(body["answer"])[:1000]}

    result = {"request": payload, "status_code": response["status_code"], "body": body}
    logger.info("Agent response summary: %s", result)
    return result


def call_infer_endpoint() -> dict[str, Any]:
    _require_capability("model_loaded", "/infer")
    payload = {
        "features": [
            [0.0 for _ in range(API_INFER_FEATURE_COUNT)] for _ in range(API_INFER_SEQUENCE_LENGTH)
        ]
    }
    response = call_api("/infer", method="POST", payload=payload, expected_statuses={200, 422})
    if response["status_code"] == 422:
        raise AirflowSkipException(
            "Payload padrao de /infer nao corresponde ao shape esperado pelo modelo. "
            "Ajuste API_INFER_SEQUENCE_LENGTH e API_INFER_FEATURE_COUNT no Compose."
        )

    result = {
        "request_shape": [API_INFER_SEQUENCE_LENGTH, API_INFER_FEATURE_COUNT],
        "status_code": response["status_code"],
        "body": response["body"],
    }
    logger.info("Raw infer response: %s", result)
    return result


def collect_api_metrics() -> dict[str, Any]:
    response = call_api("/metrics")
    metrics_text = response["body"] if isinstance(response["body"], str) else str(response["body"])
    metric_lines = [
        line for line in metrics_text.splitlines() if line.strip() and not line.startswith("#")
    ]
    result = {
        "status_code": response["status_code"],
        "content_type": response["content_type"],
        "metric_line_count": len(metric_lines),
        "sample_metrics": metric_lines[:10],
    }
    logger.info("Metrics endpoint summary: %s", result)
    return result


def generate_report(**context) -> dict[str, Any]:
    ti = context["task_instance"]

    report = {
        "run_date": datetime.now().isoformat(),
        "api_base_url": API_BASE_URL,
        "health": ti.xcom_pull(task_ids="check_api_health"),
        "training": ti.xcom_pull(task_ids="trigger_api_training"),
        "training_status": ti.xcom_pull(task_ids="wait_for_training_completion"),
        "prediction": ti.xcom_pull(task_ids="call_prediction_endpoint"),
        "agent": ti.xcom_pull(task_ids="call_agent_endpoint"),
        "infer": ti.xcom_pull(task_ids="call_infer_endpoint"),
        "metrics": ti.xcom_pull(task_ids="collect_api_metrics"),
    }

    os.makedirs(METRICS_DIR, exist_ok=True)
    report_path = os.path.join(
        METRICS_DIR,
        f"api_orchestration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False)

    logger.info("API orchestration report written to %s", report_path)
    return {"report_path": report_path, "report": report}


default_args = {
    "owner": "mlops-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="petr4_mlops_pipeline_lightweight",
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["datathon", "api", "compose", "mlops"],
) as dag:
    start = EmptyOperator(task_id="start")

    health = PythonOperator(
        task_id="check_api_health",
        python_callable=check_api_health,
    )

    train = PythonOperator(
        task_id="trigger_api_training",
        python_callable=trigger_api_training,
    )

    wait_training = PythonOperator(
        task_id="wait_for_training_completion",
        python_callable=wait_for_training_completion,
    )

    predict = PythonOperator(
        task_id="call_prediction_endpoint",
        python_callable=call_prediction_endpoint,
        trigger_rule=TriggerRule.NONE_FAILED,
    )

    agent = PythonOperator(
        task_id="call_agent_endpoint",
        python_callable=call_agent_endpoint,
        trigger_rule=TriggerRule.NONE_FAILED,
    )

    infer = PythonOperator(
        task_id="call_infer_endpoint",
        python_callable=call_infer_endpoint,
        trigger_rule=TriggerRule.NONE_FAILED,
    )

    metrics = PythonOperator(
        task_id="collect_api_metrics",
        python_callable=collect_api_metrics,
        trigger_rule=TriggerRule.NONE_FAILED,
    )

    report = PythonOperator(
        task_id="generate_report",
        python_callable=generate_report,
        trigger_rule=TriggerRule.NONE_FAILED,
    )

    end = EmptyOperator(task_id="end", trigger_rule=TriggerRule.NONE_FAILED)

    chain(start, health, train, wait_training, [predict, agent, infer, metrics], report, end)
