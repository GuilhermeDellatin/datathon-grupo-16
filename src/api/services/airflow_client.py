"""Cliente HTTP para disparar DAG runs no Airflow via REST API.

Implementa apenas o subset necessário para o endpoint POST /training/jobs:
disparo síncrono de uma DAG run (`POST /api/v1/dags/{dag_id}/dagRuns`).

A configuração é lida do ambiente para evitar acoplamento com `configs/`:

- ``AIRFLOW_BASE_URL``         (default: ``http://localhost:8080``)
- ``AIRFLOW_USERNAME``         (default: ``admin``)
- ``AIRFLOW_PASSWORD``         (default: ``admin``)
- ``AIRFLOW_TRAINING_DAG_ID``  (default: ``train_lstm_stock``)
- ``AIRFLOW_REQUEST_TIMEOUT``  (default: ``10`` segundos)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests  # type: ignore[import-untyped]
from requests.auth import HTTPBasicAuth  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

DEFAULT_TRAINING_DAG_ID = "train_lstm_stock"
DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "admin"  # nosec B105 - default for local dev only
DEFAULT_TIMEOUT = 10.0


class AirflowClientError(RuntimeError):
    """Erro genérico de comunicação com o Airflow.

    Attributes:
        status_code: Sugestão de status HTTP a propagar pela API
            (502 para falha de comunicação, 503 quando o serviço está fora).
    """

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


def get_training_dag_id() -> str:
    """Retorna o DAG id configurado para o pipeline de treino."""
    return os.getenv("AIRFLOW_TRAINING_DAG_ID", DEFAULT_TRAINING_DAG_ID)


class AirflowClient:
    """Wrapper enxuto da REST API do Airflow.

    Usa autenticação básica (compatível com o webserver default do Airflow 2.x)
    e mantém uma `requests.Session` por instância para reuso de conexões.
    """

    def __init__(
        self,
        base_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        timeout: float | None = None,
        session: requests.Session | None = None,
    ) -> None:
        resolved_base = base_url or os.getenv("AIRFLOW_BASE_URL") or DEFAULT_BASE_URL
        self.base_url = resolved_base.rstrip("/")
        self.username = username or os.getenv("AIRFLOW_USERNAME") or DEFAULT_USERNAME
        self.password = password or os.getenv("AIRFLOW_PASSWORD") or DEFAULT_PASSWORD
        env_timeout = os.getenv("AIRFLOW_REQUEST_TIMEOUT")
        if timeout is not None:
            self.timeout = timeout
        elif env_timeout:
            try:
                self.timeout = float(env_timeout)
            except ValueError:
                self.timeout = DEFAULT_TIMEOUT
        else:
            self.timeout = DEFAULT_TIMEOUT
        self._session = session or requests.Session()

    def _auth(self) -> HTTPBasicAuth:
        return HTTPBasicAuth(self.username, self.password)

    def trigger_dag_run(
        self,
        dag_id: str,
        dag_run_id: str,
        conf: dict[str, Any],
    ) -> dict[str, Any]:
        """Cria uma DAG run no Airflow.

        Args:
            dag_id: id da DAG alvo (ex.: ``train_lstm_stock``).
            dag_run_id: id estável do run; deve ser igual ao ``job_id`` da API.
            conf: payload (`dag_run.conf`) repassado às tasks.

        Returns:
            Corpo JSON da resposta do Airflow quando o run é aceito.

        Raises:
            AirflowClientError: Quando há falha de comunicação ou o Airflow
                retorna status diferente de 200/201/409. ``status_code`` na
                exceção é 503 quando o servidor está indisponível e 502 nos
                demais erros, para que a camada HTTP traduza para o status
                correto de `502 Bad Gateway` / `503 Service Unavailable`.
        """
        url = f"{self.base_url}/api/v1/dags/{dag_id}/dagRuns"
        payload = {"dag_run_id": dag_run_id, "conf": conf}

        logger.info(
            "Disparando DAG run no Airflow: dag_id=%s dag_run_id=%s url=%s",
            dag_id,
            dag_run_id,
            url,
        )

        try:
            response = self._session.post(
                url,
                json=payload,
                auth=self._auth(),
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
        except requests.exceptions.ConnectionError as exc:
            logger.error("Falha de conexão com Airflow em %s: %s", url, exc)
            raise AirflowClientError(
                f"Não foi possível conectar ao Airflow em {self.base_url}: {exc}",
                status_code=503,
            ) from exc
        except requests.exceptions.Timeout as exc:
            logger.error("Timeout ao chamar Airflow em %s: %s", url, exc)
            raise AirflowClientError(
                f"Timeout ao comunicar com o Airflow ({self.timeout}s)",
                status_code=504,
            ) from exc
        except requests.exceptions.RequestException as exc:
            logger.error("Erro inesperado ao chamar Airflow: %s", exc)
            raise AirflowClientError(
                f"Erro de comunicação com Airflow: {exc}",
                status_code=502,
            ) from exc

        if response.status_code in (200, 201):
            return self._parse_json(response)

        if response.status_code == 409:
            raise AirflowClientError(
                f"Já existe DAG run com id '{dag_run_id}' na DAG '{dag_id}'.",
                status_code=409,
            )

        if response.status_code in (401, 403):
            raise AirflowClientError(
                "Credenciais do Airflow inválidas (verifique "
                "AIRFLOW_USERNAME/AIRFLOW_PASSWORD).",
                status_code=502,
            )

        raise AirflowClientError(
            f"Airflow respondeu com status {response.status_code}: {response.text[:200]}",
            status_code=502,
        )

    @staticmethod
    def _parse_json(response: requests.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError:
            return {}
        if isinstance(data, dict):
            return data
        return {}


def get_airflow_client() -> AirflowClient:
    """Factory utilizada pelas rotas FastAPI (facilita override em testes)."""
    return AirflowClient()
