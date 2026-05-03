"""Testes do endpoint POST /training/jobs e do AirflowClient.

Os testes não tocam Airflow, DVC, MLflow, yfinance ou subprocess — todas as
chamadas externas são substituídas por fakes/monkeypatch.
"""

from __future__ import annotations

from typing import Any

import pytest
import requests
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.services.airflow_client import (
    AirflowClient,
    AirflowClientError,
    get_airflow_client,
    get_training_dag_id,
)

VALID_PAYLOAD: dict[str, Any] = {
    "ticker": "petr4.sa",
    "period": "5y",
    "model_config": {
        "sequence_length": 60,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
    },
    "training_config": {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
    },
}


class FakeAirflowClient:
    """Fake do AirflowClient que captura chamadas para inspeção em testes."""

    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = response or {"dag_run_id": "ok", "state": "queued"}

    def trigger_dag_run(
        self,
        dag_id: str,
        dag_run_id: str,
        conf: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append({"dag_id": dag_id, "dag_run_id": dag_run_id, "conf": conf})
        return self.response


class FailingAirflowClient:
    def __init__(self, error: AirflowClientError) -> None:
        self.error = error

    def trigger_dag_run(
        self,
        dag_id: str,
        dag_run_id: str,
        conf: dict[str, Any],
    ) -> dict[str, Any]:
        raise self.error


@pytest.fixture
def fake_airflow():
    """Substitui o cliente Airflow por um fake durante o teste."""
    fake = FakeAirflowClient()
    app.dependency_overrides[get_airflow_client] = lambda: fake
    try:
        yield fake
    finally:
        app.dependency_overrides.pop(get_airflow_client, None)


@pytest.fixture
def client():
    return TestClient(app)


def _override_with(failing: FailingAirflowClient) -> None:
    app.dependency_overrides[get_airflow_client] = lambda: failing


class TestTrainingJobsEndpoint:
    """Testa POST /training/jobs."""

    def test_returns_202_with_full_payload(self, client, fake_airflow):
        response = client.post("/training/jobs", json=VALID_PAYLOAD)
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "queued"
        assert body["ticker"] == "PETR4.SA"
        assert body["airflow_dag_id"] == get_training_dag_id()
        assert body["job_id"] == body["airflow_dag_run_id"]
        assert body["job_id"].startswith("train_")
        assert "petr4_sa" in body["job_id"]

    def test_dispatches_with_dag_run_id_equal_to_job_id(self, client, fake_airflow):
        response = client.post("/training/jobs", json=VALID_PAYLOAD)
        assert response.status_code == 202
        assert len(fake_airflow.calls) == 1
        call = fake_airflow.calls[0]
        body = response.json()
        assert call["dag_run_id"] == body["job_id"]
        assert call["dag_id"] == get_training_dag_id()

    def test_conf_preserves_model_config_alias(self, client, fake_airflow):
        client.post("/training/jobs", json=VALID_PAYLOAD)
        conf = fake_airflow.calls[0]["conf"]
        # `model_config` (com alias) deve ser preservado no payload da DAG.
        assert "model_config" in conf
        assert "architecture" not in conf
        assert conf["model_config"]["sequence_length"] == 60
        assert conf["training_config"]["epochs"] == 100
        assert conf["ticker"] == "PETR4.SA"
        assert conf["period"] == "5y"
        assert conf["job_id"].startswith("train_")

    def test_invalid_period_returns_422(self, client, fake_airflow):
        bad = {**VALID_PAYLOAD, "period": "7y"}
        response = client.post("/training/jobs", json=bad)
        assert response.status_code == 422

    def test_missing_field_returns_422(self, client, fake_airflow):
        partial = {k: v for k, v in VALID_PAYLOAD.items() if k != "training_config"}
        response = client.post("/training/jobs", json=partial)
        assert response.status_code == 422

    def test_dropout_out_of_range_returns_422(self, client, fake_airflow):
        bad = {
            **VALID_PAYLOAD,
            "model_config": {**VALID_PAYLOAD["model_config"], "dropout": 0.95},
        }
        response = client.post("/training/jobs", json=bad)
        assert response.status_code == 422

    def test_learning_rate_zero_returns_422(self, client, fake_airflow):
        bad = {
            **VALID_PAYLOAD,
            "training_config": {
                **VALID_PAYLOAD["training_config"],
                "learning_rate": 0.0,
            },
        }
        response = client.post("/training/jobs", json=bad)
        assert response.status_code == 422

    def test_empty_ticker_returns_422(self, client, fake_airflow):
        bad = {**VALID_PAYLOAD, "ticker": ""}
        response = client.post("/training/jobs", json=bad)
        assert response.status_code == 422

    def test_airflow_unavailable_returns_503(self, client):
        try:
            _override_with(
                FailingAirflowClient(
                    AirflowClientError("airflow down", status_code=503)
                )
            )
            response = client.post("/training/jobs", json=VALID_PAYLOAD)
            assert response.status_code == 503
            assert "airflow down" in response.json()["detail"].lower()
        finally:
            app.dependency_overrides.pop(get_airflow_client, None)

    def test_airflow_bad_gateway_returns_502(self, client):
        try:
            _override_with(
                FailingAirflowClient(
                    AirflowClientError("invalid response", status_code=502)
                )
            )
            response = client.post("/training/jobs", json=VALID_PAYLOAD)
            assert response.status_code == 502
        finally:
            app.dependency_overrides.pop(get_airflow_client, None)

    def test_airflow_conflict_returns_409(self, client):
        try:
            _override_with(
                FailingAirflowClient(
                    AirflowClientError("duplicate", status_code=409)
                )
            )
            response = client.post("/training/jobs", json=VALID_PAYLOAD)
            assert response.status_code == 409
        finally:
            app.dependency_overrides.pop(get_airflow_client, None)


# ---------------------------------------------------------------------------
# AirflowClient — testes unitários com fake `requests.Session`
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int, payload: Any | None = None) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = "" if payload is None else str(payload)

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeSession:
    def __init__(self, response: _FakeResponse | None = None, exc: Exception | None = None):
        self.response = response
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if self.exc is not None:
            raise self.exc
        assert self.response is not None
        return self.response


class TestAirflowClient:
    def _make_client(
        self,
        response: _FakeResponse | None = None,
        exc: Exception | None = None,
    ) -> tuple[AirflowClient, _FakeSession]:
        session = _FakeSession(response=response, exc=exc)
        client = AirflowClient(
            base_url="http://airflow.test",
            username="u",
            password="p",
            timeout=1.0,
            session=session,  # type: ignore[arg-type]
        )
        return client, session

    def test_trigger_success_returns_payload(self):
        client, session = self._make_client(
            response=_FakeResponse(200, {"dag_run_id": "abc"})
        )
        result = client.trigger_dag_run(
            dag_id="train_lstm_stock",
            dag_run_id="abc",
            conf={"job_id": "abc"},
        )
        assert result == {"dag_run_id": "abc"}
        assert session.calls[0]["url"] == (
            "http://airflow.test/api/v1/dags/train_lstm_stock/dagRuns"
        )
        body = session.calls[0]["json"]
        assert body == {"dag_run_id": "abc", "conf": {"job_id": "abc"}}

    def test_trigger_201_treated_as_success(self):
        client, _ = self._make_client(
            response=_FakeResponse(201, {"dag_run_id": "abc"})
        )
        assert client.trigger_dag_run("d", "abc", {}) == {"dag_run_id": "abc"}

    def test_trigger_409_raises_conflict(self):
        client, _ = self._make_client(
            response=_FakeResponse(409, {"detail": "exists"})
        )
        with pytest.raises(AirflowClientError) as exc_info:
            client.trigger_dag_run("d", "abc", {})
        assert exc_info.value.status_code == 409

    def test_trigger_401_maps_to_502(self):
        client, _ = self._make_client(response=_FakeResponse(401))
        with pytest.raises(AirflowClientError) as exc_info:
            client.trigger_dag_run("d", "abc", {})
        assert exc_info.value.status_code == 502

    def test_trigger_500_maps_to_502(self):
        client, _ = self._make_client(response=_FakeResponse(500))
        with pytest.raises(AirflowClientError) as exc_info:
            client.trigger_dag_run("d", "abc", {})
        assert exc_info.value.status_code == 502

    def test_trigger_connection_error_maps_to_503(self):
        client, _ = self._make_client(
            exc=requests.exceptions.ConnectionError("refused")
        )
        with pytest.raises(AirflowClientError) as exc_info:
            client.trigger_dag_run("d", "abc", {})
        assert exc_info.value.status_code == 503

    def test_trigger_timeout_maps_to_504(self):
        client, _ = self._make_client(exc=requests.exceptions.Timeout("slow"))
        with pytest.raises(AirflowClientError) as exc_info:
            client.trigger_dag_run("d", "abc", {})
        assert exc_info.value.status_code == 504

    def test_invalid_json_in_response_returns_empty_dict(self):
        client, _ = self._make_client(
            response=_FakeResponse(200, payload=ValueError("nope"))
        )
        result = client.trigger_dag_run("d", "abc", {})
        assert result == {}

    def test_get_training_dag_id_default(self, monkeypatch):
        monkeypatch.delenv("AIRFLOW_TRAINING_DAG_ID", raising=False)
        assert get_training_dag_id() == "train_lstm_stock"

    def test_get_training_dag_id_env_override(self, monkeypatch):
        monkeypatch.setenv("AIRFLOW_TRAINING_DAG_ID", "custom_dag")
        assert get_training_dag_id() == "custom_dag"
