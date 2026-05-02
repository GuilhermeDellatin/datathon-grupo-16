"""Utilitários para registro de modelos no MLflow Model Registry."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "lstm-petr4"
DEFAULT_ARTIFACT_PATH = "model"


def _import_mlflow():
    """Importa MLflow apenas quando uma operação de registry for executada."""
    import mlflow

    return mlflow


def _default_model_name(config_path: str = "configs/model_config.yaml") -> str:
    """Resolve o nome do modelo por env var, config YAML ou fallback local."""
    env_model_name = os.getenv("MLFLOW_MODEL_NAME")
    if env_model_name:
        return env_model_name

    try:
        from src.data.feature_engineering import load_config

        config = load_config(config_path)
        return str(config.get("mlflow", {}).get("model_name") or DEFAULT_MODEL_NAME)
    except Exception:
        logger.warning("Nao foi possivel carregar model_name da config; usando fallback.")
        return DEFAULT_MODEL_NAME


class MLflowModelRegistry:
    """Camada fina para registrar modelos no MLflow sem acoplar treino/API/DAG."""

    def __init__(self, tracking_uri: str | None = None):
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")

    def _configure_tracking(self, mlflow_module) -> None:
        if self.tracking_uri:
            mlflow_module.set_tracking_uri(self.tracking_uri)

    def register_model(self, model_uri: str, model_name: str) -> dict[str, str]:
        """Registra um model_uri no MLflow Model Registry."""
        mlflow_module = _import_mlflow()
        self._configure_tracking(mlflow_module)

        client = mlflow_module.tracking.MlflowClient()
        # Ensure registered model exists
        try:
            client.create_registered_model(model_name)
        except Exception:
            # Already exists or creation failed; continue to create a version
            logger.debug("Registered model '%s' may already exist", model_name)

        # Create a new model version from the logged model URI
        mv = client.create_model_version(name=model_name, source=model_uri)

        info = {
            "name": str(mv.name),
            "version": str(mv.version),
            "run_id": str(getattr(mv, "run_id", "") or ""),
            "model_uri": model_uri,
        }
        logger.info(
            "Modelo registrado no MLflow: %s v%s",
            info["name"],
            info["version"],
        )
        return info

    def register_run(
        self,
        run_id: str,
        model_name: str | None = None,
        artifact_path: str = DEFAULT_ARTIFACT_PATH,
    ) -> dict[str, str]:
        """Registra o artefato de modelo produzido por uma run do MLflow."""
        if not run_id:
            raise ValueError("run_id e obrigatorio para registrar modelo")

        resolved_model_name = model_name or _default_model_name()
        model_uri = f"runs:/{run_id}/{artifact_path}"
        info = self.register_model(model_uri, resolved_model_name)
        info["run_id"] = run_id
        return info


def register_model_run(
    run_id: str,
    model_name: str | None = None,
    tracking_uri: str | None = None,
    artifact_path: str = DEFAULT_ARTIFACT_PATH,
) -> dict[str, str]:
    """Registra no MLflow Registry o modelo logado em uma run de treino."""
    registry = MLflowModelRegistry(tracking_uri=tracking_uri)
    return registry.register_run(
        run_id=run_id,
        model_name=model_name,
        artifact_path=artifact_path,
    )
