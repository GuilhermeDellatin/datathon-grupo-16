"""Integração real com o MLflow Model Registry."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from src.paths import resolve_project_file

logger = logging.getLogger(__name__)

DEFAULT_METADATA_PATH = "metrics/latest_training_run.json"


def load_training_metadata(metadata_path: str | Path = DEFAULT_METADATA_PATH) -> dict:
    """Carrega metadata do último treino persistida em disco."""
    resolved_path = resolve_project_file(metadata_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Arquivo de metadata do treino não encontrado: {resolved_path}")

    with open(resolved_path, encoding="utf-8") as f:
        return json.load(f)


def find_registered_version(client: MlflowClient, model_name: str, run_id: str):
    """Retorna a versão já registrada para o run, se existir."""
    for version in client.search_model_versions(f"name='{model_name}'"):
        if version.run_id == run_id:
            return version
    return None


def register_trained_model(
    model_name: str,
    run_id: str,
    model_uri: str | None = None,
    stage: str | None = "Staging",
) -> dict[str, str]:
    """Registra um modelo treinado no MLflow Model Registry."""
    client = MlflowClient()
    effective_model_uri = model_uri or f"runs:/{run_id}/model"

    existing_version = find_registered_version(client, model_name, run_id)
    if existing_version is not None:
        version = existing_version
        logger.info(
            "Run %s já está registrado como %s v%s",
            run_id,
            model_name,
            version.version,
        )
    else:
        version = mlflow.register_model(effective_model_uri, model_name)
        logger.info("Modelo registrado: %s v%s", version.name, version.version)

    client.set_model_version_tag(model_name, version.version, "source_run_id", run_id)
    client.set_model_version_tag(model_name, version.version, "registration_source", "airflow_dag")

    if stage:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage=stage,
            archive_existing_versions=False,
        )
        logger.info("Modelo %s v%s promovido para stage %s", model_name, version.version, stage)

    result = {
        "name": model_name,
        "version": str(version.version),
        "run_id": run_id,
        "model_uri": effective_model_uri,
        "stage": stage or "",
    }

    registry_output_path = resolve_project_file("metrics/latest_registered_model.json")
    registry_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info("Resultado do registro salvo em %s", registry_output_path)
    return result


def main() -> None:
    """CLI para registro do último treino no MLflow Model Registry."""
    parser = argparse.ArgumentParser(description="Registra um modelo treinado no MLflow Registry.")
    parser.add_argument(
        "--metadata-path",
        default=DEFAULT_METADATA_PATH,
        help="Caminho do JSON com metadata do treino.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Sobrescreve o nome do modelo obtido da metadata.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Sobrescreve o run_id obtido da metadata.",
    )
    parser.add_argument(
        "--model-uri",
        default=None,
        help="Sobrescreve o model_uri obtido da metadata.",
    )
    parser.add_argument(
        "--stage",
        default="Staging",
        help="Stage de destino após o registro. Use vazio para não promover.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    metadata = load_training_metadata(args.metadata_path)
    result = register_trained_model(
        model_name=args.model_name or metadata["model_name"],
        run_id=args.run_id or metadata["run_id"],
        model_uri=args.model_uri or metadata.get("model_uri"),
        stage=args.stage or None,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
