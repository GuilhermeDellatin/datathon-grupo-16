"""Detecção de drift com Evidently e PSI.

Implementa:
- Data drift em features de entrada
- Prediction drift na saída do modelo
- PSI (Population Stability Index) com thresholds configuráveis
- Integração com Prometheus para alertas

Referência Datathon:
- PSI > 0.1 = warning
- PSI > 0.2 = retrain trigger
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from evidently import Report
from evidently.presets import DataDriftPreset

logger = logging.getLogger(__name__)


def load_monitoring_config(config_path: str = "configs/monitoring_config.yaml") -> dict:
    """Carrega configuração de monitoramento.

    Args:
        config_path: Caminho para o arquivo YAML.

    Returns:
        Dicionário com configurações.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calcula Population Stability Index (PSI).

    Args:
        reference: Distribuição de referência (treino).
        current: Distribuição atual (produção).
        n_bins: Número de bins para discretização.

    Returns:
        PSI score. > 0.1 = warning, > 0.2 = significant drift.
    """
    # Criar bins baseados na referência
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    # Contar frequências
    ref_counts = np.histogram(reference, bins=bins)[0]
    cur_counts = np.histogram(current, bins=bins)[0]

    # Normalizar para proporções (evitar zeros)
    ref_pct = (ref_counts + 1) / (len(reference) + n_bins)
    cur_pct = (cur_counts + 1) / (len(current) + n_bins)

    # PSI
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

    return float(psi)


def run_drift_detection(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    config_path: str = "configs/monitoring_config.yaml",
    output_path: str = "metrics/drift_report.json",
) -> dict:
    """Executa detecção de drift com Evidently e PSI.

    Args:
        reference_data: Dados de referência (treino).
        current_data: Dados atuais (produção).
        config_path: Caminho da config de monitoramento.
        output_path: Caminho para salvar relatório.

    Returns:
        Dicionário com resultados de drift por feature.
    """
    config = load_monitoring_config(config_path)
    psi_warning = config["drift"]["psi_warning_threshold"]
    psi_retrain = config["drift"]["psi_retrain_threshold"]
    features_to_monitor = config["drift"]["features_to_monitor"]

    # Filtrar features existentes
    common_features = [
        f
        for f in features_to_monitor
        if f in reference_data.columns and f in current_data.columns
    ]

    results = {"features": {}, "overall_drift": False, "retrain_needed": False}

    # PSI por feature
    for feature in common_features:
        ref = reference_data[feature].dropna().values
        cur = current_data[feature].dropna().values

        if len(ref) < 10 or len(cur) < 10:
            logger.warning(
                "Feature '%s' com poucos dados para drift detection", feature
            )
            continue

        psi = calculate_psi(ref, cur)

        severity = "ok"
        if psi > psi_retrain:
            severity = "critical"
            results["retrain_needed"] = True
        elif psi > psi_warning:
            severity = "warning"
            results["overall_drift"] = True

        results["features"][feature] = {
            "psi": psi,
            "severity": severity,
            "threshold_warning": psi_warning,
            "threshold_retrain": psi_retrain,
        }

        # Atualizar Prometheus
        try:
            from src.monitoring.metrics import DRIFT_ALERT, MODEL_DRIFT_PSI

            MODEL_DRIFT_PSI.labels(feature=feature).set(psi)
            if severity != "ok":
                DRIFT_ALERT.labels(severity=severity).inc()
        except ImportError:
            pass

        logger.info("Drift %s: PSI=%.4f (%s)", feature, psi, severity)

    # Evidently report para detalhamento
    try:
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=reference_data[common_features],
            current_data=current_data[common_features],
        )
        evidently_result = report.as_dict()
        drift_share = evidently_result["metrics"][0]["result"][
            "share_of_drifted_columns"
        ]
        results["evidently_drift_share"] = drift_share
        logger.info("Evidently drift share: %.2f%%", drift_share * 100)
    except Exception as e:
        logger.warning("Erro no Evidently report: %s", e)

    # Salvar relatório
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Log no MLflow
    try:
        import mlflow

        mlflow.set_experiment("datathon-petr4-monitoring")
        with mlflow.start_run(run_name="drift-check"):
            for feat, data in results["features"].items():
                mlflow.log_metric(f"psi_{feat}", data["psi"])
            mlflow.log_metric("overall_drift", float(results["overall_drift"]))
            mlflow.log_artifact(output_path)
    except Exception as e:
        logger.warning("Não foi possível logar drift no MLflow: %s", e)

    return results


def main() -> None:
    """Entry point para drift detection standalone."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Carregar dados
    df = pd.read_parquet("data/processed/petr4_features.parquet")

    # Simular split referência vs atual
    split_idx = int(len(df) * 0.8)
    reference = df.iloc[:split_idx]
    current = df.iloc[split_idx:]

    results = run_drift_detection(reference, current)
    print(f"\nDrift Detection Results:\n{json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
