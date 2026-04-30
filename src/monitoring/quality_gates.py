"""Quality gates para promoção de modelos LSTM.

Verifica se métricas críticas atendem aos thresholds antes de promover
um modelo. Sai com exit code 1 se algum gate falha — pode ser usado em
CI, pipelines DVC ou tarefas Make para bloquear modelos ruins.

Gate principal (Datathon Fase 05):
- sigma_coverage_0_5 >= 0.70
- "≥ 70 % das predições dentro de 0.5σ do preço observado"
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_METRICS_PATH = "metrics/train_metrics.json"
DEFAULT_CONFIG_PATH = "configs/monitoring_config.yaml"
DEFAULT_SIGMA_COVERAGE_THRESHOLD = 0.70
DEFAULT_SIGMA_COVERAGE_KEY = "sigma_coverage_0_5"

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_ERROR = 2


def load_metrics(metrics_path: str = DEFAULT_METRICS_PATH) -> dict[str, Any]:
    """Carrega métricas de treino de um arquivo JSON.

    Args:
        metrics_path: Caminho para o arquivo de métricas.

    Returns:
        Dicionário com métricas.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    path = Path(metrics_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de métricas não encontrado: {metrics_path}. "
            "Rode `make train` antes do quality gate."
        )
    with open(path) as f:
        return json.load(f)


def load_gate_config(
    config_path: str = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Carrega o bloco quality_gates do monitoring_config.yaml.

    Args:
        config_path: Caminho do YAML de monitoramento.

    Returns:
        Dicionário do bloco `quality_gates`. Retorna defaults se ausente.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config %s não encontrado. Usando defaults.", config_path)
        return {}
    with open(path) as f:
        cfg: dict = yaml.safe_load(f) or {}
    return cfg.get("quality_gates", {})


def check_sigma_coverage_gate(
    metrics: dict[str, Any],
    threshold: float = DEFAULT_SIGMA_COVERAGE_THRESHOLD,
    metric_key: str = DEFAULT_SIGMA_COVERAGE_KEY,
) -> tuple[bool, str]:
    """Verifica se a cobertura sigma atende ao gate.

    Args:
        metrics: Dicionário com métricas (de `metrics/train_metrics.json`).
        threshold: Cobertura mínima exigida em [0, 1] (default 0.70).
        metric_key: Chave em metrics com a coverage observada.

    Returns:
        Tupla (passed, message).
    """
    if metric_key not in metrics:
        return (
            False,
            f"Métrica '{metric_key}' ausente — execute o treinamento atualizado",
        )

    coverage_raw = metrics[metric_key]
    try:
        coverage = float(coverage_raw)
    except (TypeError, ValueError):
        return False, f"Métrica '{metric_key}' não é numérica: {coverage_raw!r}"

    if not 0.0 <= coverage <= 1.0:
        return False, f"Métrica '{metric_key}' fora do intervalo [0, 1]: {coverage}"

    passed = coverage >= threshold
    message = (
        f"sigma_coverage={coverage * 100:.2f}% "
        f"(alvo >= {threshold * 100:.2f}%) → {'PASS' if passed else 'FAIL'}"
    )
    return passed, message


def evaluate_gates(
    metrics_path: str = DEFAULT_METRICS_PATH,
    config_path: str = DEFAULT_CONFIG_PATH,
    threshold_override: float | None = None,
) -> bool:
    """Avalia todos os quality gates configurados.

    Args:
        metrics_path: Caminho do JSON de métricas.
        config_path: Caminho do YAML de monitoramento.
        threshold_override: Se fornecido, sobrescreve o threshold do config.

    Returns:
        True se TODOS os gates passaram.
    """
    metrics = load_metrics(metrics_path)
    gate_cfg = load_gate_config(config_path)

    sigma_cfg = gate_cfg.get("sigma_coverage", {})
    threshold = (
        threshold_override
        if threshold_override is not None
        else float(sigma_cfg.get("min_coverage", DEFAULT_SIGMA_COVERAGE_THRESHOLD))
    )
    metric_key = str(sigma_cfg.get("metric_key", DEFAULT_SIGMA_COVERAGE_KEY))

    sigma_passed, sigma_msg = check_sigma_coverage_gate(
        metrics, threshold=threshold, metric_key=metric_key
    )
    logger.info("[gate sigma_coverage] %s", sigma_msg)

    all_passed = sigma_passed
    if all_passed:
        logger.info("Todos os quality gates passaram.")
    else:
        logger.error(
            "Quality gate falhou. Modelo NÃO deve ser promovido para produção."
        )
    return all_passed


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argumentos da linha de comando (default sys.argv).

    Returns:
        Exit code: 0 = passou, 1 = gate falhou, 2 = erro.
    """
    parser = argparse.ArgumentParser(description="Avalia quality gates do modelo LSTM.")
    parser.add_argument(
        "--metrics", default=DEFAULT_METRICS_PATH, help="Caminho de train_metrics.json"
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH, help="Caminho de monitoring_config.yaml"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold mínimo de sigma_coverage (sobrescreve YAML)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    try:
        passed = evaluate_gates(
            metrics_path=args.metrics,
            config_path=args.config,
            threshold_override=args.threshold,
        )
    except FileNotFoundError as e:
        logger.error("%s", e)
        return EXIT_ERROR
    except Exception as e:
        logger.error("Erro ao avaliar gates: %s", e, exc_info=True)
        return EXIT_ERROR

    return EXIT_OK if passed else EXIT_GATE_FAILED


if __name__ == "__main__":
    sys.exit(main())
