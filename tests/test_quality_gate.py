"""Testes do quality gate de sigma_coverage."""

import json
from pathlib import Path

import pytest
import yaml

from src.monitoring.quality_gates import (
    EXIT_ERROR,
    EXIT_GATE_FAILED,
    EXIT_OK,
    check_sigma_coverage_gate,
    evaluate_gates,
    load_gate_config,
    load_metrics,
    main,
)


class TestCheckSigmaCoverageGate:
    """Testes da função pura check_sigma_coverage_gate."""

    def test_passes_above_threshold(self):
        """Coverage acima do threshold deve passar."""
        passed, msg = check_sigma_coverage_gate({"sigma_coverage_0_5": 0.85}, threshold=0.70)
        assert passed
        assert "PASS" in msg

    def test_fails_below_threshold(self):
        """Coverage abaixo do threshold deve falhar."""
        passed, msg = check_sigma_coverage_gate({"sigma_coverage_0_5": 0.55}, threshold=0.70)
        assert not passed
        assert "FAIL" in msg

    def test_passes_at_exact_threshold(self):
        """Coverage exatamente no threshold deve passar (>=)."""
        passed, _ = check_sigma_coverage_gate({"sigma_coverage_0_5": 0.70}, threshold=0.70)
        assert passed

    def test_missing_metric_fails(self):
        """Falta da métrica deve falhar com mensagem explicativa."""
        passed, msg = check_sigma_coverage_gate({"mae": 1.0})
        assert not passed
        assert "ausente" in msg.lower()

    def test_non_numeric_metric_fails(self):
        """Valor não numérico deve falhar."""
        passed, msg = check_sigma_coverage_gate({"sigma_coverage_0_5": "alto"})
        assert not passed
        assert "numérica" in msg.lower()

    def test_out_of_range_fails(self):
        """Valor fora de [0, 1] deve falhar."""
        passed, msg = check_sigma_coverage_gate({"sigma_coverage_0_5": 1.5})
        assert not passed
        assert "intervalo" in msg.lower()

    def test_custom_metric_key(self):
        """Deve respeitar metric_key customizado."""
        passed, _ = check_sigma_coverage_gate(
            {"sigma_coverage_1_0": 0.95},
            threshold=0.70,
            metric_key="sigma_coverage_1_0",
        )
        assert passed


class TestLoadMetrics:
    """Testes do loader de métricas."""

    def test_loads_existing_file(self, tmp_path: Path):
        """Deve carregar JSON válido."""
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps({"sigma_coverage_0_5": 0.8}))
        loaded = load_metrics(str(metrics_file))
        assert loaded["sigma_coverage_0_5"] == 0.8

    def test_raises_when_missing(self, tmp_path: Path):
        """Deve levantar FileNotFoundError quando arquivo não existe."""
        with pytest.raises(FileNotFoundError):
            load_metrics(str(tmp_path / "absent.json"))


class TestLoadGateConfig:
    """Testes do loader de config."""

    def test_returns_block_when_present(self, tmp_path: Path):
        """Deve retornar o bloco quality_gates."""
        cfg_file = tmp_path / "monitoring.yaml"
        cfg_file.write_text(
            yaml.safe_dump(
                {
                    "quality_gates": {
                        "sigma_coverage": {
                            "metric_key": "sigma_coverage_0_5",
                            "min_coverage": 0.75,
                        }
                    }
                }
            )
        )
        cfg = load_gate_config(str(cfg_file))
        assert cfg["sigma_coverage"]["min_coverage"] == 0.75

    def test_returns_empty_when_missing_file(self, tmp_path: Path):
        """Arquivo ausente retorna dict vazio sem erro."""
        cfg = load_gate_config(str(tmp_path / "absent.yaml"))
        assert cfg == {}

    def test_returns_empty_when_no_block(self, tmp_path: Path):
        """YAML sem bloco quality_gates retorna dict vazio."""
        cfg_file = tmp_path / "monitoring.yaml"
        cfg_file.write_text(yaml.safe_dump({"drift": {"psi_warning_threshold": 0.1}}))
        cfg = load_gate_config(str(cfg_file))
        assert cfg == {}


class TestEvaluateGates:
    """Testes do orquestrador evaluate_gates."""

    def _write(self, tmp_path: Path, metrics: dict, threshold: float = 0.70) -> tuple[str, str]:
        """Helper para escrever metrics + config temporários."""
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps(metrics))
        cfg_file = tmp_path / "monitoring.yaml"
        cfg_file.write_text(
            yaml.safe_dump(
                {
                    "quality_gates": {
                        "sigma_coverage": {
                            "metric_key": "sigma_coverage_0_5",
                            "min_coverage": threshold,
                        }
                    }
                }
            )
        )
        return str(metrics_file), str(cfg_file)

    def test_pass(self, tmp_path: Path):
        """Coverage suficiente deve aprovar todos os gates."""
        m, c = self._write(tmp_path, {"sigma_coverage_0_5": 0.80})
        assert evaluate_gates(metrics_path=m, config_path=c) is True

    def test_fail(self, tmp_path: Path):
        """Coverage insuficiente deve reprovar."""
        m, c = self._write(tmp_path, {"sigma_coverage_0_5": 0.50})
        assert evaluate_gates(metrics_path=m, config_path=c) is False

    def test_threshold_override(self, tmp_path: Path):
        """threshold_override deve ter precedência sobre o YAML."""
        m, c = self._write(tmp_path, {"sigma_coverage_0_5": 0.80}, threshold=0.50)
        # Config diz 0.50 (passaria), override exige 0.90 (falha)
        assert evaluate_gates(metrics_path=m, config_path=c, threshold_override=0.90) is False


class TestMainCli:
    """Testes da CLI."""

    def _setup(self, tmp_path: Path, coverage: float) -> tuple[str, str]:
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps({"sigma_coverage_0_5": coverage}))
        cfg_file = tmp_path / "monitoring.yaml"
        cfg_file.write_text(
            yaml.safe_dump(
                {
                    "quality_gates": {
                        "sigma_coverage": {
                            "metric_key": "sigma_coverage_0_5",
                            "min_coverage": 0.70,
                        }
                    }
                }
            )
        )
        return str(metrics_file), str(cfg_file)

    def test_exit_ok_on_pass(self, tmp_path: Path):
        m, c = self._setup(tmp_path, coverage=0.85)
        assert main(["--metrics", m, "--config", c]) == EXIT_OK

    def test_exit_gate_failed_on_fail(self, tmp_path: Path):
        m, c = self._setup(tmp_path, coverage=0.40)
        assert main(["--metrics", m, "--config", c]) == EXIT_GATE_FAILED

    def test_exit_error_on_missing_metrics(self, tmp_path: Path):
        cfg_file = tmp_path / "monitoring.yaml"
        cfg_file.write_text(yaml.safe_dump({"quality_gates": {}}))
        assert (
            main(["--metrics", str(tmp_path / "missing.json"), "--config", str(cfg_file)])
            == EXIT_ERROR
        )
