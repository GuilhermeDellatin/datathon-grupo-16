"""Testes do bridge entre o request da API e os stages do DVC."""

from __future__ import annotations

import yaml

from src.training_jobs import (
    TRAINING_CONFIG_PATH_ENV,
    TRAINING_PERIOD_ENV,
    TRAINING_TICKER_ENV,
    conf_to_train_overrides,
    load_training_job_conf,
    resolve_collection_params,
)

SAMPLE_CONF = {
    "job_id": "train_20260502_petr4_abc123",
    "ticker": "PETR4.SA",
    "period": "5y",
    "model_config": {
        "sequence_length": 90,
        "hidden_size": 256,
        "num_layers": 3,
        "dropout": 0.3,
    },
    "training_config": {
        "epochs": 25,
        "batch_size": 16,
        "learning_rate": 0.0005,
    },
}


class TestLoadTrainingJobConf:
    def test_returns_none_when_env_unset(self, monkeypatch):
        monkeypatch.delenv(TRAINING_CONFIG_PATH_ENV, raising=False)
        assert load_training_job_conf() is None

    def test_returns_none_when_path_missing(self, tmp_path, monkeypatch):
        missing = tmp_path / "absent.yaml"
        monkeypatch.setenv(TRAINING_CONFIG_PATH_ENV, str(missing))
        assert load_training_job_conf() is None

    def test_returns_none_when_yaml_not_dict(self, tmp_path, monkeypatch):
        bad = tmp_path / "bad.yaml"
        bad.write_text("- 1\n- 2\n")
        monkeypatch.setenv(TRAINING_CONFIG_PATH_ENV, str(bad))
        assert load_training_job_conf() is None

    def test_loads_yaml_when_path_valid(self, tmp_path, monkeypatch):
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump(SAMPLE_CONF))
        monkeypatch.setenv(TRAINING_CONFIG_PATH_ENV, str(config))

        result = load_training_job_conf()

        assert result is not None
        assert result["job_id"] == SAMPLE_CONF["job_id"]
        assert result["model_config"]["hidden_size"] == 256

    def test_explicit_path_wins_over_env(self, tmp_path, monkeypatch):
        config = tmp_path / "explicit.yaml"
        config.write_text(yaml.safe_dump(SAMPLE_CONF))
        monkeypatch.setenv(TRAINING_CONFIG_PATH_ENV, "/does/not/exist.yaml")

        result = load_training_job_conf(str(config))

        assert result is not None
        assert result["ticker"] == "PETR4.SA"


class TestConfToTrainOverrides:
    def test_full_conf_maps_all_fields(self):
        overrides = conf_to_train_overrides(SAMPLE_CONF)

        assert overrides["ticker"] == "PETR4.SA"
        assert overrides["mlflow"]["tags"]["ticker"] == "PETR4.SA"
        assert overrides["features"]["sequence_length"] == 90
        assert overrides["model"] == {
            "hidden_size": 256,
            "num_layers": 3,
            "dropout": 0.3,
        }
        assert overrides["training"] == {
            "epochs": 25,
            "batch_size": 16,
            "learning_rate": 0.0005,
        }

    def test_empty_conf_returns_empty_overrides(self):
        assert conf_to_train_overrides({}) == {}

    def test_partial_conf_only_emits_present_fields(self):
        partial = {
            "ticker": "VALE3.SA",
            "training_config": {"epochs": 10},
        }

        overrides = conf_to_train_overrides(partial)

        assert overrides["ticker"] == "VALE3.SA"
        assert overrides["mlflow"]["tags"]["ticker"] == "VALE3.SA"
        assert overrides["training"] == {"epochs": 10}
        assert "model" not in overrides
        assert "features" not in overrides

    def test_period_is_not_propagated_to_train_overrides(self):
        overrides = conf_to_train_overrides({"period": "5y"})
        assert overrides == {}

    def test_none_values_are_ignored(self):
        conf = {
            "model_config": {
                "sequence_length": None,
                "hidden_size": 128,
                "num_layers": None,
                "dropout": None,
            },
            "training_config": {
                "epochs": None,
                "batch_size": 64,
                "learning_rate": None,
            },
        }

        overrides = conf_to_train_overrides(conf)

        assert overrides["model"] == {"hidden_size": 128}
        assert overrides["training"] == {"batch_size": 64}
        assert "features" not in overrides


class TestResolveCollectionParams:
    def test_falls_back_to_yaml_when_no_job(self, monkeypatch):
        monkeypatch.delenv(TRAINING_TICKER_ENV, raising=False)
        monkeypatch.delenv(TRAINING_PERIOD_ENV, raising=False)
        monkeypatch.delenv(TRAINING_CONFIG_PATH_ENV, raising=False)

        ticker, start, end, period = resolve_collection_params(
            fallback_ticker="PETR4.SA",
            fallback_start="2020-01-01",
            fallback_end="2025-12-31",
        )

        assert ticker == "PETR4.SA"
        assert start == "2020-01-01"
        assert end == "2025-12-31"
        assert period is None

    def test_env_period_overrides_dates(self, monkeypatch):
        monkeypatch.setenv(TRAINING_TICKER_ENV, "VALE3.SA")
        monkeypatch.setenv(TRAINING_PERIOD_ENV, "1y")
        monkeypatch.delenv(TRAINING_CONFIG_PATH_ENV, raising=False)

        ticker, start, end, period = resolve_collection_params(
            fallback_ticker="PETR4.SA",
            fallback_start="2020-01-01",
            fallback_end="2025-12-31",
        )

        assert ticker == "VALE3.SA"
        assert period == "1y"
        assert start is None
        assert end is None

    def test_yaml_path_used_when_env_partial(self, tmp_path, monkeypatch):
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump(SAMPLE_CONF))
        monkeypatch.delenv(TRAINING_TICKER_ENV, raising=False)
        monkeypatch.delenv(TRAINING_PERIOD_ENV, raising=False)
        monkeypatch.setenv(TRAINING_CONFIG_PATH_ENV, str(config))

        ticker, start, end, period = resolve_collection_params(
            fallback_ticker="DEFAULT.SA",
            fallback_start="2020-01-01",
            fallback_end="2025-12-31",
        )

        assert ticker == "PETR4.SA"
        assert period == "5y"
        assert start is None
        assert end is None

    def test_env_ticker_wins_over_yaml(self, tmp_path, monkeypatch):
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump(SAMPLE_CONF))
        monkeypatch.setenv(TRAINING_TICKER_ENV, "OVERRIDE.SA")
        monkeypatch.delenv(TRAINING_PERIOD_ENV, raising=False)
        monkeypatch.setenv(TRAINING_CONFIG_PATH_ENV, str(config))

        ticker, _, _, period = resolve_collection_params(
            fallback_ticker="DEFAULT.SA",
            fallback_start="2020-01-01",
            fallback_end="2025-12-31",
        )

        assert ticker == "OVERRIDE.SA"
        assert period == "5y"
