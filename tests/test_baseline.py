"""Testes dos baselines (Ridge + MLP) para benchmark do LSTM."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.baseline import (
    MLPRegressor,
    _build_comparison,
    evaluate_predictions,
    prepare_baseline_data,
    to_price_scale,
    train_mlp_baseline,
    train_ridge,
)


class TestMLPRegressor:
    """Smoke tests do módulo PyTorch."""

    def test_forward_shape(self):
        """Saída deve ter shape (batch_size,)."""
        model = MLPRegressor(input_size=20, hidden_sizes=[16, 8], dropout=0.0)
        x = torch.randn(4, 20)
        out = model(x)
        assert out.shape == (4,)

    def test_default_hidden_sizes(self):
        """Default deve usar [128, 64]."""
        model = MLPRegressor(input_size=10)
        x = torch.randn(2, 10)
        out = model(x)
        assert out.shape == (2,)

    def test_no_dropout_when_zero(self):
        """dropout=0 não deve adicionar camadas Dropout."""
        model = MLPRegressor(input_size=10, hidden_sizes=[8], dropout=0.0)
        has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.net)
        assert not has_dropout


class TestPrepareBaselineData:
    """Testes do pipeline de dados achatado."""

    def test_shape_is_flat(self, sample_features_data: pd.DataFrame):
        """X deve estar achatado em (n_samples, seq_len * n_features)."""
        feature_cols = ["Close", "Volume", "rsi_14"]
        prepared = prepare_baseline_data(
            df=sample_features_data,
            feature_cols=feature_cols,
            sequence_length=10,
            prediction_horizon=1,
            train_ratio=0.7,
            val_ratio=0.15,
        )
        for X, _ in prepared["splits"].values():
            assert X.ndim == 2
            assert X.shape[1] == 10 * len(feature_cols)
        assert prepared["input_size"] == 10 * len(feature_cols)

    def test_close_range_within_data(self, sample_features_data: pd.DataFrame):
        """close_min/close_max devem refletir os valores reais da Close."""
        feature_cols = ["Close", "Volume", "rsi_14"]
        prepared = prepare_baseline_data(
            df=sample_features_data,
            feature_cols=feature_cols,
            sequence_length=10,
            prediction_horizon=1,
            train_ratio=0.7,
            val_ratio=0.15,
        )
        close = sample_features_data["Close"].values
        assert prepared["close_min"] == pytest.approx(float(close.min()))
        assert prepared["close_max"] == pytest.approx(float(close.max()))

    def test_temporal_ordering_preserved(self, sample_features_data: pd.DataFrame):
        """Train deve preceder val que precede test (sem shuffle)."""
        feature_cols = ["Close", "Volume", "rsi_14"]
        prepared = prepare_baseline_data(
            df=sample_features_data,
            feature_cols=feature_cols,
            sequence_length=10,
            prediction_horizon=1,
            train_ratio=0.7,
            val_ratio=0.15,
        )
        n_train = prepared["splits"]["train"][0].shape[0]
        n_val = prepared["splits"]["val"][0].shape[0]
        n_test = prepared["splits"]["test"][0].shape[0]
        assert n_train > 0 and n_val > 0 and n_test > 0


class TestToPriceScale:
    """Inverso da normalização MinMax na coluna Close."""

    def test_round_trip(self):
        """Min/max devem reverter MinMax aplicado manualmente."""
        prices = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        cmin, cmax = float(prices.min()), float(prices.max())
        scaled = (prices - cmin) / (cmax - cmin)
        recovered = to_price_scale(scaled, cmin, cmax)
        np.testing.assert_allclose(recovered, prices)


class TestEvaluatePredictions:
    """Testes do agregador de métricas em escala original."""

    def test_returns_required_keys(self):
        """Deve retornar MAE/RMSE/MAPE/sigma_*."""
        y_true = np.linspace(0.1, 0.9, 10)
        y_pred = y_true + 0.01
        metrics = evaluate_predictions(y_true, y_pred, close_min=10.0, close_max=20.0)
        for key in ("mae", "rmse", "mape", "target_sigma", "sigma_coverage_0_5"):
            assert key in metrics
        assert 0.0 <= metrics["sigma_coverage_0_5"] <= 1.0

    def test_perfect_predictions(self):
        """Predições perfeitas devem dar erro zero e cobertura 100%."""
        y = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        metrics = evaluate_predictions(y, y.copy(), close_min=10.0, close_max=50.0)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["sigma_coverage_0_5"] == 1.0


class TestTrainBaselines:
    """Smoke tests dos treinadores Ridge/MLP em dados sintéticos."""

    @pytest.fixture
    def small_splits(self):
        """Splits pequenos compatíveis com baselines tabulares."""
        rng = np.random.default_rng(0)
        # 60 amostras, 30 features achatadas (= seq_len 10 * 3 cols)
        X = rng.uniform(0, 1, size=(60, 30)).astype(np.float32)
        y = (X[:, 0] * 0.6 + X[:, 1] * 0.3 + rng.normal(0, 0.01, 60)).astype(np.float32)
        return {
            "train": (X[:40], y[:40]),
            "val": (X[40:50], y[40:50]),
            "test": (X[50:], y[50:]),
        }

    def test_ridge_returns_metrics(self, small_splits):
        """Ridge deve retornar dict de métricas com chaves esperadas."""
        model, metrics = train_ridge(
            splits=small_splits, close_min=10.0, close_max=50.0, alpha=1.0
        )
        for key in ("mae", "rmse", "mape", "sigma_coverage_0_5"):
            assert key in metrics
        # Coeficientes do Ridge devem ter dimensão do input
        assert model.coef_.shape == (30,)

    def test_mlp_returns_metrics(self, small_splits):
        """MLP deve treinar e devolver métricas válidas."""
        torch.manual_seed(0)
        _, metrics = train_mlp_baseline(
            splits=small_splits,
            close_min=10.0,
            close_max=50.0,
            input_size=30,
            hidden_sizes=[8],
            dropout=0.0,
            epochs=3,
            learning_rate=0.01,
            batch_size=8,
            weight_decay=0.0,
            device=torch.device("cpu"),
        )
        for key in ("mae", "rmse", "mape", "sigma_coverage_0_5", "best_val_loss"):
            assert key in metrics
        assert np.isfinite(metrics["mae"])


class TestBuildComparison:
    """Testes do agregador comparativo Ridge/MLP/LSTM."""

    def test_picks_lowest_error(self):
        """O melhor MAE deve ser o do modelo com menor valor."""
        cmp = _build_comparison(
            ridge={"mae": 0.5, "rmse": 0.6, "mape": 4.0, "sigma_coverage_0_5": 0.5},
            mlp={"mae": 0.3, "rmse": 0.4, "mape": 3.0, "sigma_coverage_0_5": 0.6},
            lstm={"mae": 0.2, "rmse": 0.3, "mape": 2.0, "sigma_coverage_0_5": 0.8},
        )
        assert cmp["best_mae"]["model"] == "lstm"
        assert cmp["best_sigma_coverage_0_5"]["model"] == "lstm"

    def test_handles_missing_metrics(self):
        """Modelos sem certa métrica não quebram a comparação."""
        cmp = _build_comparison(
            ridge={"mae": 0.5},
            mlp={"mae": 0.3},
            lstm={},  # vazio
        )
        assert cmp["best_mae"]["model"] == "mlp"
        assert "best_sigma_coverage_0_5" not in cmp
