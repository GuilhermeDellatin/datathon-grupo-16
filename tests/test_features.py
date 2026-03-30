"""Testes de feature engineering — schema contracts e transformações."""

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import (
    compute_features,
    compute_technical_indicators,
    create_sequences,
    split_data,
)


class TestComputeFeatures:
    """Testes para pipeline de feature engineering."""

    def test_schema_contract(self, sample_ohlcv_data: pd.DataFrame):
        """Features de saída devem conter indicadores técnicos esperados."""
        result = compute_features(sample_ohlcv_data)
        expected_cols = ["sma_20", "rsi_14", "macd", "bollinger_upper"]
        for col in expected_cols:
            assert col in result.columns, f"Coluna {col} ausente"

    def test_no_nulls(self, sample_ohlcv_data: pd.DataFrame):
        """Nenhuma feature pode ter null após transformação."""
        result = compute_features(sample_ohlcv_data)
        assert result.isnull().sum().sum() == 0, "Features com NaN encontradas"

    def test_row_count_reduced(self, sample_ohlcv_data: pd.DataFrame):
        """Número de registros deve reduzir (warmup dos indicadores)."""
        result = compute_features(sample_ohlcv_data)
        assert len(result) < len(sample_ohlcv_data)
        assert len(result) > 0

    def test_close_positive(self, sample_ohlcv_data: pd.DataFrame):
        """Preço de fechamento deve ser sempre positivo."""
        result = compute_features(sample_ohlcv_data)
        assert (result["Close"] > 0).all()

    def test_rsi_range(self, sample_ohlcv_data: pd.DataFrame):
        """RSI deve estar entre 0 e 100."""
        result = compute_features(sample_ohlcv_data)
        assert (result["rsi_14"] >= 0).all()
        assert (result["rsi_14"] <= 100).all()

    def test_volume_non_negative(self, sample_ohlcv_data: pd.DataFrame):
        """Volume não pode ser negativo."""
        result = compute_features(sample_ohlcv_data)
        assert (result["Volume"] >= 0).all()

    def test_all_indicators_present(self, sample_ohlcv_data: pd.DataFrame):
        """Todas as features técnicas devem estar presentes."""
        result = compute_features(sample_ohlcv_data)
        expected = [
            "sma_20", "sma_50", "ema_12", "ema_26", "rsi_14",
            "macd", "macd_signal", "bollinger_upper", "bollinger_lower",
            "volume_sma_20", "daily_return", "log_return",
        ]
        for col in expected:
            assert col in result.columns, f"Feature {col} ausente"


class TestCreateSequences:
    """Testes para criação de sequências LSTM."""

    def test_output_shapes(self):
        """X e y devem ter shapes corretos."""
        data = np.random.randn(100, 5).astype(np.float32)
        X, y = create_sequences(data, sequence_length=10, prediction_horizon=1)

        assert X.shape == (90, 10, 5)
        assert y.shape == (90,)

    def test_sequence_length_respected(self):
        """Cada sequência deve ter o tamanho correto."""
        data = np.random.randn(50, 3).astype(np.float32)
        X, _ = create_sequences(data, sequence_length=20)

        assert X.shape[1] == 20

    def test_prediction_horizon(self):
        """Target deve corresponder ao horizonte correto."""
        data = np.arange(30).reshape(30, 1).astype(np.float32)
        _, y = create_sequences(data, sequence_length=5, prediction_horizon=3, target_idx=0)

        # y[0] deve ser data[5+3-1, 0] = data[7, 0] = 7
        assert y[0] == 7.0

    def test_empty_data(self):
        """Dados muito curtos devem retornar arrays vazios."""
        data = np.random.randn(5, 3).astype(np.float32)
        X, y = create_sequences(data, sequence_length=10)
        assert len(X) == 0
        assert len(y) == 0

    def test_dtype_float32(self):
        """Saída deve ser float32."""
        data = np.random.randn(50, 3).astype(np.float32)
        X, y = create_sequences(data, sequence_length=10)
        assert X.dtype == np.float32
        assert y.dtype == np.float32


class TestSplitData:
    """Testes para split temporal."""

    def test_no_data_leakage(self):
        """Split temporal: treino antes de val, val antes de test."""
        X = np.arange(100).reshape(100, 1, 1).astype(np.float32)
        y = np.arange(100).astype(np.float32)
        splits = split_data(X, y, train_ratio=0.7, val_ratio=0.15)

        train_max = splits["train"][1].max()
        val_min = splits["val"][1].min()
        val_max = splits["val"][1].max()
        test_min = splits["test"][1].min()

        assert train_max < val_min, "Data leakage: treino -> validação"
        assert val_max < test_min, "Data leakage: validação -> teste"

    def test_all_data_used(self):
        """Todos os dados devem ser distribuídos nos splits."""
        X = np.random.randn(100, 1, 1).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        splits = split_data(X, y)

        total = sum(len(s[0]) for s in splits.values())
        assert total == 100

    def test_split_ratios(self):
        """Splits devem respeitar proporções aproximadas."""
        X = np.random.randn(1000, 1, 1).astype(np.float32)
        y = np.random.randn(1000).astype(np.float32)
        splits = split_data(X, y, train_ratio=0.8, val_ratio=0.1)

        assert len(splits["train"][0]) == 800
        assert len(splits["val"][0]) == 100
        assert len(splits["test"][0]) == 100
