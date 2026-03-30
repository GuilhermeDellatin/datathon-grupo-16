"""Fixtures compartilhados para testes — dados sintéticos."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.lstm_model import LSTMPredictor


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Dados OHLCV sintéticos que simulam ações (nunca dados reais em testes)."""
    np.random.seed(42)
    n = 300  # ~1 ano de trading days + buffer para indicadores

    dates = pd.bdate_range(start="2023-01-01", periods=n)

    # Simular random walk para preços
    base_price = 30.0
    returns = np.random.normal(0.0005, 0.02, n)
    close = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame(
        {
            "Open": close * (1 + np.random.uniform(-0.01, 0.01, n)),
            "High": close * (1 + np.abs(np.random.normal(0, 0.015, n))),
            "Low": close * (1 - np.abs(np.random.normal(0, 0.015, n))),
            "Close": close,
            "Volume": np.random.lognormal(mean=16, sigma=0.5, size=n).astype(float),
        },
        index=dates,
    )

    # Garantir High >= Close >= Low
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)

    return df


@pytest.fixture
def sample_features_data(sample_ohlcv_data: pd.DataFrame) -> pd.DataFrame:
    """Dados com features técnicas já calculadas."""
    from src.data.feature_engineering import compute_features

    return compute_features(sample_ohlcv_data)


@pytest.fixture
def sample_sequences(sample_features_data: pd.DataFrame):
    """Sequências prontas para LSTM."""
    from sklearn.preprocessing import MinMaxScaler

    from src.data.feature_engineering import create_sequences

    feature_cols = ["Close", "Volume", "sma_20", "rsi_14", "macd"]
    data = sample_features_data[feature_cols].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = create_sequences(data_scaled, sequence_length=30, prediction_horizon=1, target_idx=0)
    return X, y, scaler


@pytest.fixture
def sample_model() -> LSTMPredictor:
    """Modelo LSTM instanciado para testes (não treinado)."""
    return LSTMPredictor(
        input_size=5,
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
        output_size=1,
    )


@pytest.fixture
def sample_input_tensor() -> torch.Tensor:
    """Tensor de input sintético para testes de modelo."""
    batch_size = 4
    seq_len = 30
    n_features = 5
    return torch.randn(batch_size, seq_len, n_features)


@pytest.fixture
def mock_agent_response() -> dict:
    """Resposta mock do agente para testes."""
    return {
        "answer": "O preço atual da PETR4.SA é R$ 35.50.",
        "intermediate_steps": [
            {"tool": "fetch_market_data", "tool_input": "PETR4.SA", "output": "..."},
        ],
        "success": True,
    }
