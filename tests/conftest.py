"""Fixtures compartilhados para testes — dados sintéticos."""

import numpy as np
import pandas as pd
import pytest


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
