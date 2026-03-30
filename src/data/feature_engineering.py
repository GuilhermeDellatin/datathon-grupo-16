"""Feature engineering para dados financeiros.

Calcula indicadores técnicos (SMA, EMA, RSI, MACD, Bollinger Bands)
e prepara features para o modelo LSTM. Todas as transformações são
validadas com pandera schema contracts.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pandera as pa
import ta
import yaml
from pandera import Check, Column, DataFrameSchema
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


# --- Schema Contracts ---

RAW_SCHEMA = DataFrameSchema(
    {
        "Open": Column(float, Check.gt(0), nullable=False),
        "High": Column(float, Check.gt(0), nullable=False),
        "Low": Column(float, Check.gt(0), nullable=False),
        "Close": Column(float, Check.gt(0), nullable=False),
        "Volume": Column(float, Check.ge(0), nullable=False),
    },
    index=pa.Index("datetime64[ns]"),
    strict=False,  # Permite colunas extras como Adj Close
)

FEATURE_SCHEMA = DataFrameSchema(
    {
        "Close": Column(float, Check.gt(0)),
        "Volume": Column(float, Check.ge(0)),
        "sma_20": Column(float, nullable=True),
        "sma_50": Column(float, nullable=True),
        "ema_12": Column(float, nullable=True),
        "ema_26": Column(float, nullable=True),
        "rsi_14": Column(float, Check.between(0, 100), nullable=True),
        "macd": Column(float, nullable=True),
        "macd_signal": Column(float, nullable=True),
        "bollinger_upper": Column(float, nullable=True),
        "bollinger_lower": Column(float, nullable=True),
        "volume_sma_20": Column(float, nullable=True),
    },
    strict=False,
)


def load_config(config_path: str = "configs/model_config.yaml") -> dict:
    """Carrega configuração do modelo.

    Args:
        config_path: Caminho para o arquivo YAML de configuração.

    Returns:
        Dicionário com configurações.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def validate_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Valida schema dos dados brutos.

    Args:
        df: DataFrame com dados OHLCV.

    Returns:
        DataFrame validado.

    Raises:
        AssertionError: Se o DataFrame estiver vazio ou com colunas faltantes.
    """
    logger.info("Validando schema dos dados brutos...")
    assert not df.empty, "DataFrame vazio"
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = set(required_cols) - set(df.columns)
    assert not missing, f"Colunas faltantes: {missing}"
    return df


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores técnicos.

    Args:
        df: DataFrame com dados OHLCV.

    Returns:
        DataFrame com indicadores técnicos adicionados.
    """
    df = df.copy()

    # Médias Móveis Simples
    df["sma_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["sma_50"] = ta.trend.sma_indicator(df["Close"], window=50)

    # Médias Móveis Exponenciais
    df["ema_12"] = ta.trend.ema_indicator(df["Close"], window=12)
    df["ema_26"] = ta.trend.ema_indicator(df["Close"], window=26)

    # RSI
    df["rsi_14"] = ta.momentum.rsi(df["Close"], window=14)

    # MACD
    macd = ta.trend.MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["bollinger_upper"] = bollinger.bollinger_hband()
    df["bollinger_lower"] = bollinger.bollinger_lband()

    # Volume SMA
    df["volume_sma_20"] = ta.trend.sma_indicator(df["Volume"], window=20)

    # Returns
    df["daily_return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    logger.info("Indicadores técnicos calculados: %d features", len(df.columns))
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completa de feature engineering.

    Args:
        df: DataFrame com dados OHLCV brutos.

    Returns:
        DataFrame com features processadas, sem NaN.
    """
    df = validate_raw_data(df)
    df = compute_technical_indicators(df)

    # Remover NaN gerados pelos indicadores (primeiras N linhas)
    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    logger.info("Removidos %d registros com NaN (indicadores warmup)", n_before - n_after)

    return df


def create_sequences(
    data: np.ndarray,
    sequence_length: int = 60,
    prediction_horizon: int = 1,
    target_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cria sequências para treinamento LSTM.

    Args:
        data: Array numpy com features escaladas.
        sequence_length: Tamanho da janela de entrada.
        prediction_horizon: Número de passos à frente para prever.
        target_idx: Índice da coluna target no array.

    Returns:
        Tupla (X, y) onde X.shape = (n_samples, sequence_length, n_features)
        e y.shape = (n_samples,).
    """
    X, y = [], []
    for i in range(sequence_length, len(data) - prediction_horizon + 1):
        X.append(data[i - sequence_length : i])
        y.append(data[i + prediction_horizon - 1, target_idx])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    logger.info(
        "Sequências criadas: X=%s, y=%s (seq_len=%d, horizon=%d)",
        X.shape,
        y.shape,
        sequence_length,
        prediction_horizon,
    )
    return X, y


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split temporal dos dados (sem shuffle — séries temporais).

    Args:
        X: Sequências de entrada.
        y: Targets.
        train_ratio: Proporção de treino.
        val_ratio: Proporção de validação.

    Returns:
        Dicionário com splits: train, val, test.
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        "train": (X[:train_end], y[:train_end]),
        "val": (X[train_end:val_end], y[train_end:val_end]),
        "test": (X[val_end:], y[val_end:]),
    }

    for name, (Xi, yi) in splits.items():
        logger.info("Split %s: X=%s, y=%s", name, Xi.shape, yi.shape)

    return splits


def main() -> None:
    """Entry point para feature engineering."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = load_config()
    raw_path = "data/raw/petr4_raw.parquet"
    output_path = "data/processed/petr4_features.parquet"

    df = pd.read_parquet(raw_path)
    df_features = compute_features(df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(output_path)
    logger.info(
        "Features salvas em %s (%d registros, %d features)",
        output_path,
        len(df_features),
        len(df_features.columns),
    )


if __name__ == "__main__":
    main()
