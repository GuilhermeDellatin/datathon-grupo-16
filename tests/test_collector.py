"""Testes do coletor de dados — mock yfinance."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.data.collector import collect_stock_data, save_raw_data


@pytest.fixture
def mock_yf_data():
    """DataFrame mock simulando resposta do yfinance."""
    dates = pd.bdate_range("2024-01-01", periods=50)
    return pd.DataFrame(
        {
            "Open": np.random.uniform(28, 32, 50),
            "High": np.random.uniform(30, 34, 50),
            "Low": np.random.uniform(26, 30, 50),
            "Close": np.random.uniform(28, 32, 50),
            "Volume": np.random.lognormal(16, 0.5, 50),
        },
        index=dates,
    )


class TestCollectStockData:
    """Testes da coleta de dados."""

    @patch("src.data.collector.yf.download")
    def test_returns_dataframe(self, mock_download, mock_yf_data):
        """Deve retornar DataFrame com dados."""
        mock_download.return_value = mock_yf_data
        df = collect_stock_data("PETR4.SA", "2024-01-01", "2024-03-01")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    @patch("src.data.collector.yf.download")
    def test_has_ohlcv_columns(self, mock_download, mock_yf_data):
        """DataFrame deve ter colunas OHLCV."""
        mock_download.return_value = mock_yf_data
        df = collect_stock_data("PETR4.SA", "2024-01-01", "2024-03-01")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in df.columns

    @patch("src.data.collector.yf.download")
    def test_empty_data_raises(self, mock_download):
        """DataFrame vazio deve levantar ValueError."""
        mock_download.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="Nenhum dado"):
            collect_stock_data("INVALID", "2024-01-01", "2024-03-01")

    @patch("src.data.collector.yf.download")
    def test_handles_multi_index(self, mock_download, mock_yf_data):
        """Deve lidar com MultiIndex columns do yfinance."""
        multi_idx = pd.MultiIndex.from_tuples(
            [(c, "PETR4.SA") for c in mock_yf_data.columns]
        )
        multi_df = mock_yf_data.copy()
        multi_df.columns = multi_idx
        mock_download.return_value = multi_df

        df = collect_stock_data("PETR4.SA", "2024-01-01", "2024-03-01")
        assert not isinstance(df.columns, pd.MultiIndex)


class TestSaveRawData:
    """Testes de salvamento de dados."""

    def test_save_creates_file(self, tmp_path, mock_yf_data):
        """Deve criar arquivo parquet."""
        output = str(tmp_path / "data" / "test.parquet")
        save_raw_data(mock_yf_data, output)
        assert (tmp_path / "data" / "test.parquet").exists()

    def test_save_preserves_data(self, tmp_path, mock_yf_data):
        """Dados salvos devem ter mesmas colunas e tamanho."""
        output = str(tmp_path / "test.parquet")
        save_raw_data(mock_yf_data, output)
        loaded = pd.read_parquet(output)
        assert len(loaded) == len(mock_yf_data)
        assert list(loaded.columns) == list(mock_yf_data.columns)
