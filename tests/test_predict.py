"""Testes do módulo de inferência — StockPredictor com mock checkpoint."""

import numpy as np
import pandas as pd
import pytest
import tempfile
import torch

from src.models.lstm_model import LSTMPredictor
from src.models.predict import StockPredictor


@pytest.fixture
def mock_checkpoint(tmp_path):
    """Cria checkpoint mock para testes de inferência."""
    model = LSTMPredictor(input_size=3, hidden_size=16, num_layers=1, dropout=0.0)

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    dummy_data = np.random.randn(100, 3)
    scaler.fit(dummy_data)

    checkpoint_path = str(tmp_path / "test_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "hidden_size": 16,
                "num_layers": 1,
                "dropout": 0.0,
                "bidirectional": False,
                "output_size": 1,
            },
            "feature_columns": ["Close", "Volume", "sma_20"],
            "scaler_params": {
                "min_": scaler.min_.tolist(),
                "scale_": scaler.scale_.tolist(),
                "data_min_": scaler.data_min_.tolist(),
                "data_max_": scaler.data_max_.tolist(),
            },
            "sequence_length": 10,
            "prediction_horizon": 5,
        },
        checkpoint_path,
    )
    return checkpoint_path


class TestStockPredictor:
    """Testes do StockPredictor."""

    def test_load_model(self, mock_checkpoint):
        """Modelo deve carregar a partir do checkpoint."""
        predictor = StockPredictor(model_path=mock_checkpoint)
        assert predictor.model is not None
        assert predictor.sequence_length == 10
        assert predictor.prediction_horizon == 5

    def test_predict_returns_float(self, mock_checkpoint):
        """predict deve retornar float."""
        predictor = StockPredictor(model_path=mock_checkpoint)
        input_data = np.random.randn(10, 3).astype(np.float32)
        result = predictor.predict(input_data)
        assert isinstance(result, float)

    def test_predict_from_dataframe(self, mock_checkpoint):
        """predict_from_dataframe deve retornar dict com campos esperados."""
        predictor = StockPredictor(model_path=mock_checkpoint)

        df = pd.DataFrame(
            {
                "Close": np.random.randn(20) + 30,
                "Volume": np.random.lognormal(16, 0.5, 20),
                "sma_20": np.random.randn(20) + 30,
            }
        )

        result = predictor.predict_from_dataframe(df)
        assert "predicted_close" in result
        assert "predicted_scaled" in result
        assert "horizon_days" in result
        assert result["horizon_days"] == 5

    def test_predict_from_short_dataframe(self, mock_checkpoint):
        """DataFrame muito curto deve levantar ValueError."""
        predictor = StockPredictor(model_path=mock_checkpoint)

        df = pd.DataFrame(
            {"Close": [30.0], "Volume": [1e6], "sma_20": [29.5]}
        )

        with pytest.raises(ValueError, match="pelo menos"):
            predictor.predict_from_dataframe(df)

    def test_feature_columns_preserved(self, mock_checkpoint):
        """Feature columns devem ser preservadas do checkpoint."""
        predictor = StockPredictor(model_path=mock_checkpoint)
        assert predictor.feature_columns == ["Close", "Volume", "sma_20"]

    def test_model_in_eval_mode(self, mock_checkpoint):
        """Modelo deve estar em modo eval após carregamento."""
        predictor = StockPredictor(model_path=mock_checkpoint)
        assert not predictor.model.training
