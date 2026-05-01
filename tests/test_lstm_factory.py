"""Testes do Factory Method `LSTMFactory`."""

import pytest
import torch

from src.models.baseline import MLPRegressor
from src.models.lstm_factory import LSTMFactory
from src.models.lstm_model import LSTMPredictor


class TestLSTMFactory:
    """Despacho entre LSTM e MLP via factory."""

    def test_list_models(self):
        assert LSTMFactory.list_models() == ["lstm", "mlp"]

    @pytest.mark.parametrize("name", ["lstm", "LSTM", "lstm-petr4", "LSTMPredictor"])
    def test_create_lstm_aliases(self, name):
        """Vários aliases devem produzir LSTMPredictor."""
        cfg = {
            "model": {
                "input_size": 14,
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.0,
                "bidirectional": False,
                "output_size": 1,
            }
        }
        model = LSTMFactory.create(name, cfg)
        assert isinstance(model, LSTMPredictor)

    @pytest.mark.parametrize("name", ["mlp", "MLP", "mlp-baseline", "MLPRegressor"])
    def test_create_mlp_aliases(self, name):
        """Vários aliases devem produzir MLPRegressor."""
        cfg = {"baseline": {"mlp": {"input_size": 30, "hidden_sizes": [16], "dropout": 0.0}}}
        model = LSTMFactory.create(name, cfg)
        assert isinstance(model, MLPRegressor)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="não suportado"):
            LSTMFactory.create("transformer", {})

    def test_lstm_forward_works(self):
        """Modelo retornado pelo factory deve ser utilizável."""
        cfg = {
            "model": {
                "input_size": 5,
                "hidden_size": 16,
                "num_layers": 1,
                "dropout": 0.0,
                "bidirectional": False,
                "output_size": 1,
            }
        }
        model = LSTMFactory.create("lstm", cfg)
        x = torch.randn(2, 10, 5)
        out = model(x)
        assert out.shape == (2, 1)

    def test_mlp_forward_works(self):
        cfg = {"baseline": {"mlp": {"input_size": 20, "hidden_sizes": [8], "dropout": 0.0}}}
        model = LSTMFactory.create("mlp", cfg)
        x = torch.randn(3, 20)
        out = model(x)
        assert out.shape == (3,)

    def test_lstm_missing_required_key(self):
        with pytest.raises(KeyError, match="input_size"):
            LSTMFactory.create("lstm", {"model": {}})

    def test_mlp_missing_required_key(self):
        with pytest.raises(KeyError, match="input_size"):
            LSTMFactory.create("mlp", {"baseline": {"mlp": {}}})

    def test_lstm_accepts_flat_config(self):
        """Aceita dict achatado (sem o invólucro `model`)."""
        cfg = {
            "input_size": 7,
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.0,
            "bidirectional": False,
            "output_size": 1,
        }
        model = LSTMFactory.create("lstm", cfg)
        assert isinstance(model, LSTMPredictor)

    def test_mlp_accepts_flat_config(self):
        cfg = {"input_size": 12, "hidden_sizes": [4], "dropout": 0.0}
        model = LSTMFactory.create("mlp", cfg)
        assert isinstance(model, MLPRegressor)
