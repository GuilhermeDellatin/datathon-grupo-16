"""Testes do modelo LSTM — arquitetura e inferência."""

import torch
import pytest

from src.models.lstm_model import LSTMPredictor


class TestLSTMPredictor:
    """Testes da arquitetura LSTM."""

    def test_forward_shape(self, sample_model, sample_input_tensor):
        """Output shape deve ser (batch_size, output_size)."""
        output = sample_model(sample_input_tensor)
        assert output.shape == (4, 1)

    def test_predict_no_grad(self, sample_model, sample_input_tensor):
        """Predict deve funcionar sem gradientes."""
        output = sample_model.predict(sample_input_tensor)
        assert output.shape == (4, 1)
        assert not output.requires_grad

    def test_deterministic_inference(self, sample_model, sample_input_tensor):
        """Duas inferências com mesmo input devem dar mesmo resultado."""
        out1 = sample_model.predict(sample_input_tensor)
        out2 = sample_model.predict(sample_input_tensor)
        torch.testing.assert_close(out1, out2)

    def test_output_range(self, sample_model, sample_input_tensor):
        """Output deve ser finito (sem NaN ou Inf)."""
        output = sample_model(sample_input_tensor)
        assert torch.isfinite(output).all()

    def test_different_input_sizes(self):
        """Modelo deve aceitar diferentes input_size."""
        for n_features in [1, 5, 14, 20]:
            model = LSTMPredictor(input_size=n_features, hidden_size=16, num_layers=1)
            x = torch.randn(2, 30, n_features)
            output = model(x)
            assert output.shape == (2, 1)

    def test_bidirectional(self):
        """Modelo bidirecional deve funcionar."""
        model = LSTMPredictor(
            input_size=5, hidden_size=16, num_layers=1, bidirectional=True
        )
        x = torch.randn(2, 30, 5)
        output = model(x)
        assert output.shape == (2, 1)

    def test_parameter_count(self):
        """Modelo deve ter parâmetros treináveis."""
        model = LSTMPredictor(input_size=5, hidden_size=32, num_layers=2)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params > 0

    def test_gradient_flow(self):
        """Gradientes devem fluir pelo modelo."""
        model = LSTMPredictor(input_size=5, hidden_size=16, num_layers=1)
        x = torch.randn(2, 10, 5, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None

    def test_batch_size_one(self):
        """Modelo deve aceitar batch_size=1."""
        model = LSTMPredictor(input_size=5, hidden_size=16, num_layers=1)
        x = torch.randn(1, 30, 5)
        output = model(x)
        assert output.shape == (1, 1)

    def test_dropout_train_vs_eval(self):
        """Modo eval deve desativar dropout (output determinístico)."""
        model = LSTMPredictor(
            input_size=5, hidden_size=16, num_layers=2, dropout=0.5
        )
        x = torch.randn(4, 10, 5)

        model.eval()
        out1 = model(x)
        out2 = model(x)
        torch.testing.assert_close(out1, out2)
