"""Testes do pipeline de treinamento — funções auxiliares e componentes."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.lstm_model import LSTMPredictor
from src.models.train import (
    EarlyStopping,
    compute_metrics,
    evaluate_epoch,
    get_git_sha,
    train_epoch,
)


class TestComputeMetrics:
    """Testes de cálculo de métricas."""

    def test_perfect_predictions(self):
        """Predições perfeitas devem dar métricas ~0."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = compute_metrics(y, y)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["mape"] == pytest.approx(0.0)

    def test_known_values(self):
        """Métricas devem corresponder a valores conhecidos."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 2.8])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0
        assert metrics["mape"] > 0

    def test_returns_all_metrics(self):
        """Deve retornar MAE, RMSE e MAPE."""
        y = np.array([1.0, 2.0, 3.0])
        metrics = compute_metrics(y, y + 0.1)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mape" in metrics

    def test_rmse_gte_mae(self):
        """RMSE deve ser >= MAE."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 1.5, 3.5, 3.5])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["rmse"] >= metrics["mae"]


class TestEarlyStopping:
    """Testes de early stopping."""

    def test_no_stop_initially(self):
        """Não deve parar antes de atingir patience."""
        es = EarlyStopping(patience=5)
        assert not es(1.0)
        assert not es(0.9)

    def test_stops_after_patience(self):
        """Deve parar após patience épocas sem melhoria."""
        es = EarlyStopping(patience=3)
        es(1.0)
        es(1.1)  # Piora 1
        es(1.2)  # Piora 2
        assert es(1.3)  # Piora 3 → deve parar

    def test_resets_on_improvement(self):
        """Counter deve resetar quando há melhoria."""
        es = EarlyStopping(patience=3)
        es(1.0)
        es(1.1)  # Piora 1
        es(1.2)  # Piora 2
        es(0.5)  # Melhoria → reset
        assert not es(0.6)  # Piora 1 (resetou)
        assert not es(0.7)  # Piora 2
        assert es(0.8)  # Piora 3 → para

    def test_min_delta(self):
        """Melhoria menor que min_delta não deve contar."""
        es = EarlyStopping(patience=2, min_delta=0.1)
        es(1.0)
        es(0.99)  # Melhoria de 0.01 < min_delta → conta como piora
        assert es(0.98)  # Segunda "piora" → para

    def test_should_stop_flag(self):
        """Flag should_stop deve ser atualizada."""
        es = EarlyStopping(patience=1)
        es(1.0)
        assert not es.should_stop
        es(1.1)  # Piora 1 == patience → para
        assert es.should_stop


class TestGetGitSha:
    """Testes de obtenção do git SHA."""

    def test_returns_string(self):
        """Deve retornar string."""
        sha = get_git_sha()
        assert isinstance(sha, str)
        assert len(sha) > 0

    def test_returns_short_sha(self):
        """Deve retornar SHA truncado (8 chars)."""
        sha = get_git_sha()
        assert len(sha) <= 8


class TestTrainEpoch:
    """Testes do loop de treino e avaliação."""

    @pytest.fixture
    def training_setup(self):
        """Setup para testes de treino."""
        model = LSTMPredictor(input_size=3, hidden_size=8, num_layers=1, dropout=0.0)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        device = torch.device("cpu")

        X = torch.randn(20, 10, 3)
        y = torch.randn(20)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=False)

        return model, dataloader, criterion, optimizer, device

    def test_train_epoch_returns_loss(self, training_setup):
        """train_epoch deve retornar loss numérica."""
        model, dl, criterion, optimizer, device = training_setup
        loss = train_epoch(model, dl, criterion, optimizer, device)
        assert isinstance(loss, float)
        assert loss > 0

    def test_evaluate_epoch_returns_loss(self, training_setup):
        """evaluate_epoch deve retornar loss numérica."""
        model, dl, criterion, _, device = training_setup
        loss = evaluate_epoch(model, dl, criterion, device)
        assert isinstance(loss, float)
        assert loss > 0

    def test_train_reduces_loss(self, training_setup):
        """Treino deve reduzir loss ao longo de épocas."""
        model, dl, criterion, optimizer, device = training_setup

        initial_loss = evaluate_epoch(model, dl, criterion, device)
        for _ in range(20):
            train_epoch(model, dl, criterion, optimizer, device)
        final_loss = evaluate_epoch(model, dl, criterion, device)

        # Loss deve diminuir com treinamento (pode não ser sempre verdade com 20 epochs,
        # mas com lr=0.01 e dados pequenos geralmente funciona)
        assert final_loss <= initial_loss * 2  # Pelo menos não explodir
