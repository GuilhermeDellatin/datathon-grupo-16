"""Modelo LSTM para predição de séries temporais financeiras.

Arquitetura: Input → LSTM (multi-layer, dropout) → Linear → Output
Suporta configuração via YAML (hidden_size, num_layers, dropout, bidirectional).
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LSTMPredictor(nn.Module):
    """LSTM para predição de preço de fechamento de ações.

    Args:
        input_size: Número de features de entrada.
        hidden_size: Dimensão do estado oculto.
        num_layers: Número de camadas LSTM empilhadas.
        dropout: Taxa de dropout entre camadas LSTM.
        bidirectional: Se True, usa LSTM bidirecional.
        output_size: Dimensão da saída (1 para regressão univariada).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        output_size: int = 1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        # Linear recebe o output do último timestep
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

        logger.info(
            "LSTMPredictor criado: input=%d, hidden=%d, layers=%d, bidir=%s, params=%d",
            input_size,
            hidden_size,
            num_layers,
            bidirectional,
            sum(p.numel() for p in self.parameters()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor de entrada (batch_size, sequence_length, input_size).

        Returns:
            Predição (batch_size, output_size).
        """
        # LSTM output: (batch_size, seq_len, hidden_size * num_directions)
        lstm_out, _ = self.lstm(x)

        # Pegar apenas o último timestep
        last_output = lstm_out[:, -1, :]

        out = self.dropout(last_output)
        out = self.fc(out)
        return out

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inferência sem gradiente.

        Args:
            x: Tensor de entrada.

        Returns:
            Predição.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)
