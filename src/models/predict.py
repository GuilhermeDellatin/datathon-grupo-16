"""Módulo de inferência do modelo LSTM.

Carrega modelo salvo e realiza predições.
Isolado do pipeline de treinamento (sem acoplamento).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.lstm_model import LSTMPredictor

logger = logging.getLogger(__name__)


class StockPredictor:
    """Predictor para preços de ações usando LSTM treinado.

    Args:
        model_path: Caminho para o checkpoint .pt.
        device: Device para inferência.
    """

    def __init__(
        self,
        model_path: str = "models/lstm_petr4_best.pt",
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.feature_columns = checkpoint["feature_columns"]
        self.sequence_length = checkpoint["sequence_length"]
        self.prediction_horizon = checkpoint["prediction_horizon"]

        # Reconstruir modelo
        model_cfg = checkpoint["model_config"]
        self.model = LSTMPredictor(
            input_size=len(self.feature_columns),
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
            bidirectional=model_cfg["bidirectional"],
            output_size=model_cfg["output_size"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Reconstruir scaler
        from sklearn.preprocessing import MinMaxScaler

        self.scaler = MinMaxScaler()
        self.scaler.min_ = np.array(checkpoint["scaler_params"]["min_"])
        self.scaler.scale_ = np.array(checkpoint["scaler_params"]["scale_"])
        self.scaler.data_min_ = np.array(checkpoint["scaler_params"]["data_min_"])
        self.scaler.data_max_ = np.array(checkpoint["scaler_params"]["data_max_"])
        self.scaler.n_features_in_ = len(self.feature_columns)

        logger.info("Modelo carregado de %s (device=%s)", model_path, self.device)

    def predict(self, input_data: np.ndarray) -> float:
        """Realiza predição a partir de dados já escalados.

        Args:
            input_data: Array (sequence_length, n_features) já escalado.

        Returns:
            Valor predito (escalado — precisa de inverse_transform para valor real).
        """
        x = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(x).squeeze().item()
        return pred

    def predict_from_dataframe(self, df: pd.DataFrame) -> dict[str, float]:
        """Predição a partir de DataFrame com features.

        Args:
            df: DataFrame com as últimas sequence_length linhas
                contendo as feature_columns.

        Returns:
            Dicionário com predição em escala original e escalada.
        """
        if len(df) < self.sequence_length:
            raise ValueError(
                f"DataFrame precisa de pelo menos {self.sequence_length} linhas, "
                f"recebeu {len(df)}"
            )

        data = df[self.feature_columns].values[-self.sequence_length :]
        data_scaled = self.scaler.transform(data)

        pred_scaled = self.predict(data_scaled)

        # Inverse transform apenas da coluna Close (index 0)
        dummy = np.zeros((1, len(self.feature_columns)))
        dummy[0, 0] = pred_scaled
        pred_original = self.scaler.inverse_transform(dummy)[0, 0]

        return {
            "predicted_close": float(pred_original),
            "predicted_scaled": float(pred_scaled),
            "horizon_days": self.prediction_horizon,
            "ticker": "PETR4.SA",
        }
