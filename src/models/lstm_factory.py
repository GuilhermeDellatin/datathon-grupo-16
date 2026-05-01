"""Factory Method para criar modelos do projeto.

Centraliza a construção dos dois `nn.Module` que coexistem no pipeline
— LSTM principal (`LSTMPredictor`) e MLP baseline (`MLPRegressor`) —
atrás de uma única chamada ``LSTMFactory.create(name, config)``.

Por que existe:

- Permite que `train.py`, `baseline.py` e o endpoint `/train` peçam um
  modelo por nome, sem importar a classe concreta.
- Padrão Factory Method (Gamma et al., 1994) com 1 metodo de criação
  e duas classes-produto registradas.
- Adicionar um terceiro modelo é só: implementar o ``nn.Module``,
  registrar em ``_REGISTRY`` e prover um ``_create_<name>`` ou hook
  equivalente.

Não pretende substituir `lstm_model.py` nem `baseline.py`; é um ponto de
entrada uniforme.
"""

from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn

from src.models.baseline import MLPRegressor
from src.models.lstm_model import LSTMPredictor

logger = logging.getLogger(__name__)


class LSTMFactory:
    """Factory Method para os modelos suportados pelo pipeline.

    Embora o nome enfatize o LSTM (modelo principal do produto), a
    fábrica também produz o MLP baseline que vive em
    `src/models/baseline.py`. Com isso, o despacho ``create("lstm", cfg)``
    e ``create("mlp", cfg)`` é uniforme e o caller não precisa importar
    classes distintas.
    """

    LSTM_ALIASES = ("lstm", "lstm-petr4", "lstmpredictor")
    MLP_ALIASES = ("mlp", "mlp-baseline", "mlpregressor")

    @classmethod
    def list_models(cls) -> list[str]:
        """Lista os nomes canônicos dos modelos suportados.

        Returns:
            ``["lstm", "mlp"]`` (em ordem de relevância no produto).
        """
        return ["lstm", "mlp"]

    @classmethod
    def create(cls, name: str, config: dict[str, Any]) -> nn.Module:
        """Cria um modelo PyTorch a partir do nome canônico.

        Args:
            name: Identificador do modelo (case-insensitive). Aceita
                ``"lstm"``, ``"lstm-petr4"`` ou ``"mlp"``,
                ``"mlp-baseline"``.
            config: Dicionário com hiperparâmetros — ver
                ``_create_lstm`` e ``_create_mlp`` para o esquema
                esperado por modelo.

        Returns:
            Instância de ``nn.Module`` pronta para treino/inferência.

        Raises:
            ValueError: Se ``name`` não corresponder a um modelo
                conhecido.
            KeyError: Se faltar uma chave obrigatória em ``config``.
        """
        key = name.strip().lower()
        if key in cls.LSTM_ALIASES:
            return cls._create_lstm(config)
        if key in cls.MLP_ALIASES:
            return cls._create_mlp(config)
        raise ValueError(
            f"Modelo '{name}' não suportado. "
            f"Disponíveis: {cls.list_models()}"
        )

    # ------------------------------------------------------------------
    # Construtores específicos por classe-produto.
    # Mantidos como métodos privados para reduzir API pública.
    # ------------------------------------------------------------------

    @staticmethod
    def _create_lstm(config: dict[str, Any]) -> LSTMPredictor:
        """Constrói ``LSTMPredictor`` a partir do bloco ``model`` do YAML.

        Args:
            config: Dicionário com chaves obrigatórias ``input_size``,
                ``hidden_size``, ``num_layers``, ``dropout``,
                ``bidirectional``, ``output_size``. Aceita o bloco
                ``model`` cru do ``configs/model_config.yaml`` ou o
                dict achatado.

        Returns:
            ``LSTMPredictor`` instanciado.
        """
        model_cfg = config.get("model", config)
        try:
            return LSTMPredictor(
                input_size=int(model_cfg["input_size"]),
                hidden_size=int(model_cfg.get("hidden_size", 128)),
                num_layers=int(model_cfg.get("num_layers", 2)),
                dropout=float(model_cfg.get("dropout", 0.2)),
                bidirectional=bool(model_cfg.get("bidirectional", False)),
                output_size=int(model_cfg.get("output_size", 1)),
            )
        except KeyError as exc:
            raise KeyError(
                f"Config LSTM faltando chave obrigatória: {exc.args[0]}"
            ) from exc

    @staticmethod
    def _create_mlp(config: dict[str, Any]) -> MLPRegressor:
        """Constrói ``MLPRegressor`` baseline a partir do bloco YAML.

        Args:
            config: Dicionário com ``input_size`` (obrigatório) e,
                opcionalmente, ``hidden_sizes`` (lista) e ``dropout``.
                Aceita o bloco ``baseline.mlp`` ou um dict achatado.

        Returns:
            ``MLPRegressor`` instanciado.
        """
        mlp_cfg = config.get("baseline", {}).get("mlp", config)
        try:
            return MLPRegressor(
                input_size=int(mlp_cfg["input_size"]),
                hidden_sizes=list(mlp_cfg.get("hidden_sizes", [128, 64])),
                dropout=float(mlp_cfg.get("dropout", 0.1)),
            )
        except KeyError as exc:
            raise KeyError(
                f"Config MLP faltando chave obrigatória: {exc.args[0]}"
            ) from exc
