"""Coleta de dados históricos de ações via yfinance.

Responsável por baixar dados OHLCV do ticker configurado
e salvar em formato parquet para processamento posterior.
"""

import logging
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/model_config.yaml") -> dict:
    """Carrega configuração do modelo.

    Args:
        config_path: Caminho para o arquivo YAML de configuração.

    Returns:
        Dicionário com configurações.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def collect_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Coleta dados históricos de ações via yfinance.

    Args:
        ticker: Símbolo da ação (ex: PETR4.SA).
        start_date: Data inicial no formato YYYY-MM-DD.
        end_date: Data final no formato YYYY-MM-DD.

    Returns:
        DataFrame com colunas OHLCV + Date como index.

    Raises:
        ValueError: Se nenhum dado for retornado.
    """
    logger.info("Coletando dados de %s (%s a %s)", ticker, start_date, end_date)

    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df.empty:
        raise ValueError(f"Nenhum dado retornado para {ticker}")

    # Se multi-level columns (yfinance >= 0.2.36), flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    logger.info(
        "Dados coletados: %d registros, período %s a %s",
        len(df),
        df.index.min().strftime("%Y-%m-%d"),
        df.index.max().strftime("%Y-%m-%d"),
    )

    return df


def save_raw_data(df: pd.DataFrame, output_path: str = "data/raw/petr4_raw.parquet") -> None:
    """Salva dados brutos em parquet.

    Args:
        df: DataFrame com dados OHLCV.
        output_path: Caminho de saída.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    logger.info("Dados salvos em %s (%d registros)", output_path, len(df))


def main() -> None:
    """Entry point para coleta de dados."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = load_config()
    df = collect_stock_data(
        ticker=config["ticker"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )
    save_raw_data(df)


if __name__ == "__main__":
    main()
