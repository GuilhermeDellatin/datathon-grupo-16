"""Coleta de dados históricos de ações via yfinance.

Responsável por baixar dados OHLCV do ticker configurado
e salvar em formato parquet para processamento posterior.
"""

import logging

import pandas as pd
import yaml
import yfinance as yf

from src.paths import resolve_project_file

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/model_config.yaml") -> dict:
    """Carrega configuração do modelo.

    Args:
        config_path: Caminho para o arquivo YAML de configuração.

    Returns:
        Dicionário com configurações.
    """
    with open(config_path) as f:
        config: dict = yaml.safe_load(f)
    return config


def collect_stock_data(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    period: str | None = None,
) -> pd.DataFrame:
    """Coleta dados históricos de ações via yfinance.

    Suporta dois modos de janela: explícito (start_date + end_date) ou
    relativo (period, ex.: "2y", "6mo", "max"). Se `period` for fornecido,
    ele tem precedência sobre as datas.

    Args:
        ticker: Símbolo da ação (ex: PETR4.SA).
        start_date: Data inicial no formato YYYY-MM-DD (ignorada se period).
        end_date: Data final no formato YYYY-MM-DD (ignorada se period).
        period: Janela relativa do yfinance (ex.: "1y", "2y", "max").

    Returns:
        DataFrame com colunas OHLCV + Date como index.

    Raises:
        ValueError: Se nenhum dado for retornado ou se nem datas nem
            period forem fornecidos.
    """
    if period:
        logger.info("Coletando dados de %s (period=%s)", ticker, period)
        df = yf.download(ticker, period=period, progress=False)
    elif start_date and end_date:
        logger.info("Coletando dados de %s (%s a %s)", ticker, start_date, end_date)
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    else:
        raise ValueError(
            "Forneça `period` OU (`start_date` e `end_date`) para coletar dados."
        )

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
    output_file = resolve_project_file(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file)
    logger.info("Dados salvos em %s (%d registros)", output_file, len(df))


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
