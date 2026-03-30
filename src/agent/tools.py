"""Tools customizadas para o agente ReAct do Datathon.

4 tools relevantes ao domínio de análise de ações:
1. predict_stock_price — Predição via modelo LSTM
2. fetch_market_data — Dados de mercado via yfinance
3. search_financial_docs — Busca RAG em documentação financeira
4. compare_model_versions — Comparação champion-challenger via MLflow
"""

import logging

import numpy as np
import pandas as pd
import yfinance as yf
from langchain_core.tools import Tool

logger = logging.getLogger(__name__)


# --- Tool 1: Predição via LSTM ---


def _predict_stock_price(query: str) -> str:
    """Realiza predição de preço usando o modelo LSTM treinado.

    Args:
        query: Texto com ticker e horizonte (ex: "PETR4.SA 5 dias").
              Aceita também apenas o ticker.

    Returns:
        String com resultado da predição formatado.
    """
    try:
        from src.data.feature_engineering import compute_features
        from src.models.predict import StockPredictor

        ticker = "PETR4.SA"  # Default do projeto

        # Buscar dados recentes para contexto
        df = yf.download(ticker, period="6mo", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df_features = compute_features(df)

        predictor = StockPredictor()
        result = predictor.predict_from_dataframe(df_features)

        current_price = df["Close"].iloc[-1]
        predicted = result["predicted_close"]
        variation = ((predicted - current_price) / current_price) * 100

        return (
            f"Predição LSTM para {ticker}:\n"
            f"- Preço atual: R$ {current_price:.2f}\n"
            f"- Preço previsto ({result['horizon_days']} dias): R$ {predicted:.2f}\n"
            f"- Variação esperada: {variation:+.2f}%\n"
            f"- Modelo: LSTM PyTorch (seq_len={predictor.sequence_length})\n"
            f"- AVISO: Esta é uma predição de modelo, não uma recomendação de investimento."
        )
    except Exception as e:
        logger.error("Erro na predição: %s", e)
        return f"Erro ao realizar predição: {e}"


predict_stock_tool = Tool(
    name="predict_stock_price",
    func=_predict_stock_price,
    description=(
        "Realiza predição de preço de fechamento de ações da Petrobras (PETR4.SA) "
        "usando modelo LSTM treinado. Use quando o usuário perguntar sobre previsão "
        "de preço futuro. Input: texto com ticker e horizonte."
    ),
)


# --- Tool 2: Dados de Mercado ---


def _calc_rsi(series: pd.Series, period: int = 14) -> float:
    """Calcula RSI.

    Args:
        series: Série de preços de fechamento.
        period: Período do RSI.

    Returns:
        Valor do RSI.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def _fetch_market_data(query: str) -> str:
    """Busca dados recentes de mercado via yfinance.

    Args:
        query: Texto com ticker ou período (ex: "PETR4.SA últimos 5 dias").

    Returns:
        String com dados de mercado formatados.
    """
    try:
        ticker = "PETR4.SA"

        hist = yf.download(ticker, period="1mo", progress=False)

        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        if hist.empty:
            return f"Nenhum dado encontrado para {ticker}"

        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else latest

        daily_change = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100

        # Indicadores básicos
        sma_20 = hist["Close"].tail(20).mean()
        sma_50 = hist["Close"].tail(50).mean() if len(hist) >= 50 else None
        rsi = _calc_rsi(hist["Close"], 14)

        # Volume médio
        avg_volume = hist["Volume"].tail(20).mean()

        result = (
            f"Dados de Mercado — {ticker}\n"
            f"Data: {hist.index[-1].strftime('%Y-%m-%d')}\n"
            f"- Fechamento: R$ {latest['Close']:.2f}\n"
            f"- Abertura: R$ {latest['Open']:.2f}\n"
            f"- Máxima: R$ {latest['High']:.2f}\n"
            f"- Mínima: R$ {latest['Low']:.2f}\n"
            f"- Volume: {latest['Volume']:,.0f}\n"
            f"- Variação diária: {daily_change:+.2f}%\n"
            f"- SMA(20): R$ {sma_20:.2f}\n"
        )

        if sma_50:
            result += f"- SMA(50): R$ {sma_50:.2f}\n"
        result += f"- RSI(14): {rsi:.1f}\n"
        result += f"- Volume médio (20d): {avg_volume:,.0f}\n"

        # Variação no mês
        month_change = (
            (latest["Close"] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]
        ) * 100
        result += f"- Variação no mês: {month_change:+.2f}%"

        return result

    except Exception as e:
        logger.error("Erro ao buscar dados de mercado: %s", e)
        return f"Erro ao buscar dados: {e}"


fetch_market_tool = Tool(
    name="fetch_market_data",
    func=_fetch_market_data,
    description=(
        "Busca dados recentes de mercado da Petrobras (PETR4.SA) incluindo preço atual, "
        "variação, volume, médias móveis (SMA 20/50) e RSI. Use quando o usuário "
        "perguntar sobre o preço atual, dados de mercado ou indicadores técnicos."
    ),
)


# --- Tool 3: Busca RAG ---


def _search_financial_docs(query: str) -> str:
    """Busca em documentos financeiros indexados via RAG.

    Args:
        query: Pergunta sobre a empresa ou mercado.

    Returns:
        Contextos relevantes recuperados.
    """
    try:
        from src.agent.rag_pipeline import RAGPipeline

        rag = RAGPipeline()
        results = rag.retrieve(query, top_k=3)

        if not results:
            return "Nenhum documento relevante encontrado para esta consulta."

        formatted = "Documentos relevantes encontrados:\n\n"
        for i, doc in enumerate(results, 1):
            formatted += f"[{i}] {doc.page_content[:500]}\n"
            if doc.metadata:
                formatted += f"    Fonte: {doc.metadata.get('source', 'N/A')}\n\n"

        return formatted

    except Exception as e:
        logger.error("Erro na busca RAG: %s", e)
        return f"Erro ao buscar documentos: {e}"


search_docs_tool = Tool(
    name="search_financial_docs",
    func=_search_financial_docs,
    description=(
        "Busca informações em documentos financeiros indexados sobre a Petrobras, "
        "incluindo relatórios, análises de mercado e documentação do modelo. "
        "Use quando o usuário fizer perguntas sobre a empresa, resultados financeiros, "
        "estratégia, governança, ou sobre o próprio modelo preditivo."
    ),
)


# --- Tool 4: Comparar Versões de Modelo ---


def _compare_model_versions(query: str) -> str:
    """Compara versões do modelo no MLflow Registry.

    Args:
        query: Texto (ignorado — sempre compara champion vs latest).

    Returns:
        Comparação formatada entre versões.
    """
    try:
        import mlflow

        client = mlflow.tracking.MlflowClient()
        model_name = "lstm-petr4"

        versions = client.search_model_versions(f"name='{model_name}'")

        if not versions:
            return "Nenhuma versão encontrada no Model Registry."

        result = f"Versões do modelo '{model_name}':\n\n"

        for v in sorted(versions, key=lambda x: int(x.version), reverse=True)[:5]:
            run = client.get_run(v.run_id)
            metrics = run.data.metrics

            result += (
                f"v{v.version} (stage: {v.current_stage})\n"
                f"  - MAE: {metrics.get('mae', 'N/A')}\n"
                f"  - RMSE: {metrics.get('rmse', 'N/A')}\n"
                f"  - MAPE: {metrics.get('mape', 'N/A')}\n"
                f"  - Criado em: {v.creation_timestamp}\n\n"
            )

        return result

    except Exception as e:
        logger.error("Erro ao comparar modelos: %s", e)
        return f"Erro ao acessar MLflow Registry: {e}"


compare_models_tool = Tool(
    name="compare_model_versions",
    func=_compare_model_versions,
    description=(
        "Compara versões do modelo LSTM no MLflow Model Registry, mostrando "
        "métricas (MAE, RMSE, MAPE) de cada versão. Use quando o usuário "
        "perguntar sobre performance do modelo, versões ou histórico de treinamento."
    ),
)


# --- Lista de todas as tools ---

ALL_TOOLS = [
    predict_stock_tool,
    fetch_market_tool,
    search_docs_tool,
    compare_models_tool,
]
