"""Testes do agente ReAct — tools e pipeline."""

import pytest
from unittest.mock import patch, MagicMock

from src.agent.tools import ALL_TOOLS


class TestTools:
    """Testes das tools do agente."""

    def test_minimum_tools(self):
        """Deve haver >= 3 tools (requisito Datathon)."""
        assert len(ALL_TOOLS) >= 3

    def test_tool_names_unique(self):
        """Nomes das tools devem ser únicos."""
        names = [t.name for t in ALL_TOOLS]
        assert len(names) == len(set(names))

    def test_tool_descriptions_not_empty(self):
        """Todas as tools devem ter descrição."""
        for tool in ALL_TOOLS:
            assert tool.description
            assert len(tool.description) > 10

    def test_required_tools_present(self):
        """Tools obrigatórias devem estar presentes."""
        names = {t.name for t in ALL_TOOLS}
        required = {"predict_stock_price", "fetch_market_data", "search_financial_docs"}
        assert required.issubset(names)

    def test_four_tools_total(self):
        """Deve haver exatamente 4 tools."""
        assert len(ALL_TOOLS) == 4

    def test_compare_model_versions_tool(self):
        """Tool compare_model_versions deve existir."""
        names = {t.name for t in ALL_TOOLS}
        assert "compare_model_versions" in names


class TestRAGPipeline:
    """Testes do pipeline RAG."""

    def test_pipeline_init(self):
        """RAGPipeline deve ser instanciável."""
        from src.agent.rag_pipeline import RAGPipeline

        rag = RAGPipeline()
        assert rag is not None

    def test_retrieve_returns_list(self):
        """Retrieve deve retornar lista."""
        from src.agent.rag_pipeline import RAGPipeline

        rag = RAGPipeline()
        results = rag.retrieve("O que é a Petrobras?", top_k=2)
        assert isinstance(results, list)

    def test_retrieve_with_scores_returns_list(self):
        """Retrieve com scores deve retornar lista."""
        from src.agent.rag_pipeline import RAGPipeline

        rag = RAGPipeline()
        results = rag.retrieve_with_scores("Preço do petróleo", top_k=2)
        assert isinstance(results, list)
