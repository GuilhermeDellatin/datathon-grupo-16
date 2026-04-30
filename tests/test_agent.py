"""Testes do agente ReAct — tools e pipeline."""


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

    def test_pipeline_builds_index_from_clean_state(self, tmp_path):
        """Simula o passo do Dockerfile que gera `data/rag_index/` do zero.

        Garante que um build limpo (sem índice pré-existente no host) é
        capaz de criar o índice FAISS a partir dos documentos commitados.
        """
        from src.agent.rag_pipeline import RAGPipeline

        docs_dir = tmp_path / "rag_documents"
        docs_dir.mkdir()
        (docs_dir / "doc1.md").write_text(
            "# Petrobras\nA Petrobras é uma estatal brasileira de petróleo."
        )
        (docs_dir / "doc2.md").write_text(
            "# Glossário\nDividendos são distribuições de lucro aos acionistas."
        )

        index_path = tmp_path / "rag_index"
        # Deliberadamente não criar `index_path` — o pipeline deve construir.
        assert not index_path.exists()

        rag = RAGPipeline(docs_dir=str(docs_dir), index_path=str(index_path))

        assert (index_path / "index.faiss").exists()
        assert (index_path / "index.pkl").exists()

        results = rag.retrieve("dividendos", top_k=1)
        assert len(results) == 1
