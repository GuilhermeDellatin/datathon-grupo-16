"""Script para indexar documentos no FAISS."""

import logging
import shutil
from pathlib import Path

from src.agent.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)


def main() -> None:
    """Indexa documentos financeiros para o RAG pipeline."""
    index_path = Path("data/rag_index")
    if index_path.exists():
        shutil.rmtree(index_path)

    rag = RAGPipeline(docs_dir="data/rag_documents")
    print(f"Índice criado com sucesso em data/rag_index/")


if __name__ == "__main__":
    main()
