"""Pipeline RAG para busca em documentação financeira.

Implementa:
- Carregamento de documentos (PDF, MD, TXT)
- Chunking com overlap
- Embedding via sentence-transformers
- Vector store FAISS
- Retriever com score threshold

Referência: Lewis et al. (2020) — Retrieval-Augmented Generation
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Pipeline RAG para documentação financeira.

    Args:
        docs_dir: Diretório com documentos para indexar.
        embedding_model: Nome do modelo de embedding.
        index_path: Caminho para salvar/carregar o índice FAISS.
        chunk_size: Tamanho dos chunks de texto.
        chunk_overlap: Overlap entre chunks.
    """

    def __init__(
        self,
        docs_dir: str = "data/rag_documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: str = "data/rag_index",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.docs_dir = docs_dir
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Inicializar embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Carregar ou criar índice
        self.vectorstore = self._load_or_create_index()

    def _load_or_create_index(self) -> FAISS:
        """Carrega índice existente ou cria novo a partir dos documentos."""
        index_file = Path(self.index_path) / "index.faiss"

        if index_file.exists():
            logger.info("Carregando índice FAISS de %s", self.index_path)
            return FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

        logger.info("Criando novo índice FAISS a partir de %s", self.docs_dir)
        documents = self._load_documents()
        chunks = self._split_documents(documents)

        if not chunks:
            # Criar índice vazio com um documento placeholder
            logger.warning("Nenhum documento encontrado. Criando índice com placeholder.")
            chunks = [
                Document(
                    page_content="Sistema de análise de ações PETR4.SA usando LSTM.",
                    metadata={"source": "system"},
                )
            ]

        vectorstore = FAISS.from_documents(chunks, self.embeddings)

        # Salvar índice
        Path(self.index_path).mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(self.index_path)
        logger.info("Índice FAISS salvo em %s (%d chunks)", self.index_path, len(chunks))

        return vectorstore

    def _load_documents(self) -> list[Document]:
        """Carrega documentos de múltiplos formatos."""
        docs_path = Path(self.docs_dir)
        documents = []

        if not docs_path.exists():
            logger.warning("Diretório %s não encontrado", self.docs_dir)
            return documents

        # Carregar PDFs
        for pdf_file in docs_path.glob("**/*.pdf"):
            try:
                loader = PyPDFLoader(str(pdf_file))
                documents.extend(loader.load())
            except Exception as e:
                logger.warning("Erro ao carregar %s: %s", pdf_file, e)

        # Carregar Markdown
        for md_file in docs_path.glob("**/*.md"):
            try:
                loader = TextLoader(str(md_file), encoding="utf-8")
                documents.extend(loader.load())
            except Exception as e:
                logger.warning("Erro ao carregar %s: %s", md_file, e)

        # Carregar TXT
        for txt_file in docs_path.glob("**/*.txt"):
            try:
                loader = TextLoader(str(txt_file), encoding="utf-8")
                documents.extend(loader.load())
            except Exception as e:
                logger.warning("Erro ao carregar %s: %s", txt_file, e)

        logger.info("Documentos carregados: %d", len(documents))
        return documents

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        """Divide documentos em chunks com overlap.

        Args:
            documents: Lista de documentos carregados.

        Returns:
            Lista de chunks.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        logger.info("Chunks criados: %d (de %d documentos)", len(chunks), len(documents))
        return chunks

    def retrieve(self, query: str, top_k: int = 3) -> list[Document]:
        """Recupera documentos relevantes para a query.

        Args:
            query: Pergunta do usuário.
            top_k: Número máximo de documentos retornados.

        Returns:
            Lista de Documents com page_content e metadata.
        """
        results = self.vectorstore.similarity_search(query, k=top_k)
        logger.info("Retrieval: %d documentos para '%s'", len(results), query[:50])
        return results

    def retrieve_with_scores(
        self, query: str, top_k: int = 3
    ) -> list[tuple[Document, float]]:
        """Recupera documentos com scores de similaridade.

        Args:
            query: Pergunta do usuário.
            top_k: Número máximo de documentos.

        Returns:
            Lista de tuplas (Document, score).
        """
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        return results

    def add_documents(self, documents: list[Document]) -> None:
        """Adiciona documentos ao índice (incremental — não destrutivo).

        Seguindo a recomendação do GAP 03: upsert incremental,
        nunca flush + reload.

        Args:
            documents: Lista de novos documentos.
        """
        chunks = self._split_documents(documents)
        self.vectorstore.add_documents(chunks)
        self.vectorstore.save_local(self.index_path)
        logger.info("Adicionados %d chunks ao índice", len(chunks))
