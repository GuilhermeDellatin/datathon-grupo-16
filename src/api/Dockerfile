FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Source code, configs e scripts (todos commitados em git).
COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Documentos do RAG (commitados, fora do .gitignore).
COPY data/rag_documents/ data/rag_documents/

# Diretórios populados em runtime (via volume bind ou pelo passo seguinte).
# Criados vazios para que o build não dependa de artefatos do host:
#   - /app/models             → checkpoint LSTM (montado em produção)
#   - /app/data/rag_index     → índice FAISS gerado abaixo
#   - /app/data/processed     → features versionadas via DVC (volume)
#   - /app/metrics            → outputs de quality gate / ragas / llm-judge
RUN mkdir -p /app/models \
    /app/data/rag_index \
    /app/data/processed \
    /app/metrics

# Instala dependências.
RUN pip install --no-cache-dir -e .

# Pré-constrói o índice FAISS a partir de data/rag_documents/.
# Mantém o build determinístico: clones limpos não dependem de
# `data/rag_index/` no host. Em ambientes que montam ./data via bind
# (docker-compose), o índice do host tem precedência — o pré-build
# garante que builds standalone (`docker build && docker run`) também
# subam o serviço com RAG funcional.
RUN python scripts/index_documents.py

EXPOSE 8000

# Liveness check (Kubernetes-style: GET / sempre retorna 200 quando o processo
# está vivo, sem checar dependências). /ready é usado por orquestradores que
# precisam saber se a API consegue servir tráfego.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
