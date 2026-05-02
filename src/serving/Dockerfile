FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files needed for package installation
COPY pyproject.toml README.md ./
COPY src/ src/

# Install dependencies
RUN pip install --no-cache-dir -e .

# Copy runtime assets and create generated artifact directories.
# docker-compose mounts ./models, ./data, and ./configs over these paths.
COPY configs/ configs/
COPY data/rag_documents/ data/rag_documents/
RUN mkdir -p models data/raw data/processed data/rag_index metrics

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
