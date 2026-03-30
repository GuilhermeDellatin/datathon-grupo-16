.PHONY: help install lint test train serve data clean docker-up docker-down

help: ## Mostra ajuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Instala dependências
	pip install -e ".[dev]"
	pre-commit install

lint: ## Roda linters
	ruff check src/ tests/ evaluation/
	mypy src/ --ignore-missing-imports

test: ## Roda testes com coverage
	pytest tests/ -x --cov=src --cov-report=term-missing --cov-fail-under=60

test-fast: ## Roda testes excluindo lentos
	pytest tests/ -x -m "not slow" --cov=src --cov-report=term-missing

data: ## Coleta e processa dados via DVC
	dvc repro

train: ## Treina modelo LSTM
	python -m src.models.train

serve: ## Sobe API local
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

evaluate: ## Roda avaliação RAGAS + LLM-as-judge
	python -m evaluation.ragas_eval
	python -m evaluation.llm_judge

docker-up: ## Sobe stack completa (API + Prometheus + Grafana)
	docker compose up -d --build

docker-down: ## Para stack
	docker compose down

clean: ## Limpa artefatos
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov
