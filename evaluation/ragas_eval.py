"""Avaliação do pipeline RAG com RAGAS — 4 métricas obrigatórias.

Métricas:
1. Faithfulness — Resposta fiel aos contextos recuperados?
2. Answer Relevancy — Resposta relevante para a pergunta?
3. Context Precision — Contextos recuperados são precisos?
4. Context Recall — Contextos cobrem a resposta esperada?

Referência: Es et al. (2024) — RAGAS: Automated Evaluation of RAG
"""

import json
import logging
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

logger = logging.getLogger(__name__)


def load_golden_set(path: str = "data/golden_set/golden_set.json") -> list[dict]:
    """Carrega golden set.

    Args:
        path: Caminho para o JSON.

    Returns:
        Lista de dicionários com query, expected_answer, contexts.
    """
    with open(path) as f:
        golden_set = json.load(f)
    logger.info("Golden set carregado: %d pares", len(golden_set))
    return golden_set


def generate_rag_responses(golden_set: list[dict]) -> list[dict]:
    """Gera respostas do pipeline RAG para cada query do golden set.

    Args:
        golden_set: Lista de pares do golden set.

    Returns:
        Lista com respostas e contextos gerados pelo RAG.
    """
    from src.agent.rag_pipeline import RAGPipeline
    from src.agent.react_agent import create_stock_agent, query_agent

    rag = RAGPipeline()
    results = []

    for item in golden_set:
        query = item["query"]

        # Recuperar contextos via RAG
        docs = rag.retrieve(query, top_k=3)
        contexts = [doc.page_content for doc in docs]

        # Gerar resposta via agente
        try:
            agent = create_stock_agent(verbose=False)
            response = query_agent(agent, query)
            answer = response["answer"]
        except Exception as e:
            logger.warning("Erro ao gerar resposta para '%s': %s", query[:50], e)
            answer = "Erro ao processar a pergunta."

        results.append(
            {
                "question": query,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item["expected_answer"],
            }
        )

        logger.info("Processado: %s", query[:50])

    return results


def evaluate_rag_pipeline(
    golden_set_path: str = "data/golden_set/golden_set.json",
    output_path: str = "metrics/ragas_metrics.json",
) -> dict[str, float]:
    """Avalia pipeline RAG com RAGAS.

    Args:
        golden_set_path: Caminho para golden set.
        output_path: Caminho para salvar métricas.

    Returns:
        Dicionário com 4 métricas RAGAS.
    """
    golden_set = load_golden_set(golden_set_path)
    results = generate_rag_responses(golden_set)

    dataset = Dataset.from_list(results)

    # Avaliação RAGAS — 4 métricas obrigatórias
    scores = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    metrics = {
        "faithfulness": float(scores["faithfulness"]),
        "answer_relevancy": float(scores["answer_relevancy"]),
        "context_precision": float(scores["context_precision"]),
        "context_recall": float(scores["context_recall"]),
        "n_samples": len(golden_set),
    }

    # Salvar métricas
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("RAGAS scores: %s", metrics)

    # Log no MLflow
    try:
        import mlflow

        mlflow.set_experiment("datathon-petr4-evaluation")
        with mlflow.start_run(run_name="ragas-eval"):
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(output_path)
    except Exception as e:
        logger.warning("Não foi possível logar no MLflow: %s", e)

    return metrics


def main() -> None:
    """Entry point para avaliação RAGAS."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    metrics = evaluate_rag_pipeline()
    print(f"\nResultados RAGAS:\n{json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
