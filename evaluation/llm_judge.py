"""Avaliação via LLM-as-judge com 5 critérios.

Critérios:
1. Correção Técnica — A resposta é factualmente correta?
2. Relevância — A resposta aborda diretamente a pergunta?
3. Clareza — A resposta é clara e bem estruturada?
4. (Negócio) Utilidade para Investidor — A resposta auxilia na tomada de decisão?
5. (Segurança) Presença de Disclaimers — Quando aplicável, inclui avisos de risco?
"""

import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
logger = logging.getLogger(__name__)

JUDGE_PROMPT = """Você é um avaliador especializado em sistemas de análise financeira.
Avalie a resposta do assistente para a pergunta dada, considerando o contexto esperado.

PERGUNTA: {question}
RESPOSTA ESPERADA: {expected_answer}
RESPOSTA DO ASSISTENTE: {actual_answer}

Avalie nos seguintes critérios (nota de 1 a 5):

1. **Correção Técnica**: A resposta é factualmente correta? Contém erros?
2. **Relevância**: A resposta aborda diretamente a pergunta feita?
3. **Clareza**: A resposta é clara, bem organizada e fácil de entender?
4. **Utilidade para Investidor**: A resposta fornece informações úteis para tomada de decisão?
5. **Disclaimers de Risco**: Quando aplicável, a resposta inclui avisos de que não é recomendação \
de investimento?

Responda APENAS com JSON no formato:
{{
  "technical_correctness": {{"score": N, "justification": "..."}},
  "relevance": {{"score": N, "justification": "..."}},
  "clarity": {{"score": N, "justification": "..."}},
  "investor_utility": {{"score": N, "justification": "..."}},
  "risk_disclaimers": {{"score": N, "justification": "..."}},
  "overall_score": N,
  "overall_feedback": "..."
}}"""


def evaluate_with_llm_judge(
    golden_set_path: str = "data/golden_set/golden_set.json",
    output_path: str = "metrics/llm_judge_metrics.json",
    model: str = "gpt-4o-mini",
) -> dict:
    """Avalia respostas do agente usando LLM-as-judge.

    Args:
        golden_set_path: Caminho do golden set.
        output_path: Caminho de saída das métricas.
        model: Modelo para o juiz.

    Returns:
        Dicionário com scores médios e detalhados.
    """
    client = OpenAI()

    with open(golden_set_path) as f:
        golden_set = json.load(f)

    # Gerar respostas do agente
    from src.agent.react_agent import create_stock_agent, query_agent

    agent = create_stock_agent(verbose=False)

    all_scores = []

    for item in golden_set:
        response = query_agent(agent, item["query"])

        prompt = JUDGE_PROMPT.format(
            question=item["query"],
            expected_answer=item["expected_answer"],
            actual_answer=response["answer"],
        )

        try:
            judge_response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            scores = json.loads(judge_response.choices[0].message.content)
            scores["query_id"] = item["id"]
            scores["query"] = item["query"]
            all_scores.append(scores)

            logger.info(
                "Avaliado '%s': overall=%.1f",
                item["query"][:40],
                scores.get("overall_score", 0),
            )

        except Exception as e:
            logger.error("Erro ao avaliar '%s': %s", item["query"][:40], e)

    # Calcular médias
    criteria = [
        "technical_correctness",
        "relevance",
        "clarity",
        "investor_utility",
        "risk_disclaimers",
    ]

    summary = {}
    for criterion in criteria:
        scores_list = [
            s[criterion]["score"]
            for s in all_scores
            if criterion in s and "score" in s[criterion]
        ]
        if scores_list:
            summary[f"avg_{criterion}"] = sum(scores_list) / len(scores_list)

    overall_scores = [s["overall_score"] for s in all_scores if "overall_score" in s]
    if overall_scores:
        summary["avg_overall"] = sum(overall_scores) / len(overall_scores)

    summary["n_evaluated"] = len(all_scores)

    result = {"summary": summary, "detailed": all_scores}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("LLM Judge summary: %s", summary)
    return result


def main() -> None:
    """Entry point para avaliação LLM-as-judge."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    result = evaluate_with_llm_judge()
    print(f"\nLLM Judge Summary:\n{json.dumps(result['summary'], indent=2)}")


if __name__ == "__main__":
    main()
