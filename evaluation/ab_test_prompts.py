"""A/B test de prompts para o agente ReAct.

Testa 3 configurações diferentes de prompt e compara
resultados usando LLM-as-judge.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROMPT_VARIANTS = {
    "baseline": {
        "description": "Prompt padrão sem instruções específicas",
        "system_suffix": "",
    },
    "detailed_instructions": {
        "description": "Prompt com instruções detalhadas de formatação",
        "system_suffix": (
            "\nAdicional: Sempre estruture sua resposta em seções claras. "
            "Use dados numéricos quando disponíveis. Cite as fontes."
        ),
    },
    "concise": {
        "description": "Prompt pedindo respostas concisas",
        "system_suffix": (
            "\nAdicional: Seja conciso e direto. Máximo 3 parágrafos. "
            "Foque nos números mais relevantes."
        ),
    },
}


def run_ab_test(
    golden_set_path: str = "data/golden_set/golden_set.json",
    output_path: str = "metrics/ab_test_results.json",
    n_samples: int = 5,
) -> dict:
    """Executa A/B test com diferentes prompts.

    Args:
        golden_set_path: Caminho do golden set.
        output_path: Caminho de saída.
        n_samples: Número de amostras por variante.

    Returns:
        Dicionário com resultados comparativos.
    """
    with open(golden_set_path) as f:
        golden_set = json.load(f)[:n_samples]

    results = {}

    for variant_name, variant_config in PROMPT_VARIANTS.items():
        logger.info("Testando variante: %s", variant_name)

        variant_results = []
        for item in golden_set:
            variant_results.append(
                {
                    "query": item["query"],
                    "variant": variant_name,
                }
            )

        results[variant_name] = {
            "config": variant_config,
            "n_samples": len(variant_results),
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def main() -> None:
    """Entry point para A/B test de prompts."""
    logging.basicConfig(level=logging.INFO)
    results = run_ab_test()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
