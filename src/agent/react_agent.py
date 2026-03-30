"""Agente ReAct para análise de ações da Petrobras.

Combina raciocínio (Thought) e ação (Action) para responder
perguntas sobre preços, mercado e documentação financeira.

Referência: Yao et al. (2023) — ReAct: Synergizing Reasoning and Acting
"""

import logging
import os

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from src.agent.tools import ALL_TOOLS

load_dotenv()

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Você é um assistente especializado em análise de ações da Petrobras (PETR4.SA).
Você tem acesso a um modelo LSTM treinado para predição de preços, dados de mercado em tempo real,
documentação financeira indexada e histórico de versões do modelo.

REGRAS IMPORTANTES:
1. Sempre que fizer uma predição de preço, inclua o aviso de que NÃO é recomendação de investimento.
2. Use dados reais do mercado para contextualizar suas análises.
3. Quando buscar documentos, sintetize as informações de forma clara.
4. Se não tiver certeza de algo, diga explicitamente.
5. Responda SEMPRE em português brasileiro.

Ferramentas disponíveis:
{tools}

Use o formato:
Thought: pensar sobre o que fazer para responder a pergunta
Action: nome_da_ferramenta
Action Input: input para a ferramenta
Observation: resultado da ferramenta
... (repita Thought/Action/Observation quantas vezes necessário)
Thought: Agora sei a resposta final
Final Answer: resposta completa e bem formatada para o usuário

Nomes das ferramentas disponíveis: {tool_names}

Pergunta: {input}
{agent_scratchpad}"""

REACT_PROMPT = PromptTemplate.from_template(SYSTEM_PROMPT)


def create_stock_agent(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_iterations: int = 10,
    verbose: bool = True,
) -> AgentExecutor:
    """Cria agente ReAct para análise de ações.

    Args:
        model_name: Modelo OpenAI a utilizar.
        temperature: Temperatura de geração (0.0 = determinístico).
        max_iterations: Máximo de iterações do agente.
        verbose: Se True, mostra raciocínio do agente.

    Returns:
        AgentExecutor configurado com 4 tools.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY não configurada. Verifique o .env")

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
    )

    tools = ALL_TOOLS
    if len(tools) < 3:
        logger.warning("Datathon exige >= 3 tools. Fornecidas: %d", len(tools))

    agent = create_react_agent(llm=llm, tools=tools, prompt=REACT_PROMPT)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
        return_intermediate_steps=True,  # Para telemetria
    )

    logger.info(
        "Agente ReAct criado com %d tools: %s",
        len(tools),
        [t.name for t in tools],
    )

    return executor


def query_agent(agent: AgentExecutor, question: str) -> dict:
    """Executa query no agente e retorna resposta estruturada.

    Args:
        agent: AgentExecutor configurado.
        question: Pergunta do usuário.

    Returns:
        Dicionário com output, intermediate_steps e metadata.
    """
    try:
        result = agent.invoke({"input": question})

        return {
            "answer": result.get("output", ""),
            "intermediate_steps": [
                {
                    "tool": step[0].tool,
                    "tool_input": step[0].tool_input,
                    "output": str(step[1])[:500],
                }
                for step in result.get("intermediate_steps", [])
            ],
            "success": True,
        }

    except Exception as e:
        logger.error("Erro no agente: %s", e)
        return {
            "answer": f"Desculpe, ocorreu um erro ao processar sua pergunta: {e}",
            "intermediate_steps": [],
            "success": False,
        }
