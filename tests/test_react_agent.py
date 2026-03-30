"""Testes do agente ReAct — mock OpenAI para não depender de API externa."""

import pytest
from unittest.mock import patch, MagicMock

from src.agent.react_agent import REACT_PROMPT, SYSTEM_PROMPT


class TestReActPrompt:
    """Testes do prompt do agente."""

    def test_prompt_contains_required_variables(self):
        """Prompt deve conter variáveis requeridas pelo LangChain."""
        assert "{tools}" in SYSTEM_PROMPT
        assert "{tool_names}" in SYSTEM_PROMPT
        assert "{input}" in SYSTEM_PROMPT
        assert "{agent_scratchpad}" in SYSTEM_PROMPT

    def test_prompt_template_valid(self):
        """PromptTemplate deve ser válido."""
        assert REACT_PROMPT is not None
        assert hasattr(REACT_PROMPT, "format")

    def test_prompt_has_rules(self):
        """Prompt deve conter regras do sistema."""
        assert "português" in SYSTEM_PROMPT.lower() or "brasileiro" in SYSTEM_PROMPT.lower()
        assert "investimento" in SYSTEM_PROMPT.lower()


class TestCreateStockAgent:
    """Testes de criação do agente."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False)
    def test_missing_api_key_raises(self):
        """Deve levantar ValueError sem API key."""
        from src.agent.react_agent import create_stock_agent

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            create_stock_agent()

    @patch("src.agent.react_agent.ChatOpenAI")
    @patch("src.agent.react_agent.create_react_agent")
    @patch("src.agent.react_agent.AgentExecutor")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False)
    def test_creates_agent_with_tools(self, mock_executor_class, mock_create, mock_llm_class):
        """Deve criar agente com 4 tools."""
        mock_llm_class.return_value = MagicMock()
        mock_create.return_value = MagicMock()
        mock_executor_class.return_value = MagicMock()

        from src.agent.react_agent import create_stock_agent

        agent = create_stock_agent(verbose=False)
        assert agent is not None
        mock_create.assert_called_once()


class TestQueryAgent:
    """Testes de execução de query no agente."""

    def test_query_agent_success(self):
        """query_agent deve retornar dict estruturado."""
        from src.agent.react_agent import query_agent

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "output": "O preço da PETR4 é R$ 35.50.",
            "intermediate_steps": [],
        }

        result = query_agent(mock_agent, "Qual o preço da PETR4?")
        assert result["success"]
        assert "answer" in result
        assert isinstance(result["intermediate_steps"], list)

    def test_query_agent_error_handling(self):
        """query_agent deve tratar erros graciosamente."""
        from src.agent.react_agent import query_agent

        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = Exception("API error")

        result = query_agent(mock_agent, "teste")
        assert not result["success"]
        assert "erro" in result["answer"].lower()
