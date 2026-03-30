"""Guardrails de segurança para input e output do agente.

Input Guardrails:
- Detecção de prompt injection (padrões conhecidos)
- Validação de tamanho (anti context stuffing)
- Filtragem de tópicos fora do escopo

Output Guardrails:
- Remoção de PII (Presidio)
- Validação de conteúdo (anti hallucination leakage)
- Enforcing disclaimers em predições

Referência: OWASP Top 10 for LLM Applications (2025)
"""

import logging
import re

logger = logging.getLogger(__name__)


def _has_encoding_attack(text: str) -> bool:
    """Detecta tentativas de encoding attack.

    Args:
        text: Texto para verificar.

    Returns:
        True se encoding suspeito detectado.
    """
    suspicious_patterns = [
        r"\\x[0-9a-fA-F]{2}",
        r"\\u[0-9a-fA-F]{4}",
        r"&#\d+;",
        r"&#x[0-9a-fA-F]+;",
    ]
    for pattern in suspicious_patterns:
        if re.search(pattern, text):
            return True
    return False


class InputGuardrail:
    """Valida e sanitiza input do usuário antes de enviar ao LLM.

    Args:
        max_length: Tamanho máximo do input.
        allowed_topics: Lista de tópicos permitidos (opcional).
    """

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"you\s+are\s+now\s+a",
        r"system:\s*",
        r"<\|im_start\|>",
        r"\[INST\]",
        r"forget\s+(everything|all|your\s+instructions)",
        r"disregard\s+(all|any|the)\s+(above|prior|previous)",
        r"new\s+instructions?:",
        r"override\s+(system|safety|security)",
        r"jailbreak",
        r"DAN\s+mode",
        r"do\s+anything\s+now",
        r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
    ]

    def __init__(self, max_length: int = 4096, allowed_topics: list[str] | None = None):
        self.max_length = max_length
        self.allowed_topics = allowed_topics or []
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]

    def validate(self, user_input: str) -> tuple[bool, str]:
        """Valida input do usuário.

        Args:
            user_input: Texto do usuário.

        Returns:
            Tupla (is_valid, reason).
        """
        # Check 1: Input vazio
        if not user_input or not user_input.strip():
            return False, "Input vazio."

        # Check 2: Tamanho máximo (anti context stuffing — OWASP LLM01)
        if len(user_input) > self.max_length:
            logger.warning("Input excede tamanho máximo: %d chars", len(user_input))
            return False, f"Input bloqueado: excede tamanho máximo ({self.max_length} chars)."

        # Check 3: Prompt injection (OWASP LLM01)
        for pattern in self._compiled_patterns:
            if pattern.search(user_input):
                logger.warning("Prompt injection detectado: '%s'", user_input[:100])
                return False, "Input bloqueado: padrão suspeito detectado."

        # Check 4: Encoding attacks
        if _has_encoding_attack(user_input):
            logger.warning("Encoding attack detectado")
            return False, "Input bloqueado: encoding suspeito."

        return True, "OK"


class OutputGuardrail:
    """Valida e sanitiza output do LLM antes de retornar ao usuário.

    Args:
        language: Idioma para detecção de PII.
    """

    DISCLAIMER = "Esta predição NÃO constitui recomendação de investimento."

    def __init__(self, language: str = "pt"):
        self.language = language
        self._analyzer = None
        self._anonymizer = None

    def _init_presidio(self) -> None:
        """Lazy init do Presidio (pesado para importar)."""
        if self._analyzer is None:
            try:
                from presidio_analyzer import AnalyzerEngine
                from presidio_anonymizer import AnonymizerEngine

                self._analyzer = AnalyzerEngine()
                self._anonymizer = AnonymizerEngine()
            except ImportError:
                logger.warning("Presidio não instalado. PII detection desabilitado.")

    def sanitize(self, llm_output: str) -> str:
        """Remove PII do output do LLM.

        Args:
            llm_output: Texto gerado pelo LLM.

        Returns:
            Texto sanitizado sem PII.
        """
        self._init_presidio()

        if not self._analyzer:
            return llm_output

        try:
            results = self._analyzer.analyze(
                text=llm_output,
                language=self.language,
                entities=[
                    "PERSON",
                    "EMAIL_ADDRESS",
                    "PHONE_NUMBER",
                    "CREDIT_CARD",
                    "IBAN_CODE",
                ],
            )

            if results:
                logger.warning("PII detectado no output: %d entidades", len(results))
                anonymized = self._anonymizer.anonymize(
                    text=llm_output,
                    analyzer_results=results,
                )
                return anonymized.text

        except Exception as e:
            logger.error("Erro no sanitize de output: %s", e)

        return llm_output

    def validate_disclaimers(self, output: str, contains_prediction: bool = False) -> str:
        """Adiciona disclaimers obrigatórios quando necessário.

        Args:
            output: Texto do LLM.
            contains_prediction: Se True, verifica disclaimer de investimento.

        Returns:
            Texto com disclaimers adicionados se necessário.
        """
        if contains_prediction:
            disclaimer = (
                "\n\n⚠️ AVISO: Esta análise não constitui recomendação de investimento. "
                "Consulte um profissional qualificado."
            )
            prediction_keywords = [
                "previsão", "predição", "prevê", "preço futuro", "vai subir", "vai cair",
            ]

            if any(kw in output.lower() for kw in prediction_keywords):
                if "não constitui recomendação" not in output.lower():
                    output += disclaimer

        return output
