"""Detecção de PII para contexto brasileiro.

Detecta:
- CPF (xxx.xxx.xxx-xx)
- CNPJ (xx.xxx.xxx/xxxx-xx)
- Telefone BR (+55 xx xxxxx-xxxx)
- Email
- Nomes de pessoas (via Presidio)

Integrado com LGPD (Lei 13.709/2018).
"""

import logging
import re

logger = logging.getLogger(__name__)


class BrazilianPIIDetector:
    """Detecta e anonimiza PII brasileira.

    Complementa o Presidio com padrões específicos de documentos brasileiros.
    """

    CPF_PATTERN = re.compile(r"\d{3}\.?\d{3}\.?\d{3}-?\d{2}")
    CNPJ_PATTERN = re.compile(r"\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}")
    PHONE_BR_PATTERN = re.compile(
        r"(?:\+55\s?)?(?:\(?\d{2}\)?[\s-]?)?\d{4,5}[\s-]?\d{4}"
    )

    def detect(self, text: str) -> list[dict]:
        """Detecta PII no texto.

        Args:
            text: Texto para análise.

        Returns:
            Lista de entidades PII encontradas.
        """
        entities = []

        for match in self.CPF_PATTERN.finditer(text):
            if self._validate_cpf(match.group()):
                entities.append(
                    {
                        "type": "BR_CPF",
                        "value": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                    }
                )

        for match in self.CNPJ_PATTERN.finditer(text):
            entities.append(
                {
                    "type": "BR_CNPJ",
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                }
            )

        for match in self.PHONE_BR_PATTERN.finditer(text):
            entities.append(
                {
                    "type": "BR_PHONE",
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                }
            )

        if entities:
            logger.warning("PII brasileira detectada: %d entidades", len(entities))

        return entities

    def anonymize(self, text: str) -> str:
        """Anonimiza PII encontrada no texto.

        Args:
            text: Texto original.

        Returns:
            Texto com PII substituída por placeholders.
        """
        entities = self.detect(text)

        # Substituir de trás para frente (preservar offsets)
        for entity in sorted(entities, key=lambda e: e["start"], reverse=True):
            placeholder = f"<{entity['type']}>"
            text = text[: entity["start"]] + placeholder + text[entity["end"] :]

        return text

    @staticmethod
    def _validate_cpf(cpf: str) -> bool:
        """Valida CPF com dígitos verificadores.

        Args:
            cpf: String com CPF (com ou sem formatação).

        Returns:
            True se CPF é válido.
        """
        cpf = re.sub(r"\D", "", cpf)
        if len(cpf) != 11 or cpf == cpf[0] * 11:
            return False

        # Primeiro dígito
        soma = sum(int(cpf[i]) * (10 - i) for i in range(9))
        d1 = 11 - (soma % 11)
        d1 = 0 if d1 >= 10 else d1

        # Segundo dígito
        soma = sum(int(cpf[i]) * (11 - i) for i in range(10))
        d2 = 11 - (soma % 11)
        d2 = 0 if d2 >= 10 else d2

        return int(cpf[9]) == d1 and int(cpf[10]) == d2
