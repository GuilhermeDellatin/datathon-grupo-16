# Plano de Conformidade LGPD — Datathon LSTM Stock Predictor

Referência: Lei Geral de Proteção de Dados (Lei 13.709/2018)

---

## 1. Base Legal para Tratamento de Dados

### Art. 7, inciso IX — Legítimo Interesse

O tratamento de dados neste sistema se enquadra no **legítimo interesse** (Art. 7, IX) para fins de:
- Pesquisa acadêmica e desenvolvimento tecnológico (Art. 7, IV)
- Análise de dados públicos de mercado financeiro (dados anonimizados e agregados)

### Dados de mercado (OHLCV)
- São dados **públicos** disponibilizados pela B3 e acessíveis via yfinance.
- Não constituem dados pessoais segundo a LGPD (Art. 5, I).

### Dados de interação do usuário
- Queries enviadas ao agente podem conter dados pessoais.
- Base legal: **consentimento** (Art. 7, I) — usuário consente ao utilizar o sistema.

---

## 2. Dados Tratados

| Tipo de Dado | Fonte | Pessoal? | Tratamento |
|-------------|-------|----------|------------|
| Preços OHLCV (PETR4.SA) | B3 via yfinance | Não | Coleta, processamento, armazenamento |
| Indicadores técnicos | Calculados | Não | Processamento |
| Queries do usuário | Input do usuário | Possível | Processamento em memória, não persistido |
| Respostas do agente | LLM (OpenAI) | Possível | Sanitização via Presidio antes de retorno |
| Logs de API | Sistema | Possível | Armazenamento com retenção limitada |

---

## 3. Anonimização e Proteção de PII

### Mecanismos implementados

1. **Presidio (Microsoft)**:
   - Detecta: PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, IBAN_CODE
   - Anonimiza automaticamente no output do agente (`OutputGuardrail.sanitize()`)

2. **BrazilianPIIDetector**:
   - Detecta: CPF (com validação de dígitos verificadores), CNPJ, telefone BR
   - Substitui por placeholders: `<BR_CPF>`, `<BR_CNPJ>`, `<BR_PHONE>`

3. **InputGuardrail**:
   - Não persiste inputs do usuário em disco
   - Bloqueia tentativas de extrair PII do sistema

### Fluxo de dados
```
Input → InputGuardrail → LLM → OutputGuardrail (Presidio + BR PII) → Usuário
```

---

## 4. Direitos dos Titulares (Art. 18)

| Direito | Implementação |
|---------|--------------|
| **Acesso** (Art. 18, II) | Queries não são persistidas. Dados de mercado são públicos. |
| **Correção** (Art. 18, III) | Não aplicável — sistema não mantém cadastro de usuários. |
| **Eliminação** (Art. 18, VI) | Logs podem ser eliminados sob solicitação. |
| **Portabilidade** (Art. 18, V) | Dados de mercado são públicos e acessíveis por qualquer pessoa. |
| **Informação sobre compartilhamento** (Art. 18, VII) | Documentado na seção 5. |
| **Revogação de consentimento** (Art. 18, IX) | Usuário pode parar de usar o sistema a qualquer momento. |

---

## 5. Compartilhamento de Dados

| Destinatário | Dados compartilhados | Finalidade | Base legal |
|-------------|---------------------|------------|------------|
| **OpenAI API** | Query do usuário (sanitizada) | Geração de resposta do agente | Consentimento + Legítimo interesse |
| **yfinance (Yahoo Finance)** | Ticker (PETR4.SA) | Coleta de dados de mercado | Dados públicos |
| **MLflow** | Métricas e parâmetros do modelo | Experiment tracking | Dados não pessoais |
| **Prometheus/Grafana** | Métricas operacionais | Monitoramento | Dados não pessoais |

### Medidas de proteção no compartilhamento
- Queries são sanitizadas pelo InputGuardrail antes de envio à OpenAI.
- API key da OpenAI gerenciada via `.env`, nunca exposta.
- Comunicação via HTTPS.

---

## 6. Política de Retenção

| Dado | Retenção | Justificativa |
|------|----------|--------------|
| Queries do usuário | Não persistidas | Processadas apenas em memória |
| Logs de API | 30 dias | Diagnóstico e auditoria |
| Dados de mercado | Enquanto necessário para treinamento | Dados públicos |
| Métricas Prometheus | 15 dias (padrão Prometheus) | Monitoramento operacional |
| Artefatos MLflow | Indefinido | Reprodutibilidade e auditoria de modelo |
| Índice FAISS | Enquanto documentos forem relevantes | Busca RAG |

---

## 7. Procedimento em Caso de Incidente

### Plano de resposta a incidentes (Art. 48)

1. **Detecção**: Monitoramento via Prometheus/Grafana, logs estruturados.
2. **Contenção**: Desativar endpoints afetados, rotacionar API keys.
3. **Avaliação**: Determinar dados afetados, titulares impactados.
4. **Notificação**: Comunicar ANPD e titulares afetados em prazo razoável (Art. 48, §1º).
5. **Remediação**: Corrigir vulnerabilidade, atualizar guardrails.
6. **Documentação**: Registrar incidente, ações tomadas e lições aprendidas.

### Classificação de severidade
- **Baixa**: Acesso não autorizado a logs sem PII.
- **Média**: Vazamento de PII de poucos titulares via output do LLM.
- **Alta**: Comprometimento de API keys ou dados em massa.

---

## 8. Encarregado de Proteção de Dados (DPO)

- **Responsável**: Líder do Grupo 16
- **Contato**: grupo@example.com
- **Atribuições** (Art. 41):
  - Aceitar reclamações e comunicações dos titulares
  - Receber comunicações da ANPD
  - Orientar equipe sobre práticas de proteção de dados
  - Executar demais atribuições determinadas pelo controlador
