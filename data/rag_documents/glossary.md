# Glossário de Termos — Modelo LSTM e Análise Financeira

## Indicadores Técnicos

### SMA (Simple Moving Average)
Média Móvel Simples. Calcula a média aritmética dos preços de fechamento em uma janela de N dias. SMA(20) usa os últimos 20 dias. Serve como indicador de tendência: preço acima da SMA indica tendência de alta.

### EMA (Exponential Moving Average)
Média Móvel Exponencial. Similar à SMA, mas dá mais peso a dados recentes. EMA(12) e EMA(26) são usadas no cálculo do MACD. Reage mais rápido a mudanças de preço que a SMA.

### RSI (Relative Strength Index)
Índice de Força Relativa. Oscilador que mede a velocidade e magnitude de mudanças de preço. Varia de 0 a 100. RSI > 70 indica sobrecompra (possível reversão para baixo). RSI < 30 indica sobrevenda (possível reversão para cima). RSI(14) usa 14 períodos.

### MACD (Moving Average Convergence Divergence)
Indicador de momento que mostra a relação entre duas EMAs. MACD = EMA(12) - EMA(26). A linha de sinal é uma EMA(9) do MACD. Cruzamento do MACD acima da linha de sinal é sinal de compra; abaixo é sinal de venda.

### Bollinger Bands
Bandas de volatilidade. Banda central = SMA(20). Banda superior = SMA(20) + 2*desvio padrão. Banda inferior = SMA(20) - 2*desvio padrão. Preço próximo da banda superior indica possível sobrecompra; próximo da inferior indica sobrevenda. Bandas estreitas indicam baixa volatilidade (possível breakout).

## Métricas de Modelo

### MAE (Mean Absolute Error)
Erro Absoluto Médio. Média dos valores absolutos das diferenças entre predição e valor real. Fácil de interpretar: "o modelo erra em média X reais". Não penaliza erros grandes proporcionalmente mais.

### RMSE (Root Mean Squared Error)
Raiz do Erro Quadrático Médio. Penaliza erros maiores mais que o MAE. Útil quando erros grandes são particularmente indesejáveis. Na mesma unidade que a variável predita (ex: reais).

### MAPE (Mean Absolute Percentage Error)
Erro Percentual Absoluto Médio. Expressa o erro como percentual do valor real. Permite comparação entre séries de escalas diferentes. Problemático quando valores reais são próximos de zero.

### PSI (Population Stability Index)
Índice de Estabilidade Populacional. Mede o deslocamento entre duas distribuições (referência vs. atual). PSI < 0.1: sem drift significativo. PSI 0.1-0.2: drift moderado (investigar). PSI > 0.2: drift significativo (retreinar modelo).

## Deep Learning

### LSTM (Long Short-Term Memory)
Tipo de rede neural recorrente (RNN) projetada para aprender dependências de longo prazo em sequências. Resolve o problema de vanishing gradients das RNNs tradicionais usando gates (forget, input, output) que controlam o fluxo de informação. Amplamente usado em séries temporais e NLP.

### Dropout
Técnica de regularização que desativa aleatoriamente neurônios durante o treinamento. Previne overfitting. Taxa de 0.2 significa que 20% dos neurônios são desativados a cada forward pass.

### Early Stopping
Técnica para parar o treinamento quando a performance no conjunto de validação para de melhorar. Patience define quantas épocas sem melhoria são toleradas antes de parar.

### Gradient Clipping
Limita o valor máximo dos gradientes durante backpropagation. Previne o problema de exploding gradients, especialmente em RNNs. Valor típico: norma máxima de 1.0.

## Mercado Financeiro

### OHLCV
Open, High, Low, Close, Volume — os 5 dados básicos de cada candle/barra em dados de mercado. Open: preço de abertura. High: máximo do dia. Low: mínimo do dia. Close: preço de fechamento. Volume: quantidade negociada.

### B3
Brasil, Bolsa, Balcão. Bolsa de valores oficial do Brasil, sediada em São Paulo. Opera mercados de ações, derivativos, renda fixa e câmbio.

### Dividend Yield
Rendimento de dividendos. Calculado como dividendo anual por ação dividido pelo preço da ação. Expresso em percentual. PETR4 historicamente possui dividend yield elevado (10-20%).
