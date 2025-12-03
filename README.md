# R_Modelos_Predicao_Estatistica
Notebooks R explorando e implementando algoritmos de Machine Learning para Regressão e Classificação, utilizando o ecossistema Tidyverse e pacotes como caret e e1071.

# 📚 Resumo de Modelos Preditivos em Machine Learning

Este repositório contém Notebooks (em R ou Python) que demonstram a implementação e o uso de diversos algoritmos de Machine Learning para resolver problemas de **Regressão** (previsão de valores quantitativos) e **Classificação** (previsão de categorias).

## 📈 Modelos para Problemas de Regressão (Predição de Valores Quantitativos)

O objetivo principal é prever uma variável de saída **contínua** ou **quantitativa** (ex: preço, temperatura).

| Categoria | Modelo | Descrição |
| :--- | :--- | :--- |
| **Base** | **Regressão Linear** | Modelo inicial que assume uma relação linear. |
| **Base** | **Regressão Polinomial** | Usa transformações para modelar relações não lineares. |
| **Regularização** | **Regressão Ridge (L2)** | Penaliza coeficientes para evitar *overfitting* (encolhe, não zera). |
| **Regularização** | **Regressão Lasso (L1)** | Penaliza coeficientes e pode zerá-los, servindo como **seleção de *features***. |
| **Regularização** | **Elastic Net** | Combinação das penalidades Ridge e Lasso. |
| **Ensemble** | **Random Forest (Regressão)** | Combina muitas Árvores de Decisão para alta precisão e redução de variância. |
| **Ensemble** | **Gradient Boosting Machines (GBM)** | Constrói árvores sequencialmente, corrigindo erros (e.g., XGBoost, LightGBM). |
| **Distância** | **Support Vector Regression (SVR)** | Encontra um hiperplano que minimiza o erro dentro de uma margem ($\epsilon$). |
| **Distância** | **K-Nearest Neighbors (KNN Regressão)** | Prevê o valor como a média dos $K$ vizinhos mais próximos. |

---
## ⚖️ Análise Comparativa dos Modelos de Regressão Base

A transição da Regressão Linear Simples para a Múltipla ilustra como a inclusão de *features* adicionais (como Horsepower, `hp`) altera a interpretação e a precisão do modelo, especialmente devido à **multicolinearidade** existente entre as variáveis.

### Comparativo de Métricas (MPG vs. Peso e HP)

| Métrica | Simples (`mpg ~ wt`) | Múltiplo (`mpg ~ wt + hp`) | Interpretação da Mudança |
| :--- | :--- | :--- | :--- |
| **R-quadrado Ajustado** | 0.7446 | **0.8148** | **Melhoria de Ajuste:** O modelo múltiplo explica cerca de **7% a mais** da variância em MPG. |
| **RSE (Erro Residual)** | 3.046 | **2.593** | **Aumento da Precisão:** O erro médio de previsão caiu em $\approx 0.45$ unidades, tornando o modelo mais preciso. |
| **Coeficiente do Peso (`wt`)** | **-5.3445** | **-3.87783** | **Multicolinearidade:** O impacto negativo do Peso diminuiu. Isso ocorre porque o Horsepower (`hp`), que é correlacionado com o Peso, agora "explica" parte da redução no MPG. |
| **P-valor do `hp`** | N/A | $0.00145$ | **Significância:** O Horsepower é um preditor estatisticamente significativo de MPG, mesmo **após** a influência do Peso ter sido contabilizada. |

### Conclusão da Comparação

O modelo de Regressão Múltipla é **estatisticamente superior** ao modelo simples. Ele não só fornece um ajuste superior (maior R-quadrado Ajustado e menor Erro Residual), mas também oferece uma **interpretação mais precisa** do efeito isolado de cada *feature* no consumo de combustível (o princípio *ceteris paribus*).

## 📐 Análise de Forma Funcional: Linear vs. Polinomial

Esta seção compara o modelo Polinomial de 2º Grau (`mpg ~ hp + hp^2`) com o modelo de Regressão Múltipla que foi o melhor ajuste linear (`mpg ~ wt + hp`), a fim de determinar a melhor forma de modelar a relação.

### Comparativo de Desempenho (R-quadrado Ajustado e Erro)

| Métrica | Polinomial ($\text{hp} + \text{hp}^2$) | Múltipla ($\text{wt} + \text{hp}$) | Comparação/Conclusão |
| :--- | :--- | :--- | :--- |
| **R-quadrado Ajustado** | 0.7393 | **0.8148** | **Melhor Ajuste:** O modelo Múltiplo explica significativamente mais variância. |
| **RSE (Erro Residual)** | 3.077 | **2.593** | **Maior Precisão:** O modelo Múltiplo tem um erro de previsão consideravelmente menor. |
| **P-valor do $\text{I(hp}^2)$** | $\mathbf{0.000189}$ | N/A | **Validade da Curvatura:** A significância estatística confirma que a relação **não é linear**, embora a forma curvilínea não seja a que melhor prediz o alvo. |

### Conclusão sobre a Modelagem

1.  **A Relação é Não-Linear:** O termo $\text{I(hp}^2)$ ser estatisticamente significativo prova que a relação entre $\text{HP}$ e $\text{MPG}$ tem uma **curvatura**.
2.  **O Melhor Modelo é Múltiplo:** Apesar de a forma ser curva, a inclusão de um segundo *feature* linear ($\text{wt}$) no modelo **Múltiplo** resultou no **melhor ajuste geral** (maior $\text{R-quadrado Ajustado}$ e menor $\text{RSE}$).
3.  **Implicação:** Para o *dataset* `mtcars`, a **combinação de features independentes** ($\text{wt}$ e $\text{hp}$) é mais eficaz para reduzir o erro de previsão do que a tentativa de modelar a curvatura de um único *feature* ($\text{hp}$).

---
## ⚖️ Comparação da Regularização: Ridge (L2) vs. Lasso (L1)

Os modelos de regularização são cruciais para prevenir o *overfitting* em modelos lineares. Eles adicionam uma penalidade aos coeficientes ($\beta$) para mantê-los estáveis, utilizando a **Validação Cruzada (CV)** para encontrar o nível ideal dessa penalidade ($\lambda$).

| Característica | Regressão Ridge (Penalidade L2) | Regressão Lasso (Penalidade L1) |
| :--- | :--- | :--- |
| **Penalidade** | $\lambda \sum \beta_i^2$ (Soma dos Quadrados) | $\lambda \sum |\beta_i|$ (Soma dos Valores Absolutos) |
| **Efeito nos Coeficientes** | Encolhe os coeficientes em direção a zero, mas **nunca os zera**. | Pode forçar coeficientes irrelevantes a serem **exatamente zero** (Seleção de Features). |

### 🛠️ Análise dos Coeficientes e Lambda Ideal ($\lambda_{\min}$)

A tabela compara os coeficientes no $\lambda$ que minimizou o erro para cada modelo.

| Feature | Coeficiente Ridge (L2) $\lambda=0.661$ | Coeficiente Lasso (L1) $\lambda=0.005$ | Observação |
| :--- | :--- | :--- | :--- |
| (Intercept) | $28.5312$ | $10.8383$ | O Lasso tende a penalizar o Intercepto mais agressivamente. |
| **nox** | $\mathbf{-12.8894}$ | $\mathbf{-9.0319}$ | O Ridge (L2) mantém um peso maior neste feature, distribuindo a penalidade de forma mais suave. |
| **rm** | $\mathbf{4.3723}$ | $\mathbf{6.3066}$ | O Lasso (L1) mantém um peso significativamente maior para este feature, concentrando o poder preditivo no `rm` (número de quartos). |
| **age** | $-0.0036$ | $-0.0265$ | Ambos os modelos reduziram a importância de `age`. |
| **indus** | $-0.0412$ | $0.0202$ | Os sinais opostos indicam como a penalidade lida com a multicolinearidade de maneira distinta. |

**Valores Ótimos de Penalidade ($\lambda_{\min}$):**
* **Ridge (L2):** $\mathbf{0.6614818}$
* **Lasso (L1):** $\mathbf{0.005975135}$

### Conclusões sobre a Regularização

1.  **Comportamento do Ridge (L2):**
    * O $\lambda_{\min}$ encontrado demonstra que o modelo Ridge precisa de uma penalidade moderada (0.66) para estabilidade.
    * Como esperado, **nenhum coeficiente foi zerado**, apenas encolhido. O Ridge é o preferido quando todos os *features* são considerados importantes.

2.  **Comportamento do Lasso (L1):**
    * O $\lambda_{\min}$ é muito baixo (0.005), o que significa que o erro do modelo é minimizado com uma penalidade muito fraca, e por isso, **nenhum *feature* foi anulado**.
    * O Lasso é preferível quando se busca explicitamente a seleção de *features* ou quando o erro de previsão é menor (menor RMSE) em comparação com o Ridge.

## 🕸️ Análise Final: Elastic Net (Otimização de Alpha e Lambda)

O Elastic Net combina as penalidades Ridge ($L_2$) e Lasso ($L_1$), otimizando dois hiperparâmetros: $\lambda$ (força da penalidade) e $\alpha$ (mistura entre $L_1$ e $L_2$).

### 🛠️ Parâmetros Otimizados pelo `caret`

| Hiperparâmetro | Valor Ótimo | Função | Interpretação |
| :--- | :--- | :--- | :--- |
| **Melhor $\alpha$** | $\mathbf{0.1111}$ | $\alpha \in [0, 1]$. Onde 0 é Ridge puro, e 1 é Lasso puro. | **Predominância Ridge:** O $\alpha$ ótimo é muito próximo de zero. Isso significa que o modelo que obteve o melhor desempenho utiliza uma penalidade **majoritariamente Ridge (L2)**, com apenas uma leve contribuição do Lasso (L1). |
| **Melhor $\lambda$** | $\mathbf{0.2154}$ | Força total da penalidade aplicada. | O modelo encontrou o ponto de equilíbrio de regularização que minimiza o erro. |

### 🎯 Desempenho Final (RMSE)

| Modelo | Penalidade Aplicada | RMSE Típico (Conjunto de Teste) |
| :--- | :--- | :--- |
| **Elastic Net (Optimal)** | $\mathbf{\alpha \approx 0.11}$ ($\text{L}_2$ dominante) | **5.179** |
| Lasso (L1 Puro) | $\alpha = 1$ | $\approx 6.357$ |
| Regressão Múltipla (OLS) | $\lambda = 0$ | $\approx 5.5$ (sem regularização) |

### Conclusão Global da Regressão

1.  **Modelo Ideal:** O **Elastic Net** foi o modelo de melhor desempenho (menor RMSE) entre os modelos lineares testados.
2.  **Estratégia Vencedora:** O sucesso do Elastic Net reside na sua capacidade de escolher a melhor estratégia de penalização. O baixo valor de $\alpha$ indica que a melhor abordagem foi **priorizar a estabilidade** e o **encolhimento dos coeficientes (L2)** sobre a seleção agressiva de *features* (L1).
3.  **Implicação:** Para o *dataset* `Boston Housing`, a regularização é necessária para reduzir o *overfitting* (melhora em relação ao OLS simples), e a forma ideal de fazê-lo é através da **penalidade Ridge (L2)**.
---

## 🏷️ Modelos para Problemas de Classificação (Predição de Classes/Categorias)

O objetivo principal é prever uma variável de saída **categórica** ou **discreta** (ex: Sim/Não, Classe A/B/C).

| Categoria | Modelo | Descrição |
| :--- | :--- | :--- |
| **Probabilístico** | **Regressão Logística** | Calcula a probabilidade de um dado pertencer a uma classe usando a função Logit/Sigmoide. |
| **Probabilístico** | **Naive Bayes** | Baseado no Teorema de Bayes; assume independência ingênua dos *features*. |
| **Probabilístico** | **Linear Discriminant Analysis (LDA)** | Encontra uma combinação linear de *features* que melhor separa as classes. |
| **Margem** | **Support Vector Machines (SVM)** | Encontra o hiperplano que maximiza a margem entre as classes. |
| **Distância** | **K-Nearest Neighbors (KNN Classificação)** | Classifica um dado pela classe mais comum entre seus $K$ vizinhos próximos. |
| **Árvore** | **Árvores de Decisão** | Cria regras de decisão sequenciais. |
| **Ensemble** | **Random Forest (Classificação)** | Votação de múltiplas Árvores de Decisão para a classificação final. |
| **Ensemble** | **Gradient Boosting Machines (GBM)** | Constrói preditores sequencialmente para alta precisão. |

### 🎯 Regressão Logística 

Ao verificar o balanceamento do conjunto de dados, percebeu-se um desbalancemaneto entre Nao_Diabete e Diabete. Optou-se por aplicar estratégia de balancemento para mitigar o problema

### ⚖️ Estratégia de Mitigação de Desbalanceamento (Custo)

O dataset `PimaIndiansDiabetes` apresentou um **desbalanceamento de 65% (Nao_Diabete) vs. 35% (Diabete)**. Para garantir que o modelo não ignorasse a classe minoritária crítica (`Diabete`), o treinamento da Regressão Logística foi realizado em duas etapas de comparação:

1.  **Modelo Baseline:** Sem ajustes de peso.
2.  **Modelo Otimizado (Mitigação):** Treinado usando **pesos de classes inversamente proporcionais** às frequências, penalizando os erros na classe `Diabete`.

#### Análise do Impacto da Mitigação (Validação Cruzada)

| Métrica (Média CV) | Baseline (Sem Ajuste) | Otimizado (Com Ajuste de Peso) | Conclusão Estratégica |
| :--- | :--- | :--- | :--- |
| **ROC (AUC)** | $0.8392$ | $\mathbf{0.8396}$ | O poder discriminatório foi mantido. |
| **Sensibilidade (Recall) [Diabete]** | $\mathbf{0.8752}$ | $0.7857$ | **Diminuição:** Indica que o Baseline estava com **Falsos Positivos excessivos** (Especificidade muito baixa). |
| **Especificidade [Nao\_Diabete]** | $0.5553$ | $\mathbf{0.7058}$ | **Melhoria Crítica:** A Especificidade aumentou em **15 pontos percentuais**, resultando em menos Falsos Positivos no teste. |

**Conclusão:** O **Modelo Otimizado** foi selecionado, pois alcançou um **equilíbrio** mais realista entre as classes e reduziu o alto viés de Falso Positivo.

---

O modelo otimizado foi avaliado em um conjunto de dados de Teste (não visto) para confirmar sua capacidade de generalização.

### 💾 Persistência de Objetos (Reprodutibilidade)

Para garantir que o particionamento dos dados de treino e teste, as configurações de Validação Cruzada (`control`) e os modelos treinados (como o `logreg_otimizado`) possam ser reutilizados em diferentes notebooks e sessões R, utilizamos as funções de serialização de dados:

* **`saveRDS(objeto, file = "caminho/nome.rds")`**: É a função preferida para salvar **um único objeto** (como um modelo ou a configuração `control`). O arquivo `.rds` resultante é carregado usando `readRDS()`, o que garante um controle explícito sobre qual objeto está sendo injetado na sessão.
* **`save(obj1, obj2, file = "caminho/nome.RData")`**: Usada para salvar **múltiplos objetos** simultaneamente (como `train_df` e `test_df`). O arquivo `.RData` é carregado com a função `load()`, que injeta todos os objetos salvos diretamente no ambiente de trabalho.

O uso dessas funções assegura que todas as análises comparativas usem exatamente os mesmos *splits* de dados.

---

#### Matriz de Confusão

O resultado da Matriz de Confusão no Conjunto de Teste (70/30) é:

#### Métricas de Desempenho

| Métrica | Valor (Teste) | Análise / Interpretação |
| :--- | :--- | :--- |
| **AUC** | $\mathbf{0.8208}$ | **Forte Desempenho:** O modelo possui um bom poder de discriminação no conjunto não visto. |
| **Acurácia** | $0.7261$ | A taxa de acerto geral é de 72.61%. |
| **Sensibilidade (Recall)** | $0.7000$ | (Relativo à classe 'Nao\_Diabete' - Padrão do caret). |
| **Especificidade** | $0.7750$ | (Relativo à classe 'Diabete' - Padrão do caret). |
| **Recall (Diabete)** | $\mathbf{0.7750}$ | **Recálculo Crucial:** O modelo identificou corretamente **77.50%** dos pacientes que realmente têm diabetes ($62 / (62 + 18)$). |
| **Balanced Accuracy** | $\mathbf{0.7375}$ | A acurácia ajustada para o desbalanceamento das classes. |

**Conclusão da Avaliação:** O modelo final apresenta um bom equilíbrio de desempenho ($\text{AUC} > 0.80$) e conseguiu identificar quase **$78\%$** dos casos reais de diabetes (Recall da classe minoritária), sendo um excelente *baseline* para a comparação com os próximos modelos (KNN, SVM, etc.).

---

### 🎯 Naive_Bayes

O Naive Bayes (NB) foi treinado com padronização e otimizado usando Validação Cruzada (CV) repetida. O modelo escolhido foi o **`usekernel = TRUE`**, que apresentou o melhor desempenho de AUC.

| Métrica | Valor Ótimo (CV) |
| :--- | :--- |
| **ROC (AUC)** | $\mathbf{0.8301}$ |
| **Sensibilidade (Recall)** | $0.8467$ |
| **Especificidade** | $0.5794$ |

---

### 🧠 Linear Discriminant Analysis (LDA)

O modelo LDA (Linear Discriminant Analysis) não possui hiperparâmetros de *tuning* e foi ajustado diretamente no *dataset* padronizado.

| Métrica | Valor Ótimo (CV) |
| :--- | :--- |
| **ROC (AUC)** | $0.8373$ |
| **Sensibilidade (Recall)** | $\mathbf{0.8800}$ |
| **Especificidade** | $0.5469$ |

## Comparação da Categoria Probabilística

Foram comparados três modelos baseados em probabilidade para estabelecer o melhor *baseline* para modelos mais complexos.

#### Comparativo de Métricas (Validação Cruzada)

| Métrica (Média CV) | RegLog\_Opt | Naive\_Bayes | LDA | Análise |
| :--- | :--- | :--- | :--- | :--- |
| **ROC (AUC)** | $\mathbf{0.8396}$ | $0.8396$ | $0.8373$ | O poder de separação das classes é similar entre os três. |
| **Sensibilidade (Recall)** | $0.7857$ | $0.7857$ | $\mathbf{0.8800}$ | O **LDA** demonstrou a melhor capacidade de minimização de Falsos Negativos (Maior Recall). |
| **Especificidade** | $\mathbf{0.7058}$ | $0.7058$ | $0.5469$ | O **LDA** possui o maior viés de Falso Positivo, tornando-o menos prático. |

**Conclusão da Categoria Probabilística:**
A **Regressão Logística Otimizada** é escolhida como o modelo de *baseline* mais robusto. Embora o LDA tenha o maior Recall, o RegLog Otimizado oferece o **melhor equilíbrio** entre as métricas, mantendo um alto poder discriminatório (AUC) e uma Especificidade aceitável ($\mathbf{0.7058}$).

### Support Vector Machines (SVM)

### 🛡️ Resultados do Modelo Support Vector Machines (SVM)

O modelo SVM com Kernel de Função de Base Radial (RBF) foi treinado e otimizado para o parâmetro de custo (**$C$**).

| Métrica | Valor Ótimo (CV) |
| :--- | :--- |
| **ROC (AUC)** | $0.8101$ |
| **Sensibilidade (Recall)** | $\mathbf{0.8647}$ |
| **Especificidade** | $0.5737$ |

**Conclusão:** O SVM **não** alcançou o maior Recall ($0.8647$), sendo superado pelo Recall do modelo LDA ($\mathbf{0.8800}$). Seu AUC ($0.8101$) é ligeiramente inferior ao da Regressão Logística Otimizada ($0.8396$), e sua Especificidade ($0.5737$) é baixa. Isso o posiciona como um modelo que, assim como o LDA, favorece muito a identificação da doença (alto Recall) à custa de um alto número de **Falsos Positivos** (baixa Especificidade), indicando um limite de decisão agressivo.

---

### 🏘️ Resultados do Modelo K-Nearest Neighbors (KNN)

O modelo KNN (K-Vizinhos Mais Próximos) foi treinado e otimizado para o parâmetro **$k$** (o número de vizinhos).

| Métrica | Valor Ótimo (CV) |
| :--- | :--- |
| **ROC (AUC)** | $0.8030$ |
| **Sensibilidade (Recall)** | $\mathbf{0.9381}$ |
| **Especificidade** | $0.4058$ |

**Conclusão:** O KNN obteve o **maior Recall** ($\mathbf{0.9381}$) de todos os modelos testados até agora, sendo o mais eficaz em identificar corretamente os casos positivos de diabetes. No entanto, o seu AUC ($\mathbf{0.8030}$) é o mais baixo, e a **Especificidade é extremamente baixa** ($0.4058$). Este modelo é o menos equilibrado, favorecendo o Recall ao ponto de gerar uma taxa muito alta de **Falsos Positivos** (classificando pacientes saudáveis como diabéticos), o que o torna impraticável sem um ajuste rigoroso de *threshold*.

### Otimização do KNN por aumento do Threshold(Limite)

### 🏘️ Resultados do Modelo K-Nearest Neighbors (KNN) - Otimizado por Threshold

O modelo KNN ($k=33$) apresentava um Recall altíssimo (0.9381) mas uma Especificidade péssima (0.4058). Foi aplicado um **ajuste de *Threshold*** (limite de decisão) para equilibrar o desempenho e maximizar a Sensibilidade e a Especificidade.

#### Matriz de Confusão (Threshold Otimizado - Conjunto de Teste)

#### Métricas de Desempenho

| Métrica | Valor (Teste) | Análise / Conclusão |
| :--- | :--- | :--- |
| **Acurácia** | $0.7565$ | Melhora notável sobre os modelos probabilísticos. |
| **Recall (Diabete)** | $\mathbf{0.8375}$ | O Recall permanece alto ($83.75\%$), com uma queda controlada. |
| **Especificidade** | $\mathbf{0.7133}$ | A Especificidade melhorou dramaticamente, ficando no nível do RegLog Otimizado. |
| **Balanced Accuracy** | $\mathbf{0.7754}$ | **Melhor modelo até agora** em termos de equilíbrio entre classes. |

**Conclusão Final do KNN:** O ajuste de *threshold* permitiu que o modelo KNN se tornasse o **modelo de melhor desempenho balanceado** até o momento, comprovando que seu alto poder discriminatório (Curva ROC) pode ser traduzido em métricas de classificação úteis com a escolha correta do limite de decisão.

---