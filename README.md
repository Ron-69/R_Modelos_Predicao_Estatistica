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
