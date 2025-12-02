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
