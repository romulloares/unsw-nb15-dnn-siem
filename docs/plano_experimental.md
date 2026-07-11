# Plano experimental

## Objetivo

Comparar baselines clássicos e uma DNN para classificação binária no UNSW-NB15, preservando uma avaliação final isolada e priorizando métricas úteis para triagem de segurança.

## Partições

- `UNSW_NB15_training-set.csv`: dividido em treino e validação com `stratify=y` e semente 42.
- `UNSW_NB15_testing-set.csv`: reservado para o resultado final.
- proporção padrão da validação: 20% do arquivo oficial de treino.

## Prevenção de vazamento

- remover `id` e `attack_cat` antes do treinamento;
- ajustar imputação, one-hot encoding e scaler apenas no treino;
- usar validação para hiperparâmetros e limiar;
- não repetir decisões olhando os resultados do teste.

## Modelos

### B0 - Dummy

Prediz a classe majoritária. Serve para mostrar por que acurácia isolada é insuficiente.

### B1 - Regressão logística

- `class_weight="balanced"`;
- máximo de 1.000 iterações;
- probabilidade usada para curvas e análise de limiar.

### B2 - Random Forest

- 200 árvores;
- profundidade máxima inicial de 20;
- `class_weight="balanced_subsample"`;
- semente 42.

### M1 - DNN

- camadas ocultas: 128 e 64 neurônios;
- ReLU;
- dropout 0,30 e 0,20;
- Adam, `lr=1e-3`;
- batch 512;
- até 100 épocas;
- early stopping com paciência 10;
- `BCEWithLogitsLoss`;
- opção de `pos_weight` calculado no treino.

## Métricas mínimas

- accuracy;
- precision;
- recall;
- F1;
- ROC-AUC;
- average precision / PR-AUC;
- FPR;
- TN, FP, FN e TP.

## Comparação de limiares

Para cada modelo probabilístico:

1. reportar resultados com limiar 0,5;
2. selecionar na validação o limiar que maximiza F1;
3. aplicar esse limiar uma única vez no teste;
4. comparar o ganho de recall com o aumento de falsos positivos.

Uma extensão útil é selecionar um limiar sob restrição de FPR máximo, por exemplo 5%, caso a distribuição dos dados permita.

## Ablations da DNN

Executar apenas após o baseline principal:

- sem dropout;
- uma camada oculta;
- largura 64-32 versus 128-64;
- com e sem `pos_weight`;
- batch 256 versus 512.

Evitar uma busca extensa de hiperparâmetros. O objetivo é produzir comparações interpretáveis e reproduzíveis dentro do prazo.

## Artefatos a preservar

- configuração completa da execução;
- histórico de loss;
- métricas de validação e teste;
- matriz de confusão;
- curvas ROC e PR;
- pré-processador ajustado;
- pesos/modelo;
- tempo de execução;
- observações no diário de experimentos.
