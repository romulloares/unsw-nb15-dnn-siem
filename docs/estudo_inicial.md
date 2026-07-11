# Estudo inicial do projeto

## 1. Contexto

Soluções de SIEM agregam eventos de múltiplas fontes e podem gerar um volume elevado de alertas. O objetivo deste projeto não é substituir regras, correlação ou o julgamento de analistas, mas estudar um classificador capaz de atribuir uma probabilidade de comportamento malicioso a registros de tráfego de rede. Essa pontuação poderia servir como sinal adicional de priorização em uma fila de triagem.

## 2. Definição da tarefa

A tarefa inicial será de **classificação binária supervisionada**:

- `0`: tráfego normal;
- `1`: ataque.

Dado um vetor de características de uma conexão, o modelo deve estimar `P(ataque | características)`.

A classificação binária foi escolhida para manter o projeto executável no prazo e permitir uma análise mais direta do custo de falsos negativos e falsos positivos. A classificação multiclasse por categoria de ataque fica como extensão.

## 3. Dataset UNSW-NB15

O UNSW-NB15 foi produzido no Cyber Range Lab da UNSW Canberra e combina atividade normal moderna com comportamentos de ataque sintetizados. A página oficial descreve nove famílias de ataque: Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode e Worms.

A base disponibiliza partições prontas com:

- **175.341 registros de treino**;
- **82.332 registros de teste**.

Os arquivos contêm características de fluxo e rótulos. Para o primeiro ciclo experimental, serão usados apenas os CSVs de treino e teste já preparados pela fonte oficial.

## 4. Hipóteses e perguntas de pesquisa

### H1

Uma DNN totalmente conectada consegue capturar relações não lineares entre as características do tráfego e superar uma regressão logística em F1, recall e ROC-AUC.

### H2

O melhor modelo em acurácia não necessariamente será o mais adequado para um cenário de SOC; a taxa de falsos positivos e a curva Precision-Recall devem orientar a escolha.

### Perguntas

1. A DNN supera os baselines clássicos no conjunto de teste isolado?
2. Qual é o custo dessa melhora em falsos positivos?
3. O ajuste do limiar de decisão na validação produz um ponto operacional mais útil para priorização de alertas?
4. Quais categorias de ataque aparecem com maior frequência entre os falsos negativos, mesmo que o treinamento seja binário?

## 5. Abordagem técnica inicial

### Pré-processamento

1. validar nomes, tipos e rótulos;
2. remover `id` por ser identificador;
3. remover `attack_cat` da entrada binária para evitar vazamento do rótulo;
4. imputar valores ausentes;
5. aplicar one-hot encoding às variáveis categóricas, em especial `proto`, `service` e `state`;
6. padronizar variáveis numéricas;
7. ajustar todo o pré-processamento somente no subconjunto de treino.

### Baselines

- regressão logística com pesos de classe;
- Random Forest com pesos balanceados;
- opcionalmente, um classificador ingênuo da classe majoritária para contextualizar a acurácia.

### Modelo principal

DNN inicial:

```text
Entrada
  -> Dense(128) + ReLU
  -> Dropout(0,30)
  -> Dense(64) + ReLU
  -> Dropout(0,20)
  -> Dense(1), produzindo logit
```

Treinamento:

- `BCEWithLogitsLoss`;
- Adam;
- mini-batches;
- early stopping pela loss de validação;
- até 100 épocas, com interrupção antecipada;
- pesos de classe positivos calculados no treino como opção para lidar com desbalanceamento.

A implementação usa logits por estabilidade numérica. A sigmoide é aplicada somente na inferência para transformar o logit em probabilidade.

## 6. Protocolo experimental

A partição oficial de treino será subdividida de forma estratificada:

- 80% para ajuste dos modelos;
- 20% para validação e seleção de hiperparâmetros/limiar.

O arquivo oficial de teste será usado somente após a definição do modelo e de seu limiar.

Todos os modelos devem receber a mesma partição, a mesma semente e o mesmo protocolo de métricas.

## 7. Métricas

Métricas principais:

- **recall de ataques**: proporção de ataques detectados;
- **precision**: proporção dos alertas positivos que são realmente ataques;
- **F1-score**: equilíbrio entre precision e recall;
- **taxa de falsos positivos (FPR)**: tráfego normal marcado como ataque;
- **PR-AUC**: útil quando há desbalanceamento;
- **ROC-AUC**: capacidade de ordenação em diferentes limiares.

Métricas complementares:

- acurácia;
- matriz de confusão;
- tempo aproximado de treino e inferência;
- número de parâmetros da DNN.

## 8. Riscos metodológicos

### Vazamento de informação

`attack_cat` descreve diretamente a categoria do ataque e não deve ser usada como entrada na classificação binária. O ajuste do encoder e do scaler antes do split também causaria vazamento.

### Desbalanceamento

A proporção entre tráfego normal e ataque pode influenciar a acurácia. Pesos de classe e análise de PR-AUC serão considerados antes de técnicas mais invasivas de reamostragem.

### Sobreposição de classes

Certos fluxos normais e maliciosos podem apresentar características semelhantes, limitando a separação possível. Isso deve aparecer nos erros e nas curvas de avaliação.

### Generalização

O UNSW-NB15 é um benchmark controlado e não reproduz integralmente o tráfego de uma empresa real. Portanto, os resultados devem ser apresentados como evidência experimental no dataset, não como garantia de desempenho em produção.

### Relação com SIEM

Os registros são fluxos de rede rotulados, não alertas de SIEM. O vínculo com SIEM é uma simulação de priorização: a probabilidade do modelo funciona como um sinal de risco, mas integração, contexto, correlação temporal e explicabilidade operacional não serão implementados nesta primeira versão.

## 9. Critério de sucesso

O projeto será considerado tecnicamente bem-sucedido se:

1. o pipeline for reproduzível a partir dos dois CSVs oficiais;
2. houver comparação justa entre pelo menos dois baselines e a DNN;
3. os resultados incluírem análise de falsos positivos e falsos negativos;
4. as limitações e experimentos malsucedidos forem documentados;
5. o relatório puder ser reconstruído a partir dos artefatos salvos no repositório.

## 10. Cronograma sugerido

| Período | Atividade |
|---|---|
| 11-13 jul. | Estrutura do repositório, download e validação dos dados |
| 14-20 jul. | EDA, pré-processamento e baselines |
| 21-27 jul. | DNN inicial, early stopping e métricas |
| 28 jul.-3 ago. | Tuning controlado e análise de limiar |
| 4-9 ago. | Tabelas, gráficos, erros e lições aprendidas |
| 10-13 ago. | Relatório SBC, revisão e teste de reprodução |
| 14 ago. | Entrega final |

## 11. Próximo passo imediato

1. baixar os dois CSVs oficiais;
2. executar o notebook `01_eda_unsw_nb15.ipynb`;
3. confirmar distribuição das classes, valores ausentes e cardinalidade das colunas categóricas;
4. registrar a primeira execução no diário de experimentos;
5. executar o baseline de regressão logística antes da DNN.
