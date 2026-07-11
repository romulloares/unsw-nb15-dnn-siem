# Dados do projeto

Baixe os arquivos oficiais do UNSW-NB15 em:

- https://research.unsw.edu.au/projects/unsw-nb15-dataset

Para este projeto, use preferencialmente as partições prontas:

```text
UNSW_NB15_training-set.csv
UNSW_NB15_testing-set.csv
```

Salve os arquivos em `data/raw/` sem alterar os nomes.

## Por que os CSVs não estão no Git?

- evitam aumentar o tamanho do repositório;
- mantêm clara a separação entre código e dados externos;
- respeitam as condições de uso e citação definidas pela fonte oficial;
- tornam o pipeline reprodutível sem redistribuir os dados.

## Validação rápida

Após posicionar os arquivos, execute:

```bash
python -c "from pathlib import Path; from src.data import load_official_splits; a,b=load_official_splits(Path('data/raw')); print(a.shape, b.shape)"
```
