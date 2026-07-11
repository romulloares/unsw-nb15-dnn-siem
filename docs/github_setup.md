# Publicação inicial no GitHub

## Opção recomendada: repositório privado durante o desenvolvimento

Na raiz do projeto, execute no PowerShell:

```powershell
git init
git branch -M main
git add .
git commit -m "chore: estrutura inicial do projeto UNSW-NB15"
```

Com o GitHub CLI instalado e autenticado:

```powershell
gh repo create unsw-nb15-dnn-siem --private --source=. --remote=origin --push
```

Sem GitHub CLI:

1. crie um repositório vazio no GitHub com o nome `unsw-nb15-dnn-siem`;
2. não marque a criação automática de README, `.gitignore` ou licença;
3. execute os comandos exibidos pelo GitHub, normalmente:

```powershell
git remote add origin https://github.com/SEU_USUARIO/unsw-nb15-dnn-siem.git
git push -u origin main
```

## Sequência de commits sugerida

```text
chore: estrutura inicial do projeto
 docs: adiciona estudo inicial e protocolo experimental
 data: documenta obtenção e validação do UNSW-NB15
 feat: implementa baseline de regressão logística
 feat: implementa random forest
 feat: implementa DNN em PyTorch
 experiment: registra resultados do experimento E01
 report: adiciona versão inicial do artigo SBC
```

## Antes de cada push

```powershell
pytest -q
git status
git diff
git add .
git commit -m "mensagem objetiva"
git push
```

Não adicione os CSVs, ambientes virtuais nem modelos treinados. O `.gitignore` já cobre esses itens.
