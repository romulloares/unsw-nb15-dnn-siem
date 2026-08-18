# Como compilar o relatório SBC

1. Baixe o template oficial de artigos da SBC.
2. Copie `sbc-template.sty` e `sbc.bst` para esta pasta (`report/`).
3. Gere os experimentos e figuras a partir da raiz do repositório.
4. Substitua `INSERIR-URL-DO-GITHUB` em `main.tex` pela URL do repositório.
5. Preencha nas tabelas os valores reais dos JSONs em `outputs/*/metrics/`.
6. Compile dentro de `report/`:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Alternativamente, use `latexmk`:

```powershell
latexmk -pdf main.tex
```

## Figuras geradas automaticamente

Os scripts de treinamento já produzem para cada modelo:

- `*_confusion_matrix.png`
- `*_roc_curve.png`
- `*_precision_recall_curve.png`

O comando abaixo adiciona as figuras consolidadas do relatório:

```powershell
python -m src.generate_report_figures --data-dir data/raw --outputs-dir outputs --report-dir outputs/report_figures --baseline-dir E00_E02 --main-dnn E03
```

Arquivos esperados em `outputs/report_figures/`:

- `dnn_architecture.png`
- `class_distribution.png`
- `attack_categories.png`
- `distribution_shift.png`
- `correlation_heatmap.png`
- `dnn_training_history.png`
- `threshold_comparison.png`
- `model_comparison.png`
- `ablation_comparison.png` (quando houver pelo menos duas DNNs executadas)
