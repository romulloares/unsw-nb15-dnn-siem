# Execute a partir da raiz do repositório.
# Pré-requisito: ambiente virtual ativo e dependências instaladas.

$ErrorActionPreference = "Stop"

Write-Host "[1/3] Baselines (E00-E02)"
python -m src.train_baselines `
    --data-dir data/raw `
    --output-dir outputs/E00_E02

Write-Host "[2/3] DNN principal (E03)"
python -m src.train_dnn `
    --data-dir data/raw `
    --output-dir outputs/E03 `
    --hidden-dims 128 64 `
    --dropout 0.30 0.20 `
    --learning-rate 0.001 `
    --batch-size 512 `
    --epochs 100 `
    --patience 10

Write-Host "[3/3] Figuras consolidadas do relatório"
python -m src.generate_report_figures `
    --data-dir data/raw `
    --outputs-dir outputs `
    --report-dir outputs/report_figures `
    --baseline-dir E00_E02 `
    --main-dnn E03

Write-Host "Concluído. Veja outputs/report_figures e os diretórios figures/ de cada experimento."
