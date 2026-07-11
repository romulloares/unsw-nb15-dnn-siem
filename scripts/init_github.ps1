$ErrorActionPreference = "Stop"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git não encontrado. Instale o Git antes de executar este script."
}

if (-not (Test-Path ".git")) {
    git init
}

git add .
git commit -m "chore: estrutura inicial do projeto UNSW-NB15"

Write-Host "Repositório local iniciado."
Write-Host "Para criar no GitHub com GitHub CLI:"
Write-Host "  gh repo create unsw-nb15-dnn-siem --private --source=. --remote=origin --push"
Write-Host "Mantenha privado durante o desenvolvimento, salvo orientação diferente do professor."
