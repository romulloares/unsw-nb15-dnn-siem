"""Gera figuras consolidadas para o relatório final do projeto.

As figuras de avaliação por modelo (matriz de confusão, ROC e PR) já são
criadas por ``src.train_baselines`` e ``src.train_dnn``. Este módulo adiciona
figuras de EDA e comparações consolidadas a partir dos artefatos salvos.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import ATTACK_CATEGORY_COLUMN, TARGET_COLUMN
from .data import infer_column_groups, load_official_splits, split_features_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--report-dir", type=Path, default=Path("outputs/report_figures")
    )
    parser.add_argument("--baseline-dir", default="E00_E02")
    parser.add_argument("--main-dnn", default="E03")
    parser.add_argument(
        "--experiments", nargs="*", default=["E03", "E04", "E05", "E06"]
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")


def generate_eda(data_dir: Path, report_dir: Path) -> None:
    train_df, test_df = load_official_splits(data_dir)

    # 1) Distribuição binária treino x teste
    comparison = pd.DataFrame(
        {
            "Treino": train_df[TARGET_COLUMN]
            .value_counts(normalize=True)
            .sort_index(),
            "Teste": test_df[TARGET_COLUMN]
            .value_counts(normalize=True)
            .sort_index(),
        }
    )
    comparison.index = ["Normal", "Ataque"]
    fig, ax = plt.subplots(figsize=(7, 4))
    comparison.plot(kind="bar", ax=ax)
    ax.set_ylabel("Proporção")
    ax.set_title("Distribuição binária: treino versus teste")
    ax.tick_params(axis="x", rotation=0)
    _save(fig, report_dir / "class_distribution.png")

    # 2) Categorias de ataque
    if ATTACK_CATEGORY_COLUMN in train_df.columns:
        attack_counts = (
            train_df[ATTACK_CATEGORY_COLUMN]
            .fillna("Normal/ausente")
            .astype(str)
            .value_counts()
            .sort_values()
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        attack_counts.plot(kind="barh", ax=ax)
        ax.set_xlabel("Registros")
        ax.set_title("Categorias no conjunto oficial de treino")
        _save(fig, report_dir / "attack_categories.png")

    # Colunas numéricas do conjunto de entrada, sem id/attack_cat/label
    x_train_raw, _ = split_features_target(train_df)
    _, numeric_columns = infer_column_groups(x_train_raw)

    # 3) Sinal exploratório de mudança de distribuição
    train_means = train_df[numeric_columns].mean()
    test_means = test_df[numeric_columns].mean()
    pooled_scale = train_df[numeric_columns].std().replace(0, np.nan)
    shift = ((test_means - train_means) / pooled_scale).abs().sort_values(
        ascending=False
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    shift.head(15).sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel("|média_teste - média_treino| / desvio_treino")
    ax.set_title("Sinal exploratório de mudança de distribuição")
    _save(fig, report_dir / "distribution_shift.png")

    # 4) Correlação entre os atributos numéricos mais associados ao rótulo
    correlations = (
        train_df[numeric_columns + [TARGET_COLUMN]]
        .corr(numeric_only=True)[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .sort_values(key=lambda series: series.abs(), ascending=False)
    )
    selected = correlations.head(15).index.tolist()
    matrix = train_df[selected].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, aspect="auto")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(selected)), selected, rotation=90)
    ax.set_yticks(range(len(selected)), selected)
    ax.set_title("Correlação entre os 15 atributos mais associados ao rótulo")
    _save(fig, report_dir / "correlation_heatmap.png")


def generate_architecture(report_dir: Path) -> None:
    labels = [
        "Entrada\n(n atributos)",
        "Dense 128\nReLU",
        "Dropout\n0,30",
        "Dense 64\nReLU",
        "Dropout\n0,20",
        "Dense 1\nLogit",
        "Sigmoide\nP(ataque)",
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(-0.7, len(labels) - 0.3)
    ax.set_ylim(-0.8, 0.8)
    ax.axis("off")
    for index, label in enumerate(labels):
        ax.text(
            index,
            0,
            label,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "black"},
        )
        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=(index + 0.72, 0),
                xytext=(index + 0.28, 0),
                arrowprops={"arrowstyle": "->"},
            )
    ax.set_title("Arquitetura da DNN principal")
    _save(fig, report_dir / "dnn_architecture.png")


def generate_training_history(outputs_dir: Path, main_dnn: str, report_dir: Path) -> None:
    history_path = outputs_dir / main_dnn / "metrics" / "dnn_history.csv"
    if not history_path.exists():
        print(f"[AVISO] Histórico não encontrado: {history_path}")
        return
    history = pd.read_csv(history_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["epoch"], history["train_loss"], label="Treino")
    ax.plot(history["epoch"], history["validation_loss"], label="Validação")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_title("Histórico de treinamento da DNN")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, report_dir / "dnn_training_history.png")


def generate_threshold_comparison(
    outputs_dir: Path, main_dnn: str, report_dir: Path
) -> None:
    result = _load_json(outputs_dir / main_dnn / "metrics" / "dnn_pytorch.json")
    if result is None:
        print(f"[AVISO] Métricas da DNN não encontradas em {main_dnn}.")
        return
    default = result["test_default"]
    tuned = result["test_tuned"]
    metric_names = ["precision", "recall", "f1", "false_positive_rate"]
    labels = ["Precision", "Recall", "F1", "FPR"]
    frame = pd.DataFrame(
        {
            "Limiar 0,5": [default[name] for name in metric_names],
            f"Limiar ajustado ({tuned['threshold']:.3f})": [
                tuned[name] for name in metric_names
            ],
        },
        index=labels,
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    frame.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Valor")
    ax.set_title("Efeito do ajuste de limiar na DNN")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, report_dir / "threshold_comparison.png")


def generate_model_comparison(
    outputs_dir: Path, baseline_dir: str, main_dnn: str, report_dir: Path
) -> None:
    paths = {
        "Dummy": outputs_dir / baseline_dir / "metrics" / "dummy.json",
        "Logistic": outputs_dir / baseline_dir / "metrics" / "logistic.json",
        "Random Forest": outputs_dir / baseline_dir / "metrics" / "random_forest.json",
        "DNN": outputs_dir / main_dnn / "metrics" / "dnn_pytorch.json",
    }
    rows: list[dict[str, Any]] = []
    for model, path in paths.items():
        result = _load_json(path)
        if result is None:
            continue
        metrics = result["test_tuned"]
        rows.append(
            {
                "Modelo": model,
                "F1": metrics["f1"],
                "Recall": metrics["recall"],
                "FPR": metrics["false_positive_rate"],
            }
        )
    if len(rows) < 2:
        print("[AVISO] Resultados insuficientes para comparação de modelos.")
        return
    frame = pd.DataFrame(rows).set_index("Modelo")
    fig, ax = plt.subplots(figsize=(9, 5))
    frame.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Valor")
    ax.set_title("Comparação dos modelos no conjunto de teste")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, report_dir / "model_comparison.png")


def generate_ablation_comparison(
    outputs_dir: Path, experiments: list[str], report_dir: Path
) -> None:
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        result = _load_json(outputs_dir / experiment / "metrics" / "dnn_pytorch.json")
        if result is None:
            continue
        metrics = result["test_tuned"]
        config = result.get("config", {})
        hidden_dims = config.get("hidden_dims", [])
        dropout = config.get("dropout", [])
        rows.append(
            {
                "Experimento": experiment,
                "F1": metrics["f1"],
                "Recall": metrics["recall"],
                "FPR": metrics["false_positive_rate"],
                "Arquitetura": "-".join(map(str, hidden_dims)),
                "Dropout": "/".join(f"{value:.2f}" for value in dropout),
            }
        )
    if len(rows) < 2:
        print("[AVISO] Menos de duas DNNs disponíveis; gráfico de ablação ignorado.")
        return
    frame = pd.DataFrame(rows).set_index("Experimento")[["F1", "Recall", "FPR"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    frame.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Valor")
    ax.set_title("Comparação das ablações da DNN")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, report_dir / "ablation_comparison.png")


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    # A arquitetura independe da execução dos experimentos.
    generate_architecture(args.report_dir)

    try:
        generate_eda(args.data_dir, args.report_dir)
    except FileNotFoundError as exc:
        print(f"[AVISO] EDA não gerada: {exc}")

    generate_training_history(args.outputs_dir, args.main_dnn, args.report_dir)
    generate_threshold_comparison(args.outputs_dir, args.main_dnn, args.report_dir)
    generate_model_comparison(
        args.outputs_dir, args.baseline_dir, args.main_dnn, args.report_dir
    )
    generate_ablation_comparison(args.outputs_dir, args.experiments, args.report_dir)

    print("\nFiguras consolidadas em:", args.report_dir)
    print(
        "As matrizes de confusão, ROC e Precision-Recall permanecem nos diretórios "
        "figures/ de cada experimento, pois são geradas pelos scripts de treinamento."
    )


if __name__ == "__main__":
    main()
