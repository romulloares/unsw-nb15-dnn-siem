"""Treina uma rede neural densa em PyTorch no UNSW-NB15."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .constants import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, RANDOM_STATE
from .data import load_official_splits, prepare_data
from .metrics import (
    evaluate_binary,
    save_evaluation_plots,
    save_metrics,
    threshold_maximizing_f1,
)


@dataclass(frozen=True)
class TrainConfig:
    hidden_dims: tuple[int, ...]
    dropout: tuple[float, ...]
    learning_rate: float
    batch_size: int
    epochs: int
    patience: int
    validation_size: float
    use_pos_weight: bool
    random_state: int = RANDOM_STATE


class DenseNetwork(nn.Module):
    """MLP para classificação binária com saída em logits."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: tuple[float, ...],
    ) -> None:
        super().__init__()
        if len(hidden_dims) != len(dropout):
            raise ValueError("hidden_dims e dropout devem ter o mesmo comprimento.")

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim, dropout_rate in zip(hidden_dims, dropout, strict=True):
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 64])
    parser.add_argument("--dropout", nargs="+", type=float, default=[0.30, 0.20])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument(
        "--no-pos-weight",
        action="store_true",
        help="Desativa o peso automático para a classe positiva.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32, copy=False)),
        torch.from_numpy(labels.astype(np.float32, copy=False)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def epoch_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(features)
            loss = loss_function(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * labels.shape[0]
            total_examples += labels.shape[0]

    return total_loss / max(total_examples, 1)


def predict_probabilities(
    model: nn.Module,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        torch.from_numpy(features.astype(np.float32, copy=False)),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.to(device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities)


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        hidden_dims=tuple(args.hidden_dims),
        dropout=tuple(args.dropout),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        validation_size=args.validation_size,
        use_pos_weight=not args.no_pos_weight,
    )
    if len(config.hidden_dims) != len(config.dropout):
        raise ValueError("Informe uma taxa de dropout para cada camada oculta.")
    if any(not 0.0 <= value < 1.0 for value in config.dropout):
        raise ValueError("Cada taxa de dropout deve estar no intervalo [0, 1).")

    set_seed(config.random_state)
    device = select_device()
    print(f"Dispositivo: {device}")

    train_frame, test_frame = load_official_splits(args.data_dir)
    data = prepare_data(
        train_frame,
        test_frame,
        validation_size=config.validation_size,
        random_state=config.random_state,
    )

    train_loader = make_loader(
        data.x_train, data.y_train, config.batch_size, shuffle=True
    )
    val_loader = make_loader(data.x_val, data.y_val, config.batch_size, shuffle=False)

    model = DenseNetwork(
        input_dim=data.x_train.shape[1],
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)

    if config.use_pos_weight:
        positives = max(int(data.y_train.sum()), 1)
        negatives = max(int(data.y_train.shape[0] - data.y_train.sum()), 1)
        pos_weight = torch.tensor([negatives / positives], device=device)
        loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        pos_weight_value: float | None = float(pos_weight.item())
    else:
        loss_function = nn.BCEWithLogitsLoss()
        pos_weight_value = None

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        train_loss = epoch_loss(
            model, train_loader, loss_function, device, optimizer=optimizer
        )
        val_loss = epoch_loss(model, val_loader, loss_function, device)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss}
        )
        print(
            f"Época {epoch:03d} | treino={train_loss:.6f} | validação={val_loss:.6f}"
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print("Early stopping acionado.")
                break

    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)

    val_probability = predict_probabilities(
        model, data.x_val, config.batch_size, device
    )
    test_probability = predict_probabilities(
        model, data.x_test, config.batch_size, device
    )
    tuned_threshold = threshold_maximizing_f1(data.y_val, val_probability)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "model": "dnn_pytorch",
        "device": str(device),
        "config": asdict(config),
        "pos_weight": pos_weight_value,
        "input_features": int(data.x_train.shape[1]),
        "parameters": int(parameter_count),
        "epochs_ran": len(history),
        "best_validation_loss": best_val_loss,
        "training_seconds": training_seconds,
        "validation_default": evaluate_binary(
            data.y_val, val_probability, threshold=0.5
        ),
        "validation_tuned": evaluate_binary(
            data.y_val, val_probability, threshold=tuned_threshold
        ),
        "test_default": evaluate_binary(
            data.y_test, test_probability, threshold=0.5
        ),
        "test_tuned": evaluate_binary(
            data.y_test, test_probability, threshold=tuned_threshold
        ),
    }

    models_dir = args.output_dir / "models"
    metrics_dir = args.output_dir / "metrics"
    figures_dir = args.output_dir / "figures"
    for directory in (models_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(data.x_train.shape[1]),
            "hidden_dims": config.hidden_dims,
            "dropout": config.dropout,
            "threshold": tuned_threshold,
        },
        models_dir / "dnn_pytorch.pt",
    )
    joblib.dump(data.preprocessor, models_dir / "dnn_preprocessor.joblib")
    (metrics_dir / "dnn_feature_names.txt").write_text(
        "\n".join(data.feature_names), encoding="utf-8"
    )
    pd.DataFrame(history).to_csv(metrics_dir / "dnn_history.csv", index=False)
    save_metrics(result, metrics_dir / "dnn_pytorch.json")
    save_evaluation_plots(
        data.y_test,
        test_probability,
        tuned_threshold,
        figures_dir,
        prefix="dnn_pytorch_test",
    )

    (metrics_dir / "dnn_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    tuned = result["test_tuned"]
    print(
        f"Teste | F1={tuned['f1']:.4f} Recall={tuned['recall']:.4f} "
        f"FPR={tuned['false_positive_rate']:.4f} Threshold={tuned_threshold:.4f}"
    )


if __name__ == "__main__":
    main()
