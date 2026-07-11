"""Treina baselines clássicos no UNSW-NB15."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .constants import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, RANDOM_STATE
from .data import load_official_splits, prepare_data
from .metrics import (
    evaluate_binary,
    save_evaluation_plots,
    save_metrics,
    threshold_maximizing_f1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("dummy", "logistic", "random_forest"),
        default=("dummy", "logistic", "random_forest"),
    )
    return parser.parse_args()


def build_models(selected: list[str] | tuple[str, ...]):
    available = {
        "dummy": DummyClassifier(strategy="prior", random_state=RANDOM_STATE),
        "logistic": LogisticRegression(
            max_iter=1_000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }
    return {name: available[name] for name in selected}


def main() -> None:
    args = parse_args()
    train_frame, test_frame = load_official_splits(args.data_dir)
    data = prepare_data(
        train_frame,
        test_frame,
        validation_size=args.validation_size,
        random_state=RANDOM_STATE,
    )

    models_dir = args.output_dir / "models"
    metrics_dir = args.output_dir / "metrics"
    figures_dir = args.output_dir / "figures"
    for directory in (models_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    joblib.dump(data.preprocessor, models_dir / "baseline_preprocessor.joblib")
    (metrics_dir / "baseline_feature_names.txt").write_text(
        "\n".join(data.feature_names), encoding="utf-8"
    )

    for name, model in build_models(args.models).items():
        started = time.perf_counter()
        model.fit(data.x_train, data.y_train)
        elapsed = time.perf_counter() - started

        val_probability = model.predict_proba(data.x_val)[:, 1]
        test_probability = model.predict_proba(data.x_test)[:, 1]
        selected_threshold = threshold_maximizing_f1(data.y_val, val_probability)

        result = {
            "model": name,
            "training_seconds": elapsed,
            "input_features": int(data.x_train.shape[1]),
            "validation_default": evaluate_binary(
                data.y_val, val_probability, threshold=0.5
            ),
            "validation_tuned": evaluate_binary(
                data.y_val, val_probability, threshold=selected_threshold
            ),
            "test_default": evaluate_binary(
                data.y_test, test_probability, threshold=0.5
            ),
            "test_tuned": evaluate_binary(
                data.y_test, test_probability, threshold=selected_threshold
            ),
        }

        save_metrics(result, metrics_dir / f"{name}.json")
        save_evaluation_plots(
            data.y_test,
            test_probability,
            selected_threshold,
            figures_dir,
            prefix=f"{name}_test",
        )
        joblib.dump(model, models_dir / f"{name}.joblib")

        tuned = result["test_tuned"]
        print(
            f"[{name}] F1={tuned['f1']:.4f} "
            f"Recall={tuned['recall']:.4f} FPR={tuned['false_positive_rate']:.4f} "
            f"Threshold={selected_threshold:.4f}"
        )


if __name__ == "__main__":
    main()
