"""Evaluate a trained sign model on landmark CSV data."""

from __future__ import annotations

import argparse

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

try:
    from .modeling import (
        DEFAULT_DATA_PATH,
        DEFAULT_MODEL_PATH,
        load_landmark_dataset,
        suggest_stratify_target,
    )
except ImportError:  # pragma: no cover - supports `python src/evaluate_model.py`
    import os
    import sys

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)

    from modeling import (
        DEFAULT_DATA_PATH,
        DEFAULT_MODEL_PATH,
        load_landmark_dataset,
        suggest_stratify_target,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained sign model.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="If >0, evaluate on a random holdout split from the dataset.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--show-confusion", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    X_raw, y, _, dropped = load_landmark_dataset(args.data)
    if dropped:
        print(f"Dropped invalid rows during load: {dropped}")

    model = joblib.load(args.model)

    if args.test_size > 0:
        stratify_target = suggest_stratify_target(y)
        _, X_eval, _, y_eval = train_test_split(
            X_raw,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=stratify_target,
        )
    else:
        X_eval, y_eval = X_raw, y

    preds = model.predict(X_eval)
    acc = accuracy_score(y_eval, preds)

    print(f"Evaluation samples: {len(y_eval)}")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_eval, preds, digits=4, zero_division=0))

    if args.show_confusion:
        label_order = sorted(np.unique(np.concatenate([y_eval, preds])))
        cm = confusion_matrix(y_eval, preds, labels=label_order)
        print("Confusion matrix labels:", label_order)
        print(cm)


if __name__ == "__main__":
    main()
