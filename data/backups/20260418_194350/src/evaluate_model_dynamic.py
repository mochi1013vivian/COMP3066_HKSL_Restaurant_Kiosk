"""Evaluate dynamic sequence classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

try:
    from .extract_sequences import sequence_feature_dim
except ImportError:  # pragma: no cover
    import os
    import sys

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)

    from extract_sequences import sequence_feature_dim


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = str(PROJECT_ROOT / "data" / "raw" / "landmarks_dynamic.csv")
DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "models" / "sign_model_dynamic.joblib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dynamic sequence model.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--include-deltas", action="store_true", default=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--show-confusion", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(f"Missing dynamic dataset: {args.data}")
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Missing dynamic model: {args.model}")

    df = pd.read_csv(args.data)
    expected_dim = sequence_feature_dim(args.window_size, args.include_deltas)
    feature_cols = [f"f{i}" for i in range(expected_dim)]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df["label"].astype(str).str.strip()

    valid = (~X.isna().any(axis=1)) & y.ne("")
    X = X[valid]
    y = y[valid]

    if len(X) == 0:
        raise ValueError("No valid dynamic samples to evaluate.")

    model = joblib.load(args.model)

    _, X_eval, _, y_eval = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if y.nunique() > 1 and y.value_counts().min() >= 2 else None,
    )

    preds = model.predict(X_eval)
    acc = accuracy_score(y_eval, preds)

    print(f"Dynamic eval samples: {len(y_eval)}")
    print(f"Dynamic accuracy: {acc:.4f}")
    print(classification_report(y_eval, preds, digits=4, zero_division=0))

    if args.show_confusion:
        labels = sorted(set(y_eval) | set(preds))
        cm = confusion_matrix(y_eval, preds, labels=labels)
        print("Confusion matrix labels:", labels)
        print(cm)


if __name__ == "__main__":
    main()
