"""Train a RandomForest sign classifier from landmark CSV data."""

from __future__ import annotations

import argparse
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

try:
    from .modeling import (
        DEFAULT_DATA_PATH,
        DEFAULT_LABEL_PATH,
        DEFAULT_MODEL_PATH,
        build_random_forest_pipeline,
        ensure_parent_dir,
        load_landmark_dataset,
        suggest_stratify_target,
    )
except ImportError:  # pragma: no cover - supports `python src/train_model.py`
    import os
    import sys

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)

    from modeling import (
        DEFAULT_DATA_PATH,
        DEFAULT_LABEL_PATH,
        DEFAULT_MODEL_PATH,
        build_random_forest_pipeline,
        ensure_parent_dir,
        load_landmark_dataset,
        suggest_stratify_target,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sign model from landmarks CSV.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--labels", default=DEFAULT_LABEL_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=2,
        help="Drop classes with fewer samples than this value.",
    )
    return parser.parse_args()


def _get_model_classes(model, y_fallback) -> list[str]:
    if hasattr(model, "classes_"):
        return [str(x) for x in model.classes_]
    clf = getattr(model, "named_steps", {}).get("clf")
    if clf is not None and hasattr(clf, "classes_"):
        return [str(x) for x in clf.classes_]
    return [str(x) for x in sorted(np.unique(y_fallback).tolist())]


def main() -> None:
    args = parse_args()

    print("Loading dataset...")
    X_raw, y, feature_columns, dropped = load_landmark_dataset(args.data)
    print(f"Loaded samples: {len(y)}")
    print(f"Feature columns: {len(feature_columns)}")
    if dropped:
        print(f"Dropped invalid rows: {dropped}")

    labels, counts = np.unique(y, return_counts=True)
    print("Label counts before filtering:")
    for label, count in zip(labels.tolist(), counts.tolist()):
        print(f"  {label}: {count}")

    if args.min_samples_per_class > 1:
        keep_labels = set(labels[counts >= args.min_samples_per_class])
        keep_mask = np.array([label in keep_labels for label in y])
        X_raw = X_raw[keep_mask]
        y = y[keep_mask]

        removed_classes = [
            label for label, count in zip(labels.tolist(), counts.tolist()) if count < args.min_samples_per_class
        ]
        if removed_classes:
            print(
                "Dropped underrepresented classes "
                f"(< {args.min_samples_per_class} samples): {removed_classes}"
            )

    unique_after = np.unique(y)
    if len(unique_after) < 2:
        raise ValueError(
            "Need at least 2 classes to train a multiclass classifier after filtering."
        )

    stratify_target = suggest_stratify_target(y)

    if args.test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=stratify_target,
        )
    else:
        X_train, y_train = X_raw, y
        X_test, y_test = X_raw, y

    print("Building model pipeline...")
    model = build_random_forest_pipeline(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )

    print("Training model...")
    model.fit(X_train, y_train)
    print("Training complete.")

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Evaluation accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, digits=4, zero_division=0))

    ensure_parent_dir(args.model)
    ensure_parent_dir(args.labels)

    class_names = _get_model_classes(model, y)
    joblib.dump(model, args.model)
    joblib.dump(class_names, args.labels)

    print(f"Saved model: {args.model}")
    print(f"Saved class names: {args.labels}")


if __name__ == "__main__":
    main()