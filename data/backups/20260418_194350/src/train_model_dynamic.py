"""Train temporal RandomForest model for dynamic sign recognition."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
DEFAULT_LABEL_PATH = str(PROJECT_ROOT / "models" / "class_names_dynamic.joblib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dynamic sequence classifier.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--labels", default=DEFAULT_LABEL_PATH)
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--include-deltas", action="store_true", default=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--classifier", choices=["rf", "svm"], default="rf")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--svm-c", type=float, default=10.0)
    parser.add_argument("--svm-gamma", type=str, default="scale")
    parser.add_argument(
        "--use-pca",
        action="store_true",
        help="Enable PCA dimensionality reduction before classifier.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=0,
        help="Number of PCA components (0 means auto=min(64, n_features)).",
    )
    return parser.parse_args()


def build_model(args: argparse.Namespace, *, n_features: int) -> Tuple[Pipeline, str]:
    steps = [("scaler", StandardScaler())]
    model_desc_parts = [f"classifier={args.classifier}"]

    if args.use_pca:
        n_components = args.pca_components if args.pca_components > 0 else min(64, n_features)
        n_components = max(2, min(n_components, n_features))
        steps.append(("pca", PCA(n_components=n_components, random_state=args.random_state)))
        model_desc_parts.append(f"pca={n_components}")

    if args.classifier == "svm":
        steps.append(
            (
                "clf",
                SVC(
                    C=args.svm_c,
                    gamma=args.svm_gamma,
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=args.random_state,
                ),
            )
        )
        model_desc_parts.append(f"svm_c={args.svm_c}")
        model_desc_parts.append(f"svm_gamma={args.svm_gamma}")
    else:
        steps.append(
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=args.n_estimators,
                    random_state=args.random_state,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            )
        )
        model_desc_parts.append(f"rf_estimators={args.n_estimators}")

    return Pipeline(steps), ", ".join(model_desc_parts)


def main() -> None:
    args = parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(f"Missing dynamic dataset: {args.data}")

    print("Loading dynamic dataset...")
    df = pd.read_csv(args.data)
    if df.empty:
        raise ValueError(f"Dataset is empty: {args.data}")
    if "label" not in df.columns:
        raise ValueError("Dynamic dataset must contain a 'label' column")

    expected_dim = sequence_feature_dim(args.window_size, args.include_deltas)
    feature_cols = [f"f{i}" for i in range(expected_dim)]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected feature columns in dynamic dataset: first missing={missing[0]}"
        )

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df["label"].astype(str).str.strip()

    valid = (~X.isna().any(axis=1)) & y.ne("")
    dropped = int((~valid).sum())
    if dropped:
        print(f"Dropped invalid rows: {dropped}")

    X = X[valid].to_numpy(dtype=np.float32)
    y = y[valid].to_numpy(dtype=str)

    labels, counts = np.unique(y, return_counts=True)
    print("Label counts:")
    for label, count in zip(labels.tolist(), counts.tolist()):
        print(f"  {label}: {count}")

    if len(labels) < 2:
        raise ValueError("Need at least 2 labels to train dynamic model.")

    stratify = y if counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    model, model_desc = build_model(args, n_features=X_train.shape[1])

    print(f"Training dynamic model ({model_desc})...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Dynamic model accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, digits=4, zero_division=0))

    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    Path(args.labels).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)

    class_names = list(model.named_steps["clf"].classes_)
    joblib.dump(class_names, args.labels)

    print(f"Saved dynamic model: {args.model}")
    print(f"Saved dynamic class names: {args.labels}")


if __name__ == "__main__":
    main()
