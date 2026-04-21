"""Shared model/data helpers for training and evaluation scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

try:
    from .extract_landmarks import TOTAL_FEATURES, preprocess_feature_matrix
except ImportError:  # pragma: no cover - supports `python src/*.py`
    from extract_landmarks import TOTAL_FEATURES, preprocess_feature_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_PATH = str(PROJECT_ROOT / "data" / "raw" / "landmarks.csv")
DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "models" / "sign_model.joblib")
DEFAULT_LABEL_PATH = str(PROJECT_ROOT / "models" / "class_names.joblib")


def expected_feature_columns() -> List[str]:
    return [f"f{i}" for i in range(TOTAL_FEATURES)]


def _resolve_feature_columns(df: pd.DataFrame) -> List[str]:
    expected = expected_feature_columns()
    if all(col in df.columns for col in expected):
        return expected

    # Fallback for datasets that still have 126 non-label columns but different names.
    fallback = [col for col in df.columns if col != "label"]
    if len(fallback) != TOTAL_FEATURES:
        raise ValueError(
            "Feature column mismatch. "
            f"Expected {TOTAL_FEATURES} features, found {len(fallback)}."
        )
    return fallback


def load_landmark_dataset(data_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """Load and validate landmark CSV dataset.

    Returns:
        X_raw: shape (n_samples, 126)
        y: shape (n_samples,)
        feature_columns: list of selected feature columns
        dropped_rows: number of rows dropped due to invalid values
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing dataset: {data_path}")

    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {data_path}")
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    feature_columns = _resolve_feature_columns(df)

    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = df["label"].astype(str).str.strip()

    valid_mask = (~X.isna().any(axis=1)) & y.ne("")
    dropped_rows = int((~valid_mask).sum())

    X = X[valid_mask].to_numpy(dtype=np.float32)
    y = y[valid_mask].to_numpy(dtype=str)

    if X.shape[0] == 0:
        raise ValueError("No valid samples remained after cleaning the dataset.")

    if X.shape[1] != TOTAL_FEATURES:
        raise ValueError(
            f"Expected {TOTAL_FEATURES} features, got {X.shape[1]} after loading."
        )

    return X, y, feature_columns, dropped_rows


def build_random_forest_pipeline(
    n_estimators: int = 300,
    random_state: int = 42,
) -> Pipeline:
    """Build a landmark classifier pipeline.

    Pipeline stages:
    1) Landmark normalization (shared preprocessing logic)
    2) Standard scaling
    3) Random Forest classifier
    """
    return Pipeline(
        [
            (
                "landmark_preprocess",
                FunctionTransformer(preprocess_feature_matrix, validate=False),
            ),
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )


def suggest_stratify_target(y: Sequence[str] | np.ndarray) -> np.ndarray | None:
    """Return y for stratification only when class counts are sufficient."""
    arr = np.asarray(y)
    labels, counts = np.unique(arr, return_counts=True)
    if len(labels) < 2:
        return None
    if counts.min() < 2:
        return None
    return arr


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
