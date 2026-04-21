"""Sequence feature utilities for dynamic sign recognition."""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    from .extract_landmarks import TOTAL_FEATURES
except ImportError:  # pragma: no cover
    from extract_landmarks import TOTAL_FEATURES


def sequence_feature_dim(window_size: int, include_deltas: bool = True) -> int:
    base = window_size * TOTAL_FEATURES
    if not include_deltas:
        return base
    delta = (window_size - 1) * TOTAL_FEATURES
    return base + delta


def sequence_to_feature_vector(
    sequence_rows: Sequence[Sequence[float]],
    *,
    window_size: int,
    include_deltas: bool = True,
) -> np.ndarray:
    """Convert a sequence (T, 126) into a flat feature vector.

    - Pads with zeros if T < window_size
    - Truncates if T > window_size (keeps the most recent window)
    - Optional temporal delta features improve motion separability
    """
    arr = np.asarray(sequence_rows, dtype=np.float32)

    if arr.ndim != 2 or arr.shape[1] != TOTAL_FEATURES:
        raise ValueError(
            f"Expected sequence shape (T, {TOTAL_FEATURES}), got {arr.shape}"
        )

    if arr.shape[0] < window_size:
        pad = np.zeros((window_size - arr.shape[0], TOTAL_FEATURES), dtype=np.float32)
        arr = np.vstack([arr, pad])
    elif arr.shape[0] > window_size:
        arr = arr[-window_size:]

    base = arr.reshape(-1)
    if not include_deltas:
        return base.astype(np.float32)

    deltas = np.diff(arr, axis=0).reshape(-1)
    return np.concatenate([base, deltas], axis=0).astype(np.float32)


def validate_flat_feature_length(
    feature_vector: Sequence[float], *, window_size: int, include_deltas: bool = True
) -> None:
    expected = sequence_feature_dim(window_size, include_deltas)
    actual = len(feature_vector)
    if actual != expected:
        raise ValueError(f"Expected feature length {expected}, got {actual}")
