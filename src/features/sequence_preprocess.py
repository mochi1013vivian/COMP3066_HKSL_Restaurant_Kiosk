"""Sequence preprocessing helpers for PyTorch training and inference parity."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .mediapipe_extractor import (
    FEATURE_MODE_HANDS,
    get_model_feature_dim,
    get_raw_feature_dim,
    preprocess_feature_row,
)


def expected_feature_dim(window_size: int, feature_mode: str = FEATURE_MODE_HANDS, normalized: bool = False) -> int:
    per_frame = get_model_feature_dim(feature_mode) if normalized else get_raw_feature_dim(feature_mode)
    return window_size * per_frame


def flatten_sequence(sequence: np.ndarray, feature_mode: str = FEATURE_MODE_HANDS) -> np.ndarray:
    raw_dim = get_raw_feature_dim(feature_mode)
    if sequence.ndim != 2 or sequence.shape[1] != raw_dim:
        raise ValueError(f"Expected shape (T, {raw_dim}), got {sequence.shape}")
    return sequence.reshape(-1).astype(np.float32)


def unflatten_sequence(feature_row: Sequence[float], window_size: int, feature_mode: str = FEATURE_MODE_HANDS) -> np.ndarray:
    arr = np.asarray(feature_row, dtype=np.float32).reshape(-1)
    raw_dim = get_raw_feature_dim(feature_mode)
    expected = expected_feature_dim(window_size, feature_mode=feature_mode, normalized=False)
    if arr.shape[0] != expected:
        raise ValueError(f"Expected {expected} values, got {arr.shape[0]}")
    return arr.reshape(window_size, raw_dim).astype(np.float32)


def normalize_sequence(sequence: np.ndarray, feature_mode: str = FEATURE_MODE_HANDS) -> np.ndarray:
    return np.vstack([preprocess_feature_row(frame, feature_mode=feature_mode) for frame in sequence]).astype(np.float32)
