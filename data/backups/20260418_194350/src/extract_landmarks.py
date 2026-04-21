"""Hand landmark extraction and preprocessing utilities.

This module is the shared core for:
- data collection
- model training
- real-time inference

Using one common implementation keeps training-time and inference-time preprocessing
consistent (critical for reliable Lab-8-style pipelines).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np

NUM_LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
FEATURES_PER_HAND = NUM_LANDMARKS_PER_HAND * COORDS_PER_LANDMARK  # 63
TOTAL_FEATURES = FEATURES_PER_HAND * 2  # 126 (left + right)


@dataclass
class HandExtractorConfig:
    """Configuration for MediaPipe hand landmark extraction."""

    static_image_mode: bool = False
    max_num_hands: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


def flatten_hand_landmarks(hand_landmarks: object) -> np.ndarray:
    """Convert one detected hand into a flat 63-dim vector (x, y, z for 21 points)."""
    row = []
    for lm in hand_landmarks.landmark:
        row.extend([float(lm.x), float(lm.y), float(lm.z)])
    return np.asarray(row, dtype=np.float32)


def normalize_single_hand(hand_row: Sequence[float]) -> np.ndarray:
    """Normalize one 63-dim hand vector with wrist-relative and scale normalization.

    Steps:
    1) Translate all points so wrist landmark (id=0) becomes origin.
    2) Scale by max 2D wrist-relative distance to reduce sensitivity to hand size.
    """
    arr = np.asarray(hand_row, dtype=np.float32)
    if arr.shape[0] != FEATURES_PER_HAND:
        raise ValueError(
            f"Expected {FEATURES_PER_HAND} values for one hand, got {arr.shape[0]}"
        )

    if np.allclose(arr, 0.0):
        return np.zeros(FEATURES_PER_HAND, dtype=np.float32)

    pts = arr.reshape(NUM_LANDMARKS_PER_HAND, COORDS_PER_LANDMARK)
    wrist = pts[0].copy()
    pts = pts - wrist

    scale = float(np.linalg.norm(pts[:, :2], axis=1).max())
    if scale < 1e-6:
        scale = 1.0

    pts = pts / scale
    return pts.reshape(-1).astype(np.float32)


def preprocess_feature_row(raw_row: Sequence[float]) -> np.ndarray:
    """Preprocess a raw 126-dim (left+right) feature row into normalized features."""
    arr = np.asarray(raw_row, dtype=np.float32).reshape(-1)
    if arr.shape[0] != TOTAL_FEATURES:
        raise ValueError(f"Expected {TOTAL_FEATURES} features, got {arr.shape[0]}")

    left_raw = arr[:FEATURES_PER_HAND]
    right_raw = arr[FEATURES_PER_HAND:]

    left = normalize_single_hand(left_raw)
    right = normalize_single_hand(right_raw)
    return np.concatenate([left, right], axis=0).astype(np.float32)


def preprocess_feature_matrix(raw_matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """Vectorized wrapper for preprocessing multiple 126-dim samples.

    This function is designed to be used inside scikit-learn transformers.
    """
    arr = np.asarray(raw_matrix, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] != TOTAL_FEATURES:
        raise ValueError(
            f"Expected matrix with {TOTAL_FEATURES} columns, got {arr.shape[1]}"
        )

    return np.vstack([preprocess_feature_row(row) for row in arr]).astype(np.float32)


class HandLandmarkExtractor:
    """MediaPipe-based extractor that returns left+right hand feature rows."""

    def __init__(self, config: Optional[HandExtractorConfig] = None) -> None:
        self.config = config or HandExtractorConfig()
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils

        self._hands = self._mp_hands.Hands(
            static_image_mode=self.config.static_image_mode,
            max_num_hands=self.config.max_num_hands,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "HandLandmarkExtractor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def process_frame(self, frame_bgr: np.ndarray) -> object:
        """Run MediaPipe Hands on one BGR frame and return raw result object."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self._hands.process(rgb)

    def _build_two_hand_row(self, result: object) -> Optional[np.ndarray]:
        left = np.zeros(FEATURES_PER_HAND, dtype=np.float32)
        right = np.zeros(FEATURES_PER_HAND, dtype=np.float32)

        multi_hands = getattr(result, "multi_hand_landmarks", None)
        if not multi_hands:
            return None

        handedness_list = getattr(result, "multi_handedness", None) or []

        for idx, hand_landmarks in enumerate(multi_hands):
            row = flatten_hand_landmarks(hand_landmarks)
            label = None

            if idx < len(handedness_list):
                label = handedness_list[idx].classification[0].label.lower()

            if label == "left":
                left = row
            elif label == "right":
                right = row
            else:
                # Fallback if handedness is missing.
                wrist_x = hand_landmarks.landmark[0].x
                if wrist_x < 0.5:
                    left = row
                else:
                    right = row

        return np.concatenate([left, right], axis=0).astype(np.float32)

    def extract_raw_features(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], object]:
        """Return raw 126-dim feature row and raw MediaPipe result."""
        result = self.process_frame(frame_bgr)
        row = self._build_two_hand_row(result)
        return row, result

    def extract_preprocessed_features(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], object]:
        """Return normalized 126-dim row and raw MediaPipe result."""
        raw_row, result = self.extract_raw_features(frame_bgr)
        if raw_row is None:
            return None, result
        return preprocess_feature_row(raw_row), result

    def draw_landmarks(self, frame_bgr: np.ndarray, result: object) -> np.ndarray:
        """Draw detected hand landmarks in-place and return the frame."""
        multi_hands = getattr(result, "multi_hand_landmarks", None)
        if not multi_hands:
            return frame_bgr

        for hand_landmarks in multi_hands:
            self._mp_draw.draw_landmarks(
                frame_bgr, hand_landmarks, self._mp_hands.HAND_CONNECTIONS
            )

        return frame_bgr
