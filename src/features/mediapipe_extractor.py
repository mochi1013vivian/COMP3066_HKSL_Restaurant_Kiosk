"""MediaPipe landmark extraction and normalization utilities.

Supports two feature modes:
- hands: baseline, two-hand landmarks only
- hands_pose: hands + selected upper-body pose landmarks and engineered features
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np

NUM_LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
FEATURES_PER_HAND = NUM_LANDMARKS_PER_HAND * COORDS_PER_LANDMARK

FEATURE_MODE_HANDS = "hands"
FEATURE_MODE_HANDS_POSE = "hands_pose"
SUPPORTED_FEATURE_MODES = (FEATURE_MODE_HANDS, FEATURE_MODE_HANDS_POSE)

TOTAL_HAND_FEATURES = FEATURES_PER_HAND * 2
# Backward-compatible alias used by the baseline pipeline.
TOTAL_FEATURES = TOTAL_HAND_FEATURES

POSE_LANDMARK_COUNT = 6  # left/right shoulder, elbow, wrist
POSE_RAW_FEATURES = POSE_LANDMARK_COUNT * COORDS_PER_LANDMARK
TOTAL_RAW_FEATURES_HANDS_POSE = TOTAL_HAND_FEATURES + POSE_RAW_FEATURES

# Engineered pose/body-relative features (hands_pose mode)
POSE_ENGINEERED_DISTANCE_FEATURES = 8  # (2 hands) x (to L/R shoulder + L/R elbow)
POSE_ENGINEERED_WRIST_HEIGHT_FEATURES = 2  # left/right wrist y relative to shoulder line
POSE_ENGINEERED_ELBOW_ANGLE_FEATURES = 2  # left/right elbow angle
POSE_ENGINEERED_MISSING_FLAG_FEATURES = 1  # any-pose-present flag
POSE_ENGINEERED_FEATURES = (
    POSE_ENGINEERED_DISTANCE_FEATURES
    + POSE_ENGINEERED_WRIST_HEIGHT_FEATURES
    + POSE_ENGINEERED_ELBOW_ANGLE_FEATURES
    + POSE_ENGINEERED_MISSING_FLAG_FEATURES
)
TOTAL_MODEL_FEATURES_HANDS_POSE = TOTAL_RAW_FEATURES_HANDS_POSE + POSE_ENGINEERED_FEATURES

POSE_LANDMARK_INDEX_TO_NAME: Dict[int, str] = {
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
}


def get_raw_feature_dim(feature_mode: str = FEATURE_MODE_HANDS) -> int:
    if feature_mode == FEATURE_MODE_HANDS:
        return TOTAL_HAND_FEATURES
    if feature_mode == FEATURE_MODE_HANDS_POSE:
        return TOTAL_RAW_FEATURES_HANDS_POSE
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def get_model_feature_dim(feature_mode: str = FEATURE_MODE_HANDS) -> int:
    if feature_mode == FEATURE_MODE_HANDS:
        return TOTAL_HAND_FEATURES
    if feature_mode == FEATURE_MODE_HANDS_POSE:
        return TOTAL_MODEL_FEATURES_HANDS_POSE
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


@dataclass
class ExtractionResult:
    hands_result: object
    pose_result: Optional[object] = None
    pose_debug: Dict[str, object] = field(default_factory=dict)


@dataclass
class HandExtractorConfig:
    static_image_mode: bool = False
    max_num_hands: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    use_pose: bool = False
    pose_model_complexity: int = 1
    min_pose_detection_confidence: float = 0.35
    min_pose_tracking_confidence: float = 0.35
    pose_visibility_threshold: float = 0.35
    pose_presence_threshold: float = 0.0
    smooth_pose_landmarks: bool = True


def flatten_hand_landmarks(hand_landmarks: object) -> np.ndarray:
    row = []
    for lm in hand_landmarks.landmark:
        row.extend([float(lm.x), float(lm.y), float(lm.z)])
    return np.asarray(row, dtype=np.float32)


def normalize_single_hand(hand_row: Sequence[float]) -> np.ndarray:
    arr = np.asarray(hand_row, dtype=np.float32)
    if arr.shape[0] != FEATURES_PER_HAND:
        raise ValueError(f"Expected {FEATURES_PER_HAND}, got {arr.shape[0]}")

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


def _is_valid_point(pt: np.ndarray) -> bool:
    return bool(not np.allclose(pt, 0.0))


def _safe_shoulder_scale(left_shoulder: np.ndarray, right_shoulder: np.ndarray) -> float:
    if _is_valid_point(left_shoulder) and _is_valid_point(right_shoulder):
        scale = float(np.linalg.norm(right_shoulder[:2] - left_shoulder[:2]))
        if scale > 1e-6:
            return scale
    return 1.0


def _safe_pose_origin(left_shoulder: np.ndarray, right_shoulder: np.ndarray) -> np.ndarray:
    if _is_valid_point(left_shoulder) and _is_valid_point(right_shoulder):
        return (left_shoulder + right_shoulder) / 2.0
    if _is_valid_point(left_shoulder):
        return left_shoulder
    if _is_valid_point(right_shoulder):
        return right_shoulder
    return np.zeros(3, dtype=np.float32)


def _hand_center(hand_row: np.ndarray) -> Optional[np.ndarray]:
    if np.allclose(hand_row, 0.0):
        return None
    pts = hand_row.reshape(NUM_LANDMARKS_PER_HAND, COORDS_PER_LANDMARK)
    return pts.mean(axis=0).astype(np.float32)


def _distance2d_scaled(a: Optional[np.ndarray], b: np.ndarray, scale: float) -> float:
    if a is None or not _is_valid_point(b):
        return 0.0
    d = float(np.linalg.norm(a[:2] - b[:2]))
    return d / max(scale, 1e-6)


def _elbow_angle(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float:
    if not (_is_valid_point(shoulder) and _is_valid_point(elbow) and _is_valid_point(wrist)):
        return 0.0
    v1 = shoulder[:2] - elbow[:2]
    v2 = wrist[:2] - elbow[:2]
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_theta = float(np.dot(v1, v2) / (n1 * n2))
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    return theta / np.pi


def preprocess_feature_row(raw_row: Sequence[float], feature_mode: str = FEATURE_MODE_HANDS) -> np.ndarray:
    arr = np.asarray(raw_row, dtype=np.float32).reshape(-1)

    expected_raw_dim = get_raw_feature_dim(feature_mode)
    if arr.shape[0] != expected_raw_dim:
        raise ValueError(f"Expected {expected_raw_dim} features, got {arr.shape[0]}")

    left_raw = arr[:FEATURES_PER_HAND]
    right_raw = arr[FEATURES_PER_HAND:TOTAL_HAND_FEATURES]
    left = normalize_single_hand(left_raw)
    right = normalize_single_hand(right_raw)

    if feature_mode == FEATURE_MODE_HANDS:
        return np.concatenate([left, right], axis=0).astype(np.float32)

    if feature_mode != FEATURE_MODE_HANDS_POSE:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")

    pose_raw = arr[TOTAL_HAND_FEATURES:TOTAL_RAW_FEATURES_HANDS_POSE]
    pose_pts = pose_raw.reshape(POSE_LANDMARK_COUNT, COORDS_PER_LANDMARK).astype(np.float32)

    left_shoulder = pose_pts[0]
    right_shoulder = pose_pts[1]
    left_elbow = pose_pts[2]
    right_elbow = pose_pts[3]
    left_wrist = pose_pts[4]
    right_wrist = pose_pts[5]

    shoulder_scale = _safe_shoulder_scale(left_shoulder, right_shoulder)
    pose_origin = _safe_pose_origin(left_shoulder, right_shoulder)

    pose_norm = np.zeros_like(pose_pts, dtype=np.float32)
    for i, pt in enumerate(pose_pts):
        if _is_valid_point(pt):
            pose_norm[i] = (pt - pose_origin) / shoulder_scale

    left_hand_center = _hand_center(left_raw)
    right_hand_center = _hand_center(right_raw)

    dist_features = np.asarray(
        [
            _distance2d_scaled(left_hand_center, left_shoulder, shoulder_scale),
            _distance2d_scaled(left_hand_center, right_shoulder, shoulder_scale),
            _distance2d_scaled(left_hand_center, left_elbow, shoulder_scale),
            _distance2d_scaled(left_hand_center, right_elbow, shoulder_scale),
            _distance2d_scaled(right_hand_center, left_shoulder, shoulder_scale),
            _distance2d_scaled(right_hand_center, right_shoulder, shoulder_scale),
            _distance2d_scaled(right_hand_center, left_elbow, shoulder_scale),
            _distance2d_scaled(right_hand_center, right_elbow, shoulder_scale),
        ],
        dtype=np.float32,
    )

    shoulder_mid_y = 0.0
    if _is_valid_point(left_shoulder) and _is_valid_point(right_shoulder):
        shoulder_mid_y = float((left_shoulder[1] + right_shoulder[1]) / 2.0)
    elif _is_valid_point(left_shoulder):
        shoulder_mid_y = float(left_shoulder[1])
    elif _is_valid_point(right_shoulder):
        shoulder_mid_y = float(right_shoulder[1])

    wrist_height_features = np.asarray(
        [
            ((shoulder_mid_y - float(left_wrist[1])) / shoulder_scale) if _is_valid_point(left_wrist) else 0.0,
            ((shoulder_mid_y - float(right_wrist[1])) / shoulder_scale) if _is_valid_point(right_wrist) else 0.0,
        ],
        dtype=np.float32,
    )

    elbow_angle_features = np.asarray(
        [
            _elbow_angle(left_shoulder, left_elbow, left_wrist),
            _elbow_angle(right_shoulder, right_elbow, right_wrist),
        ],
        dtype=np.float32,
    )

    pose_present_flag = np.asarray(
        [1.0 if np.any(np.abs(pose_pts) > 1e-8) else 0.0],
        dtype=np.float32,
    )

    return np.concatenate(
        [
            left,
            right,
            pose_norm.reshape(-1),
            dist_features,
            wrist_height_features,
            elbow_angle_features,
            pose_present_flag,
        ],
        axis=0,
    ).astype(np.float32)


class HandLandmarkExtractor:
    def __init__(self, config: Optional[HandExtractorConfig] = None) -> None:
        self.config = config or HandExtractorConfig()
        self._mp_hands = mp.solutions.hands
        self._mp_pose = mp.solutions.pose
        self._mp_draw = mp.solutions.drawing_utils

        self._hands = self._mp_hands.Hands(
            static_image_mode=self.config.static_image_mode,
            max_num_hands=self.config.max_num_hands,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._pose = None
        if self.config.use_pose:
            self._pose = self._mp_pose.Pose(
                static_image_mode=self.config.static_image_mode,
                model_complexity=self.config.pose_model_complexity,
                smooth_landmarks=self.config.smooth_pose_landmarks,
                min_detection_confidence=self.config.min_pose_detection_confidence,
                min_tracking_confidence=self.config.min_pose_tracking_confidence,
            )

    def close(self) -> None:
        self._hands.close()
        if self._pose is not None:
            self._pose.close()

    def __enter__(self) -> "HandLandmarkExtractor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def process_frame(self, frame_bgr: np.ndarray) -> ExtractionResult:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        hands_result = self._hands.process(rgb)
        pose_result = self._pose.process(rgb) if self._pose is not None else None
        return ExtractionResult(hands_result=hands_result, pose_result=pose_result)

    def _build_two_hand_row(self, hands_result: object) -> Optional[np.ndarray]:
        left = np.zeros(FEATURES_PER_HAND, dtype=np.float32)
        right = np.zeros(FEATURES_PER_HAND, dtype=np.float32)

        multi_hands = getattr(hands_result, "multi_hand_landmarks", None)
        if not multi_hands:
            return None

        handedness_list = getattr(hands_result, "multi_handedness", None) or []
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
                wrist_x = hand_landmarks.landmark[0].x
                if wrist_x < 0.5:
                    left = row
                else:
                    right = row

        return np.concatenate([left, right], axis=0).astype(np.float32)

    def _build_pose_row(self, pose_result: Optional[object]) -> Tuple[np.ndarray, Dict[str, object]]:
        pose_row = np.zeros(POSE_RAW_FEATURES, dtype=np.float32)
        pose_debug: Dict[str, object] = {
            "present": False,
            "missing_points": list(POSE_LANDMARK_INDEX_TO_NAME.values()),
            "visibilities": {},
            "presence_scores": {},
            "in_frame_ratio": 0.0,
        }
        if pose_result is None:
            return pose_row, pose_debug

        landmarks = getattr(pose_result, "pose_landmarks", None)
        if landmarks is None:
            return pose_row, pose_debug

        selected_indices = [11, 12, 13, 14, 15, 16]  # L/R shoulder, elbow, wrist
        values: List[float] = []
        missing_points: List[str] = []
        visibilities: Dict[str, float] = {}
        presence_scores: Dict[str, float] = {}
        in_frame_count = 0
        valid_count = 0

        for idx in selected_indices:
            lm = landmarks.landmark[idx]
            name = POSE_LANDMARK_INDEX_TO_NAME[idx]
            vis = float(getattr(lm, "visibility", 0.0))
            pres = float(getattr(lm, "presence", 1.0))
            visibilities[name] = vis
            presence_scores[name] = pres

            is_valid = vis >= self.config.pose_visibility_threshold and pres >= self.config.pose_presence_threshold
            if is_valid:
                x, y, z = float(lm.x), float(lm.y), float(lm.z)
                values.extend([x, y, z])
                valid_count += 1
                if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                    in_frame_count += 1
            else:
                values.extend([0.0, 0.0, 0.0])
                missing_points.append(name)

        pose_debug = {
            "present": valid_count > 0,
            "missing_points": missing_points,
            "visibilities": visibilities,
            "presence_scores": presence_scores,
            "in_frame_ratio": float(in_frame_count / max(1, valid_count)),
        }
        return np.asarray(values, dtype=np.float32), pose_debug

    def extract_raw_features(
        self, frame_bgr: np.ndarray, feature_mode: str = FEATURE_MODE_HANDS
    ) -> Tuple[Optional[np.ndarray], ExtractionResult]:
        result = self.process_frame(frame_bgr)
        hands_row = self._build_two_hand_row(result.hands_result)

        if hands_row is None:
            return None, result

        if feature_mode == FEATURE_MODE_HANDS:
            return hands_row, result
        if feature_mode == FEATURE_MODE_HANDS_POSE:
            pose_row, pose_debug = self._build_pose_row(result.pose_result)
            result.pose_debug = pose_debug
            return np.concatenate([hands_row, pose_row], axis=0).astype(np.float32), result
        raise ValueError(f"Unsupported feature mode: {feature_mode}")

    def get_pose_debug_info(self, result: ExtractionResult) -> Dict[str, object]:
        return dict(getattr(result, "pose_debug", {}) or {})

    def draw_landmarks(self, frame_bgr: np.ndarray, result: ExtractionResult, draw_pose: bool = False) -> np.ndarray:
        multi_hands = getattr(result.hands_result, "multi_hand_landmarks", None)
        if not multi_hands:
            if draw_pose and result.pose_result is not None:
                return self._draw_pose_keypoints(frame_bgr, result.pose_result)
            return frame_bgr

        for hand_landmarks in multi_hands:
            self._mp_draw.draw_landmarks(frame_bgr, hand_landmarks, self._mp_hands.HAND_CONNECTIONS)

        if draw_pose and result.pose_result is not None:
            frame_bgr = self._draw_pose_keypoints(frame_bgr, result.pose_result)
        return frame_bgr

    def _draw_pose_keypoints(self, frame_bgr: np.ndarray, pose_result: object) -> np.ndarray:
        pose_landmarks = getattr(pose_result, "pose_landmarks", None)
        if pose_landmarks is None:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        idxs = [11, 12, 13, 14, 15, 16]
        points = {}
        for idx in idxs:
            lm = pose_landmarks.landmark[idx]
            x_raw, y_raw = float(lm.x), float(lm.y)
            x, y = int(np.clip(x_raw * w, 0, w - 1)), int(np.clip(y_raw * h, 0, h - 1))
            points[idx] = (x, y)
            vis = float(getattr(lm, "visibility", 0.0))
            color = (0, 220, 0) if vis >= self.config.pose_visibility_threshold else (0, 0, 255)
            cv2.circle(frame_bgr, (x, y), 7, color, -1)
            cv2.putText(
                frame_bgr,
                POSE_LANDMARK_INDEX_TO_NAME[idx].replace("_", ""),
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
            )

        for a, b in [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12)]:
            if a in points and b in points:
                cv2.line(frame_bgr, points[a], points[b], (255, 40, 255), 3)
        return frame_bgr
