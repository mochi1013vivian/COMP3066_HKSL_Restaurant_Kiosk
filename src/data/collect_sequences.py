"""Sequence data collection for closed-domain HKSL restaurant signs.

Behavior requested:
- Press C once to arm recording session
- 3-second countdown
- Continuous automatic recording until target sample count is reached
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

try:
    from ..config.labels import load_label_set
    from ..config.settings import DEFAULT_SEQUENCE_CSV, DEFAULT_WINDOW_SIZE
    from ..features.mediapipe_extractor import (
        FEATURE_MODE_HANDS,
        FEATURE_MODE_HANDS_POSE,
        HandExtractorConfig,
        HandLandmarkExtractor,
        get_raw_feature_dim,
    )
    from ..features.sequence_preprocess import flatten_sequence
except ImportError:  # pragma: no cover
    import os
    import sys

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_ROOT = os.path.dirname(CURRENT_DIR)
    if SRC_ROOT not in sys.path:
        sys.path.insert(0, SRC_ROOT)

    from config.labels import load_label_set
    from config.settings import DEFAULT_SEQUENCE_CSV, DEFAULT_WINDOW_SIZE
    from features.mediapipe_extractor import (
        FEATURE_MODE_HANDS,
        FEATURE_MODE_HANDS_POSE,
        HandExtractorConfig,
        HandLandmarkExtractor,
        get_raw_feature_dim,
    )
    from features.sequence_preprocess import flatten_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect sequence samples for one label.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--label-file", default=None)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_SEQUENCE_CSV)
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--cooldown", type=float, default=0.25)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--min-pose-detection-confidence", type=float, default=0.35)
    parser.add_argument("--min-pose-tracking-confidence", type=float, default=0.35)
    parser.add_argument("--pose-visibility-threshold", type=float, default=0.35)
    parser.add_argument(
        "--feature-mode",
        choices=[FEATURE_MODE_HANDS, FEATURE_MODE_HANDS_POSE],
        default=FEATURE_MODE_HANDS_POSE,
        help="Feature mode: hands baseline or hands+upper-body pose",
    )
    parser.add_argument(
        "--with-arms",
        action="store_true",
        help="Convenience switch for arms+hands collection (equivalent to --feature-mode hands_pose).",
    )
    return parser.parse_args()


def ensure_csv_header(path: Path, window_size: int, feature_mode: str) -> None:
    expected_dim = window_size * get_raw_feature_dim(feature_mode)
    header = ["label", "seq_len"] + [f"f{i}" for i in range(expected_dim)]

    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)
        return

    with open(path, "r", newline="", encoding="utf-8") as f:
        first = next(csv.reader(f), None)
    if first is None or len(first) != len(header):
        raise ValueError(
            f"CSV header mismatch for {path}. Expected {len(header)} columns, got {len(first) if first else 0}."
        )


def append_sequence(path: Path, label: str, seq: np.ndarray, feature_mode: str) -> None:
    row = [label, seq.shape[0]] + flatten_sequence(seq, feature_mode=feature_mode).tolist()
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def main() -> None:
    args = parse_args()
    feature_mode = FEATURE_MODE_HANDS_POSE if args.with_arms else args.feature_mode
    labels = load_label_set(args.label_file)
    if args.label not in labels:
        raise SystemExit(f"Label '{args.label}' is not in label set: {labels}")

    output_path = args.output
    try:
        ensure_csv_header(output_path, args.window_size, feature_mode=feature_mode)
    except ValueError as exc:
        alt_path = output_path.with_name(f"{output_path.stem}_{feature_mode}{output_path.suffix}")
        print(
            f"[WARN] Output schema mismatch for {output_path} ({exc}). "
            f"Switching to mode-specific file: {alt_path}"
        )
        output_path = alt_path
        ensure_csv_header(output_path, args.window_size, feature_mode=feature_mode)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check permissions/index.")

    config = HandExtractorConfig(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        use_pose=(feature_mode == FEATURE_MODE_HANDS_POSE),
        min_pose_detection_confidence=args.min_pose_detection_confidence,
        min_pose_tracking_confidence=args.min_pose_tracking_confidence,
        pose_visibility_threshold=args.pose_visibility_threshold,
    )

    collected = 0
    is_armed = False
    is_recording = False
    countdown_start_ts = 0.0
    frame_buffer: List[np.ndarray] = []

    with HandLandmarkExtractor(config) as extractor:
        while cap.isOpened() and collected < args.samples:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            raw_row, result = extractor.extract_raw_features(frame, feature_mode=feature_mode)
            extractor.draw_landmarks(frame, result, draw_pose=(feature_mode == FEATURE_MODE_HANDS_POSE))
            pose_debug = extractor.get_pose_debug_info(result)

            now = time.time()
            countdown_remaining = 0.0

            if is_armed and not is_recording:
                elapsed = now - countdown_start_ts
                countdown_remaining = max(0.0, args.countdown - elapsed)
                if countdown_remaining <= 1e-6:
                    is_recording = True
                    frame_buffer.clear()

            if is_recording:
                if raw_row is None:
                    frame_buffer.clear()
                else:
                    frame_buffer.append(raw_row)

                if len(frame_buffer) >= args.window_size:
                    seq = np.asarray(frame_buffer[: args.window_size], dtype=np.float32)
                    append_sequence(output_path, args.label, seq, feature_mode=feature_mode)
                    collected += 1
                    frame_buffer.clear()
                    time.sleep(args.cooldown)

            state = "IDLE"
            if is_armed and not is_recording:
                state = "COUNTDOWN"
            elif is_recording:
                state = "RECORDING"

            detected = "hands detected" if raw_row is not None else "no hands"
            cv2.putText(frame, f"Label: {args.label} | {collected}/{args.samples}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"State: {state} | {detected}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(
                frame,
                f"Features: {feature_mode} ({'arms ON' if feature_mode == FEATURE_MODE_HANDS_POSE else 'arms OFF'})",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.66,
                (180, 230, 255) if feature_mode == FEATURE_MODE_HANDS_POSE else (170, 170, 170),
                2,
            )

            if state == "COUNTDOWN":
                cv2.putText(frame, f"Recording starts in: {countdown_remaining:.1f}s", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 220, 255), 3)

            if state == "RECORDING":
                cv2.putText(frame, f"Auto recording... frames: {len(frame_buffer)}/{args.window_size}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 80, 255), 2)

            if feature_mode == FEATURE_MODE_HANDS_POSE:
                present = bool(pose_debug.get("present", False))
                missing = pose_debug.get("missing_points", [])
                in_frame_ratio = float(pose_debug.get("in_frame_ratio", 0.0))
                pose_text = f"Pose: {'OK' if present else 'MISSING'} | in-frame: {in_frame_ratio:.2f}"
                cv2.putText(frame, pose_text, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 120), 2)
                if missing:
                    missing_text = "Missing: " + ",".join(missing)
                    cv2.putText(frame, missing_text[:100], (20, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 120), 2)

                h, w = frame.shape[:2]
                x1, y1 = int(0.14 * w), int(0.08 * h)
                x2, y2 = int(0.86 * w), int(0.92 * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 0), 2)
                cv2.putText(
                    frame,
                    "Keep both shoulders/elbows/wrists inside box",
                    (x1 + 5, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 170, 0),
                    2,
                )

            cv2.putText(
                frame,
                "C: start session  S: stop session  Q: quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (220, 220, 220),
                2,
            )

            cv2.imshow("Collect HKSL Sequences", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("c") and not is_armed and not is_recording:
                is_armed = True
                countdown_start_ts = time.time()
            if key == ord("s"):
                is_armed = False
                is_recording = False
                frame_buffer.clear()

            if is_recording and collected >= args.samples:
                is_recording = False
                is_armed = False

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {collected} sequence samples for label '{args.label}' to {output_path} | feature_mode={feature_mode}")


if __name__ == "__main__":
    main()
