"""Collect dynamic sign samples as short landmark sequences."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

try:
    from .extract_landmarks import HandExtractorConfig, HandLandmarkExtractor, TOTAL_FEATURES
    from .extract_sequences import sequence_feature_dim, sequence_to_feature_vector
    from .labels import load_label_set
except ImportError:  # pragma: no cover
    import sys

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)

    from extract_landmarks import HandExtractorConfig, HandLandmarkExtractor, TOTAL_FEATURES
    from extract_sequences import sequence_feature_dim, sequence_to_feature_vector
    from labels import load_label_set


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = str(PROJECT_ROOT / "data" / "raw" / "landmarks_dynamic.csv")


def ensure_header(path: str, *, window_size: int, include_deltas: bool) -> None:
    dim = sequence_feature_dim(window_size, include_deltas)
    expected_header = ["label", "sequence_len"] + [f"f{i}" for i in range(dim)]

    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(expected_header)
        return

    with open(path, "r", newline="", encoding="utf-8") as f:
        first_row = next(csv.reader(f), None)

    if first_row is None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(expected_header)
        return

    if len(first_row) != len(expected_header):
        raise ValueError(
            "Existing dynamic CSV header length mismatch. "
            f"Expected {len(expected_header)} columns, found {len(first_row)}."
        )


def append_sequence_sample(
    output_csv: str,
    *,
    label: str,
    sequence_rows: List[np.ndarray],
    window_size: int,
    include_deltas: bool,
) -> None:
    feature_vector = sequence_to_feature_vector(
        sequence_rows, window_size=window_size, include_deltas=include_deltas
    )

    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([label, len(sequence_rows)] + feature_vector.tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect dynamic landmark sequences for one label.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--label-file", default=None)
    parser.add_argument("--samples", type=int, default=40, help="Number of sequences to collect.")
    parser.add_argument("--window-size", type=int, default=15, help="Frames per sequence window.")
    parser.add_argument("--include-deltas", action="store_true", default=True)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--cooldown", type=float, default=0.25)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    available_labels = load_label_set(args.label_file)
    if args.label not in available_labels:
        joined = ", ".join(available_labels)
        raise SystemExit(
            f"Label '{args.label}' is not in allowed label set. Available labels: {joined}"
        )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ensure_header(args.output, window_size=args.window_size, include_deltas=args.include_deltas)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions/device index.")

    config = HandExtractorConfig(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    collected = 0
    is_recording = False
    frame_buffer: List[np.ndarray] = []

    with HandLandmarkExtractor(config) as extractor:
        while cap.isOpened() and collected < args.samples:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            raw_row, result = extractor.extract_raw_features(frame)
            extractor.draw_landmarks(frame, result)

            if is_recording and raw_row is not None:
                frame_buffer.append(raw_row)
                if len(frame_buffer) >= args.window_size:
                    append_sequence_sample(
                        args.output,
                        label=args.label,
                        sequence_rows=frame_buffer,
                        window_size=args.window_size,
                        include_deltas=args.include_deltas,
                    )
                    collected += 1
                    is_recording = False
                    frame_buffer.clear()
                    time.sleep(args.cooldown)

            status = "recording" if is_recording else "idle"
            detected = "hands detected" if raw_row is not None else "no hands"
            cv2.putText(
                frame,
                f"Label: {args.label} | Sequences: {collected}/{args.samples}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"State: {status} | {detected}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Buffered frames: {len(frame_buffer)}/{args.window_size}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 220),
                2,
            )
            cv2.putText(
                frame,
                "Press C to start sequence capture | Q to quit",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (200, 200, 200),
                2,
            )

            cv2.imshow("Collect Dynamic HKSL Data", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("c") and not is_recording:
                is_recording = True
                frame_buffer.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
