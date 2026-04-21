"""Webcam data collection for landmark-based sign recognition."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import cv2

try:
    from .extract_landmarks import HandExtractorConfig, HandLandmarkExtractor, TOTAL_FEATURES
    from .labels import load_label_set
except ImportError:  # pragma: no cover - supports `python src/collect_data.py`
    import os
    import sys

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)

    from extract_landmarks import HandExtractorConfig, HandLandmarkExtractor, TOTAL_FEATURES
    from labels import load_label_set


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = str(PROJECT_ROOT / "data" / "raw" / "landmarks.csv")


def ensure_header(path: str) -> None:
    """Create CSV header if needed, and validate if file already exists."""
    expected_header = ["label"] + [f"f{i}" for i in range(TOTAL_FEATURES)]

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
            "Existing CSV header length mismatch. "
            f"Expected {len(expected_header)} columns, found {len(first_row)}."
        )


def append_sample(output_csv: str, label: str, raw_row) -> None:
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([label] + list(raw_row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect hand landmarks for one label.")
    parser.add_argument("--label", required=True, help="Target token label to capture.")
    parser.add_argument("--label-file", default=None, help="Optional text file of allowed labels.")
    parser.add_argument("--samples", type=int, default=80, help="Number of samples to capture.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--cooldown", type=float, default=0.15, help="Pause after each capture.")
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
    ensure_header(args.output)

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
    with HandLandmarkExtractor(config) as extractor:
        while cap.isOpened() and collected < args.samples:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            raw_row, result = extractor.extract_raw_features(frame)
            extractor.draw_landmarks(frame, result)

            status = "hands detected" if raw_row is not None else "no hands"
            cv2.putText(
                frame,
                f"Label: {args.label} | Collected: {collected}/{args.samples}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Status: {status}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Press C to capture | Q to quit",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 220, 220),
                2,
            )

            cv2.imshow("Collect HKSL Data", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c"):
                if raw_row is None:
                    continue
                append_sample(args.output, args.label, raw_row)
                collected += 1
                time.sleep(args.cooldown)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()