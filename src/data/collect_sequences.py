"""Sequence data collection for closed-domain HKSL restaurant signs.

Behavior requested:
- Press C once to arm recording session
- 3-second countdown
- Continuous automatic recording until target sample count is reached
"""

from __future__ import annotations

import argparse
import csv
import queue
import threading
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
    parser.add_argument(
        "--voice-control",
        action="store_true",
        help='Enable voice commands: say "start" to arm, "stop" to cancel, "quit" to exit.',
    )
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
    parser.add_argument(
        "--auto-segment-motion",
        action="store_true",
        help="Use motion-energy auto-segmentation to capture action windows instead of purely frame-count windows.",
    )
    parser.add_argument(
        "--motion-energy-threshold",
        type=float,
        default=0.010,
        help="Smoothed motion-energy threshold for active movement (normalized landmark units).",
    )
    parser.add_argument(
        "--motion-energy-alpha",
        type=float,
        default=0.60,
        help="EMA smoothing factor for motion energy in [0,1]. Higher reacts faster.",
    )
    parser.add_argument(
        "--motion-tail-frames",
        type=int,
        default=4,
        help="Extra quiet frames to keep after motion drops below threshold.",
    )
    parser.add_argument(
        "--min-motion-frames",
        type=int,
        default=8,
        help="Minimum active-motion frames required to accept a segmented sample.",
    )
    return parser.parse_args()


def _voice_listener(cmd_queue: "queue.Queue[str]", stop_event: threading.Event) -> None:
    """Background thread: push 'start', 'stop', or 'quit' into cmd_queue on voice command."""
    try:
        import speech_recognition as sr  # type: ignore
    except ImportError:
        print("[voice-control] SpeechRecognition not installed. Run: pip install SpeechRecognition pyaudio")
        return

    recognizer = sr.Recognizer()
    try:
        mic = sr.Microphone()
    except Exception as exc:
        print(f"[voice-control] No microphone found: {exc}")
        return

    print("[voice-control] Listening. Say 'start', 'stop', or 'quit'.")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

    while not stop_event.is_set():
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
            text = recognizer.recognize_google(audio).lower().strip()
            print(f"[voice-control] Heard: {text!r}")
            for cmd in ("start", "stop", "quit"):
                if cmd in text:
                    cmd_queue.put(cmd)
                    break
        except sr.WaitTimeoutError:
            pass
        except sr.UnknownValueError:
            pass
        except Exception as exc:
            print(f"[voice-control] Error: {exc}")


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


def _motion_energy(prev_row: np.ndarray | None, curr_row: np.ndarray | None) -> float:
    """Compute frame-to-frame motion energy over valid landmarks only."""
    if prev_row is None or curr_row is None:
        return 0.0

    prev = np.asarray(prev_row, dtype=np.float32).reshape(-1)
    curr = np.asarray(curr_row, dtype=np.float32).reshape(-1)
    if prev.shape != curr.shape or prev.size == 0 or prev.size % 3 != 0:
        return 0.0

    prev_pts = prev.reshape(-1, 3)
    curr_pts = curr.reshape(-1, 3)

    prev_valid = np.linalg.norm(prev_pts, axis=1) > 1e-8
    curr_valid = np.linalg.norm(curr_pts, axis=1) > 1e-8
    valid = prev_valid & curr_valid
    if not np.any(valid):
        return 0.0

    delta_xy = curr_pts[valid, :2] - prev_pts[valid, :2]
    return float(np.mean(np.linalg.norm(delta_xy, axis=1)))


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

    # Voice control setup
    voice_queue: "queue.Queue[str]" = queue.Queue()
    voice_stop = threading.Event()
    if args.voice_control:
        vt = threading.Thread(target=_voice_listener, args=(voice_queue, voice_stop), daemon=True)
        vt.start()

    collected = 0
    is_armed = False
    is_recording = False
    countdown_start_ts = 0.0
    frame_buffer: List[np.ndarray] = []
    prev_raw_row: np.ndarray | None = None
    motion_energy_ema = 0.0

    # Auto-segmentation state
    segment_open = False
    segment_buffer: List[np.ndarray] = []
    quiet_tail_count = 0
    motion_frame_hits = 0

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
                    segment_open = False
                    segment_buffer.clear()
                    quiet_tail_count = 0
                    motion_frame_hits = 0
                    prev_raw_row = None
                    motion_energy_ema = 0.0

            if is_recording:
                if not args.auto_segment_motion:
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
                else:
                    alpha = float(np.clip(args.motion_energy_alpha, 0.0, 1.0))
                    energy_now = _motion_energy(prev_raw_row, raw_row if raw_row is not None else None)
                    motion_energy_ema = alpha * energy_now + (1.0 - alpha) * motion_energy_ema
                    is_motion_active = motion_energy_ema >= args.motion_energy_threshold

                    if raw_row is not None:
                        if is_motion_active:
                            if not segment_open:
                                segment_open = True
                                segment_buffer = []
                                quiet_tail_count = 0
                                motion_frame_hits = 0
                            segment_buffer.append(raw_row)
                            motion_frame_hits += 1
                            quiet_tail_count = 0
                        elif segment_open:
                            # Keep a small tail after motion drops so action endings are not clipped.
                            if quiet_tail_count < max(0, args.motion_tail_frames):
                                segment_buffer.append(raw_row)
                                quiet_tail_count += 1
                            else:
                                segment_open = False
                                segment_buffer.clear()
                                quiet_tail_count = 0
                                motion_frame_hits = 0

                        if segment_open and len(segment_buffer) >= args.window_size:
                            if motion_frame_hits >= max(1, args.min_motion_frames):
                                seq = np.asarray(segment_buffer[: args.window_size], dtype=np.float32)
                                append_sequence(output_path, args.label, seq, feature_mode=feature_mode)
                                collected += 1
                                time.sleep(args.cooldown)

                            segment_open = False
                            segment_buffer.clear()
                            quiet_tail_count = 0
                            motion_frame_hits = 0

                    prev_raw_row = raw_row.copy() if raw_row is not None else None

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
                if args.auto_segment_motion:
                    cv2.putText(
                        frame,
                        f"Motion auto-seg: buf {len(segment_buffer)}/{args.window_size} | active {motion_frame_hits} | E {motion_energy_ema:.4f}/{args.motion_energy_threshold:.4f}",
                        (20, 145),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.66,
                        (0, 180, 255),
                        2,
                    )
                else:
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

            controls_hint = "C/[start]: arm  S/[stop]: cancel  Q/[quit]: exit" if args.voice_control else "C: start session  S: stop session  Q: quit"
            cv2.putText(
                frame,
                controls_hint,
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (220, 220, 220),
                2,
            )

            cv2.imshow("Collect HKSL Sequences", frame)
            key = cv2.waitKey(1) & 0xFF

            # Drain voice commands (non-blocking)
            voice_cmd: str | None = None
            try:
                voice_cmd = voice_queue.get_nowait()
            except queue.Empty:
                pass

            if key == ord("q") or voice_cmd == "quit":
                break
            if (key == ord("c") or voice_cmd == "start") and not is_armed and not is_recording:
                is_armed = True
                countdown_start_ts = time.time()
            if key == ord("s") or voice_cmd == "stop":
                is_armed = False
                is_recording = False
                frame_buffer.clear()
                segment_open = False
                segment_buffer.clear()
                quiet_tail_count = 0
                motion_frame_hits = 0
                prev_raw_row = None
                motion_energy_ema = 0.0

            if is_recording and collected >= args.samples:
                is_recording = False
                is_armed = False

    voice_stop.set()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {collected} sequence samples for label '{args.label}' to {output_path} | feature_mode={feature_mode}")


if __name__ == "__main__":
    main()
