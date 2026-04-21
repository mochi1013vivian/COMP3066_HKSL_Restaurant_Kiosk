"""Real-time webcam inference for sign-based food ordering."""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import joblib
import numpy as np

try:
    from .extract_landmarks import HandExtractorConfig, HandLandmarkExtractor
    from .extract_sequences import sequence_to_feature_vector
    from .modeling import DEFAULT_LABEL_PATH, DEFAULT_MODEL_PATH
    from .sentence_builder import OrderSentenceBuilder
    from .ui_display import draw_runtime_ui
except ImportError:  # pragma: no cover - supports `python src/inference_realtime.py`
    import os
    import sys

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)

    from extract_landmarks import HandExtractorConfig, HandLandmarkExtractor
    from extract_sequences import sequence_to_feature_vector
    from modeling import DEFAULT_LABEL_PATH, DEFAULT_MODEL_PATH
    from sentence_builder import OrderSentenceBuilder
    from ui_display import draw_runtime_ui


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DYNAMIC_MODEL_PATH = str(PROJECT_ROOT / "models" / "sign_model_dynamic.joblib")
DEFAULT_DYNAMIC_LABEL_PATH = str(PROJECT_ROOT / "models" / "class_names_dynamic.joblib")


@dataclass
class PredictionSmoother:
    """Stabilize noisy frame-by-frame predictions using a short temporal window."""

    window_size: int = 7
    min_consensus: float = 0.6
    min_confidence: float = 0.55
    history: deque[Tuple[Optional[str], float]] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.history = deque(maxlen=self.window_size)

    def update(
        self, token: Optional[str], confidence: float
    ) -> Tuple[Optional[str], float, float]:
        self.history.append((token, confidence))

        valid = [(t, c) for t, c in self.history if t is not None]
        if not valid:
            return None, 0.0, 0.0

        counts = Counter(t for t, _ in valid)
        stable_token, stable_count = counts.most_common(1)[0]
        consensus = stable_count / len(valid)

        stable_conf = float(np.mean([c for t, c in valid if t == stable_token]))
        return stable_token, stable_conf, consensus

    def is_reliable(self, token: Optional[str], confidence: float, consensus: float) -> bool:
        if token is None:
            return False
        if confidence < self.min_confidence:
            return False
        if consensus < self.min_consensus:
            return False
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time sign recognition app.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--labels", default=DEFAULT_LABEL_PATH)
    parser.add_argument("--model-dynamic", default=DEFAULT_DYNAMIC_MODEL_PATH)
    parser.add_argument("--labels-dynamic", default=DEFAULT_DYNAMIC_LABEL_PATH)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--dynamic-window-size", type=int, default=15)
    parser.add_argument("--min-consensus", type=float, default=0.6)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--min-dynamic-confidence", type=float, default=0.7)
    parser.add_argument("--add-cooldown", type=float, default=0.8)
    parser.add_argument("--auto-add", action="store_true", help="Auto append stable tokens.")
    parser.add_argument("--use-dynamic", action="store_true", default=True)
    parser.add_argument("--disable-static-fallback", action="store_true")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    return parser.parse_args()


def _get_model_classes(model) -> list[str]:
    if hasattr(model, "classes_"):
        return [str(x) for x in model.classes_]
    clf = getattr(model, "named_steps", {}).get("clf")
    if clf is not None and hasattr(clf, "classes_"):
        return [str(x) for x in clf.classes_]
    return []


def predict_top1(model, raw_row) -> Tuple[str, float]:
    """Predict one token from a raw 126-dim landmark row."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([raw_row])[0]
        classes = _get_model_classes(model)
        if not classes:
            pred = str(model.predict([raw_row])[0])
            return pred, 1.0

        best_idx = int(np.argmax(probs))
        return classes[best_idx], float(probs[best_idx])

    pred = str(model.predict([raw_row])[0])
    return pred, 1.0


def predict_dynamic_top1(model_dynamic, frame_buffer, *, window_size: int) -> Tuple[str, float]:
    feature_vector = sequence_to_feature_vector(
        frame_buffer, window_size=window_size, include_deltas=True
    )
    if hasattr(model_dynamic, "predict_proba"):
        probs = model_dynamic.predict_proba([feature_vector])[0]
        classes = list(getattr(model_dynamic, "classes_", []))
        if not classes:
            clf = getattr(model_dynamic, "named_steps", {}).get("clf")
            classes = list(getattr(clf, "classes_", [])) if clf is not None else []
        if classes:
            best_idx = int(np.argmax(probs))
            return str(classes[best_idx]), float(probs[best_idx])

    pred = str(model_dynamic.predict([feature_vector])[0])
    return pred, 1.0


def main() -> None:
    args = parse_args()

    model = joblib.load(args.model)
    model_dynamic = None
    if args.use_dynamic and Path(args.model_dynamic).exists():
        model_dynamic = joblib.load(args.model_dynamic)
        print(f"Loaded dynamic model: {args.model_dynamic}")
    elif args.use_dynamic:
        print(f"Dynamic model not found at {args.model_dynamic}; running static-only fallback.")

    # Optional labels file is loaded for compatibility with existing project artifacts.
    try:
        label_names = joblib.load(args.labels)
        print(f"Loaded label names: {label_names}")
    except Exception:
        label_names = _get_model_classes(model)
        print(f"Using model classes: {label_names}")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions/device index.")

    config = HandExtractorConfig(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    sentence_builder = OrderSentenceBuilder()
    smoother = PredictionSmoother(
        window_size=args.window_size,
        min_consensus=args.min_consensus,
        min_confidence=args.min_confidence,
    )

    auto_add_enabled = args.auto_add
    last_add_ts = 0.0
    last_frame_ts = time.time()
    dynamic_buffer: deque[np.ndarray] = deque(maxlen=args.dynamic_window_size)

    with HandLandmarkExtractor(config) as extractor:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            raw_row, result = extractor.extract_raw_features(frame)
            extractor.draw_landmarks(frame, result)

            predicted_token: Optional[str] = None
            predicted_confidence = 0.0
            dynamic_token: Optional[str] = None
            dynamic_confidence = 0.0

            if raw_row is not None:
                predicted_token, predicted_confidence = predict_top1(model, raw_row)
                dynamic_buffer.append(raw_row)
            else:
                dynamic_buffer.clear()

            if model_dynamic is not None and len(dynamic_buffer) == args.dynamic_window_size:
                dynamic_token, dynamic_confidence = predict_dynamic_top1(
                    model_dynamic,
                    list(dynamic_buffer),
                    window_size=args.dynamic_window_size,
                )

            stable_token, stable_confidence, consensus = smoother.update(
                predicted_token, predicted_confidence
            )

            chosen_token: Optional[str] = None
            chosen_confidence = 0.0
            chosen_source = "none"

            if (
                model_dynamic is not None
                and dynamic_token is not None
                and dynamic_confidence >= args.min_dynamic_confidence
            ):
                chosen_token = dynamic_token
                chosen_confidence = dynamic_confidence
                chosen_source = "dynamic"
            elif not args.disable_static_fallback and smoother.is_reliable(
                stable_token, stable_confidence, consensus
            ):
                chosen_token = stable_token
                chosen_confidence = stable_confidence
                chosen_source = "static"

            is_reliable = chosen_token is not None

            now = time.time()
            can_add_now = (now - last_add_ts) >= args.add_cooldown
            if auto_add_enabled and is_reliable and can_add_now:
                sentence_builder.add_token(chosen_token or "")
                last_add_ts = now

            order_text = sentence_builder.build_staff_text()

            fps = 1.0 / max(1e-6, now - last_frame_ts)
            last_frame_ts = now

            instructions = [
                f"Reliability: {'OK' if is_reliable else 'low'} | consensus={consensus:.2f}",
                f"Dynamic: {dynamic_token or '-'} ({dynamic_confidence:.2f}) | source={chosen_source}",
                "A/Space: add stable token",
                "U: undo  |  C: clear",
                "T: toggle auto-add",
                "Q or ESC: quit",
            ]

            draw_runtime_ui(
                frame,
                predicted_token=predicted_token,
                predicted_confidence=predicted_confidence,
                stable_token=chosen_token,
                stable_confidence=chosen_confidence,
                hands_detected=(raw_row is not None),
                order_text=order_text,
                instructions=instructions,
                fps=fps,
            )

            cv2.imshow("HKSL Food Ordering Assistant", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):  # q or ESC
                break
            elif key in (ord("a"), ord(" ")):
                if is_reliable and can_add_now:
                    sentence_builder.add_token(chosen_token or "")
                    last_add_ts = time.time()
            elif key == ord("u"):
                sentence_builder.undo()
            elif key == ord("c"):
                sentence_builder.clear()
            elif key == ord("t"):
                auto_add_enabled = not auto_add_enabled
                print(f"Auto-add: {'ON' if auto_add_enabled else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
