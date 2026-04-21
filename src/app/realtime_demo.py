"""Realtime PyTorch GRU inference app for HKSL restaurant ordering & staff communication.

Supports deaf customer food ordering (primary) and staff communication with coworkers (secondary).
Low-latency presentation mode available for real-time demonstrations.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from dataclasses import fields
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

try:
    from ..config.settings import DEFAULT_MODEL_PATH
    from ..config.labels import token_to_emoji
    from ..features.mediapipe_extractor import (
        FEATURE_MODE_HANDS,
        FEATURE_MODE_HANDS_POSE,
        FEATURES_PER_HAND,
        HandExtractorConfig,
        HandLandmarkExtractor,
        TOTAL_FEATURES,
    )
    from ..features.sequence_preprocess import normalize_sequence
    from ..models.gru_classifier import GRUClassifier
    from ..utils.audio_feedback import play_accept_sound, play_confirmation_sound, speak_text
    from .sentence_builder import OrderSentenceBuilder
    from .ui import compute_demo_layout, draw_demo_ui
except ImportError:  # pragma: no cover
    import os
    import sys

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_ROOT = os.path.dirname(CURRENT_DIR)
    APP_DIR = CURRENT_DIR
    if SRC_ROOT not in sys.path:
        sys.path.insert(0, SRC_ROOT)
    if APP_DIR not in sys.path:
        sys.path.insert(0, APP_DIR)

    from config.settings import DEFAULT_MODEL_PATH
    from config.labels import token_to_emoji
    from features.mediapipe_extractor import (
        FEATURE_MODE_HANDS,
        FEATURE_MODE_HANDS_POSE,
        FEATURES_PER_HAND,
        HandExtractorConfig,
        HandLandmarkExtractor,
        TOTAL_FEATURES,
    )
    from features.sequence_preprocess import normalize_sequence
    from models.gru_classifier import GRUClassifier
    from utils.audio_feedback import play_accept_sound, play_confirmation_sound, speak_text
    from sentence_builder import OrderSentenceBuilder
    from ui import compute_demo_layout, draw_demo_ui


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime HKSL GRU demo for customer ordering & staff communication."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--accept-confidence", type=float, default=0.78)
    parser.add_argument("--stable-frames", type=int, default=6, help="Frames needed before word acceptance (6-8 normal, 8 low-latency)")
    parser.add_argument("--accept-cooldown", type=float, default=1.0, help="Cooldown after accepting (1.0 normal, 0.4 low-latency)")
    parser.add_argument("--repeat-block-seconds", type=float, default=2.0, help="Seconds before same word can repeat (2.0-2.5 normal)")
    parser.add_argument("--no-sign-frames", type=int, default=11, help="No-sign frames before word can repeat (10-12 frames)")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--min-pose-detection-confidence", type=float, default=0.35)
    parser.add_argument("--min-pose-tracking-confidence", type=float, default=0.35)
    parser.add_argument("--pose-visibility-threshold", type=float, default=0.35)
    parser.add_argument("--sound", action="store_true", help="Enable confirmation chimes on word acceptance")
    parser.add_argument("--presentation-mode", action="store_true", help="Low-latency mode: 640x480, max_1_hand, model_lite, skip frames")
    parser.add_argument("--skip-frames", type=int, default=0, help="Run inference every Nth frame (0=every frame)")
    parser.add_argument("--camera-width", type=int, default=0, help="Camera width (0=auto, 1280 normal, 640 presentation)")
    parser.add_argument("--camera-height", type=int, default=0, help="Camera height (0=auto, 720 normal, 480 presentation)")
    parser.add_argument("--tts", action="store_true", help="Enable text-to-speech on Enter key")
    parser.add_argument("--debug-ui", action="store_true", help="Enable instrumented UI/debug logs and hardcoded panel markers")
    parser.add_argument(
        "--show-pose-debug",
        action="store_true",
        help="Draw upper-body pose keypoints/links for debugging even when UI debug is off.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["auto", FEATURE_MODE_HANDS, FEATURE_MODE_HANDS_POSE],
        default="auto",
        help="Feature mode for realtime preprocessing. 'auto' follows checkpoint metadata (fallback hands_pose).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Apply presentation mode defaults if enabled
    if args.presentation_mode:
        # Override for low-latency presentation
        args.camera_width = args.camera_width or 960
        args.camera_height = args.camera_height or 540
        args.stable_frames = 6
        args.accept_cooldown = 0.35
        args.repeat_block_seconds = 1.8
        args.no_sign_frames = 11
        args.skip_frames = args.skip_frames or 1  # Predict every other frame, display stays smooth
        args.min_detection_confidence = 0.6
        args.accept_confidence = 0.8
    else:
        # Normal mode defaults
        args.camera_width = args.camera_width or 1280
        args.camera_height = args.camera_height or 720

    ckpt = torch.load(args.model, map_location="cpu")
    feature_mode = ckpt.get("feature_mode", FEATURE_MODE_HANDS_POSE) if args.feature_mode == "auto" else args.feature_mode
    classes = list(ckpt["classes"])
    print(f"[INFO] Realtime feature_mode={feature_mode} | pose_enabled={feature_mode == FEATURE_MODE_HANDS_POSE or args.show_pose_debug}")
    model = GRUClassifier(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        num_classes=len(classes),
        dropout=ckpt.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    window_size = int(ckpt["window_size"])

    def read_latest_frame(cap: cv2.VideoCapture, drain_count: int = 2) -> tuple[bool, Optional[np.ndarray]]:
        ok = False
        for _ in range(max(1, drain_count)):
            ok = cap.grab()
            if not ok:
                return False, None
        ok, frame = cap.retrieve()
        return ok, frame

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions/index.")
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    
    # LOW-LATENCY: Minimize internal buffer to discard old frames and use latest only
    # This prevents lag from queued frames in the camera driver
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Keep only 1 frame in buffer (latest only)

    # Keep two-hand contract consistent across collection/training/realtime.
    max_hands = 2
    cfg_kwargs = {
        "static_image_mode": False,
        "max_num_hands": max_hands,
        "min_detection_confidence": args.min_detection_confidence,
        "min_tracking_confidence": args.min_tracking_confidence,
        "use_pose": (feature_mode == FEATURE_MODE_HANDS_POSE) or args.show_pose_debug,
        "min_pose_detection_confidence": args.min_pose_detection_confidence,
        "min_pose_tracking_confidence": args.min_pose_tracking_confidence,
        "pose_visibility_threshold": args.pose_visibility_threshold,
    }
    # Backward/forward compatibility: only pass fields supported by this project config class.
    supported_fields = {f.name for f in fields(HandExtractorConfig)}
    if "model_complexity" in supported_fields:
        cfg_kwargs["model_complexity"] = 0 if args.presentation_mode else 1

    extractor_cfg = HandExtractorConfig(**cfg_kwargs)

    frame_buffer: deque[np.ndarray] = deque(maxlen=window_size)
    pred_history: deque[str] = deque(maxlen=max(12, args.stable_frames + 4))
    
    # Track sentence building with duplicate blocking
    sentence = OrderSentenceBuilder()
    
    # Tracking variables for word-append logic
    last_accept_ts = 0.0
    last_accepted_word = ""
    no_sign_frame_count = 0
    frame_count = 0
    top_predictions: list[tuple[str, float]] = []

    # Restaurant-style confirmation flow
    order_confirmed = False
    frozen_order_text = ""
    confirmed_order_number = ""
    next_order_seq = 12
    
    last_frame_ts = time.time()

    # Persistent display state (prevents panel from going blank between inference windows)
    display_live_token: Optional[str] = None
    display_live_conf: float = 0.0
    display_confirmed_token: Optional[str] = None
    display_confirmed_count: int = 0

    def reset_live_order_state() -> None:
        nonlocal sentence, frame_buffer, pred_history
        nonlocal last_accept_ts, last_accepted_word, no_sign_frame_count, top_predictions
        nonlocal display_live_token, display_live_conf, display_confirmed_token, display_confirmed_count
        sentence.clear()
        frame_buffer.clear()
        pred_history.clear()
        last_accept_ts = 0.0
        last_accepted_word = ""
        no_sign_frame_count = 0
        top_predictions = []
        display_live_token = None
        display_live_conf = 0.0
        display_confirmed_token = None
        display_confirmed_count = 0

    def confirm_current_order() -> bool:
        nonlocal order_confirmed, frozen_order_text, confirmed_order_number, next_order_seq
        current_text = sentence.build_text()
        if order_confirmed or current_text == "(empty)":
            return False
        frozen_order_text = current_text
        confirmed_order_number = f"A{next_order_seq:02d}"
        next_order_seq += 1
        order_confirmed = True
        play_confirmation_sound(enabled=args.sound)
        if args.tts:
            speak_text(f"Order confirmed. Your order number is {confirmed_order_number}.")
        return True

    def start_new_order() -> None:
        nonlocal order_confirmed, frozen_order_text, confirmed_order_number
        order_confirmed = False
        frozen_order_text = ""
        confirmed_order_number = ""
        reset_live_order_state()

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        state = userdata if isinstance(userdata, dict) else {}
        layout = state.get("layout")
        if layout is None:
            return
        confirm_enabled = bool(state.get("confirm_enabled", False))
        order_confirmed_flag = bool(state.get("order_confirmed", False))

        def in_rect(rect: tuple[int, int, int, int]) -> bool:
            rx, ry, rw, rh = rect
            return rx <= x <= rx + rw and ry <= y <= ry + rh

        if confirm_enabled and not order_confirmed_flag and in_rect(layout.confirm_button):
            state["confirm_clicked"] = True
        elif order_confirmed_flag and in_rect(layout.new_order_button):
            state["start_new_clicked"] = True

    def consume_ui_actions() -> bool:
        """Apply pending UI actions immediately; return True if state changed."""
        changed = False
        if mouse_state.get("confirm_clicked"):
            mouse_state["confirm_clicked"] = False
            changed = confirm_current_order() or changed
        if mouse_state.get("start_new_clicked"):
            mouse_state["start_new_clicked"] = False
            start_new_order()
            changed = True
        return changed

    window_name = "HKSL Realtime: Deaf Customer Ordering + Staff Communication" if args.presentation_mode else "HKSL Restaurant Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Fixed 16:9 composition (PPT-like) for presentation clarity.
    ui_w, ui_h = ((960, 540) if args.presentation_mode else (1280, 720))
    mouse_state = {
        "layout": None,
        "confirm_enabled": False,
        "order_confirmed": False,
        "confirm_clicked": False,
        "start_new_clicked": False,
    }
    cv2.setMouseCallback(window_name, on_mouse, mouse_state)

    with HandLandmarkExtractor(extractor_cfg) as extractor:
        first_debug_print_done = False
        while cap.isOpened():
            # Handle pending UI clicks before doing heavy processing for snappier button response.
            consume_ui_actions()

            ok, frame = read_latest_frame(cap, drain_count=3 if args.presentation_mode else 1)
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (ui_w, ui_h), interpolation=cv2.INTER_LINEAR)
            raw_row, result = extractor.extract_raw_features(frame, feature_mode=feature_mode)
            extractor.draw_landmarks(
                frame,
                result,
                draw_pose=args.show_pose_debug or (args.debug_ui and feature_mode == FEATURE_MODE_HANDS_POSE),
            )
            pose_debug = extractor.get_pose_debug_info(result)
            if args.show_pose_debug:
                h, w = frame.shape[:2]
                x1, y1 = int(0.14 * w), int(0.08 * h)
                x2, y2 = int(0.86 * w), int(0.92 * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 0), 2)
                cv2.putText(
                    frame,
                    "Framing: keep shoulders/elbows/wrists in box",
                    (x1 + 5, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 170, 0),
                    2,
                )

            # ---- Instrumented two-hand diagnostics
            multi_hands = getattr(result.hands_result, "multi_hand_landmarks", None) or []
            hand_count = len(multi_hands)
            handedness_labels = []
            handedness_list = getattr(result.hands_result, "multi_handedness", None) or []
            for hs in handedness_list:
                try:
                    handedness_labels.append(hs.classification[0].label)
                except Exception:
                    continue

            if raw_row is not None:
                left_nonzero = int(np.any(np.abs(raw_row[:FEATURES_PER_HAND]) > 1e-8))
                right_nonzero = int(np.any(np.abs(raw_row[FEATURES_PER_HAND:]) > 1e-8))
                feature_len = int(raw_row.shape[0])
            else:
                left_nonzero = 0
                right_nonzero = 0
                feature_len = 0

            live_token: Optional[str] = None
            live_conf = 0.0
            confirmed_token: Optional[str] = None
            confirmed_count = 0

            # Track no-sign frames for duplicate blocking
            if raw_row is None:
                no_sign_frame_count += 1
            else:
                no_sign_frame_count = 0

            if order_confirmed:
                # Freeze live-order updates while in confirmed state.
                frame_buffer.clear()
                pred_history.clear()
            elif raw_row is not None:
                frame_buffer.append(raw_row)
            else:
                frame_buffer.clear()
                pred_history.clear()

            # Run inference every Nth frame (default: every frame)
            should_infer = (args.skip_frames == 0) or (frame_count % (args.skip_frames + 1) == 0)
            
            if len(frame_buffer) == window_size and should_infer:
                seq = np.asarray(list(frame_buffer), dtype=np.float32)
                seq = normalize_sequence(seq, feature_mode=feature_mode)
                x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                best_idx = int(np.argmax(probs))
                live_token = classes[best_idx]
                live_conf = float(probs[best_idx])

                ranked_idx = np.argsort(probs)[::-1][:3]
                top_predictions = [(classes[int(i)], float(probs[int(i)])) for i in ranked_idx]
                pred_history.append(live_token)

                if pred_history:
                    confirmed_token, confirmed_count = Counter(pred_history).most_common(1)[0]

                # Update persistent display state when inference is available.
                display_live_token = live_token
                display_live_conf = live_conf
                display_confirmed_token = confirmed_token
                display_confirmed_count = confirmed_count

                # NEW LOGIC: Word is accepted only if:
                # 1. Confirmed for 6-8 stable frames (confirmed_count >= stable_frames)
                # 2. Confidence is high enough (live_conf >= accept_confidence)
                # 3. NOT a consecutive duplicate (don't append same word twice in a row)
                # 4. Either: no_sign_frames >= threshold, OR time_since_last >= repeat_block_seconds
                
                now = time.time()
                is_stable = confirmed_count >= args.stable_frames and live_conf >= args.accept_confidence
                not_consecutive_dup = (confirmed_token != sentence.last_token) if sentence.last_token else True
                time_ok = (now - last_accept_ts) >= args.accept_cooldown
                repeat_blocked = (confirmed_token == last_accepted_word) and not (
                    (no_sign_frame_count >= args.no_sign_frames) or 
                    ((now - last_accept_ts) >= args.repeat_block_seconds)
                )
                
                if (is_stable and not_consecutive_dup and time_ok and not repeat_blocked and not order_confirmed):
                    sentence.add_token(confirmed_token)
                    last_accept_ts = now
                    last_accepted_word = confirmed_token
                    play_confirmation_sound(enabled=args.sound)

            current_emoji = token_to_emoji(display_live_token or display_confirmed_token or "")

            now = time.time()
            fps = 1.0 / max(1e-6, now - last_frame_ts)
            last_frame_ts = now

            mode_label = "PRESENTATION (low-latency)" if args.presentation_mode else "NORMAL"
            status_lines = [
                f"Mode: {mode_label}  |  Controls: X=reset | C=confirm | N=new order | Enter=speak | Q/ESC=quit",
                f"No-sign frames: {no_sign_frame_count}/{args.no_sign_frames}  |  Cooldown: {max(0, args.accept_cooldown - (now - last_accept_ts)):.1f}s",
                f"Features: {feature_mode}",
            ]
            if (feature_mode == FEATURE_MODE_HANDS_POSE or args.show_pose_debug) and args.debug_ui:
                present = bool(pose_debug.get("present", False))
                missing = pose_debug.get("missing_points", [])
                in_frame_ratio = float(pose_debug.get("in_frame_ratio", 0.0))
                status_lines.append(
                    f"Pose: {'OK' if present else 'MISSING'} | in-frame={in_frame_ratio:.2f} | missing={','.join(missing) if missing else 'none'}"
                )

            # IMPORTANT: use returned composed frame for display (camera + right panel)
            frame = draw_demo_ui(
                frame,
                live_token=display_live_token,
                live_conf=display_live_conf,
                current_emoji=current_emoji,
                confirmed_token=display_confirmed_token,
                confirmed_count=display_confirmed_count,
                sentence_text=frozen_order_text if order_confirmed else sentence.build_text(),
                sentence_tokens=sentence.tokens,
                top_predictions=top_predictions,
                status_lines=status_lines,
                fps=fps,
                order_confirmed=order_confirmed,
                order_number=confirmed_order_number or f"A{next_order_seq:02d}",
                kitchen_status="Kitchen status: sent to kitchen" if order_confirmed else "Kitchen status: awaiting confirmation",
                confirm_enabled=bool(sentence.tokens) and not order_confirmed,
                debug_ui=args.debug_ui,
            )

            mouse_state["layout"] = compute_demo_layout(frame.shape)
            mouse_state["confirm_enabled"] = bool(sentence.tokens) and not order_confirmed
            mouse_state["order_confirmed"] = order_confirmed

            # Consume clicks again right after drawing in case user clicked this frame.
            if consume_ui_actions():
                frame = draw_demo_ui(
                    frame,
                    live_token=display_live_token,
                    live_conf=display_live_conf,
                    current_emoji=current_emoji,
                    confirmed_token=display_confirmed_token,
                    confirmed_count=display_confirmed_count,
                    sentence_text=frozen_order_text if order_confirmed else sentence.build_text(),
                    sentence_tokens=sentence.tokens,
                    top_predictions=top_predictions,
                    status_lines=status_lines,
                    fps=fps,
                    order_confirmed=order_confirmed,
                    order_number=confirmed_order_number or f"A{next_order_seq:02d}",
                    kitchen_status="Kitchen status: sent to kitchen" if order_confirmed else "Kitchen status: awaiting confirmation",
                    confirm_enabled=bool(sentence.tokens) and not order_confirmed,
                    debug_ui=args.debug_ui,
                )

            if args.debug_ui:
                # Print frame/panel coordinates and pipeline values periodically for proof.
                if not first_debug_print_done:
                    h, w = frame.shape[:2]
                    panel_x = int(w * 0.75)
                    print(f"[DEBUG] imshow frame var=frame, shape={frame.shape}, panel_x={panel_x}, panel_w={w-panel_x}")
                    print(
                        f"[DEBUG] detector max_num_hands={max_hands}, TOTAL_FEATURES={TOTAL_FEATURES}, "
                        f"FEATURES_PER_HAND={FEATURES_PER_HAND}, feature_mode={feature_mode}"
                    )
                    first_debug_print_done = True
                if frame_count % 10 == 0:
                    pose_vis = pose_debug.get("visibilities", {})
                    print(
                        "[DEBUG] "
                        f"hands={hand_count} labels={handedness_labels} "
                        f"feat_len={feature_len} slots(L,R)=({left_nonzero},{right_nonzero}) "
                        f"pose_present={pose_debug.get('present', False)} missing={pose_debug.get('missing_points', [])} "
                        f"pose_vis={pose_vis} "
                        f"live={display_live_token} confirmed={display_confirmed_token} "
                        f"sentence='{sentence.build_text()}' top2={top_predictions[:2]}"
                    )

            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # Q or ESC
                break
            if key == ord("x"):  # X: clear sentence
                start_new_order()
            if key in (ord("z"), 8, 127):  # Z or Backspace: undo last word
                if not order_confirmed:
                    sentence.undo()
            if key == ord("c"):
                confirm_current_order()
            if key == ord("n"):
                start_new_order()
            if key == ord("\r"):  # Enter: speak sentence if TTS enabled
                if args.tts:
                    text = frozen_order_text if order_confirmed else sentence.build_text()
                    if text != "(empty)":
                        speak_text(text, voice="Victoria")
            
            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
