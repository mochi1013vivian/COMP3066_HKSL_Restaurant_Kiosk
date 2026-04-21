"""OpenCV UI rendering helpers for real-time inference."""

from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np


def _draw_text_lines(
    frame: np.ndarray,
    lines: Iterable[str],
    origin: tuple[int, int],
    line_height: int = 28,
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.7,
    thickness: int = 2,
) -> None:
    x, y = origin
    for idx, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (x, y + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_runtime_ui(
    frame: np.ndarray,
    *,
    predicted_token: Optional[str],
    predicted_confidence: float,
    stable_token: Optional[str],
    stable_confidence: float,
    hands_detected: bool,
    order_text: str,
    instructions: list[str],
    fps: Optional[float] = None,
) -> np.ndarray:
    """Draw status and order text overlays on the current frame."""
    h, w = frame.shape[:2]

    # Top status panel
    cv2.rectangle(frame, (0, 0), (w, 140), (20, 20, 20), thickness=-1)

    status_lines = [
        f"Hands: {'detected' if hands_detected else 'not detected'}",
        f"Prediction: {predicted_token or '-'} ({predicted_confidence:.2f})",
        f"Stable: {stable_token or '-'} ({stable_confidence:.2f})",
    ]
    if fps is not None:
        status_lines.append(f"FPS: {fps:.1f}")

    _draw_text_lines(
        frame,
        status_lines,
        origin=(14, 30),
        line_height=28,
        color=(230, 230, 230),
        scale=0.65,
    )

    # Bottom order panel
    panel_height = 140
    cv2.rectangle(frame, (0, h - panel_height), (w, h), (20, 20, 20), thickness=-1)

    order_preview = order_text if order_text else "(empty order)"
    cv2.putText(
        frame,
        "Order preview:",
        (14, h - panel_height + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (120, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        order_preview,
        (14, h - panel_height + 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Instructions at lower-right area
    _draw_text_lines(
        frame,
        instructions,
        origin=(w - 380, h - panel_height + 30),
        line_height=24,
        color=(200, 200, 200),
        scale=0.55,
        thickness=1,
    )

    return frame
