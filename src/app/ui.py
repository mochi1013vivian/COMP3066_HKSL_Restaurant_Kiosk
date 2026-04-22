"""Realtime UI renderer for classroom presentation.

Layout:
- Left ~72%: large live camera area (clean with only light overlays)
- Right ~28%: warm, card-based kiosk / order panel
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None


def _rounded_card(img: np.ndarray, x: int, y: int, w: int, h: int, color: Tuple[int, int, int], r: int = 16) -> None:
    """Draw a rounded rectangle card (filled)."""
    x2, y2 = x + w, y + h
    cv2.rectangle(img, (x + r, y), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x, y + r), (x2, y2 - r), color, -1)
    cv2.circle(img, (x + r, y + r), r, color, -1)
    cv2.circle(img, (x2 - r, y + r), r, color, -1)
    cv2.circle(img, (x + r, y2 - r), r, color, -1)
    cv2.circle(img, (x2 - r, y2 - r), r, color, -1)


def _draw_card_title(canvas: np.ndarray, title: str, x: int, y: int) -> None:
    cv2.putText(canvas, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (72, 84, 84), 2, cv2.LINE_AA)


def _draw_unicode_text(canvas: np.ndarray, text: str, x: int, y: int, *, font_size: int, color: Tuple[int, int, int], anchor: str = "mm") -> None:
    """Draw Unicode text (including emoji) using Pillow when available."""
    if not text:
        return

    if Image is None or ImageDraw is None or ImageFont is None:
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        return

    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    font_paths = [
        "/System/Library/Fonts/Apple Color Emoji.ttc",
        "/System/Library/Fonts/Apple Symbols.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    font = None
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    draw.text((x, y), text, font=font, fill=(int(color[2]), int(color[1]), int(color[0])), anchor=anchor)
    canvas[:, :] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


@dataclass(frozen=True)
class DemoUiLayout:
    """Shared geometry for draw + mouse hit-testing."""

    h: int
    w: int
    cam_w: int
    panel_x: int
    panel_w: int
    pad: int
    header: Tuple[int, int, int, int]
    word_card: Tuple[int, int, int, int]
    sentence_card: Tuple[int, int, int, int]
    order_card: Tuple[int, int, int, int]
    controls_card: Tuple[int, int, int, int]
    confirm_button: Tuple[int, int, int, int]
    new_order_button: Tuple[int, int, int, int]


def compute_demo_layout(frame_shape: Tuple[int, int, int]) -> DemoUiLayout:
    """Compute the exact card/button geometry for the current frame."""
    h, w = frame_shape[:2]

    # Keep a fuller right panel, especially at 960x540 presentation size.
    cam_ratio = 0.62 if w <= 1000 else 0.66
    cam_w = int(w * cam_ratio)
    panel_w = w - cam_w
    panel_x = cam_w
    pad = max(8, min(14, int(round(h * 0.018))))

    available_content = max(int(h * 0.30), h - (pad * 6))
    keys = ("header", "word", "sentence", "order", "controls")

    ratio = {
        "header": 0.11,
        "word": 0.17,
        "sentence": 0.30,
        "order": 0.29,
        "controls": 0.13,
    }

    min_ratio = {
        "header": 0.08,
        "word": 0.14,
        "sentence": 0.22,
        "order": 0.22,
        "controls": 0.09,
    }
    floor_ratio = {
        "header": 0.06,
        "word": 0.11,
        "sentence": 0.18,
        "order": 0.19,
        "controls": 0.07,
    }
    min_h = {k: max(30, int(round(available_content * min_ratio[k]))) for k in keys}
    floor_h = {k: max(24, int(round(available_content * floor_ratio[k]))) for k in keys}

    heights = {k: int(round(available_content * ratio[k])) for k in keys}
    for k in keys:
        heights[k] = max(min_h[k], heights[k])

    used = sum(heights.values())
    if used < available_content:
        extra = available_content - used
        heights["sentence"] += int(extra * 0.38)
        heights["order"] += int(extra * 0.42)
        heights["word"] += int(extra * 0.08)
        heights["controls"] += int(extra * 0.12)
        spill = available_content - sum(heights.values())
        if spill > 0:
            heights["sentence"] += spill
    elif used > available_content:
        overflow = used - available_content
        for k in ("sentence", "order", "word", "controls", "header"):
            room = max(0, heights[k] - floor_h[k])
            cut = min(overflow, room)
            heights[k] -= cut
            overflow -= cut
            if overflow <= 0:
                break
        if overflow > 0:
            heights["sentence"] = max(40, heights["sentence"] - overflow)

    h_header = heights["header"]
    h_word = heights["word"]
    h_sentence = heights["sentence"]
    h_order = heights["order"]
    h_controls = heights["controls"]

    cx = panel_x + pad
    cw = panel_w - pad * 2
    y = pad

    header = (cx, y, cw, h_header)
    y += h_header + pad
    word_card = (cx, y, cw, h_word)
    y += h_word + pad
    sentence_card = (cx, y, cw, h_sentence)
    y += h_sentence + pad
    order_card = (cx, y, cw, h_order)
    y += h_order + pad
    controls_card = (cx, y, cw, h_controls)

    inner_pad = max(8, int(round(cw * 0.04)))
    btn_margin = max(6, int(round(h_order * 0.06)))
    btn_w = max(int(round(cw * 0.60)), cw - (inner_pad * 2))
    btn_h_confirm = max(30, min(int(round(h_order * 0.40)), int(round(h_order * 0.52))))
    btn_h_new = max(26, min(int(round(h_order * 0.34)), int(round(h_order * 0.46))))
    confirm_y = order_card[1] + order_card[3] - btn_h_confirm - btn_margin
    new_order_y = order_card[1] + order_card[3] - btn_h_new - btn_margin
    confirm_button = (order_card[0] + inner_pad, confirm_y, btn_w, btn_h_confirm)
    new_order_button = (order_card[0] + inner_pad, new_order_y, btn_w, btn_h_new)

    return DemoUiLayout(
        h=h,
        w=w,
        cam_w=cam_w,
        panel_x=panel_x,
        panel_w=panel_w,
        pad=pad,
        header=header,
        word_card=word_card,
        sentence_card=sentence_card,
        order_card=order_card,
        controls_card=controls_card,
        confirm_button=confirm_button,
        new_order_button=new_order_button,
    )


def _clamp_lines(lines: List[str], max_lines: int) -> List[str]:
    if len(lines) <= max_lines:
        return lines
    out = lines[:max_lines]
    if out:
        tail = out[-1]
        out[-1] = (tail[:-1] + "...") if len(tail) > 1 else "..."
    return out


def _truncate_one_line(text: str, max_width: int, scale: float, thickness: int) -> str:
    txt = text
    while txt:
        tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        if tw <= max_width:
            return txt
        txt = txt[:-1]
    return ""


def _is_clean_token(token: str) -> bool:
    if not token:
        return False
    if "?" in token:
        return False
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ -")
    return all(ch in allowed for ch in token.upper())


def _draw_badge(canvas: np.ndarray, text: str, x: int, y: int, w: int, h: int, color: Tuple[int, int, int], text_color: Tuple[int, int, int]) -> None:
    _rounded_card(canvas, x, y, w, h, color, r=max(8, h // 2))
    scale = _fit_text_scale(text, target_px=max(14, int(h * 0.58)), max_width=w - 10, max_height=h - 4, thickness=1, min_scale=0.38, max_scale=1.4)
    tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
    tx = x + max(5, (w - tw) // 2)
    ty = y + max(th + 1, (h + th) // 2)
    cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 1, cv2.LINE_AA)


def _draw_button(canvas: np.ndarray, rect: Tuple[int, int, int, int], text: str, enabled: bool, *, primary: bool = True) -> None:
    x, y, w, h = rect
    if enabled:
        color = (34, 124, 255) if primary else (72, 170, 82)
        text_color = (248, 250, 252)
        border = (255, 255, 255)
    else:
        color = (183, 192, 204)
        text_color = (88, 96, 106)
        border = (206, 214, 223)

    _rounded_card(canvas, x, y, w, h, color, r=max(8, h // 2))
    border_thickness = 1 if h < 40 else 2
    cv2.rectangle(canvas, (x, y), (x + w, y + h), border, border_thickness)
    txt_thickness = 1 if h < 42 else 2
    scale = _fit_text_scale(
        text,
        target_px=max(14, int(round(h * 0.52))),
        max_width=w - max(10, int(round(w * 0.08))),
        max_height=h - max(4, int(round(h * 0.18))),
        thickness=txt_thickness,
        min_scale=0.48,
        max_scale=1.20,
    )
    tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, txt_thickness)[0]
    tx = x + max(8, (w - tw) // 2)
    ty = y + (h + th) // 2 - 1
    cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, txt_thickness, cv2.LINE_AA)


def _point_in_rect(px: int, py: int, rect: Tuple[int, int, int, int]) -> bool:
    x, y, w, h = rect
    return x <= px <= x + w and y <= py <= y + h


def _fit_text_scale(
    text: str,
    *,
    target_px: int,
    max_width: int,
    max_height: int,
    thickness: int,
    min_scale: float = 0.45,
    max_scale: float = 4.2,
) -> float:
    """Compute a readable scale that fits target height AND available box size.

    This prevents invisible text when the right panel is narrow (e.g. 640x480 presentation mode).
    """
    if not text:
        return 1.0

    base = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, 1.0, thickness)[0]
    bh = max(1, base[1])
    scale = float(target_px) / float(bh)
    scale = min(max_scale, max(min_scale, scale))

    for _ in range(30):
        tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
        if tw <= max_width and th <= max_height:
            break
        scale *= 0.92
        if scale <= min_scale:
            scale = min_scale
            break
    return scale


def _wrap_text_lines(text: str, max_width: int, scale: float, thickness: int) -> List[str]:
    """Wrap text into as many lines as needed without truncation."""
    if not text:
        return ["(empty)"]
    words = text.split()
    if not words:
        return ["(empty)"]

    lines: List[str] = []
    current_words: List[str] = []

    for word in words:
        trial = " ".join(current_words + [word]).strip()
        tw = cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        if tw <= max_width or not current_words:
            current_words.append(word)
        else:
            lines.append(" ".join(current_words))
            current_words = [word]

    if current_words:
        lines.append(" ".join(current_words))

    return lines or ["(empty)"]


def _fit_multiline_sentence(
    text: str,
    *,
    max_width: int,
    max_height: int,
    thickness: int,
    start_scale: float = 1.0,
    min_scale: float = 0.22,
) -> Tuple[List[str], float, int]:
    """Fit full sentence into a box by reducing font scale if needed."""
    scale = max(start_scale, min_scale)
    while scale >= min_scale:
        lines = _wrap_text_lines(text, max_width=max_width, scale=scale, thickness=thickness)
        line_h = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][1] + 6
        total_h = line_h * len(lines)
        widest = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] for line in lines)

        if total_h <= max_height and widest <= max_width:
            return lines, scale, line_h

        scale *= 0.90

    # Final fallback at minimum scale.
    scale = min_scale
    lines = _wrap_text_lines(text, max_width=max_width, scale=scale, thickness=thickness)
    line_h = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][1] + 6
    return lines, scale, line_h


def _resize_cover(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize while preserving aspect ratio, then center-crop to target box."""
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0 or target_w <= 0 or target_h <= 0:
        return np.zeros((max(1, target_h), max(1, target_w), 3), dtype=np.uint8)

    scale = max(target_w / float(w), target_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    x0 = max(0, (new_w - target_w) // 2)
    y0 = max(0, (new_h - target_h) // 2)
    return resized[y0 : y0 + target_h, x0 : x0 + target_w]


def draw_demo_ui(
    frame: np.ndarray,
    *,
    live_token: Optional[str],
    live_conf: float,
    current_emoji: Optional[str],
    confirmed_token: Optional[str],
    confirmed_count: int,
    sentence_text: str,
    sentence_tokens: List[str],
    top_predictions: Sequence[Tuple[str, float]],
    status_lines: List[str],
    fps: float,
    order_confirmed: bool,
    order_number: str,
    kitchen_status: str,
    confirm_enabled: bool,
    debug_ui: bool = False,
) -> np.ndarray:
    """Draw clean left-camera + right-dashboard UI for realtime demo."""
    layout = compute_demo_layout(frame.shape)
    h, w = layout.h, layout.w
    cam_w = layout.cam_w
    panel_w = layout.panel_w
    panel_x = layout.panel_x
    pad = layout.pad

    # ---- Warm palette (clean fast-food / kiosk)
    cream_bg = (244, 241, 232)
    panel_bg = (248, 237, 223)
    card_bg = (246, 250, 254)
    accent_tomato = (44, 89, 227)
    accent_orange = (29, 160, 245)
    accent_green = (95, 163, 102)
    text_dark = (36, 48, 62)
    text_muted = (90, 103, 116)

    canvas = np.full((h, w, 3), cream_bg, dtype=np.uint8)

    # ---- Left large camera area (clean, minimal overlays)
    cam = _resize_cover(frame, cam_w, h)
    canvas[:, :cam_w] = cam

    # Subtle camera frame and compact top label (resolution-aware)
    cam_margin = max(6, int(round(min(cam_w, h) * 0.012)))
    cv2.rectangle(canvas, (cam_margin, cam_margin), (cam_w - cam_margin, h - cam_margin), (242, 248, 252), 2)

    label_x = cam_margin + max(6, int(round(cam_w * 0.012)))
    label_y = cam_margin + max(4, int(round(h * 0.010)))
    label_w = max(180, int(round(cam_w * 0.58)))
    label_h = max(30, int(round(h * 0.068)))
    cv2.rectangle(canvas, (label_x, label_y), (label_x + label_w, label_y + label_h), (250, 252, 255), -1)

    cam_label = f"Live signer view   FPS {fps:.1f}"
    cam_scale = _fit_text_scale(
        cam_label,
        target_px=max(13, int(round(label_h * 0.50))),
        max_width=label_w - 12,
        max_height=label_h - 6,
        thickness=1,
        min_scale=0.40,
        max_scale=0.78,
    )
    cam_text_y = label_y + max(16, int(round(label_h * 0.70)))
    cv2.putText(canvas, cam_label, (label_x + 10, cam_text_y), cv2.FONT_HERSHEY_SIMPLEX, cam_scale, (74, 92, 106), 1, cv2.LINE_AA)

    # ---- Right dashboard panel
    cv2.rectangle(canvas, (panel_x, 0), (w, h), panel_bg, -1)
    cv2.line(canvas, (panel_x, 0), (panel_x, h), (190, 206, 219), 2)

    # Header strip
    _rounded_card(canvas, *layout.header, card_bg, r=12)
    hx, hy, hw, hh = layout.header
    header_pad_x = max(8, int(round(hw * 0.03)))
    live_h = max(20, int(round(hh * 0.44)))
    live_w = max(64, int(round(hw * 0.24)))
    live_x = hx + hw - header_pad_x - live_w
    live_y = hy + max(6, int(round(hh * 0.14)))
    _draw_badge(canvas, "LIVE", live_x, live_y, live_w, live_h, accent_tomato, (248, 250, 252))

    title = "Restaurant Kiosk"
    subtitle = "Sign-language ordering"
    title_max_w = max(80, (live_x - 8) - (hx + header_pad_x))
    title_scale = _fit_text_scale(
        title,
        target_px=max(16, int(round(hh * 0.38))),
        max_width=title_max_w,
        max_height=max(16, int(round(hh * 0.34))),
        thickness=2,
        min_scale=0.46,
        max_scale=0.95,
    )
    title_y = hy + max(16, int(round(hh * 0.40)))
    cv2.putText(canvas, title, (hx + header_pad_x, title_y), cv2.FONT_HERSHEY_SIMPLEX, title_scale, text_dark, 2, cv2.LINE_AA)

    sub_scale = _fit_text_scale(
        subtitle,
        target_px=max(11, int(round(hh * 0.22))),
        max_width=title_max_w,
        max_height=max(11, int(round(hh * 0.20))),
        thickness=1,
        min_scale=0.35,
        max_scale=0.7,
    )
    sub_y = hy + max(30, int(round(hh * 0.74)))
    cv2.putText(canvas, subtitle, (hx + header_pad_x, sub_y), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, text_muted, 1, cv2.LINE_AA)
    # 1) Current detected word card
    wx0, wy0, ww0, wh0 = layout.word_card
    _rounded_card(canvas, wx0, wy0, ww0, wh0, card_bg, r=14)
    word_pad_x = max(8, int(round(ww0 * 0.035)))
    _draw_card_title(canvas, "Current word", wx0 + word_pad_x, wy0 + max(18, int(round(wh0 * 0.22))))
    live_word_raw = (live_token or "").strip().upper()
    stable_word_raw = (confirmed_token or "").strip().upper()
    live_word = live_word_raw if _is_clean_token(live_word_raw) else ""
    stable_word = stable_word_raw if _is_clean_token(stable_word_raw) else ""

    if stable_word:
        current_word = stable_word
        subline = f"confirmed: {stable_word}"
        state_color = accent_green
    elif live_word_raw:
        current_word = live_word_raw
        subline = f"detected: {live_word_raw}"
        state_color = accent_orange
    else:
        current_word = "WAITING FOR SIGN"
        subline = "show the next sign clearly"
        state_color = text_muted

    # Main word: clean stable word only (or detecting/waiting placeholder), never broken token.
    word_box_x = wx0 + word_pad_x
    word_box_w = ww0 - (word_pad_x * 2)
    word_box_y = wy0 + max(28, int(round(wh0 * 0.33)))
    word_box_h = max(30, wh0 - int(round(wh0 * 0.45)))
    _rounded_card(canvas, word_box_x, word_box_y, word_box_w, word_box_h, (224, 235, 255), r=14)

    scale_word = _fit_text_scale(
        current_word,
        target_px=max(30, int(wh0 * 0.30)),
        max_width=max(80, word_box_w - 16),
        max_height=max(24, word_box_h - 22),
        thickness=2,
        min_scale=0.46,
        max_scale=1.35,
    )
    tw, th = cv2.getTextSize(current_word, cv2.FONT_HERSHEY_SIMPLEX, scale_word, 2)[0]
    wx = word_box_x + max(6, (word_box_w - tw) // 2)
    wy = word_box_y + max(th + 4, int(word_box_h * 0.60))
    cv2.putText(canvas, current_word, (wx, wy), cv2.FONT_HERSHEY_SIMPLEX, scale_word, text_dark, 2, cv2.LINE_AA)

    sub_scale = _fit_text_scale(
        subline,
        target_px=max(10, int(round(wh0 * 0.11))),
        max_width=ww0 - (word_pad_x * 2),
        max_height=16,
        thickness=1,
        min_scale=0.34,
        max_scale=0.62,
    )
    cv2.putText(
        canvas,
        subline,
        (wx0 + word_pad_x, wy0 + wh0 - max(8, int(round(wh0 * 0.08)))),
        cv2.FONT_HERSHEY_SIMPLEX,
        sub_scale,
        state_color,
        1,
        cv2.LINE_AA,
    )

    # 2) Final sentence card (full sentence, auto-wrapped)
    sx0, sy0, sw0, sh0 = layout.sentence_card
    _rounded_card(canvas, sx0, sy0, sw0, sh0, card_bg, r=14)
    sentence_pad_x = max(8, int(round(sw0 * 0.035)))
    _draw_card_title(canvas, "Order summary", sx0 + sentence_pad_x, sy0 + max(18, int(round(sh0 * 0.16))))

    sentence_clean = (sentence_text or "").strip() or "(empty)"
    content_top = sy0 + max(34, int(round(sh0 * 0.24)))
    content_bottom = sy0 + sh0 - max(8, int(round(sh0 * 0.06)))
    content_h = max(24, content_bottom - content_top)
    lines, sent_scale, line_h = _fit_multiline_sentence(
        sentence_clean,
        max_width=sw0 - (sentence_pad_x * 2),
        max_height=content_h,
        thickness=2,
        start_scale=0.90,
        min_scale=0.22,
    )
    content_left = sx0 + sentence_pad_x
    for i, line in enumerate(lines):
        ly = content_top + (i + 1) * line_h - 4
        if ly > content_bottom:
            break
        cv2.putText(canvas, line, (content_left, ly), cv2.FONT_HERSHEY_SIMPLEX, sent_scale, text_dark, 1, cv2.LINE_AA)

    # 3) Order confirmation card
    ax0, ay0, aw0, ah0 = layout.order_card
    _rounded_card(canvas, ax0, ay0, aw0, ah0, card_bg, r=14)
    next_order_num = order_number or "A12"
    order_pad_x = max(8, int(round(aw0 * 0.035)))
    title_y = ay0 + max(18, int(round(ah0 * 0.18)))
    badge_h = max(20, int(round(ah0 * 0.16)))
    badge_y = ay0 + max(26, int(round(ah0 * 0.24)))
    next_badge_w = max(78, int(round(aw0 * 0.26)))
    detail_top = badge_y + badge_h + max(6, int(round(ah0 * 0.05)))
    detail_step = max(14, int(round(ah0 * 0.12)))

    state = "sent" if order_confirmed else ("ready" if confirm_enabled else "detecting")
    _draw_card_title(canvas, "Order confirmation", ax0 + order_pad_x, title_y)

    if state == "sent":
        sent_badge_w = min(max(96, int(round(aw0 * 0.40))), aw0 - (order_pad_x * 2) - next_badge_w - 8)
        _draw_badge(canvas, "ORDER SENT", ax0 + order_pad_x, badge_y, sent_badge_w, badge_h, (74, 138, 78), (248, 250, 252))
        _draw_badge(canvas, next_order_num, ax0 + aw0 - order_pad_x - next_badge_w, badge_y, next_badge_w, badge_h, accent_tomato, (248, 250, 252))
        info_bottom = layout.new_order_button[1] - max(6, int(round(ah0 * 0.05)))
        sent_lines = ["Your order has been sent.", kitchen_status]
        baseline = detail_top
        for line in sent_lines:
            if baseline > info_bottom:
                break
            scale_line = _fit_text_scale(
                line,
                target_px=max(11, int(round(ah0 * 0.10))),
                max_width=aw0 - (order_pad_x * 2),
                max_height=15,
                thickness=1,
                min_scale=0.30,
                max_scale=0.50,
            )
            cv2.putText(canvas, line, (ax0 + order_pad_x, baseline), cv2.FONT_HERSHEY_SIMPLEX, scale_line, text_dark if line.startswith("Your") else text_muted, 1, cv2.LINE_AA)
            baseline += detail_step
        _draw_button(canvas, layout.new_order_button, "Start New Order", True, primary=False)
    elif state == "ready":
        _draw_badge(canvas, f"Next: {next_order_num}", ax0 + order_pad_x, badge_y, next_badge_w, badge_h, accent_tomato, (248, 250, 252))
        details = [
            "Tap Confirm Order to send to kitchen.",
            "Order summary will be frozen.",
            f"Detected confidence: {live_conf:.0%}",
        ]
        baseline = detail_top
        limit = layout.confirm_button[1] - max(6, int(round(ah0 * 0.05)))
        for idx, line in enumerate(details):
            if baseline > limit:
                break
            scale_line = _fit_text_scale(
                line,
                target_px=max(10, int(round(ah0 * 0.095))),
                max_width=aw0 - (order_pad_x * 2),
                max_height=14,
                thickness=1,
                min_scale=0.30,
                max_scale=0.48,
            )
            color = text_dark if idx == 0 else text_muted
            cv2.putText(canvas, line, (ax0 + order_pad_x, baseline), cv2.FONT_HERSHEY_SIMPLEX, scale_line, color, 1, cv2.LINE_AA)
            baseline += detail_step
        _draw_button(canvas, layout.confirm_button, "Confirm Order", True, primary=True)
    else:
        _draw_badge(canvas, f"Next: {next_order_num}", ax0 + order_pad_x, badge_y, next_badge_w, badge_h, accent_tomato, (248, 250, 252))
        details = [
            "Tap Confirm Order to send to kitchen.",
            "Order summary will be frozen.",
            f"Detected confidence: {live_conf:.0%}",
        ]
        baseline = detail_top
        limit = ay0 + ah0 - max(8, int(round(ah0 * 0.08)))
        for idx, line in enumerate(details):
            if baseline > limit:
                break
            scale_line = _fit_text_scale(
                line,
                target_px=max(10, int(round(ah0 * 0.095))),
                max_width=aw0 - (order_pad_x * 2),
                max_height=14,
                thickness=1,
                min_scale=0.30,
                max_scale=0.48,
            )
            color = text_dark if idx == 0 else text_muted
            cv2.putText(canvas, line, (ax0 + order_pad_x, baseline), cv2.FONT_HERSHEY_SIMPLEX, scale_line, color, 1, cv2.LINE_AA)
            baseline += detail_step

    # 4) Controls/help footer card
    fx0, fy0, fw0, fh0 = layout.controls_card
    _rounded_card(canvas, fx0, fy0, fw0, fh0, (236, 229, 214), r=12)
    controls_pad_x = max(8, int(round(fw0 * 0.035)))
    if fh0 >= 50:
        _draw_card_title(canvas, "Menu board & controls", fx0 + controls_pad_x, fy0 + max(16, int(round(fh0 * 0.34))))
        baseline = fy0 + max(30, int(round(fh0 * 0.62)))
    else:
        compact_title = "Menu & controls"
        scale_title = _fit_text_scale(compact_title, target_px=max(10, int(round(fh0 * 0.28))), max_width=fw0 - (controls_pad_x * 2), max_height=max(10, int(round(fh0 * 0.40))), thickness=1, min_scale=0.32, max_scale=0.52)
        cv2.putText(canvas, compact_title, (fx0 + controls_pad_x, fy0 + max(16, int(round(fh0 * 0.70)))), cv2.FONT_HERSHEY_SIMPLEX, scale_title, text_dark, 1, cv2.LINE_AA)
        baseline = fy0 + max(22, int(round(fh0 * 0.52)))

    if fh0 >= 60:
        footer_lines = [
            "Menu: hamburger, fries, hash brown, apple pie",
            "X reset | Z undo | C confirm | N new order | Enter speak | Q/ESC quit",
        ]
    else:
        footer_lines = ["X reset | C confirm | N new | Q/ESC quit"]

    line_step = max(12, int(round(fh0 * 0.24)))
    bottom_limit = fy0 + fh0 - max(4, int(round(fh0 * 0.08)))
    for line in footer_lines:
        if baseline > bottom_limit:
            break
        scale_line = _fit_text_scale(line, target_px=max(10, int(round(fh0 * 0.20))), max_width=fw0 - (controls_pad_x * 2), max_height=max(10, int(round(fh0 * 0.24))), thickness=1, min_scale=0.30, max_scale=0.48)
        cv2.putText(canvas, line, (fx0 + controls_pad_x, baseline), cv2.FONT_HERSHEY_SIMPLEX, scale_line, text_muted, 1, cv2.LINE_AA)
        baseline += line_step

    # Keep debug_ui as a no-op so the presentation stays clean.
    # Pose/keypoint debug is handled by the separate show-pose-debug overlay on the camera.

    return canvas
