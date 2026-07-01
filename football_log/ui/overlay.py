"""在视频帧上绘制检测框与 HUD。"""

from typing import Any, Dict, List

import cv2


_LABEL_COLORS = {
    "Team A": (0, 0, 255),
    "Team B": (255, 0, 0),
    "Ball": (0, 200, 200),
    "Player": (180, 180, 180),
}


def _label_color(label: str) -> tuple:
    for key, color in _LABEL_COLORS.items():
        if key in label:
            return color
    return (0, 255, 255)


def _draw_badge(frame, text: str, x: int, y: int, bg: tuple) -> None:
    """Draw a filled pill-shaped label chip like Image 3 (e.g. '#21')."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.42, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    pad_x, pad_y = 4, 3
    bx1, by1 = x, y - th - pad_y * 2 - baseline
    bx2, by2 = x + tw + pad_x * 2, y
    # Clamp to frame
    bx1, by1 = max(0, bx1), max(0, by1)
    bx2, by2 = min(frame.shape[1] - 1, bx2), min(frame.shape[0] - 1, by2)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), bg, -1, cv2.LINE_AA)
    cv2.putText(frame, text, (bx1 + pad_x, by2 - baseline - 1), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_tracking_overlay(frame, tracked_objects: List[Dict[str, Any]]) -> None:
    for obj in tracked_objects:
        x, y, w, h = [int(v) for v in obj["bbox"]]
        label = obj["label"]
        color = obj.get("box_color") or _label_color(label)
        # Thinner box — less visual clutter
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
        # Badge chip above the box
        tid = obj.get("track_id", obj.get("id", ""))
        _draw_badge(frame, f"#{tid}", x, y - 2, color)


def draw_frame_hud(frame, frame_idx: int, detect_every_n: int) -> None:
    h, w, _ = frame.shape
    cv2.putText(frame, f"Frame: {frame_idx}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(
        frame,
        f"detect_every_n={detect_every_n}",
        (w - 260, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
