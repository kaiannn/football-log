"""在视频帧上绘制检测框与 HUD。"""

from typing import TYPE_CHECKING, Any, Dict, List

import cv2
import numpy as np

if TYPE_CHECKING:
    from football_log.pitch.observation import PitchObservation


def draw_pitch_observation(
    frame,
    obs: "PitchObservation",
    *,
    draw_grass: bool = True,
    draw_lines: bool = True,
    draw_quad: bool = True,
    grass_alpha: float = 0.25,
) -> None:
    """叠加草地半透明绿、场内线段、场区四边形（供调试与多模块可视化）。"""
    h, w = frame.shape[:2]
    if draw_grass and obs.grass_mask is not None and obs.grass_mask.shape[:2] == (h, w):
        color = np.zeros_like(frame)
        color[:, :, 1] = obs.grass_mask  # G channel
        m = (obs.grass_mask > 127).astype(np.float32)[..., None]
        a = float(np.clip(grass_alpha, 0.0, 1.0))
        frame[:] = (frame.astype(np.float32) * (1.0 - m * a) + color.astype(np.float32) * (m * a)).astype(np.uint8)

    if draw_lines and obs.line_segments is not None and len(obs.line_segments) > 0:
        for seg in obs.line_segments:
            x1, y1, x2, y2 = [int(round(v)) for v in seg[:4]]
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)

    if draw_quad and obs.field_quad_xy is not None:
        q = obs.field_quad_xy.astype(np.float32).reshape(-1, 1, 2)
        cv2.polylines(frame, [q.astype(np.int32)], True, (0, 165, 255), 2, cv2.LINE_AA)
        for i, p in enumerate(obs.field_quad_xy.reshape(-1, 2)):
            cv2.circle(frame, (int(p[0]), int(p[1])), 4, (0, 165, 255), -1, cv2.LINE_AA)

    cv2.putText(
        frame,
        f"pitch conf={obs.confidence:.2f} grass={obs.grass_area_ratio:.2f}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


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


def draw_tracking_overlay(frame, tracked_objects: List[Dict[str, Any]]) -> None:
    for obj in tracked_objects:
        x, y, w, h = [int(v) for v in obj["bbox"]]
        label = obj["label"]
        color = obj.get("box_color") or _label_color(label)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        conf = obj.get("conf", 0)
        cv2.putText(
            frame,
            f"ID {obj['id']} {label} {conf:.2f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


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
