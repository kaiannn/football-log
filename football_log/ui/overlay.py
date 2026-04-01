"""在视频帧上绘制检测框与 HUD。"""

from typing import Any, Dict, List

import cv2


def draw_tracking_overlay(frame, tracked_objects: List[Dict[str, Any]]) -> None:
    for obj in tracked_objects:
        x, y, w, h = [int(v) for v in obj["bbox"]]
        label = obj["label"]
        color = (0, 0, 255) if "Red" in label else (255, 0, 0) if "Blue" in label else (0, 255, 255)
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
