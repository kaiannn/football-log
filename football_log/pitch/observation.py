"""场地识别一帧输出（可被跟踪、标定、UI 共用）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PitchObservation:
    """
    单帧场地观测。

    - grass_mask: 与输入帧同尺寸的 uint8 掩膜（0/255），大致对应可踢草皮区域。
    - line_segments: (N, 4)，每行 (x1, y1, x2, y2) 像素，场内线段候选。
    - field_quad_xy: 若可估计，为场地区域近似四边形四角，顺序为左上、右上、右下、左下（图像坐标）。
    - confidence: 0~1 启发式质量分，用于决定是否信任四边形/掩膜。
    """

    grass_mask: np.ndarray
    grass_area_ratio: float
    line_segments: np.ndarray
    field_quad_xy: Optional[np.ndarray] = None
    confidence: float = 0.0
    meta: dict = field(default_factory=dict)

    def grass_roi_bbox_xywh(self) -> Optional[tuple[float, float, float, float]]:
        """草地掩膜外接轴对齐矩形 (x, y, w, h)，无有效前景时 None。"""
        if self.grass_mask is None or self.grass_mask.size == 0:
            return None
        m = self.grass_mask
        ys, xs = np.where(m > 127)
        if len(xs) == 0:
            return None
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        return x0, y0, x1 - x0 + 1, y1 - y0 + 1
