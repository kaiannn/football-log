"""像素平面与球场世界平面之间的单应变换工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Homography:
    """
    图像像素 → 世界平面 (米) 的 3×3 单应矩阵 H。

    满足 [wx', wy', w]^T = H @ [x, y, 1]^T，世界点为 (wx'/w, wy'/w)。
    由 cv2.findHomography(image_points, world_points) 等方式得到。
    """

    matrix: np.ndarray  # shape (3, 3)

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=np.float64).reshape(3, 3)

    def pixel_to_world(self, u: float, v: float) -> Tuple[float, float]:
        """像素 (u, v) → 世界坐标 (wx, wy)，单位米。"""
        p = self.matrix @ np.array([u, v, 1.0], dtype=np.float64)
        if abs(p[2]) < 1e-12:
            return float("nan"), float("nan")
        return float(p[0] / p[2]), float(p[1] / p[2])

    def world_to_pixel(self, wx: float, wy: float) -> Tuple[float, float]:
        """世界 (wx, wy) → 像素 (u, v)。需要 H 可逆时可靠。"""
        Hinv = np.linalg.inv(self.matrix)
        p = Hinv @ np.array([wx, wy, 1.0], dtype=np.float64)
        if abs(p[2]) < 1e-12:
            return float("nan"), float("nan")
        return float(p[0] / p[2]), float(p[1] / p[2])


def bbox_foot_point(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    从 bbox (x, y, w, h) 取「脚底」近似：底边中点。
    用于把球员投影到地面平面。
    """
    x, y, w, h = bbox
    return float(x + 0.5 * w), float(y + h)


def project_foot_to_world(
    bbox: Tuple[float, float, float, float],
    label: str,
    H: Optional[Homography],
) -> Tuple[Optional[float], Optional[float]]:
    """
    若 label 为球员且 H 有效，返回脚底世界坐标；否则返回 (None, None)。
    球可改用 bbox 中心，此处与业务约定一致时再扩展。
    """
    if H is None:
        return None, None
    if "Player" not in label and label != "Ball":
        return None, None
    u, v = bbox_foot_point(bbox) if "Player" in label else (
        float(bbox[0] + 0.5 * bbox[2]),
        float(bbox[1] + 0.5 * bbox[3]),
    )
    wx, wy = H.pixel_to_world(u, v)
    if np.isnan(wx) or np.isnan(wy):
        return None, None
    return wx, wy
