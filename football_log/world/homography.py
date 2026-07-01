"""像素平面与球场世界平面之间的单应变换工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from football_log.vision.label_utils import bbox_anchor, bbox_foot_point, is_person_label

__all__ = ["Homography", "HomographyProjector", "project_foot_to_world"]


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
        self._inv_matrix: Optional[np.ndarray] = None

    def pixel_to_world(self, u: float, v: float) -> Tuple[float, float]:
        """像素 (u, v) → 世界坐标 (wx, wy)，单位米。"""
        p = self.matrix @ np.array([u, v, 1.0], dtype=np.float64)
        if abs(p[2]) < 1e-12:
            return float("nan"), float("nan")
        return float(p[0] / p[2]), float(p[1] / p[2])

    def world_to_pixel(self, wx: float, wy: float) -> Tuple[float, float]:
        """世界 (wx, wy) → 像素 (u, v)。需要 H 可逆时可靠。"""
        if self._inv_matrix is None:
            try:
                self._inv_matrix = np.linalg.inv(self.matrix)
            except np.linalg.LinAlgError:
                self._inv_matrix = np.linalg.pinv(self.matrix)
        p = self._inv_matrix @ np.array([wx, wy, 1.0], dtype=np.float64)
        if abs(p[2]) < 1e-12:
            return float("nan"), float("nan")
        return float(p[0] / p[2]), float(p[1] / p[2])


def project_foot_to_world(
    bbox: Tuple[float, float, float, float],
    label: str,
    H: Optional[Homography],
) -> Tuple[Optional[float], Optional[float]]:
    """若 label 为球员或球且 H 有效，返回锚点世界坐标；否则返回 (None, None)。"""
    if H is None:
        return None, None
    if not is_person_label(label) and label != "Ball":
        return None, None
    u, v = bbox_anchor(bbox, label)
    wx, wy = H.pixel_to_world(u, v)
    if np.isnan(wx) or np.isnan(wy):
        return None, None
    return wx, wy


class HomographyProjector:
    """WorldProjector 协议实现：基于 3x3 单应矩阵的坐标映射。"""

    def __init__(self, H: Homography) -> None:
        self._H = H

    def project(
        self,
        bbox: Tuple[int, int, int, int],
        label: str,
    ) -> Tuple[Optional[float], Optional[float]]:
        return project_foot_to_world(bbox, label, self._H)
