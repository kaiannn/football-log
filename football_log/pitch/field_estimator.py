"""
草地分割 + 场内线段（Hough）+ 场区四边形近似。

文献与任务背景（便于与重型学习方法区分）：
- **SoccerNet**（Deliège et al., CVPR Workshop 等）将广播足球视频结构化，后续 **Camera Calibration**
  子任务显式利用**场平面与场线几何**评价单应/相机参数；场线检测是经典基线之一。
- **TVCalib**（Theiner et al., WACV 2023）等近期工作用可学习模型做全场注册；本模块刻意采用
  **无深度模型**的 OpenCV 管线，便于 CPU 部署与跨项目复用，可作为粗初始化或与标定模块级联。

实现要点：HSV 草皮掩膜 → 形态学 → 最大连通域 → 可选凸包/多边形近似四边形；
在草皮掩膜内对灰度图做 Canny + **概率 Hough** 提取白线线段，供可视化或后续线聚类/消失点估计。
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

import cv2
import numpy as np

from football_log.pitch.config import PitchFieldConfig
from football_log.pitch.observation import PitchObservation


def _sort_quad_tl_tr_br_bl(points: np.ndarray) -> np.ndarray:
    """四点排序：左上→右上→右下→左下（透视四边形常用 sum/diff 规则）。"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) != 4:
        return pts
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float64)


def _quad_from_contour(c: np.ndarray, epsilon_ratio: float) -> Optional[np.ndarray]:
    peri = cv2.arcLength(c, True)
    if peri < 1e-6:
        return None
    eps = float(epsilon_ratio) * peri
    approx = cv2.approxPolyDP(c, eps, True)
    if len(approx) == 4:
        return _sort_quad_tl_tr_br_bl(approx.reshape(4, 2))
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    return _sort_quad_tl_tr_br_bl(box)


def _confidence(area_ratio: float, n_lines: int, has_quad: bool) -> float:
    a = min(1.0, area_ratio / 0.35)
    l = min(1.0, n_lines / 25.0)
    q = 1.0 if has_quad else 0.0
    return float(0.35 * a + 0.35 * l + 0.3 * q)


class PitchFieldEstimator:
    def __init__(self, config: Optional[PitchFieldConfig] = None):
        self.config = config or PitchFieldConfig()

    def estimate(self, frame_bgr: np.ndarray) -> PitchObservation:
        h, w = frame_bgr.shape[:2]
        cfg = self.config
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, cfg.hsv_lower, cfg.hsv_upper)

        k_open = max(1, cfg.morph_open_ksize | 1)
        k_close = max(1, cfg.morph_close_ksize | 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))

        frame_area = float(h * w)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        grass_area_ratio = 0.0
        field_quad: Optional[np.ndarray] = None
        grass_mask = np.zeros((h, w), dtype=np.uint8)

        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            grass_area_ratio = area / frame_area if frame_area > 0 else 0.0
            if grass_area_ratio >= cfg.min_grass_area_ratio and area > 100:
                cv2.drawContours(grass_mask, [c], -1, 255, thickness=-1)
                field_quad = _quad_from_contour(c, cfg.approx_poly_epsilon_ratio)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, cfg.canny_low, cfg.canny_high)
        edges = cv2.bitwise_and(edges, edges, mask=grass_mask)

        theta = np.deg2rad(cfg.hough_theta_deg)
        lines = cv2.HoughLinesP(
            edges,
            cfg.hough_rho,
            theta,
            cfg.hough_threshold,
            minLineLength=cfg.hough_min_line_length,
            maxLineGap=cfg.hough_max_line_gap,
        )
        segs: List[List[float]] = []
        if lines is not None:
            for ln in lines:
                x1, y1, x2, y2 = ln[0].astype(float).tolist()
                segs.append([x1, y1, x2, y2])
        line_segments = np.array(segs, dtype=np.float32).reshape(-1, 4) if segs else np.zeros((0, 4), dtype=np.float32)

        conf = _confidence(grass_area_ratio, len(segs), field_quad is not None)
        meta = {
            "n_line_segments": len(segs),
            "frame_size": (w, h),
        }
        return PitchObservation(
            grass_mask=grass_mask,
            grass_area_ratio=float(grass_area_ratio),
            line_segments=line_segments,
            field_quad_xy=field_quad,
            confidence=conf,
            meta=meta,
        )


class TemporalPitchSmoother:
    """对四边形角点做指数平滑，减轻抖动（视频部署常用）。"""

    def __init__(self, alpha: float = 0.25, history: int = 5):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self._history: Deque[float] = deque(maxlen=history)
        self._quad: Optional[np.ndarray] = None

    def smooth(self, obs: PitchObservation) -> PitchObservation:
        if obs.field_quad_xy is None or obs.confidence < 0.2:
            self._history.append(obs.confidence)
            return obs

        q = obs.field_quad_xy.astype(np.float64)
        if self._quad is None:
            self._quad = q.copy()
        else:
            self._quad = self.alpha * q + (1.0 - self.alpha) * self._quad

        self._history.append(obs.confidence)
        out = PitchObservation(
            grass_mask=obs.grass_mask,
            grass_area_ratio=obs.grass_area_ratio,
            line_segments=obs.line_segments,
            field_quad_xy=self._quad.copy(),
            confidence=obs.confidence,
            meta={**obs.meta, "temporal_smoothed": True},
        )
        return out
