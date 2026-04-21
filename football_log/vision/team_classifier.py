"""球衣颜色自动聚类分队 + 时序平滑。

特征提取流程：
  1. bbox 中 20%-50% 高度、15%-85% 宽度的球衣区域
  2. HSV 空间 mask 掉草地绿色像素（H 35-85, S>40）
  3. 对剩余像素取 H (×2 映射到 0-360) 和 S 的均值作为 2D 特征
  4. K-Means (k=2) 自动聚类

可选覆盖：通过 team_colors 传入两队 BGR 颜色，跳过自动聚类。
"""

from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

_GRASS_H_LOW = 35
_GRASS_H_HIGH = 85
_GRASS_S_MIN = 40
_MIN_NON_GRASS_RATIO = 0.15


def _extract_jersey_patch(frame: np.ndarray, bbox: Tuple[int, ...]) -> Optional[np.ndarray]:
    x, y, w, h = bbox
    y_top = y + int(h * 0.20)
    y_bot = y + int(h * 0.50)
    x_left = x + int(w * 0.15)
    x_right = x + int(w * 0.85)
    fh, fw = frame.shape[:2]
    y_top = max(0, y_top)
    y_bot = min(fh, y_bot)
    x_left = max(0, x_left)
    x_right = min(fw, x_right)
    if y_bot <= y_top or x_right <= x_left:
        return None
    patch = frame[y_top:y_bot, x_left:x_right]
    if patch.size == 0:
        return None
    return patch


def _grass_mask(patch_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    return (h >= _GRASS_H_LOW) & (h <= _GRASS_H_HIGH) & (s >= _GRASS_S_MIN)


def _patch_to_hs_feature(patch_bgr: np.ndarray) -> Optional[np.ndarray]:
    mask = ~_grass_mask(patch_bgr)
    total = mask.size
    non_grass = int(mask.sum())
    if non_grass < max(10, int(total * _MIN_NON_GRASS_RATIO)):
        return None
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    h_vals = hsv[:, :, 0][mask].astype(np.float32) * 2.0
    s_vals = hsv[:, :, 1][mask].astype(np.float32)
    return np.array([float(h_vals.mean()), float(s_vals.mean())], dtype=np.float32)


def get_dominant_color(frame: np.ndarray, bbox: Tuple[int, ...]) -> Optional[Tuple[int, int, int]]:
    patch = _extract_jersey_patch(frame, bbox)
    if patch is None:
        return None
    mask = ~_grass_mask(patch)
    if mask.sum() < 10:
        return None
    pixels = patch[mask]
    median_bgr = np.median(pixels, axis=0).astype(int)
    return (int(median_bgr[0]), int(median_bgr[1]), int(median_bgr[2]))


def _bgr_to_hs_center(bgr: Tuple[int, int, int]) -> np.ndarray:
    pixel = np.array([[list(bgr)]], dtype=np.uint8)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
    return np.array([float(hsv[0, 0, 0]) * 2.0, float(hsv[0, 0, 1])], dtype=np.float32)


class TeamClassifier:
    TEAM_A = "Team A"
    TEAM_B = "Team B"
    UNKNOWN = "Player"

    def __init__(
        self,
        history_len: int = 12,
        warmup_frames: int = 50,
        min_samples: int = 20,
        team_colors: Optional[List[Tuple[int, int, int]]] = None,
    ):
        self._history: Dict[int, Deque[str]] = defaultdict(lambda: deque(maxlen=history_len))
        self._warmup_frames = warmup_frames
        self._min_samples = min_samples
        self._frame_count = 0

        self._samples: List[np.ndarray] = []
        self._centers: Optional[np.ndarray] = None
        self._fitted = False

        if team_colors is not None and len(team_colors) >= 2:
            c0 = _bgr_to_hs_center(team_colors[0])
            c1 = _bgr_to_hs_center(team_colors[1])
            self._centers = np.vstack([c0, c1]).astype(np.float32)
            self._fitted = True

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _try_fit(self) -> None:
        if self._fitted or len(self._samples) < self._min_samples:
            return
        data = np.vstack(self._samples).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            data, 2, None, criteria, 20, cv2.KMEANS_PP_CENTERS,
        )
        dist = float(np.linalg.norm(centers[0] - centers[1]))
        if dist < 15.0:
            print(f"[TeamClassifier] 聚类中心距离过近 ({dist:.1f})，延长采样")
            return
        self._centers = centers
        self._fitted = True
        print(
            f"[TeamClassifier] K-Means 完成: "
            f"A HS=({centers[0][0]:.0f},{centers[0][1]:.0f}), "
            f"B HS=({centers[1][0]:.0f},{centers[1][1]:.0f}), "
            f"距离={dist:.1f}, 样本={len(self._samples)}"
        )

    def _classify(self, feat: np.ndarray) -> str:
        if self._centers is None:
            return self.UNKNOWN
        d0 = float(np.linalg.norm(feat - self._centers[0]))
        d1 = float(np.linalg.norm(feat - self._centers[1]))
        ratio = min(d0, d1) / (max(d0, d1) + 1e-6)
        if ratio > 0.80:
            return self.UNKNOWN
        return self.TEAM_A if d0 < d1 else self.TEAM_B

    def instant_label(self, frame: np.ndarray, bbox: Tuple[int, ...]) -> str:
        patch = _extract_jersey_patch(frame, bbox)
        if patch is None:
            return self.UNKNOWN

        feat = _patch_to_hs_feature(patch)
        if feat is None:
            return self.UNKNOWN

        if not self._fitted:
            self._samples.append(feat)
            self._frame_count += 1
            if self._frame_count >= self._warmup_frames:
                self._try_fit()
            return self.UNKNOWN

        return self._classify(feat)

    def smooth_label(self, track_id: int, instant_label: str) -> str:
        if "Player" not in instant_label and "Team" not in instant_label:
            return instant_label
        history = self._history[track_id]
        history.append(instant_label)
        a_votes = sum(1 for x in history if x == self.TEAM_A)
        b_votes = sum(1 for x in history if x == self.TEAM_B)
        if a_votes > b_votes:
            return self.TEAM_A
        if b_votes > a_votes:
            return self.TEAM_B
        if a_votes == 0 and b_votes == 0:
            return self.UNKNOWN
        return instant_label
