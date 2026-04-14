"""球衣颜色自动聚类分队 + 时序平滑。

默认模式：开场 warmup 阶段收集球员球衣 LAB 色度特征，K-Means（k=2）
自动聚成 Team A / Team B；后续帧用聚类中心做最近邻分类。

可选覆盖：通过 team_colors 传入两队 BGR 颜色，跳过自动聚类。
"""

from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _extract_jersey_patch(frame: np.ndarray, bbox: Tuple[int, ...]) -> Optional[np.ndarray]:
    x, y, w, h = bbox
    y_top = y + int(h * 0.15)
    y_bot = y + int(h * 0.55)
    x_left = x + int(w * 0.1)
    x_right = x + int(w * 0.9)
    fh, fw = frame.shape[:2]
    if y_top < 0 or y_bot > fh or x_left < 0 or x_right > fw or y_bot <= y_top or x_right <= x_left:
        return None
    patch = frame[y_top:y_bot, x_left:x_right]
    if patch.size == 0:
        return None
    return patch


def _patch_to_ab_feature(patch_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB)
    a_mean = float(lab[:, :, 1].mean())
    b_mean = float(lab[:, :, 2].mean())
    return np.array([a_mean, b_mean], dtype=np.float32)


def _patch_to_lab_feature(patch_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB)
    return np.array([
        float(lab[:, :, 0].mean()),
        float(lab[:, :, 1].mean()),
        float(lab[:, :, 2].mean()),
    ], dtype=np.float32)


def _bgr_to_ab_center(bgr: Tuple[int, int, int]) -> np.ndarray:
    pixel = np.array([[list(bgr)]], dtype=np.uint8)
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)
    return np.array([float(lab[0, 0, 1]), float(lab[0, 0, 2])], dtype=np.float32)


class TeamClassifier:
    TEAM_A = "Team A"
    TEAM_B = "Team B"
    UNKNOWN = "Player"

    def __init__(
        self,
        history_len: int = 8,
        warmup_frames: int = 30,
        min_samples: int = 10,
        team_colors: Optional[List[Tuple[int, int, int]]] = None,
    ):
        self._history: Dict[int, Deque[str]] = defaultdict(lambda: deque(maxlen=history_len))
        self._warmup_frames = warmup_frames
        self._min_samples = min_samples
        self._frame_count = 0

        self._samples: List[np.ndarray] = []
        self._centers: Optional[np.ndarray] = None  # (2, 2) in ab space
        self._fitted = False

        if team_colors is not None and len(team_colors) >= 2:
            c0 = _bgr_to_ab_center(team_colors[0])
            c1 = _bgr_to_ab_center(team_colors[1])
            self._centers = np.vstack([c0, c1]).astype(np.float32)
            self._fitted = True

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _try_fit(self) -> None:
        if self._fitted or len(self._samples) < self._min_samples:
            return
        data = np.vstack(self._samples).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _, labels, centers = cv2.kmeans(
            data, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS,
        )
        self._centers = centers  # (2, 2)
        self._fitted = True
        c0_lab = centers[0]
        c1_lab = centers[1]
        print(f"[TeamClassifier] K-Means 拟合完成: "
              f"Team A ab=({c0_lab[0]:.1f},{c0_lab[1]:.1f}), "
              f"Team B ab=({c1_lab[0]:.1f},{c1_lab[1]:.1f}), "
              f"样本数={len(self._samples)}")

    def _classify_ab(self, ab: np.ndarray) -> str:
        if self._centers is None:
            return self.UNKNOWN
        d0 = float(np.linalg.norm(ab - self._centers[0]))
        d1 = float(np.linalg.norm(ab - self._centers[1]))
        ratio = min(d0, d1) / (max(d0, d1) + 1e-6)
        if ratio > 0.85:
            return self.UNKNOWN
        return self.TEAM_A if d0 < d1 else self.TEAM_B

    def instant_label(self, frame: np.ndarray, bbox: Tuple[int, ...]) -> str:
        patch = _extract_jersey_patch(frame, bbox)
        if patch is None:
            return self.UNKNOWN

        ab = _patch_to_ab_feature(patch)

        if not self._fitted:
            self._samples.append(ab)
            self._frame_count += 1
            if self._frame_count >= self._warmup_frames:
                self._try_fit()
            return self.UNKNOWN

        return self._classify_ab(ab)

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
