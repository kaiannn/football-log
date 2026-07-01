"""Keypoint-based team classifier — pose-driven torso-pixel voting in LAB color space.

Drop-in alternative to the default HSV K-Means TeamClassifier. Implements the
TeamClassifierProto interface (instant_label + smooth_label).

Why this exists:
    The HSV K-Means classifier collapses on low-saturation kits (black vs white,
    grey vs grey) because in HSV with H+S features, both have S≈0 and arbitrary H.
    LAB color space separates lightness from color, so black (L=0) and white
    (L=100) are far apart on the L axis alone.

How it works:
    1. Pose model emits torso keypoints (shoulders + hips) per person per frame.
    2. For each player bbox, we form a torso quadrilateral from those keypoints
       (or fall back to a bbox heuristic if pose fails).
    3. Sample N pixels inside the quad, convert each to LAB.
    4. Each pixel votes for the nearest of two cluster centers (also in LAB).
    5. The bbox label is the majority vote.

Compared to HSV K-Means:
    + Robust to black-vs-white kit confusion
    + Pixel-level voting tolerates skin / shadow / hair pollution better
      than averaging-then-classifying
    - Requires a pose model (extra inference pass)
    - Slower than HSV (~2x with default YOLOv8n-pose on CPU)

The pose model is constructed lazily — it's not loaded unless this classifier
is actually used.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

from football_log.vision.label_utils import try_fit_kmeans
from football_log.vision.pose import (
    PoseEstimator,
    TorsoKeypoints,
    match_pose_to_bbox,
    torso_keypoints_from_bbox_heuristic,
)


# Saturation threshold for LAB sampling: pixels too close to neutral grey
# (low chroma in the a/b channels) are dropped — they're likely shadow or
# washed-out pixels rather than jersey.
LAB_CHROMA_FLOOR = 8.0
DEFAULT_NUM_SAMPLES = 80
DEFAULT_HISTORY_LEN = 12
DEFAULT_WARMUP_FRAMES = 50
DEFAULT_MIN_SAMPLES = 20


def _bgr_to_lab(bgr: Tuple[int, int, int]) -> np.ndarray:
    pixel = np.array([[list(bgr)]], dtype=np.uint8)
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)
    return lab[0, 0].astype(np.float32)


def _sample_pixels_in_quad(
    frame: np.ndarray,
    quad: np.ndarray,
    n_samples: int,
) -> Optional[np.ndarray]:
    """Return up to n_samples LAB pixels inside the quadrilateral.

    Returns None if the quad is outside the frame, degenerate, or yields
    fewer than 5 valid (sufficiently chromatic) pixels.
    """
    fh, fw = frame.shape[:2]
    x_min = max(0, int(quad[:, 0].min()))
    x_max = min(fw, int(quad[:, 0].max()) + 1)
    y_min = max(0, int(quad[:, 1].min()))
    y_max = min(fh, int(quad[:, 1].max()) + 1)
    if x_max <= x_min or y_max <= y_min:
        return None
    local_quad = quad.copy()
    local_quad[:, 0] -= x_min
    local_quad[:, 1] -= y_min
    mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
    cv2.fillPoly(mask, [local_quad], 255)
    local_ys, local_xs = np.nonzero(mask)
    ys = local_ys + y_min
    xs = local_xs + x_min
    if len(ys) < 5:
        return None
    if len(ys) > n_samples:
        sel = np.linspace(0, len(ys) - 1, n_samples).astype(int)
        ys, xs = ys[sel], xs[sel]
    bgr_pixels = frame[ys, xs]
    lab_pixels = cv2.cvtColor(bgr_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    chroma = np.hypot(lab_pixels[:, 1] - 128.0, lab_pixels[:, 2] - 128.0)
    keep = chroma >= LAB_CHROMA_FLOOR
    if int(keep.sum()) < 5:
        # Fall back: use all pixels (kits with no chroma like black/white still
        # have informative L values).
        return lab_pixels
    return lab_pixels[keep]


def _vote_label(
    samples: np.ndarray,
    centers_lab: np.ndarray,
    team_a: str,
    team_b: str,
    unknown: str,
) -> str:
    """Pixel-level majority vote between two LAB cluster centers."""
    if centers_lab is None or len(samples) == 0:
        return unknown
    d0 = np.linalg.norm(samples - centers_lab[0], axis=1)
    d1 = np.linalg.norm(samples - centers_lab[1], axis=1)
    a_votes = int((d0 < d1).sum())
    b_votes = int((d1 <= d0).sum())
    total = a_votes + b_votes
    if total == 0:
        return unknown
    margin = abs(a_votes - b_votes) / total
    if margin < 0.10:
        return unknown
    return team_a if a_votes > b_votes else team_b


class KeypointTeamClassifier:
    """Pose-based team classifier — implements TeamClassifierProto."""

    TEAM_A = "Team A"
    TEAM_B = "Team B"
    UNKNOWN = "Player"

    def __init__(
        self,
        pose_estimator: Optional[PoseEstimator] = None,
        history_len: int = DEFAULT_HISTORY_LEN,
        warmup_frames: int = DEFAULT_WARMUP_FRAMES,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        num_pixel_samples: int = DEFAULT_NUM_SAMPLES,
        team_colors: Optional[List[Tuple[int, int, int]]] = None,
    ):
        self._pose: Optional[PoseEstimator] = pose_estimator  # lazily constructed if None
        self._frame_pose_results: List[Tuple[Tuple[int, int, int, int], TorsoKeypoints]] = []
        self._cached_frame_id: int = -1

        self._history: Dict[int, Deque[str]] = defaultdict(lambda: deque(maxlen=history_len))
        self._warmup_frames = warmup_frames
        self._min_samples = min_samples
        self._num_pixel_samples = num_pixel_samples
        self._frame_count = 0

        self._samples: List[np.ndarray] = []
        self._centers: Optional[np.ndarray] = None
        self._fitted = False

        if team_colors is not None and len(team_colors) >= 2:
            c0 = _bgr_to_lab(team_colors[0])
            c1 = _bgr_to_lab(team_colors[1])
            self._centers = np.vstack([c0, c1]).astype(np.float32)
            self._fitted = True

    # ------ lazy pose model ------

    def _ensure_pose(self) -> PoseEstimator:
        if self._pose is None:
            self._pose = PoseEstimator()
        return self._pose

    def update_pose_for_frame(self, frame: np.ndarray, frame_id: int) -> None:
        """Run pose on this frame and cache results.

        Caller is expected to invoke this once per frame before any
        per-detection instant_label calls. If skipped, instant_label falls
        back to the bbox-heuristic torso.
        """
        self._frame_pose_results = self._ensure_pose().predict(frame)
        self._cached_frame_id = frame_id

    # ------ TeamClassifierProto ------

    def instant_label(self, frame: np.ndarray, bbox: Tuple[int, ...]) -> str:
        bbox_t = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

        # 1. Get torso keypoints — from pose if available, else heuristic.
        kpts = match_pose_to_bbox(bbox_t, self._frame_pose_results)
        if kpts is None or not kpts.is_complete:
            kpts = torso_keypoints_from_bbox_heuristic(bbox_t)
        quad = kpts.quadrilateral()
        if quad is None:
            return self.UNKNOWN

        # 2. Sample pixels inside the torso quad → LAB.
        samples = _sample_pixels_in_quad(frame, quad, self._num_pixel_samples)
        if samples is None or len(samples) == 0:
            return self.UNKNOWN

        # 3. Warmup: collect mean LAB per bbox until we have enough samples.
        if not self._fitted:
            self._samples.append(samples.mean(axis=0))
            self._frame_count += 1
            if self._frame_count >= self._warmup_frames:
                self._try_fit()
            return self.UNKNOWN

        # 4. Vote among torso pixels.
        return _vote_label(
            samples, self._centers, self.TEAM_A, self.TEAM_B, self.UNKNOWN
        )

    def smooth_label(self, track_id: int, instant_label: str) -> str:
        from football_log.vision.label_utils import smooth_label as _sl
        return _sl(track_id, instant_label, self._history, self.TEAM_A, self.TEAM_B, self.UNKNOWN)

    # ------ internals ------

    def _try_fit(self) -> None:
        if self._fitted:
            return
        centers = try_fit_kmeans(self._samples, self._min_samples, 12.0)
        if centers is None:
            return
        dist = float(np.linalg.norm(centers[0] - centers[1]))
        if dist < 12.0:
            logger.info("LAB cluster distance too small (%.1f); extending warmup", dist)
            return
        self._centers = centers.astype(np.float32)
        self._fitted = True
        logger.info(
            "LAB K-Means fitted: A=(%.0f,%.0f,%.0f), B=(%.0f,%.0f,%.0f), dist=%.1f, samples=%d",
            centers[0, 0], centers[0, 1], centers[0, 2],
            centers[1, 0], centers[1, 1], centers[1, 2],
            dist, len(self._samples),
        )

    @property
    def is_fitted(self) -> bool:
        return self._fitted
