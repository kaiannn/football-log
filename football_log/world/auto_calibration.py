"""Per-frame automatic homography estimation.

When a broadcast camera pans / tilts / zooms, a static `--homography` baked
in at startup goes stale within a few frames. This module computes the
homography for *each* frame so projection stays accurate.

Three layered components:

1. `Keyframe` + `load_keyframes_json` — labeled (image_pixel, world_meter)
   point pairs at specific frames. The user labels these once via
   `app/calibrate_reference.py` or any external tool, dumps to JSON.

2. `KeyframeOpticalFlowSource` — between keyframes, propagates the
   labeled image points using Lucas–Kanade optical flow on grayscale
   frames. Re-anchors on every new keyframe.

3. `HomographySmoother` — exponential moving average on the 8 free
   parameters of H (matrix is normalized so H[2,2] = 1). Suppresses
   the per-frame jitter that would otherwise leak into world coords.

`AutoCalibrationProjector` ties them together and implements the
`WorldProjector` Protocol — drop-in replacement for the static
`HomographyProjector`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2  # noqa: F401
except ImportError:  # pragma: no cover - cv2 is a hard runtime dep
    cv2 = None  # type: ignore

from football_log.vision.label_utils import bbox_anchor, is_person_label
from football_log.world.homography import Homography


# ---- keyframe loader -----------------------------------------------------------


@dataclass
class Keyframe:
    """Hand-labeled image↔world correspondences at one frame."""

    frame_idx: int
    image_points: np.ndarray   # (N, 2), float32, pixel coords
    world_points: np.ndarray   # (N, 2), float32, world meters
    names: List[str] = field(default_factory=list)


def load_keyframes_json(path: Path) -> List[Keyframe]:
    """Parse a keyframes JSON file.

    Schema:

        {
          "video_path": "match.mp4",
          "pitch_dimensions": {"length_m": 105.0, "width_m": 68.0},
          "keyframes": [
            {
              "frame_idx": 0,
              "points": [
                {"name": "corner_NW",      "image_uv": [120,  80], "world_xy_m": [0,    0]},
                {"name": "corner_NE",      "image_uv": [1800, 90], "world_xy_m": [105,  0]},
                {"name": "corner_SW",      "image_uv": [80,  980], "world_xy_m": [0,   68]},
                {"name": "corner_SE",      "image_uv": [1840,990], "world_xy_m": [105, 68]},
                {"name": "center_circle",  "image_uv": [960, 540], "world_xy_m": [52.5,34]}
              ]
            },
            { "frame_idx": 250, "points": [...] }
          ]
        }
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    raw_keyframes = data.get("keyframes")
    if not isinstance(raw_keyframes, list) or not raw_keyframes:
        raise ValueError(f"{path}: 'keyframes' must be a non-empty list")

    out: List[Keyframe] = []
    for kf in raw_keyframes:
        frame_idx_raw = kf.get("frame_idx")
        if frame_idx_raw is None:
            raise ValueError(f"{path}: keyframe missing 'frame_idx'")
        frame_idx = int(frame_idx_raw)
        points = kf.get("points", [])
        if len(points) < 4:
            raise ValueError(
                f"{path}: keyframe at frame {frame_idx} has {len(points)} points; need >= 4"
            )
        for i, p in enumerate(points):
            if "image_uv" not in p:
                raise ValueError(f"{path}: keyframe {frame_idx} point {i} missing 'image_uv'")
            if "world_xy_m" not in p:
                raise ValueError(f"{path}: keyframe {frame_idx} point {i} missing 'world_xy_m'")
        image_pts = np.array([p["image_uv"] for p in points], dtype=np.float32)
        world_pts = np.array([p["world_xy_m"] for p in points], dtype=np.float32)
        names = [str(p.get("name", f"pt{i}")) for i, p in enumerate(points)]
        out.append(Keyframe(frame_idx=frame_idx, image_points=image_pts, world_points=world_pts, names=names))
    return sorted(out, key=lambda k: k.frame_idx)


# ---- pure homography fit -------------------------------------------------------


def fit_homography(
    image_pts: np.ndarray,
    world_pts: np.ndarray,
    ransac_threshold_px: float = 3.0,
) -> Optional[np.ndarray]:
    """Fit a 3x3 homography from (image, world) point pairs via RANSAC.

    Returns the matrix or None if too few points / degenerate configuration.
    """
    if cv2 is None:
        raise RuntimeError("cv2 not installed")
    if image_pts is None or world_pts is None:
        return None
    if len(image_pts) < 4 or len(world_pts) < 4 or len(image_pts) != len(world_pts):
        return None
    H, _mask = cv2.findHomography(
        image_pts.astype(np.float32),
        world_pts.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold_px,
    )
    if H is None:
        return None
    return H.astype(np.float64)


# ---- EMA smoother --------------------------------------------------------------


class HomographySmoother:
    """Exponential moving average on the 8 free parameters of H.

    H is normalized so H[2, 2] = 1 before smoothing — this keeps the EMA
    well-defined despite homographies being defined up to scale.
    """

    def __init__(self, alpha: float = 0.3):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self._smoothed: Optional[np.ndarray] = None

    @staticmethod
    def _normalize(H: np.ndarray) -> np.ndarray:
        if abs(H[2, 2]) < 1e-12:
            return H.copy()
        return H / H[2, 2]

    def reset(self) -> None:
        self._smoothed = None

    def smooth(self, H: np.ndarray) -> np.ndarray:
        Hn = self._normalize(np.asarray(H, dtype=np.float64))
        if self._smoothed is None:
            self._smoothed = Hn
        else:
            self._smoothed = self.alpha * Hn + (1.0 - self.alpha) * self._smoothed
        return self._smoothed.copy()


# ---- homography sources --------------------------------------------------------


class KeyframeOpticalFlowSource:
    """Propagates labeled image points across frames via Lucas–Kanade.

    Re-anchors on every keyframe (so drift is bounded by inter-keyframe gap).
    """

    def __init__(self, keyframes: List[Keyframe], ransac_threshold_px: float = 3.0):
        if cv2 is None:
            raise RuntimeError("cv2 not installed")
        if not keyframes:
            raise ValueError("KeyframeOpticalFlowSource needs at least one keyframe")
        self._keyframes = sorted(keyframes, key=lambda k: k.frame_idx)
        self._ransac_threshold = float(ransac_threshold_px)

        self._prev_gray: Optional[np.ndarray] = None
        self._image_pts: Optional[np.ndarray] = None      # (N, 1, 2), tracks the image points
        self._world_pts: Optional[np.ndarray] = None      # (N, 2)
        self._anchored_keyframe_idx: int = -1

    def _lk_params(self) -> dict:
        return dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def _find_active_keyframe(self, frame_idx: int) -> Optional[Keyframe]:
        active: Optional[Keyframe] = None
        for kf in self._keyframes:
            if kf.frame_idx <= frame_idx:
                active = kf
            else:
                break
        return active

    def _anchor_to(self, keyframe: Keyframe, frame_gray: np.ndarray) -> None:
        self._image_pts = keyframe.image_points.reshape(-1, 1, 2).astype(np.float32)
        self._world_pts = keyframe.world_points.astype(np.float32)
        self._prev_gray = frame_gray
        self._anchored_keyframe_idx = keyframe.frame_idx

    def prepare_for_frame(self, frame_idx: int, frame_bgr: np.ndarray) -> None:
        """Update internal state with the current frame."""
        if cv2 is None:
            raise RuntimeError("cv2 not installed")
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        active_kf = self._find_active_keyframe(frame_idx)
        if active_kf is None:
            # Before the first keyframe — nothing to project from.
            return

        # Re-anchor whenever we hit / pass a new keyframe.
        if active_kf.frame_idx > self._anchored_keyframe_idx:
            self._anchor_to(active_kf, gray)
            return

        # Propagate via LK from the previous frame.
        if self._prev_gray is None or self._image_pts is None:
            self._anchor_to(active_kf, gray)
            return

        new_pts, status, _err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._image_pts, None, **self._lk_params(),
        )
        if new_pts is None or status is None:
            # Tracking failed — re-anchor on the active keyframe to recover.
            self._anchor_to(active_kf, gray)
            return

        valid = status.flatten() == 1
        if int(valid.sum()) < 4:
            self._anchor_to(active_kf, gray)
            return

        # Keep only the valid tracked points (and corresponding world points).
        self._image_pts = new_pts[valid].reshape(-1, 1, 2).astype(np.float32)
        self._world_pts = self._world_pts[valid].astype(np.float32)
        self._prev_gray = gray

    def current_homography(self) -> Optional[np.ndarray]:
        if self._image_pts is None or self._world_pts is None:
            return None
        image_2d = self._image_pts.reshape(-1, 2)
        return fit_homography(image_2d, self._world_pts, self._ransac_threshold)


class HomographySequenceSource:
    """Pre-computed per-frame H from a .npy of shape (N, 3, 3).

    Convenient when an external tool (e.g. TVCalib) has dumped per-frame
    homographies offline. Frames beyond N reuse the last entry.
    """

    def __init__(self, path: Path):
        arr = np.load(path)
        if arr.ndim != 3 or arr.shape[1:] != (3, 3):
            raise ValueError(f"{path}: expected .npy of shape (N, 3, 3), got {arr.shape}")
        self._H_seq = arr.astype(np.float64)

    def prepare_for_frame(self, frame_idx: int, frame_bgr: np.ndarray) -> None:
        # Stateless — index lookup happens in current_homography()
        self._current_idx = frame_idx

    def current_homography(self) -> Optional[np.ndarray]:
        idx = min(self._current_idx, len(self._H_seq) - 1)
        if idx < 0:
            return None
        return self._H_seq[idx].copy()


# ---- WorldProjector adapter ----------------------------------------------------


class AutoCalibrationProjector:
    """WorldProjector Protocol implementation backed by a time-varying H source.

    Caller is expected to call `prepare_for_frame(frame_idx, frame_bgr)` once
    per frame BEFORE invoking `project()` for any detection in that frame.
    """

    def __init__(
        self,
        source,
        smoother: Optional[HomographySmoother] = None,
    ):
        self._source = source
        self._smoother = smoother
        self._current_H: Optional[Homography] = None
        self.n_frames_seen: int = 0
        self.n_frames_with_homography: int = 0

    @property
    def current_homography(self) -> Optional[Homography]:
        """The most recent Homography object, or None if unavailable."""
        return self._current_H

    def prepare_for_frame(self, frame_idx: int, frame_bgr: np.ndarray) -> None:
        self.n_frames_seen += 1
        self._source.prepare_for_frame(frame_idx, frame_bgr)
        H_raw = self._source.current_homography()
        if H_raw is None:
            self._current_H = None
            return
        if self._smoother is not None:
            H_raw = self._smoother.smooth(H_raw)
        self._current_H = Homography(matrix=H_raw)
        self.n_frames_with_homography += 1

    def project(
        self,
        bbox: Tuple[int, int, int, int],
        label: str,
    ) -> Tuple[Optional[float], Optional[float]]:
        if self._current_H is None:
            return None, None
        if not is_person_label(label) and label != "Ball":
            return None, None
        u, v = bbox_anchor(bbox, label)
        wx, wy = self._current_H.pixel_to_world(u, v)
        if np.isnan(wx) or np.isnan(wy):
            return None, None
        return wx, wy
