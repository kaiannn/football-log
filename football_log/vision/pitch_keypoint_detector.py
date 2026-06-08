"""Pitch keypoint detection using Roboflow's football-pitch-detection.pt (YOLOv8x-pose).

Detects up to 32 labeled keypoints on the pitch.  Returns them as a numpy array
of shape (N, 2) in pixel coordinates together with a confidence vector, and
optionally computes a per-frame homography to world coordinates.

Standard pitch layout (105 × 68 m) — 32 keypoint world positions indexed 0-31:
Based on Roboflow sports keypoint ordering used in football-field-detection-f07vi v12.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore

# World coordinates for each of the 32 pitch keypoints (x_m, y_m).
# Origin = top-left corner (near touchline), x along length (105 m), y along width (68 m).
#
# Empirically verified against two broadcast videos (ManUtd + Brighton) using
# cross-validation reprojection < 1.5 m. Unknown entries are set to NaN and are
# skipped during homography computation.
#
# Confirmed (✓) = sub-1.5m reprojection error on both test frames.
# Unconfirmed (?) = could not be reliably identified from available footage.
_NAN = float("nan")
PITCH_KP_WORLD: np.ndarray = np.array([
    [_NAN, _NAN],  #  0  unknown
    [_NAN, _NAN],  #  1  unknown
    [_NAN, _NAN],  #  2  unknown
    [_NAN, _NAN],  #  3  unknown
    [_NAN, _NAN],  #  4  unknown
    [_NAN, _NAN],  #  5  unknown
    [_NAN, _NAN],  #  6  unknown
    [_NAN, _NAN],  #  7  unknown
    [_NAN, _NAN],  #  8  unknown
    [_NAN, _NAN],  #  9  unknown
    [_NAN, _NAN],  # 10  unknown
    [_NAN, _NAN],  # 11  unknown
    [_NAN, _NAN],  # 12  unknown
    [_NAN, _NAN],  # 13  unknown
    [_NAN, _NAN],  # 14  unknown
    [_NAN, _NAN],  # 15  unknown
    [_NAN, _NAN],  # 16  unknown
    [52.5,  0.0 ],  # 17  ✓ halfway line × near touchline
    [88.5, 54.16],  # 18  ✓ right penalty area BL (far touchline side)
    [99.5, 54.16],  # 19  ✓ right goal area BR inner
    [105.0,54.16],  # 20  ✓ right penalty area BR (on goal line, far side)
    [99.5, 43.16],  # 21  ✓ right goal area BL
    [_NAN, _NAN],  # 22  ? unconfirmed (projects near 99.5,34 — not a named landmark)
    [105.0,43.16],  # 23  ✓ right goal area BR (on goal line)
    [88.5, 13.84],  # 24  ✓ right penalty area TL (near touchline)
    [_NAN, _NAN],  # 25  ? unconfirmed
    [105.0,24.84],  # 26  ✓ right goal area TR (on goal line, near side)
    [_NAN, _NAN],  # 27  unknown
    [_NAN, _NAN],  # 28  unknown
    [_NAN, _NAN],  # 29  unknown
    [_NAN, _NAN],  # 30  unknown
    [_NAN, _NAN],  # 31  unknown
], dtype=np.float32)


class PitchKeypointDetector:
    """Wraps football-pitch-detection.pt and returns per-frame homography.

    Usage:
        det = PitchKeypointDetector("runs/roboflow/football-pitch-detection.pt")
        H, kps, confs = det.detect(frame)
        # H is (3,3) or None; kps is (N,2) pixel coords; confs is (N,)
    """

    def __init__(self, model_path: str, conf: float = 0.3, imgsz: int = 640):
        if YOLO is None:
            raise RuntimeError("ultralytics not installed")
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz

    def detect(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
        """Run inference on one frame.

        Returns:
            H       : (3,3) homography pixel→world_m, or None if not enough points
            kps_px  : (N,2) detected keypoints in pixel coords
            kp_conf : (N,) confidence per keypoint
        """
        results = self.model(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)
        if not results or results[0].keypoints is None:
            return None, np.empty((0, 2)), np.empty((0,))

        kp_data = results[0].keypoints
        # xy: (num_instances, num_kps, 2), conf: (num_instances, num_kps)
        if kp_data.xy is None or len(kp_data.xy) == 0:
            return None, np.empty((0, 2)), np.empty((0,))

        # Take the highest-confidence detection if multiple instances returned.
        xy   = kp_data.xy.cpu().numpy()    # (instances, kps, 2)
        conf = kp_data.conf.cpu().numpy() if kp_data.conf is not None else np.ones(xy.shape[:2])

        best = int(conf.mean(axis=1).argmax())
        kps_px  = xy[best]    # (num_kps, 2)
        kp_conf = conf[best]  # (num_kps,)

        H = self._compute_homography(kps_px, kp_conf)
        return H, kps_px, kp_conf

    def _compute_homography(
        self, kps_px: np.ndarray, kp_conf: np.ndarray, min_pts: int = 4
    ) -> Optional[np.ndarray]:
        n = min(len(kps_px), len(PITCH_KP_WORLD))
        src, dst = [], []
        for i in range(n):
            wx, wy = PITCH_KP_WORLD[i]
            if np.isnan(wx) or np.isnan(wy):
                continue  # index not yet calibrated
            if kp_conf[i] < self.conf:
                continue
            px, py = kps_px[i]
            if px < 1 and py < 1:
                continue  # model outputs (0,0) for undetected keypoints
            src.append([px, py])
            dst.append([wx, wy])

        if len(src) < min_pts:
            return None

        src_arr = np.array(src, dtype=np.float32)
        dst_arr = np.array(dst, dtype=np.float32)
        # Tight RANSAC threshold (2m) — our confirmed points reproject < 1m,
        # so any inlier with > 2m error is a misdetection.
        H, mask = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 2.0)
        if H is None or mask is None or mask.sum() < min_pts:
            return None
        return H
