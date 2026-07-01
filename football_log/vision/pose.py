"""Pose estimation wrapper for torso-pixel sampling in team classification.

Uses ultralytics YOLOv8-pose. Only loaded when explicitly requested
(via `--team-classifier keypoint`) so the default pipeline stays
pose-free and CPU-friendly.

YOLOv8-pose returns 17 COCO keypoints per detected person. We use four:
- 5: left_shoulder
- 6: right_shoulder
- 11: left_hip
- 12: right_hip

Together they form the torso quadrilateral we sample for jersey color.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:
    YOLO = None  # type: ignore
    _import_error = exc
else:
    _import_error = None


# COCO keypoint indices we care about
COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_HIP = 11
COCO_RIGHT_HIP = 12

KEYPOINT_CONF_THRESHOLD = 0.4


@dataclass
class TorsoKeypoints:
    """Four keypoints defining the torso quadrilateral, image coordinates."""

    left_shoulder: Optional[Tuple[float, float]]
    right_shoulder: Optional[Tuple[float, float]]
    left_hip: Optional[Tuple[float, float]]
    right_hip: Optional[Tuple[float, float]]

    @property
    def is_complete(self) -> bool:
        return all(
            kp is not None
            for kp in (self.left_shoulder, self.right_shoulder, self.left_hip, self.right_hip)
        )

    def quadrilateral(self) -> Optional[np.ndarray]:
        """Return the torso quad as a (4, 2) int array in (x, y) order, or None.

        Order: left_shoulder, right_shoulder, right_hip, left_hip
        (clockwise starting from top-left, suitable for cv2.fillPoly).
        """
        if not self.is_complete:
            return None
        return np.array(
            [
                self.left_shoulder,
                self.right_shoulder,
                self.right_hip,
                self.left_hip,
            ],
            dtype=np.int32,
        )


class PoseEstimator:
    """Lazy YOLOv8-pose wrapper. Constructed only when keypoint classifier is enabled."""

    def __init__(self, model_name: str = "yolov8n-pose.pt", conf: float = 0.25, imgsz: int = 640):
        if YOLO is None:
            raise RuntimeError(
                "ultralytics not installed, cannot enable keypoint team classifier: "
                "pip install ultralytics"
            ) from _import_error
        self.model = YOLO(model_name)
        self.conf = conf
        self.imgsz = imgsz

    def predict(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], TorsoKeypoints]]:
        """Run pose on the full frame; return list of (person_bbox, torso_keypoints)."""
        results = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, verbose=False
        )
        if not results:
            return []
        r = results[0]
        if r.keypoints is None or r.boxes is None or len(r.keypoints) == 0:
            return []

        kpts_xy = r.keypoints.xy.cpu().numpy()
        kpts_conf = (
            r.keypoints.conf.cpu().numpy()
            if r.keypoints.conf is not None
            else np.ones(kpts_xy.shape[:2], dtype=np.float32)
        )
        boxes_xyxy = r.boxes.xyxy.cpu().numpy().astype(int)

        out: List[Tuple[Tuple[int, int, int, int], TorsoKeypoints]] = []
        for i in range(len(kpts_xy)):
            tk = TorsoKeypoints(
                left_shoulder=_kp_or_none(kpts_xy[i], kpts_conf[i], COCO_LEFT_SHOULDER),
                right_shoulder=_kp_or_none(kpts_xy[i], kpts_conf[i], COCO_RIGHT_SHOULDER),
                left_hip=_kp_or_none(kpts_xy[i], kpts_conf[i], COCO_LEFT_HIP),
                right_hip=_kp_or_none(kpts_xy[i], kpts_conf[i], COCO_RIGHT_HIP),
            )
            x1, y1, x2, y2 = boxes_xyxy[i]
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            out.append((bbox, tk))
        return out


def _kp_or_none(
    kp_xy: np.ndarray, kp_conf: np.ndarray, idx: int
) -> Optional[Tuple[float, float]]:
    if idx >= len(kp_xy) or float(kp_conf[idx]) < KEYPOINT_CONF_THRESHOLD:
        return None
    x, y = float(kp_xy[idx, 0]), float(kp_xy[idx, 1])
    if x <= 0.0 and y <= 0.0:
        return None
    return x, y


# ---------------------------------------------------------------------------
# Pure helpers (testable without ultralytics)
# ---------------------------------------------------------------------------


def torso_keypoints_from_bbox_heuristic(
    bbox: Tuple[int, int, int, int]
) -> TorsoKeypoints:
    """Fallback when no pose data is available: estimate torso corners from bbox proportions.

    Used when pose model fails to detect a person inside the bbox (e.g. heavy
    occlusion, unusual pose). The four heuristic points roughly approximate
    where shoulders and hips would be in a standing player.
    """
    x, y, w, h = bbox
    shoulder_y = y + int(0.20 * h)
    hip_y = y + int(0.50 * h)
    left_x = x + int(0.30 * w)
    right_x = x + int(0.70 * w)
    return TorsoKeypoints(
        left_shoulder=(left_x, shoulder_y),
        right_shoulder=(right_x, shoulder_y),
        left_hip=(left_x, hip_y),
        right_hip=(right_x, hip_y),
    )


def bbox_iou(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> float:
    """IoU between two (x, y, w, h) bboxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter) / union if union > 0 else 0.0


def match_pose_to_bbox(
    target_bbox: Tuple[int, int, int, int],
    pose_results: List[Tuple[Tuple[int, int, int, int], TorsoKeypoints]],
    iou_threshold: float = 0.3,
) -> Optional[TorsoKeypoints]:
    """Find the pose result whose bbox best matches `target_bbox` by IoU."""
    best_iou = 0.0
    best_kpts: Optional[TorsoKeypoints] = None
    for pose_bbox, kpts in pose_results:
        score = bbox_iou(target_bbox, pose_bbox)
        if score > best_iou:
            best_iou = score
            best_kpts = kpts
    if best_iou < iou_threshold:
        return None
    return best_kpts
