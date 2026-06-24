"""Dedicated ball detection using Roboflow's football-ball-detection.pt (YOLOv8x).

Uses InferenceSlicer to tile the full frame into 640×640 crops for small-ball recall.
Only returns ball detections — meant to be merged with the main tracker's output.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from football_log.protocols import Detection

try:
    from ultralytics import YOLO
    from inference.models.utils import get_roboflow_model  # noqa: F401 — not required
except ImportError:
    pass

try:
    import supervision as sv
    _SV_AVAILABLE = True
except ImportError:
    _SV_AVAILABLE = False


class BallDetector:
    """Runs football-ball-detection.pt with optional InferenceSlicer tiling.

    Returns a list of Detection objects (typically 0 or 1 ball per frame).

    slicer=True  → accurate, ~4–8 YOLO passes per frame, slower (~4 fps on M4)
    slicer=False → single pass at imgsz, faster but may miss distant balls
    """

    def __init__(
        self,
        model_path: str = "runs/roboflow/football-ball-detection.pt",
        conf: float = 0.3,
        imgsz: int = 640,
        slicer: bool = False,
    ):
        try:
            from ultralytics import YOLO as _YOLO
        except ImportError:
            raise RuntimeError("ultralytics not installed")

        self.model = _YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.use_slicer = slicer and _SV_AVAILABLE

        if slicer and not _SV_AVAILABLE:
            print(
                "[BallDetector] supervision not installed — falling back to single-pass "
                "mode. Install with: pip install supervision"
            )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.use_slicer:
            return self._detect_sliced(frame)
        return self._detect_single(frame)

    # ------------------------------------------------------------------
    def _detect_single(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)
        return self._parse(results, frame)

    def _detect_sliced(self, frame: np.ndarray) -> List[Detection]:
        import supervision as sv

        def _callback(crop: np.ndarray) -> sv.Detections:
            results = self.model(crop, imgsz=self.imgsz, conf=self.conf, verbose=False)
            return sv.Detections.from_ultralytics(results[0])

        slicer = sv.InferenceSlicer(
            callback=_callback,
            slice_wh=(640, 640),
            overlap_ratio_wh=(0.2, 0.2),
            iou_threshold=0.5,
        )
        detections = slicer(frame)
        if len(detections) == 0:
            return []

        best = int(detections.confidence.argmax())
        x1, y1, x2, y2 = [int(v) for v in detections.xyxy[best]]
        return [Detection(
            track_id=-1,
            bbox=(x1, y1, max(0, x2 - x1), max(0, y2 - y1)),
            label="Ball",
            conf=float(detections.confidence[best]),
        )]

    # ------------------------------------------------------------------
    @staticmethod
    def _parse(results, frame: np.ndarray) -> List[Detection]:
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        if len(xyxy) == 0:
            return []
        best = int(confs.argmax())
        x1, y1, x2, y2 = [int(v) for v in xyxy[best]]
        return [Detection(
            track_id=-1,
            bbox=(x1, y1, max(0, x2 - x1), max(0, y2 - y1)),
            label="Ball",
            conf=float(confs[best]),
        )]
