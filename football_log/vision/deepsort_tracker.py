"""YOLO detection + standalone DeepSORT tracker (Module 2).

Separates detection from association: YOLO predicts bounding boxes,
DeepSORT handles Kalman-filter state and Re-ID appearance matching.

This avoids the skip-frame Kalman stale-state bug that affects
ultralytics' built-in tracker (setback 2.1): because we drive the
Kalman update ourselves via update_tracks(), skipped frames simply
don't trigger an update — lost tracks age out gracefully instead of
being re-initialised with a new ID on the next detected frame.

Install dep:
    pip install deep-sort-realtime>=1.3
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from football_log.protocols import Detection, TeamClassifierProto
    from football_log.vision.reid import ReIDExtractor

try:
    from ultralytics import YOLO
except ImportError as exc:
    YOLO = None  # type: ignore
    _yolo_import_error: Optional[Exception] = exc
else:
    _yolo_import_error = None





from football_log.vision.label_utils import BaseDetector, bbox_too_small, coerce_ids


class DeepSortTracker(BaseDetector):
    """Detector Protocol: YOLO detection + standalone DeepSORT association.

    Use: set_team_classifier(tc) then call detect(frame) → List[Detection].

    Compared to YoloByteTrackTracker / botsort+reid:
    - Re-ID appearance matching reduces ID switches under occlusion
    - Skip-frame tracking is handled correctly (Kalman ages out cleanly)
    - ~2–3x slower than bytetrack due to Re-ID embedding extraction
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf: float = 0.3,
        imgsz: int = 640,
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_distance: float = 0.4,
        embedder: str = "mobilenet",
        half: bool = False,
        player_class_ids: Sequence[int] = (0,),
        ball_class_ids: Sequence[int] = (32,),
        referee_class_ids: Optional[Sequence[int]] = None,
        team_a_class_ids: Optional[Sequence[int]] = None,
        team_b_class_ids: Optional[Sequence[int]] = None,
        reid_extractor: Optional["ReIDExtractor"] = None,
    ):
        if YOLO is None:
            raise RuntimeError(
                "ultralytics not installed. Run: pip install ultralytics"
            ) from _yolo_import_error

        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError as exc:
            raise RuntimeError(
                "--tracker deepsort requires deep-sort-realtime. "
                "Run: pip install 'deep-sort-realtime>=1.3'"
            ) from exc

        super().__init__(
            coerce_ids(player_class_ids), coerce_ids(ball_class_ids),
            coerce_ids(referee_class_ids), coerce_ids(team_a_class_ids),
            coerce_ids(team_b_class_ids),
        )
        self.model = YOLO(model_name)
        self._conf = conf
        self._imgsz = imgsz
        self._reid_extractor = reid_extractor
        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=1.0,
            max_cosine_distance=max_cosine_distance,
            nn_budget=None,
            embedder=embedder if reid_extractor is None else None,
            half=half,
            bgr=True,
        )

    # ------ Detector Protocol ------

    def detect(self, frame: np.ndarray) -> List["Detection"]:
        from football_log.protocols import Detection

        results = self.model.predict(
            frame,
            conf=self._conf,
            imgsz=self._imgsz,
            classes=self.all_class_ids or None,
            verbose=False,
        )

        raw_dets: list = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                raw_dets.append((
                    [float(x1), float(y1), float(x2), float(y2)],
                    float(confs[i]),
                    int(clss[i]),
                ))

        embeds = None
        if self._reid_extractor is not None and raw_dets:
            embeds = []
            for det in raw_dets:
                ltrb = det[0]
                x, y = int(ltrb[0]), int(ltrb[1])
                w, h = int(ltrb[2] - ltrb[0]), int(ltrb[3] - ltrb[1])
                emb = self._reid_extractor.extract(frame, (x, y, w, h))
                embeds.append(emb.astype(np.float32))

        tracks = self._tracker.update_tracks(raw_dets, frame=frame, embeds=embeds)

        detections: List[Detection] = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            bbox = (x1, y1, w, h)
            if bbox_too_small(bbox):
                continue
            cls = int(track.get_det_cls() or 0)
            conf_val = track.get_det_conf()
            label, color = self._assign_label(cls, frame, bbox, track.track_id)
            detections.append(Detection(
                track_id=track.track_id,
                bbox=bbox,
                label=label,
                conf=float(conf_val) if conf_val is not None else 0.0,
                box_color=color,
            ))
        return detections
