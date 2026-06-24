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

try:
    from ultralytics import YOLO
except ImportError as exc:
    YOLO = None  # type: ignore
    _yolo_import_error: Optional[Exception] = exc
else:
    _yolo_import_error = None





from football_log.vision.label_utils import all_class_ids_from, bbox_too_small, coerce_ids


class DeepSortTracker:
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

        self.model = YOLO(model_name)
        self._conf = conf
        self._imgsz = imgsz
        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=1.0,  # YOLO has already applied NMS
            max_cosine_distance=max_cosine_distance,
            nn_budget=None,
            embedder=embedder,
            half=half,
            bgr=True,
        )
        self.player_class_ids: Tuple[int, ...] = coerce_ids(player_class_ids)
        self.ball_class_ids: Tuple[int, ...] = coerce_ids(ball_class_ids)
        self.referee_class_ids: Tuple[int, ...] = coerce_ids(referee_class_ids)
        self.team_a_class_ids: Tuple[int, ...] = coerce_ids(team_a_class_ids)
        self.team_b_class_ids: Tuple[int, ...] = coerce_ids(team_b_class_ids)
        self._team_classifier: Optional["TeamClassifierProto"] = None
        self._all_class_ids: List[int] = all_class_ids_from(
            self.player_class_ids, self.ball_class_ids,
            self.referee_class_ids, self.team_a_class_ids, self.team_b_class_ids,
        )

    def set_team_classifier(self, tc: "TeamClassifierProto") -> None:
        self._team_classifier = tc

    @property
    def all_class_ids(self) -> List[int]:
        return self._all_class_ids

    # ------ Detector Protocol ------

    def detect(self, frame: np.ndarray) -> List["Detection"]:
        from football_log.protocols import Detection

        # 1. YOLO detection only (no per-frame tracking state inside YOLO)
        results = self.model.predict(
            frame,
            conf=self._conf,
            imgsz=self._imgsz,
            classes=self.all_class_ids or None,
            verbose=False,
        )

        # 2. Convert to deep_sort_realtime input: [(ltrb, conf, cls), ...]
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

        # 3. DeepSORT update: Kalman predict + Re-ID association
        tracks = self._tracker.update_tracks(raw_dets, frame=frame)

        # 4. Build Detection list from confirmed tracks
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

    def _assign_label(
        self,
        obj_cls: int,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        track_id: int,
    ) -> Tuple[str, Optional[Tuple[int, int, int]]]:
        from football_log.vision.label_utils import assign_label as _al
        return _al(
            obj_cls, frame, bbox, track_id,
            self.ball_class_ids, self.referee_class_ids,
            self.team_a_class_ids, self.team_b_class_ids,
            self._team_classifier,
        )
