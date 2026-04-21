"""YOLO + ByteTrack/BoT-SORT 跟踪封装。

实现 protocols.Detector 接口，同时保留旧版 track_frame 以兼容已有调用方。
"""

from __future__ import annotations

from typing import Any, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from football_log.protocols import Detection

try:
    from ultralytics import YOLO
except ImportError as exc:
    YOLO = None  # type: ignore
    _import_error = exc
else:
    _import_error = None

from football_log.vision.team_classifier import TeamClassifier, get_dominant_color


class YoloByteTrackTracker:
    """Detector 协议实现：YOLO 检测 + ByteTrack/BoT-SORT 跟踪。

    同时满足两种使用方式：
    - 新接口 detect(frame) → List[Detection]（需要先 set_team_classifier）
    - 旧接口 track_frame(frame, team_classifier) → List[dict]
    """

    person_class_id = 0
    ball_class_id = 32

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf: float = 0.3,
        imgsz: int = 640,
        tracker: str = "bytetrack.yaml",
    ):
        if YOLO is None:
            raise RuntimeError("未安装 ultralytics，请先执行: pip install ultralytics") from _import_error
        self.model = YOLO(model_name)
        self.conf = conf
        self.imgsz = imgsz
        self.tracker = tracker
        self._team_classifier: Optional[TeamClassifier] = None

    def set_team_classifier(self, tc: Any) -> None:
        self._team_classifier = tc

    # ------ 新接口 (Detector Protocol) ------

    def detect(self, frame: np.ndarray) -> List["Detection"]:
        from football_log.protocols import Detection

        raw = self._track_raw(frame)
        tc = self._team_classifier
        results: List[Detection] = []
        for item in raw:
            bbox = item["bbox"]
            obj_cls = item["cls"]
            track_id = item["id"]
            conf = item["conf"]

            if obj_cls == self.ball_class_id:
                label = "Ball"
                box_color = None
            else:
                if tc is not None:
                    instant = tc.instant_label(frame, bbox)
                    label = tc.smooth_label(track_id, instant)
                    box_color = get_dominant_color(frame, bbox)
                else:
                    label = "Player"
                    box_color = get_dominant_color(frame, bbox)

            results.append(Detection(
                track_id=track_id,
                bbox=bbox,
                label=label,
                conf=conf,
                box_color=box_color,
            ))
        return results

    # ------ 旧接口（向后兼容） ------

    def track_frame(self, frame, team_classifier) -> List[dict]:
        raw = self._track_raw(frame)
        tracked: List[dict] = []
        for item in raw:
            bbox = item["bbox"]
            obj_cls = item["cls"]
            track_id = item["id"]
            conf = item["conf"]

            if obj_cls == self.ball_class_id:
                label = "Ball"
                box_color = None
            else:
                instant = team_classifier.instant_label(frame, bbox)
                label = team_classifier.smooth_label(track_id, instant)
                box_color = get_dominant_color(frame, bbox)

            tracked.append({"id": track_id, "bbox": bbox, "label": label, "conf": conf, "box_color": box_color})
        return tracked

    # ------ 内部方法 ------

    def _track_raw(self, frame: np.ndarray) -> List[dict]:
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            classes=[self.person_class_id, self.ball_class_id],
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        cls_arr = boxes.cls.cpu().numpy().astype(int)
        conf_arr = boxes.conf.cpu().numpy()
        id_arr = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.full(len(xyxy), -1)

        items: List[dict] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if w < 2 or h < 2:
                continue
            items.append({
                "id": int(id_arr[i]) if i < len(id_arr) else -1,
                "bbox": (x1, y1, w, h),
                "cls": int(cls_arr[i]),
                "conf": float(conf_arr[i]),
            })
        return items
