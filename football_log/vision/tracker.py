"""YOLO + ByteTrack/BoT-SORT 跟踪封装。"""

from typing import Any, List

import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:
    YOLO = None  # type: ignore
    _import_error = exc
else:
    _import_error = None


class YoloByteTrackTracker:
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

    def track_frame(self, frame, team_classifier) -> List[dict]:
        """
        单帧跟踪，返回 dict 列表: id, bbox (x,y,w,h), label, conf。
        team_classifier: TeamClassifier，用于球员分队。
        """
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

        tracked: List[dict] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if w < 2 or h < 2:
                continue
            obj_cls = int(cls_arr[i])
            track_id = int(id_arr[i]) if i < len(id_arr) else -1
            conf = float(conf_arr[i])
            bbox = (x1, y1, w, h)

            if obj_cls == self.ball_class_id:
                label = "Ball"
            else:
                instant = team_classifier.instant_label(frame, bbox)
                label = team_classifier.smooth_label(track_id, instant)

            tracked.append({"id": track_id, "bbox": bbox, "label": label, "conf": conf})
        return tracked
