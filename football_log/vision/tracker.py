"""YOLO + ByteTrack/BoT-SORT 跟踪封装。

实现 protocols.Detector 接口。

支持配置 player / ball / referee 三类的源 class IDs，便于切换到自训练权重：
- COCO 默认（yolov8n.pt）：player=[0], ball=[32], referee=None
- Module 1 自训练权重（顺序 player, ball, referee）：player=[0], ball=[1], referee=[2]
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
    _import_error = exc
else:
    _import_error = None

from football_log.vision.label_utils import all_class_ids_from, bbox_too_small, coerce_ids
from football_log.vision.team_classifier import TeamClassifier
from football_log.vision.tracker_registry import resolve_tracker


class YoloByteTrackTracker:
    """Detector 协议实现：YOLO 检测 + ByteTrack/BoT-SORT 跟踪。

    使用：先 set_team_classifier(tc) 注入分队器，再调用 detect(frame) → List[Detection]。
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf: float = 0.3,
        imgsz: int = 640,
        tracker: str = "bytetrack.yaml",
        player_class_ids: Sequence[int] = (0,),
        ball_class_ids: Sequence[int] = (32,),
        referee_class_ids: Optional[Sequence[int]] = None,
        team_a_class_ids: Optional[Sequence[int]] = None,
        team_b_class_ids: Optional[Sequence[int]] = None,
    ):
        if YOLO is None:
            raise RuntimeError("未安装 ultralytics，请先执行: pip install ultralytics") from _import_error
        self.model = YOLO(model_name)
        self.conf = conf
        self.imgsz = imgsz
        self.tracker = resolve_tracker(tracker)
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

        raw = self._track_raw(frame)
        results: List[Detection] = []
        for item in raw:
            label, box_color = self._assign_label(
                item["cls"], frame, item["bbox"], item["id"]
            )
            results.append(Detection(
                track_id=item["id"],
                bbox=item["bbox"],
                label=label,
                conf=item["conf"],
                box_color=box_color,
            ))
        return results

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

    # ------ 内部方法 ------

    def _track_raw(self, frame: np.ndarray) -> List[dict]:
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            classes=self.all_class_ids,
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
            bbox = (x1, y1, w, h)
            if bbox_too_small(bbox):
                continue
            items.append({
                "id": int(id_arr[i]) if i < len(id_arr) else -1,
                "bbox": bbox,
                "cls": int(cls_arr[i]),
                "conf": float(conf_arr[i]),
            })
        return items
