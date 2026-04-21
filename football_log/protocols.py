"""流水线各环节的接口协议与统一数据结构。

第三方只需实现对应 Protocol 即可替换默认组件，例如：
- 用 RT-DETR 替代 YOLO → 实现 Detector
- 用 Re-ID 方案替代 K-Means → 实现 TeamClassifierProto
- 用 TVCalib 替代 Homography → 实现 WorldProjector
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np


@dataclass
class Detection:
    """流水线内部统一的单目标检测结果。

    所有 Detector 实现都应返回此结构的列表，
    下游的分队、映射、导出均依赖这些字段。
    """

    track_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    label: str  # "player" / "ball" / "Team A" / "Team B" / ...
    conf: float = 0.0
    box_color: Optional[Tuple[int, int, int]] = None
    world_x_m: Optional[float] = None
    world_y_m: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.track_id,
            "bbox": self.bbox,
            "label": self.label,
            "conf": self.conf,
            "box_color": self.box_color,
            "world_x_m": self.world_x_m,
            "world_y_m": self.world_y_m,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Detection:
        return cls(
            track_id=int(d.get("id", -1)),
            bbox=tuple(d["bbox"]),  # type: ignore[arg-type]
            label=str(d.get("label", "")),
            conf=float(d.get("conf", 0.0)),
            box_color=d.get("box_color"),
            world_x_m=d.get("world_x_m"),
            world_y_m=d.get("world_y_m"),
        )


# ---------------------------------------------------------------------------
# Protocol 定义
# ---------------------------------------------------------------------------


@runtime_checkable
class Detector(Protocol):
    """检测 + 跟踪：输入一帧图像，输出 Detection 列表。"""

    def detect(self, frame: np.ndarray) -> List[Detection]: ...


@runtime_checkable
class TeamClassifierProto(Protocol):
    """分队：给单个检测结果打上队伍标签。"""

    def classify(
        self,
        frame: np.ndarray,
        detection: Detection,
    ) -> str:
        """返回标签字符串，如 "Team A" / "Team B" / "Player"。"""
        ...


@runtime_checkable
class PitchEstimator(Protocol):
    """场地估计：从一帧图像输出场地观测。"""

    def estimate(self, frame: np.ndarray) -> Any:
        """返回 PitchObservation 或兼容结构。"""
        ...


@runtime_checkable
class WorldProjector(Protocol):
    """坐标映射：像素坐标 → 世界坐标（米）。"""

    def project(
        self,
        bbox: Tuple[int, int, int, int],
        label: str,
    ) -> Tuple[Optional[float], Optional[float]]:
        """返回 (world_x_m, world_y_m)，无法映射时返回 (None, None)。"""
        ...


@runtime_checkable
class Exporter(Protocol):
    """导出：将每帧 Detection 列表持久化。"""

    def write_frame(self, frame_idx: int, detections: List[Detection]) -> None: ...

    def close(self) -> None: ...

    @property
    def records_written(self) -> int: ...
