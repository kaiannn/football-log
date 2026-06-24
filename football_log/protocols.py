"""流水线各环节的接口协议与统一数据结构。

第三方只需实现对应 Protocol 即可替换默认组件：

  Protocol               默认实现                           替换场景
  ──────────────────────────────────────────────────────────────────────────
  Detector               YoloByteTrackTracker               RT-DETR、DeepSORT、6-class 分队模型
  TeamClassifierProto    TeamClassifier (HSV K-Means)       KeypointTeamClassifier、SAM
  PitchEstimator         PitchFieldEstimator                自定义线检测
  WorldProjector         HomographyProjector                PinholeGroundProjector、AutoCalibrationProjector
  PitchCalibratorProto   PitchKeypointDetector (32pt pose)  TVCalib、SoccerNet Camera Calibration
  BallDetectorProto      BallDetector (Roboflow ball model)  专用小目标模型、InferenceSlicer
  Exporter               TrackingDataWriter (JSONL/CSV)     数据库写入、实时推流

注：BallDetector.detect() 已统一返回 List[Detection]，与本 Protocol 一致。
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
    world_x_m_smoothed: Optional[float] = None
    world_y_m_smoothed: Optional[float] = None
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
            "world_x_m_smoothed": self.world_x_m_smoothed,
            "world_y_m_smoothed": self.world_y_m_smoothed,
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
            world_x_m_smoothed=d.get("world_x_m_smoothed"),
            world_y_m_smoothed=d.get("world_y_m_smoothed"),
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
    """分队：给单个检测打上队伍标签（瞬时判定 + 跨帧平滑）。

    默认 Detector（YoloByteTrackTracker）按以下顺序调用：
        instant = tc.instant_label(frame, bbox)
        label   = tc.smooth_label(track_id, instant)
    自定义实现需同时提供这两个方法。
    """

    def instant_label(self, frame: np.ndarray, bbox: Tuple[int, ...]) -> str:
        """根据当前帧 + 边界框给出瞬时标签。"""
        ...

    def smooth_label(self, track_id: int, instant_label: str) -> str:
        """基于历史瞬时标签，对同一 track_id 做时序平滑后输出最终标签。"""
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


@runtime_checkable
class BallDetectorProto(Protocol):
    """专用球检测器：输入一帧图像，返回最多一个球的检测结果。

    与 Detector 的区别：只检测球，不做跟踪，不返回 track_id；结果用于
    覆盖主 Detector 输出的球检测，提升小目标召回率。

    默认实现: football_log.vision.ball_detector.BallDetector
    替换场景: 不同供应商的专用球模型、InferenceSlicer 分块检测
    """

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """返回检测到的球，通常只有 0 或 1 个元素。track_id 可为 -1。"""
        ...


@runtime_checkable
class PitchCalibratorProto(Protocol):
    """逐帧场地标定器：从图像中估计像素→世界坐标的单应矩阵（H 的生产者）。

    与 WorldProjector 的区别：
    - WorldProjector: 持有已有 H，对每个 bbox 做坐标投影（消费者）
    - PitchCalibratorProto: 从当前帧视觉内容中估计新的 H（生产者）

    两者配合使用：PitchCalibratorProto 每 N 帧更新 H，
    WorldProjector（或 runner 直接）用该 H 做投影。

    默认实现: football_log.vision.pitch_keypoint_detector.PitchKeypointDetector
    替换场景: TVCalib、SoccerNet Camera Calibration baseline、手标关键帧光流传播
    """

    def detect(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
        """从一帧图像估计场地单应矩阵。

        Returns:
            H        : (3,3) float32，像素坐标→世界坐标（米）单应矩阵；
                       关键点不足或 RANSAC 失败时返回 None
            kps_px   : (N,2) float32，检测到的关键点像素坐标
            kp_conf  : (N,)  float32，每个关键点的置信度
        """
        ...
