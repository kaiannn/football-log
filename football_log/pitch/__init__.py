"""
场地（球场）感知模块。

当前仅保留 PitchObservation 数据结构，供 PitchEstimator 协议的自定义实现使用。
默认的 HSV 草地分割 + Hough 提线实现已移除，推荐使用球场关键点模型（PitchKeypointDetector）。
"""

from football_log.pitch.observation import PitchObservation

__all__ = [
    "PitchObservation",
]
