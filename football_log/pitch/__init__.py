"""
场地（球场）感知：草地分割 + 场内线段检测 + 场区四边形近似。

方法学上与广播足球分析中常见的「平面场模型 + 场线/绿色区域特征」一致，便于部署时
**不依赖 GPU 与额外权重**；与 SoccerNet 系任务中基于**场线/场平面**做相机标定或单应估计
的思路对齐（见模块内 `field_estimator` 文档字符串中的参考文献说明）。

可被多个模块复用：
- 可视化叠加、调试；
- 为单应/外参提供**初始四边形**或 ROI；
- 在草地掩膜内限制检测/跟踪区域以降低误检。
"""

from football_log.pitch.config import PitchFieldConfig
from football_log.pitch.field_estimator import PitchFieldEstimator, TemporalPitchSmoother
from football_log.pitch.integration import bbox_anchor_pixel, filter_objects_in_grass_mask
from football_log.pitch.observation import PitchObservation

__all__ = [
    "PitchFieldConfig",
    "PitchObservation",
    "PitchFieldEstimator",
    "TemporalPitchSmoother",
    "bbox_anchor_pixel",
    "filter_objects_in_grass_mask",
]
