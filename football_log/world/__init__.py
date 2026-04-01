"""球场世界坐标系、尺寸模型与单应变换（标定后像素 ↔ 米）。"""

from football_log.world.pitch_model import PitchSpec
from football_log.world.homography import Homography, bbox_foot_point, project_foot_to_world

__all__ = ["PitchSpec", "Homography", "bbox_foot_point", "project_foot_to_world"]
