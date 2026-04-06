"""场地估计可调参数。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PitchFieldConfig:
    """草地 HSV、形态学、Canny、Hough 等；可按机位/联赛微调。"""

    hsv_lower: np.ndarray = field(default_factory=lambda: np.array([35, 40, 40], dtype=np.uint8))
    hsv_upper: np.ndarray = field(default_factory=lambda: np.array([85, 255, 255], dtype=np.uint8))

    morph_open_ksize: int = 5
    morph_close_ksize: int = 15
    min_grass_area_ratio: float = 0.05

    canny_low: int = 40
    canny_high: int = 120
    hough_rho: float = 1.0
    hough_theta_deg: float = 1.0
    hough_threshold: int = 40
    hough_min_line_length: int = 30
    hough_max_line_gap: int = 10

    approx_poly_epsilon_ratio: float = 0.02
