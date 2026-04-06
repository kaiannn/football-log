"""与其它模块衔接的薄工具（ROI 过滤等）。"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from football_log.world.homography import bbox_foot_point


def bbox_anchor_pixel(bbox: tuple, label: str) -> Tuple[int, int]:
    """与 project_foot_to_world 一致：球员脚底中点，球用框中心。"""
    if "Player" in label:
        u, v = bbox_foot_point(bbox)
    else:
        x, y, w, h = bbox
        u, v = float(x + 0.5 * w), float(y + 0.5 * h)
    return int(round(u)), int(round(v))


def filter_objects_in_grass_mask(
    objects: List[Dict[str, Any]],
    grass_mask: np.ndarray,
    *,
    require_nonzero: bool = True,
) -> List[Dict[str, Any]]:
    """
    仅保留锚点落在草地掩膜内的目标（草地为 0 则全部通过）。
    用于降低观众席/替补席误检，可与跟踪结果级联。
    """
    if grass_mask is None or grass_mask.size == 0:
        return objects
    h, w = grass_mask.shape[:2]
    out: List[Dict[str, Any]] = []
    for o in objects:
        u, v = bbox_anchor_pixel(tuple(o["bbox"]), str(o.get("label", "")))
        if u < 0 or v < 0 or u >= w or v >= h:
            if not require_nonzero:
                out.append(o)
            continue
        if grass_mask[v, u] > 127:
            out.append(o)
    return out
