"""Shared label-resolution utilities used by multiple vision modules."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from football_log.protocols import TeamClassifierProto

# ---------------------------------------------------------------------------
# get_dominant_color — jersey colour extraction (moved from team_classifier)
# ---------------------------------------------------------------------------

_GRASS_H_LOW = 35
_GRASS_H_HIGH = 85
_GRASS_S_MIN = 40


def _extract_jersey_patch(frame: np.ndarray, bbox: Tuple[int, ...]) -> Optional[np.ndarray]:
    x, y, w, h = bbox
    y_top = y + int(h * 0.20)
    y_bot = y + int(h * 0.50)
    x_left = x + int(w * 0.15)
    x_right = x + int(w * 0.85)
    fh, fw = frame.shape[:2]
    y_top = max(0, y_top)
    y_bot = min(fh, y_bot)
    x_left = max(0, x_left)
    x_right = min(fw, x_right)
    if y_bot <= y_top or x_right <= x_left:
        return None
    patch = frame[y_top:y_bot, x_left:x_right]
    if patch.size == 0:
        return None
    return patch


def _grass_mask(patch_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    return (h >= _GRASS_H_LOW) & (h <= _GRASS_H_HIGH) & (s >= _GRASS_S_MIN)


def get_dominant_color(frame: np.ndarray, bbox: Tuple[int, ...]) -> Optional[Tuple[int, int, int]]:
    patch = _extract_jersey_patch(frame, bbox)
    if patch is None:
        return None
    mask = ~_grass_mask(patch)
    if mask.sum() < 10:
        return None
    pixels = patch[mask]
    median_bgr = np.median(pixels, axis=0).astype(int)
    return (int(median_bgr[0]), int(median_bgr[1]), int(median_bgr[2]))


# ---------------------------------------------------------------------------
# smooth_label — temporal voting for team assignment
# ---------------------------------------------------------------------------

def smooth_label(
    track_id: int,
    instant_label: str,
    history: Dict[int, Deque[str]],
    team_a: str = "Team A",
    team_b: str = "Team B",
    unknown: str = "Player",
) -> str:
    """Majority-vote smoothing over recent instant labels for one track.

    Used by both TeamClassifier (HSV) and KeypointTeamClassifier (LAB).
    """
    if "Player" not in instant_label and "Team" not in instant_label:
        return instant_label
    h = history[track_id]
    h.append(instant_label)
    a_votes = sum(1 for x in h if x == team_a)
    b_votes = sum(1 for x in h if x == team_b)
    if a_votes > b_votes:
        return team_a
    if b_votes > a_votes:
        return team_b
    if a_votes == 0 and b_votes == 0:
        return unknown
    return instant_label


# ---------------------------------------------------------------------------
# assign_label — class-id → human-readable label routing
# ---------------------------------------------------------------------------

def assign_label(
    obj_cls: int,
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    track_id: int,
    ball_class_ids: Tuple[int, ...],
    referee_class_ids: Tuple[int, ...],
    team_a_class_ids: Tuple[int, ...],
    team_b_class_ids: Tuple[int, ...],
    team_classifier: Optional["TeamClassifierProto"] = None,
) -> Tuple[str, Optional[Tuple[int, int, int]]]:
    """Route a detection class ID to a label + dominant color.

    Shared by YoloByteTrackTracker and DeepSortTracker.
    """
    if obj_cls in ball_class_ids:
        return "Ball", None
    if obj_cls in referee_class_ids:
        return "Referee", None
    if obj_cls in team_a_class_ids:
        return "Team A", get_dominant_color(frame, bbox)
    if obj_cls in team_b_class_ids:
        return "Team B", get_dominant_color(frame, bbox)

    if team_classifier is not None:
        instant = team_classifier.instant_label(frame, bbox)
        label = team_classifier.smooth_label(track_id, instant)
    else:
        label = "Player"
    return label, get_dominant_color(frame, bbox)


# ---------------------------------------------------------------------------
# parse_team_colors — CLI / Web shared parser
# ---------------------------------------------------------------------------

def parse_team_colors(raw: Optional[str]) -> Optional[List[Tuple[int, int, int]]]:
    """Parse 'B,G,R;B,G,R' string into two BGR tuples. Returns None on failure."""
    if not raw or not raw.strip():
        return None
    parts = raw.strip().split(";")
    if len(parts) < 2:
        return None
    colors = []
    for p in parts[:2]:
        try:
            nums = [int(x.strip()) for x in p.split(",")]
        except ValueError:
            return None
        if len(nums) != 3:
            return None
        colors.append(tuple(nums))
    return colors


# ---------------------------------------------------------------------------
# coerce_ids / all_class_ids / bbox_too_small — shared tracker helpers
# ---------------------------------------------------------------------------

def coerce_ids(value: Optional[Iterable[int] | int]) -> Tuple[int, ...]:
    """Normalize optional int/iterable to a tuple of ints."""
    if value is None:
        return ()
    if isinstance(value, int):
        return (value,)
    return tuple(int(v) for v in value)


def all_class_ids_from(
    player: Tuple[int, ...],
    ball: Tuple[int, ...],
    referee: Tuple[int, ...],
    team_a: Tuple[int, ...],
    team_b: Tuple[int, ...],
) -> List[int]:
    """Concatenate all class ID tuples into a single sorted-deduped list."""
    return sorted(set(player + ball + referee + team_a + team_b))


def bbox_too_small(bbox: Tuple[int, int, int, int], min_side: int = 2) -> bool:
    """Return True if bbox width or height is below min_side pixels."""
    return bbox[2] < min_side or bbox[3] < min_side


class BaseDetector:
    """Tracker 公共逻辑 mixin：class ID 管理、team classifier 注入、标签分配。"""

    def __init__(
        self,
        player_class_ids: Tuple[int, ...],
        ball_class_ids: Tuple[int, ...],
        referee_class_ids: Tuple[int, ...],
        team_a_class_ids: Tuple[int, ...],
        team_b_class_ids: Tuple[int, ...],
    ) -> None:
        self.player_class_ids = player_class_ids
        self.ball_class_ids = ball_class_ids
        self.referee_class_ids = referee_class_ids
        self.team_a_class_ids = team_a_class_ids
        self.team_b_class_ids = team_b_class_ids
        self._team_classifier: Optional["TeamClassifierProto"] = None
        self._all_class_ids: List[int] = all_class_ids_from(
            player_class_ids, ball_class_ids,
            referee_class_ids, team_a_class_ids, team_b_class_ids,
        )

    def set_team_classifier(self, tc: "TeamClassifierProto") -> None:
        self._team_classifier = tc

    @property
    def all_class_ids(self) -> List[int]:
        return self._all_class_ids

    def _assign_label(
        self,
        obj_cls: int,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        track_id: int,
    ) -> Tuple[str, Optional[Tuple[int, int, int]]]:
        return assign_label(
            obj_cls, frame, bbox, track_id,
            self.ball_class_ids, self.referee_class_ids,
            self.team_a_class_ids, self.team_b_class_ids,
            self._team_classifier,
        )


# ---------------------------------------------------------------------------
# bbox_foot_point / is_person_label / bbox_anchor — shared anchor geometry
# ---------------------------------------------------------------------------


def bbox_foot_point(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """从 bbox (x, y, w, h) 取「脚底」近似：底边中点。"""
    x, y, w, h = bbox
    return float(x + 0.5 * w), float(y + h)


def is_person_label(label: str) -> bool:
    return "Player" in label or "Team" in label


def bbox_anchor(bbox: Tuple[float, float, float, float], label: str) -> Tuple[float, float]:
    """统一锚点约定：球员用脚底中点，球用 bbox 中心。"""
    if is_person_label(label):
        return bbox_foot_point(bbox)
    return float(bbox[0] + 0.5 * bbox[2]), float(bbox[1] + 0.5 * bbox[3])


# ---------------------------------------------------------------------------
# sort_quad_tl_tr_br_bl — shared quadrilateral ordering
# ---------------------------------------------------------------------------


def try_fit_kmeans(
    samples: List[np.ndarray],
    min_samples: int,
    distance_threshold: float,
) -> Optional[np.ndarray]:
    """尝试用 K-Means 聚类两类颜色特征。成功返回 centers (2, C)，失败返回 None。"""
    if len(samples) < min_samples:
        return None
    data = np.vstack(samples).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, _, centers = cv2.kmeans(data, 2, None, criteria, 20, cv2.KMEANS_PP_CENTERS)
    dist = float(np.linalg.norm(centers[0] - centers[1]))
    if dist < distance_threshold:
        return None
    return centers


def sort_quad_tl_tr_br_bl(points: np.ndarray) -> np.ndarray:
    """四点排序：左上→右上→右下→左下（透视四边形常用 sum/diff 规则）。"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) != 4:
        return pts
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float64)
