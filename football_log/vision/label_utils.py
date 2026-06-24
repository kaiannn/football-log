"""Shared label-resolution utilities used by multiple vision modules."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

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
    team_classifier: Optional[Any] = None,
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
