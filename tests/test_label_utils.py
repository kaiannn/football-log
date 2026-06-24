"""Tests for football_log.vision.label_utils — shared label utilities."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pytest

from football_log.vision.label_utils import (
    assign_label,
    get_dominant_color,
    parse_team_colors,
    smooth_label,
)


# ---- parse_team_colors --------------------------------------------------------


def test_parse_valid_colors():
    assert parse_team_colors("255,255,255;0,255,255") == [(255, 255, 255), (0, 255, 255)]


def test_parse_with_spaces():
    assert parse_team_colors(" 100, 50, 200 ; 10, 20, 30 ") == [(100, 50, 200), (10, 20, 30)]


def test_parse_returns_none_for_empty():
    assert parse_team_colors(None) is None
    assert parse_team_colors("") is None
    assert parse_team_colors("   ") is None


def test_parse_returns_none_for_single_color():
    assert parse_team_colors("255,255,255") is None


def test_parse_returns_none_for_bad_int():
    assert parse_team_colors("abc,0,0;0,0,0") is None


def test_parse_returns_none_for_wrong_component_count():
    assert parse_team_colors("255,255;0,0,0") is None


# ---- smooth_label -------------------------------------------------------------


def _make_history(maxlen: int = 12) -> Dict[int, Deque[str]]:
    from collections import defaultdict
    return defaultdict(lambda: deque(maxlen=maxlen))


def test_smooth_label_majority_team_a():
    h = _make_history()
    smooth_label(1, "Team A", h)
    smooth_label(1, "Team A", h)
    smooth_label(1, "Team B", h)
    result = smooth_label(1, "Team A", h)
    assert result == "Team A"


def test_smooth_label_majority_team_b():
    h = _make_history()
    smooth_label(1, "Team B", h)
    smooth_label(1, "Team B", h)
    result = smooth_label(1, "Team A", h)
    assert result == "Team B"


def test_smooth_label_passthrough_non_player():
    h = _make_history()
    assert smooth_label(1, "Ball", h) == "Ball"
    assert smooth_label(1, "Referee", h) == "Referee"


def test_smooth_label_unknown_when_no_team_votes():
    h = _make_history()
    result = smooth_label(1, "Player", h)
    assert result == "Player"


def test_smooth_label_tie_returns_instant():
    h = _make_history()
    smooth_label(1, "Team A", h)
    result = smooth_label(1, "Team B", h)
    assert result == "Team B"


# ---- get_dominant_color -------------------------------------------------------


def test_get_dominant_color_returns_none_for_empty_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # bbox outside frame
    assert get_dominant_color(frame, (200, 200, 10, 10)) is None


def test_get_dominant_color_extracts_median_bgr():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Fill the jersey region (20-50% height, 15-85% width) with a known color
    frame[20:50, 15:85] = (50, 100, 200)  # BGR
    result = get_dominant_color(frame, (0, 0, 100, 100))
    assert result is not None
    b, g, r = result
    assert abs(b - 50) < 5
    assert abs(g - 100) < 5
    assert abs(r - 200) < 5


# ---- assign_label -------------------------------------------------------------


class _StubTeamClassifier:
    def __init__(self, label: str = "Team A"):
        self._label = label
        self.instant_calls: list = []
        self.smooth_calls: list = []

    def instant_label(self, frame, bbox):
        self.instant_calls.append(bbox)
        return self._label

    def smooth_label(self, track_id, instant_label):
        self.smooth_calls.append((track_id, instant_label))
        return self._label


def _frame() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_assign_label_ball():
    label, color = assign_label(
        1, _frame(), (10, 10, 10, 10), 1,
        ball_class_ids=(1,), referee_class_ids=(),
        team_a_class_ids=(), team_b_class_ids=(),
    )
    assert label == "Ball"
    assert color is None


def test_assign_label_referee():
    label, color = assign_label(
        2, _frame(), (10, 10, 10, 10), 1,
        ball_class_ids=(1,), referee_class_ids=(2,),
        team_a_class_ids=(), team_b_class_ids=(),
    )
    assert label == "Referee"
    assert color is None


def test_assign_label_team_a_class_id():
    label, _ = assign_label(
        0, _frame(), (10, 10, 10, 10), 1,
        ball_class_ids=(5,), referee_class_ids=(4,),
        team_a_class_ids=(0,), team_b_class_ids=(1,),
    )
    assert label == "Team A"


def test_assign_label_team_b_class_id():
    label, _ = assign_label(
        1, _frame(), (10, 10, 10, 10), 1,
        ball_class_ids=(5,), referee_class_ids=(4,),
        team_a_class_ids=(0,), team_b_class_ids=(1,),
    )
    assert label == "Team B"


def test_assign_label_delegates_to_team_classifier():
    tc = _StubTeamClassifier("Team B")
    label, _ = assign_label(
        99, _frame(), (10, 10, 30, 50), 7,
        ball_class_ids=(1,), referee_class_ids=(2,),
        team_a_class_ids=(), team_b_class_ids=(),
        team_classifier=tc,
    )
    assert label == "Team B"
    assert len(tc.instant_calls) == 1
    assert tc.smooth_calls == [(7, "Team B")]


def test_assign_label_defaults_to_player_without_classifier():
    label, _ = assign_label(
        99, _frame(), (10, 10, 30, 50), 1,
        ball_class_ids=(1,), referee_class_ids=(2,),
        team_a_class_ids=(), team_b_class_ids=(),
        team_classifier=None,
    )
    assert label == "Player"
