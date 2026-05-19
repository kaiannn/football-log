"""Tests for YoloByteTrackTracker label assignment.

We bypass __init__ to avoid loading ultralytics weights; we test the pure
labeling logic which is the part that changed in Module 1 wiring.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from football_log.vision.tracker import YoloByteTrackTracker


class _RecordingTeamClassifier:
    """Stand-in for TeamClassifierProto that records every call it sees."""

    def __init__(self):
        self.instant_calls: List[Tuple[int, ...]] = []
        self.smooth_calls: List[Tuple[int, str]] = []

    def instant_label(self, frame, bbox):
        self.instant_calls.append(tuple(bbox))
        return "Team A"

    def smooth_label(self, track_id, instant_label):
        self.smooth_calls.append((track_id, instant_label))
        return "Team A"


def _make_tracker(
    player_ids=(0,),
    ball_ids=(32,),
    referee_ids=(),
    tc=None,
) -> YoloByteTrackTracker:
    t = YoloByteTrackTracker.__new__(YoloByteTrackTracker)
    t.player_class_ids = tuple(player_ids)
    t.ball_class_ids = tuple(ball_ids)
    t.referee_class_ids = tuple(referee_ids)
    t.team_a_class_ids = ()
    t.team_b_class_ids = ()
    t._team_classifier = tc
    return t


def _frame() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_coco_defaults_route_class_0_to_player(monkeypatch):
    tc = _RecordingTeamClassifier()
    t = _make_tracker(tc=tc)

    label, _ = t._assign_label(0, _frame(), (10, 10, 30, 50), track_id=1)
    assert label == "Team A"
    assert tc.smooth_calls == [(1, "Team A")]


def test_coco_defaults_route_class_32_to_ball():
    tc = _RecordingTeamClassifier()
    t = _make_tracker(tc=tc)

    label, color = t._assign_label(32, _frame(), (40, 40, 10, 10), track_id=2)
    assert label == "Ball"
    assert color is None
    assert tc.smooth_calls == []


def test_referee_class_routes_to_referee_label_and_skips_team_classifier():
    tc = _RecordingTeamClassifier()
    t = _make_tracker(player_ids=(0,), ball_ids=(1,), referee_ids=(2,), tc=tc)

    label, color = t._assign_label(2, _frame(), (10, 10, 30, 50), track_id=7)
    assert label == "Referee"
    assert color is None
    assert tc.instant_calls == []
    assert tc.smooth_calls == []


def test_finetuned_class_layout_player_ball_referee():
    """Module 1 self-trained weights with class order [player, ball, referee]."""
    tc = _RecordingTeamClassifier()
    t = _make_tracker(player_ids=(0,), ball_ids=(1,), referee_ids=(2,), tc=tc)

    assert t._assign_label(0, _frame(), (5, 5, 10, 20), 1)[0] == "Team A"
    assert t._assign_label(1, _frame(), (5, 5, 10, 10), 2)[0] == "Ball"
    assert t._assign_label(2, _frame(), (5, 5, 10, 20), 3)[0] == "Referee"


def test_multiple_player_class_ids_all_route_to_player():
    """Some datasets keep team A and team B as distinct detector classes."""
    tc = _RecordingTeamClassifier()
    t = _make_tracker(player_ids=(0, 1), ball_ids=(2,), referee_ids=(3,), tc=tc)

    assert t._assign_label(0, _frame(), (5, 5, 10, 20), 1)[0] == "Team A"
    assert t._assign_label(1, _frame(), (5, 5, 10, 20), 2)[0] == "Team A"
    assert tc.smooth_calls == [(1, "Team A"), (2, "Team A")]


def test_player_label_when_team_classifier_missing():
    t = _make_tracker(tc=None)
    label, color = t._assign_label(0, _frame(), (5, 5, 10, 20), track_id=1)
    assert label == "Player"
    # color is computed via get_dominant_color on a black frame; should at least exist
    assert color is not None


def test_all_class_ids_property_returns_union():
    t = _make_tracker(player_ids=(0,), ball_ids=(1,), referee_ids=(2, 3))
    assert t.all_class_ids == [0, 1, 2, 3]


def test_unknown_class_falls_through_as_player():
    """Class IDs not in any named set are treated as players (defensive default)."""
    tc = _RecordingTeamClassifier()
    t = _make_tracker(player_ids=(0,), ball_ids=(32,), referee_ids=(), tc=tc)

    label, _ = t._assign_label(99, _frame(), (5, 5, 10, 20), track_id=1)
    assert label == "Team A"


def test_team_a_class_id_returns_team_a_directly():
    tc = _RecordingTeamClassifier()
    t = _make_tracker(player_ids=(), ball_ids=(5,), referee_ids=(4,), tc=tc)
    t.team_a_class_ids = (0, 2)
    t.team_b_class_ids = (1, 3)

    label, color = t._assign_label(0, _frame(), (5, 5, 10, 20), track_id=1)
    assert label == "Team A"
    assert tc.instant_calls == []   # classifier not called

    label2, _ = t._assign_label(2, _frame(), (5, 5, 10, 20), track_id=2)
    assert label2 == "Team A"  # goalkeeper_a also → Team A


def test_team_b_class_id_returns_team_b_directly():
    tc = _RecordingTeamClassifier()
    t = _make_tracker(player_ids=(), ball_ids=(5,), referee_ids=(4,), tc=tc)
    t.team_a_class_ids = (0, 2)
    t.team_b_class_ids = (1, 3)

    label, _ = t._assign_label(1, _frame(), (5, 5, 10, 20), track_id=3)
    assert label == "Team B"

    label2, _ = t._assign_label(3, _frame(), (5, 5, 10, 20), track_id=4)
    assert label2 == "Team B"  # goalkeeper_b also → Team B


def test_6class_full_layout_all_six_classes():
    """Full Module 3B class layout: 0=team_a, 1=team_b, 2=gk_a, 3=gk_b, 4=ref, 5=ball."""
    t = _make_tracker(player_ids=(), ball_ids=(5,), referee_ids=(4,), tc=None)
    t.team_a_class_ids = (0, 2)
    t.team_b_class_ids = (1, 3)

    assert t._assign_label(0, _frame(), (5, 5, 10, 20), 1)[0] == "Team A"
    assert t._assign_label(1, _frame(), (5, 5, 10, 20), 2)[0] == "Team B"
    assert t._assign_label(2, _frame(), (5, 5, 10, 20), 3)[0] == "Team A"
    assert t._assign_label(3, _frame(), (5, 5, 10, 20), 4)[0] == "Team B"
    assert t._assign_label(4, _frame(), (5, 5, 10, 20), 5)[0] == "Referee"
    assert t._assign_label(5, _frame(), (5, 5, 10, 10), 6)[0] == "Ball"
