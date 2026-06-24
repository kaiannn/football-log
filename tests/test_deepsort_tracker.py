"""Tests for DeepSortTracker label assignment.

We bypass __init__ to avoid loading ultralytics / deep-sort-realtime;
the pure labeling logic is tested in isolation, mirroring test_tracker_labels.py.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from football_log.vision.deepsort_tracker import DeepSortTracker


class _RecordingTeamClassifier:
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
) -> DeepSortTracker:
    t = DeepSortTracker.__new__(DeepSortTracker)
    t.player_class_ids = tuple(player_ids)
    t.ball_class_ids = tuple(ball_ids)
    t.referee_class_ids = tuple(referee_ids)
    t.team_a_class_ids = ()
    t.team_b_class_ids = ()
    t._team_classifier = tc
    return t


def _frame() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


# ------ _assign_label routing ------

def test_ball_class_returns_ball_no_color():
    t = _make_tracker(ball_ids=(32,))
    label, color = t._assign_label(32, _frame(), (10, 10, 10, 10), track_id=1)
    assert label == "Ball"
    assert color is None


def test_referee_class_returns_referee_skips_team_classifier():
    tc = _RecordingTeamClassifier()
    t = _make_tracker(referee_ids=(2,), tc=tc)
    label, color = t._assign_label(2, _frame(), (10, 10, 20, 40), track_id=5)
    assert label == "Referee"
    assert color is None
    assert tc.instant_calls == []
    assert tc.smooth_calls == []


def test_player_class_without_classifier_returns_player():
    t = _make_tracker(player_ids=(0,), tc=None)
    label, color = t._assign_label(0, _frame(), (5, 5, 10, 20), track_id=3)
    assert label == "Player"
    assert color is not None


def test_player_class_with_classifier_calls_both_methods():
    tc = _RecordingTeamClassifier()
    t = _make_tracker(player_ids=(0,), tc=tc)
    label, _ = t._assign_label(0, _frame(), (5, 5, 10, 20), track_id=7)
    assert label == "Team A"
    assert tc.instant_calls == [(5, 5, 10, 20)]
    assert tc.smooth_calls == [(7, "Team A")]


def test_finetuned_layout_player_ball_referee():
    """Module 1 weights [player=0, ball=1, referee=2]."""
    tc = _RecordingTeamClassifier()
    t = _make_tracker(player_ids=(0,), ball_ids=(1,), referee_ids=(2,), tc=tc)

    assert t._assign_label(0, _frame(), (5, 5, 10, 20), 1)[0] == "Team A"
    assert t._assign_label(1, _frame(), (5, 5, 10, 10), 2)[0] == "Ball"
    assert t._assign_label(2, _frame(), (5, 5, 10, 20), 3)[0] == "Referee"


def test_unknown_class_falls_through_as_player():
    tc = _RecordingTeamClassifier()
    t = _make_tracker(player_ids=(0,), ball_ids=(32,), tc=tc)
    label, _ = t._assign_label(99, _frame(), (5, 5, 10, 20), track_id=1)
    assert label == "Team A"


def test_all_class_ids_property():
    t = _make_tracker(player_ids=(0,), ball_ids=(1,), referee_ids=(2, 3))
    assert t.all_class_ids == [0, 1, 2, 3]


def test_multiple_player_class_ids_all_route_to_player():
    tc = _RecordingTeamClassifier()
    t = _make_tracker(player_ids=(0, 1), ball_ids=(2,), referee_ids=(3,), tc=tc)
    assert t._assign_label(0, _frame(), (5, 5, 10, 20), 1)[0] == "Team A"
    assert t._assign_label(1, _frame(), (5, 5, 10, 20), 2)[0] == "Team A"
    assert len(tc.smooth_calls) == 2


# ------ import-failure path ------

def test_missing_deep_sort_realtime_raises_helpful_error(monkeypatch):
    """If deep-sort-realtime is not installed, __init__ raises RuntimeError."""
    import sys

    class _FakeYOLO:
        def __init__(self, _):
            pass

    import football_log.vision.deepsort_tracker as mod
    orig_yolo = mod.YOLO
    mod.YOLO = _FakeYOLO  # type: ignore

    # Block import of deep_sort_realtime even if installed
    monkeypatch.setitem(sys.modules, "deep_sort_realtime", None)
    monkeypatch.setitem(sys.modules, "deep_sort_realtime.deepsort_tracker", None)

    try:
        with pytest.raises(RuntimeError, match="deep-sort-realtime"):
            DeepSortTracker.__init__(
                DeepSortTracker.__new__(DeepSortTracker),
                model_name="dummy.pt",
            )
    finally:
        mod.YOLO = orig_yolo
