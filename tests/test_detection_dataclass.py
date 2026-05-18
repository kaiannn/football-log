"""Stability tests for the Detection dataclass — to_dict / from_dict shape."""

from __future__ import annotations

from football_log.protocols import Detection


def test_to_dict_has_documented_keys():
    det = Detection(track_id=3, bbox=(10, 20, 30, 40), label="Team A", conf=0.5)
    d = det.to_dict()
    for k in ("id", "bbox", "label", "conf", "box_color", "world_x_m", "world_y_m"):
        assert k in d


def test_to_dict_spreads_extra_at_top_level():
    det = Detection(
        track_id=1,
        bbox=(0, 0, 10, 10),
        label="Player",
        extra={"jersey_number": 9},
    )
    d = det.to_dict()
    assert d["jersey_number"] == 9


def test_from_dict_restores_core_fields():
    det = Detection(
        track_id=7,
        bbox=(1, 2, 3, 4),
        label="Ball",
        conf=0.91,
        box_color=(255, 0, 0),
        world_x_m=12.3,
        world_y_m=-4.5,
    )
    restored = Detection.from_dict(det.to_dict())
    assert restored.track_id == 7
    assert restored.bbox == (1, 2, 3, 4)
    assert restored.label == "Ball"
    assert restored.conf == 0.91
    assert restored.box_color == (255, 0, 0)
    assert restored.world_x_m == 12.3
    assert restored.world_y_m == -4.5


def test_from_dict_handles_missing_optionals_gracefully():
    restored = Detection.from_dict({"id": 1, "bbox": [0, 0, 10, 10], "label": "Player"})
    assert restored.conf == 0.0
    assert restored.box_color is None
    assert restored.world_x_m is None
    assert restored.world_y_m is None


def test_default_runtime_protocol_match_for_team_classifier():
    """TeamClassifierProto contract: instant_label + smooth_label.

    If this regresses (e.g. someone reverts protocols.py), default
    TeamClassifier will silently stop matching the Protocol and plugin users
    will hit confusing errors. Lock it here.
    """
    from football_log.protocols import TeamClassifierProto
    from football_log.vision.team_classifier import TeamClassifier

    assert isinstance(TeamClassifier(), TeamClassifierProto)
