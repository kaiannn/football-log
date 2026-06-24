"""Tests for vision/team_classifier.py — HSV K-Means team classifier."""

from __future__ import annotations

import numpy as np
import pytest

from football_log.vision.team_classifier import TeamClassifier, _bgr_to_hs_center, _patch_to_hs_feature


# ---- _bgr_to_hs_center -------------------------------------------------------


def test_bgr_to_hs_center_white():
    hs = _bgr_to_hs_center((255, 255, 255))
    assert hs.shape == (2,)
    assert hs[1] < 10  # white has low saturation


def test_bgr_to_hs_center_red():
    hs = _bgr_to_hs_center((0, 0, 255))  # pure red in BGR
    assert hs[0] < 20 or hs[0] > 340  # red hue is near 0/360


# ---- _patch_to_hs_feature ----------------------------------------------------


def test_patch_to_hs_feature_returns_none_for_all_green():
    patch = np.zeros((50, 50, 3), dtype=np.uint8)
    patch[:, :, 1] = 200  # green channel only → grass-like
    assert _patch_to_hs_feature(patch) is None


def test_patch_to_hs_feature_returns_hs_for_non_green():
    patch = np.zeros((50, 50, 3), dtype=np.uint8)
    patch[:, :] = (0, 0, 200)  # red in BGR → not grass
    feat = _patch_to_hs_feature(patch)
    assert feat is not None
    assert feat.shape == (2,)


# ---- TeamClassifier ----------------------------------------------------------


def test_manual_team_colors_immediately_fitted():
    clf = TeamClassifier(team_colors=[(0, 0, 255), (255, 0, 0)])
    assert clf.is_fitted


def test_manual_team_colors_classify_red_vs_blue():
    clf = TeamClassifier(team_colors=[(0, 0, 255), (255, 0, 0)])
    # Red patch should classify
    red_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    red_frame[:, :] = (0, 0, 200)
    bbox = (10, 10, 80, 80)
    label = clf.instant_label(red_frame, bbox)
    assert label in ("Team A", "Team B", "Player")


def test_warmup_returns_unknown():
    clf = TeamClassifier(warmup_frames=5, min_samples=10)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :] = (0, 0, 200)
    bbox = (10, 10, 80, 80)
    for _ in range(3):
        label = clf.instant_label(frame, bbox)
        assert label == "Player"
    assert not clf.is_fitted


def test_smooth_label_majority_vote():
    clf = TeamClassifier(team_colors=[(0, 0, 255), (255, 0, 0)])
    tid = 1
    clf.smooth_label(tid, "Team A")
    clf.smooth_label(tid, "Team A")
    clf.smooth_label(tid, "Team B")
    result = clf.smooth_label(tid, "Team A")
    assert result == "Team A"


def test_smooth_label_passthrough_non_player():
    clf = TeamClassifier(team_colors=[(0, 0, 255), (255, 0, 0)])
    assert clf.smooth_label(1, "Ball") == "Ball"
    assert clf.smooth_label(1, "Referee") == "Referee"


def test_classify_ambiguous_returns_unknown():
    clf = TeamClassifier(team_colors=[(0, 0, 255), (255, 0, 0)])
    # Feature equidistant from both centers → ratio ≈ 1.0 > 0.80 → unknown
    clf._centers = np.array([[100.0, 100.0], [200.0, 200.0]], dtype=np.float32)
    feat = np.array([150.0, 150.0], dtype=np.float32)
    label = clf._classify(feat)
    assert label == "Player"  # ratio > 0.80 → unknown


def test_empty_bbox_returns_unknown():
    clf = TeamClassifier(team_colors=[(0, 0, 255), (255, 0, 0)])
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # bbox with zero area
    assert clf.instant_label(frame, (50, 50, 0, 0)) == "Player"


def test_protocol_conformance():
    from football_log.protocols import TeamClassifierProto
    clf = TeamClassifier(team_colors=[(0, 0, 255), (255, 0, 0)])
    assert isinstance(clf, TeamClassifierProto)
