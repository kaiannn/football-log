"""Tests for the keypoint-based team classifier and pose helpers.

We test the pure logic — torso quadrilateral construction, pixel sampling,
voting, IoU matching — without loading YOLOv8-pose. The pose model itself
is exercised at runtime; here we verify the algorithmic core.
"""

from __future__ import annotations

import numpy as np
import pytest

from football_log.vision.pose import (
    TorsoKeypoints,
    bbox_iou,
    match_pose_to_bbox,
    torso_keypoints_from_bbox_heuristic,
)
from football_log.vision.team_classifier_keypoint import (
    KeypointTeamClassifier,
    _bgr_to_lab,
    _sample_pixels_in_quad,
    _vote_label,
)


# ---- TorsoKeypoints ------------------------------------------------------------


def test_torso_keypoints_complete_when_all_four_present():
    tk = TorsoKeypoints(
        left_shoulder=(10, 20),
        right_shoulder=(40, 20),
        left_hip=(15, 50),
        right_hip=(35, 50),
    )
    assert tk.is_complete


def test_torso_keypoints_incomplete_when_any_missing():
    tk = TorsoKeypoints(
        left_shoulder=(10, 20),
        right_shoulder=None,
        left_hip=(15, 50),
        right_hip=(35, 50),
    )
    assert not tk.is_complete
    assert tk.quadrilateral() is None


def test_quadrilateral_clockwise_order():
    """The returned quad order must be valid for cv2.fillPoly (no self-intersect)."""
    tk = TorsoKeypoints(
        left_shoulder=(10, 20),
        right_shoulder=(40, 20),
        left_hip=(15, 50),
        right_hip=(35, 50),
    )
    quad = tk.quadrilateral()
    assert quad.shape == (4, 2)
    # Order: LS, RS, RH, LH — going clockwise from top-left.
    assert quad.tolist() == [[10, 20], [40, 20], [35, 50], [15, 50]]


# ---- bbox heuristic fallback ---------------------------------------------------


def test_heuristic_torso_inside_bbox():
    tk = torso_keypoints_from_bbox_heuristic((100, 200, 80, 200))
    assert tk.is_complete
    quad = tk.quadrilateral()
    # All four points should be inside the bbox
    for x, y in quad:
        assert 100 <= x <= 180
        assert 200 <= y <= 400


def test_heuristic_torso_proportions():
    """Check we use ~20-50% height, 30-70% width for the torso ROI."""
    tk = torso_keypoints_from_bbox_heuristic((0, 0, 100, 200))
    assert tk.left_shoulder == (30, 40)
    assert tk.right_shoulder == (70, 40)
    assert tk.left_hip == (30, 100)
    assert tk.right_hip == (70, 100)


# ---- bbox IoU and pose matching ------------------------------------------------


def test_bbox_iou_perfect_overlap():
    bbox = (10, 20, 50, 80)
    assert bbox_iou(bbox, bbox) == pytest.approx(1.0)


def test_bbox_iou_no_overlap():
    assert bbox_iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_bbox_iou_half_overlap_horizontally():
    a = (0, 0, 100, 100)
    b = (50, 0, 100, 100)  # shifted right by 50% of width
    # Intersection = 50 × 100 = 5000; union = 2 × 10000 - 5000 = 15000
    assert bbox_iou(a, b) == pytest.approx(5000 / 15000)


def test_match_pose_to_bbox_returns_best_match():
    tk_close = TorsoKeypoints((10, 20), (40, 20), (15, 50), (35, 50))
    tk_far = TorsoKeypoints((100, 200), (140, 200), (105, 250), (135, 250))
    pose_results = [
        ((100, 200, 50, 80), tk_far),       # far from target
        ((10, 20, 40, 60), tk_close),       # closer to target (10, 20, 40, 60)
    ]
    matched = match_pose_to_bbox((10, 20, 40, 60), pose_results, iou_threshold=0.5)
    assert matched is tk_close


def test_match_pose_to_bbox_returns_none_below_threshold():
    tk = TorsoKeypoints((10, 20), (40, 20), (15, 50), (35, 50))
    matched = match_pose_to_bbox(
        (200, 200, 50, 80),
        [((10, 20, 40, 60), tk)],
        iou_threshold=0.3,
    )
    assert matched is None


def test_match_pose_to_bbox_handles_empty_list():
    assert match_pose_to_bbox((0, 0, 10, 10), []) is None


# ---- pixel sampling ------------------------------------------------------------


def test_sample_pixels_in_quad_returns_lab_pixels():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20:50, 30:70] = (50, 100, 200)  # pure-ish color in BGR
    quad = np.array([[30, 20], [70, 20], [70, 50], [30, 50]], dtype=np.int32)

    samples = _sample_pixels_in_quad(frame, quad, n_samples=20)
    assert samples is not None
    assert samples.shape[1] == 3
    assert samples.shape[0] > 0


def test_sample_pixels_in_quad_returns_none_for_offscreen_quad():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    quad = np.array([[200, 200], [300, 200], [300, 300], [200, 300]], dtype=np.int32)
    samples = _sample_pixels_in_quad(frame, quad, n_samples=20)
    assert samples is None


# ---- voting --------------------------------------------------------------------


def test_vote_label_clear_majority_for_team_a():
    samples = np.array([[50, 130, 130], [55, 132, 128], [48, 128, 131]], dtype=np.float32)
    centers = np.array([[50, 130, 130], [200, 130, 130]], dtype=np.float32)
    label = _vote_label(samples, centers, "Team A", "Team B", "Player")
    assert label == "Team A"


def test_vote_label_clear_majority_for_team_b():
    samples = np.array([[200, 130, 130], [195, 128, 132], [205, 132, 128]], dtype=np.float32)
    centers = np.array([[50, 130, 130], [200, 130, 130]], dtype=np.float32)
    label = _vote_label(samples, centers, "Team A", "Team B", "Player")
    assert label == "Team B"


def test_vote_label_unknown_when_split_close():
    """If half the pixels go each way, return 'Player' (ambiguous)."""
    samples = np.array([[50, 130, 130], [200, 130, 130]], dtype=np.float32)
    centers = np.array([[50, 130, 130], [200, 130, 130]], dtype=np.float32)
    label = _vote_label(samples, centers, "Team A", "Team B", "Player")
    assert label == "Player"


def test_vote_label_unknown_when_no_centers():
    samples = np.array([[50, 130, 130]], dtype=np.float32)
    label = _vote_label(samples, None, "Team A", "Team B", "Player")
    assert label == "Player"


# ---- LAB conversion is sane for black/white ------------------------------------


def test_lab_conversion_separates_black_and_white():
    """The whole point of LAB: black and white have very different L values."""
    black = _bgr_to_lab((0, 0, 0))
    white = _bgr_to_lab((255, 255, 255))
    assert black[0] < 30
    assert white[0] > 220
    # Distance in LAB should be large enough for K-Means to separate.
    assert np.linalg.norm(black - white) > 100


# ---- pre-fitted classifier with manual team_colors -----------------------------


def test_classifier_with_manual_team_colors_is_immediately_fitted():
    clf = KeypointTeamClassifier(
        team_colors=[(0, 0, 0), (255, 255, 255)],  # black vs white
    )
    assert clf.is_fitted
    assert clf._centers is not None


def test_classifier_smooth_label_majority_vote():
    clf = KeypointTeamClassifier(team_colors=[(0, 0, 0), (255, 255, 255)])
    track_id = 1
    # Three Team A votes, one Team B → should resolve to Team A
    clf.smooth_label(track_id, "Team A")
    clf.smooth_label(track_id, "Team A")
    clf.smooth_label(track_id, "Team B")
    final = clf.smooth_label(track_id, "Team A")
    assert final == "Team A"


def test_classifier_smooth_label_passes_through_non_player_labels():
    clf = KeypointTeamClassifier(team_colors=[(0, 0, 0), (255, 255, 255)])
    assert clf.smooth_label(99, "Ball") == "Ball"
    assert clf.smooth_label(99, "Referee") == "Referee"


# ---- protocol conformance ------------------------------------------------------


def test_keypoint_classifier_satisfies_team_classifier_protocol():
    from football_log.protocols import TeamClassifierProto

    clf = KeypointTeamClassifier(team_colors=[(0, 0, 0), (255, 255, 255)])
    assert isinstance(clf, TeamClassifierProto)
