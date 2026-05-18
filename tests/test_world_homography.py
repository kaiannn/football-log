"""Round-trip and known-correspondence tests for Homography."""

from __future__ import annotations

import numpy as np
import pytest

from football_log.world.homography import (
    Homography,
    HomographyProjector,
    bbox_foot_point,
    project_foot_to_world,
)


def test_identity_matrix_is_pixel_equals_world():
    H = Homography(matrix=np.eye(3))
    wx, wy = H.pixel_to_world(123.4, 56.7)
    assert wx == pytest.approx(123.4)
    assert wy == pytest.approx(56.7)


def test_pixel_to_world_round_trip_with_random_homography():
    rng = np.random.default_rng(0)
    M = np.eye(3) + 0.05 * rng.standard_normal((3, 3))
    M[2, 2] = 1.0
    H = Homography(matrix=M)
    for _ in range(20):
        u, v = float(rng.uniform(0, 1920)), float(rng.uniform(0, 1080))
        wx, wy = H.pixel_to_world(u, v)
        u2, v2 = H.world_to_pixel(wx, wy)
        assert u2 == pytest.approx(u, abs=1e-6)
        assert v2 == pytest.approx(v, abs=1e-6)


def test_homography_from_known_correspondences():
    # 1920x1080 image to a 105x68 m pitch, no perspective: pure scale.
    sx = 105.0 / 1920.0
    sy = 68.0 / 1080.0
    M = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    H = Homography(matrix=M)
    assert H.pixel_to_world(0, 0) == (0.0, 0.0)
    assert H.pixel_to_world(1920, 1080) == pytest.approx((105.0, 68.0))
    assert H.pixel_to_world(960, 540) == pytest.approx((52.5, 34.0))


def test_bbox_foot_point_is_bottom_center():
    assert bbox_foot_point((100, 200, 50, 80)) == (125.0, 280.0)


def test_project_foot_to_world_player_uses_foot_anchor():
    H = Homography(matrix=np.eye(3))
    wx, wy = project_foot_to_world((100, 200, 50, 80), "Team A", H)
    assert (wx, wy) == (125.0, 280.0)


def test_project_foot_to_world_ball_uses_bbox_center():
    H = Homography(matrix=np.eye(3))
    wx, wy = project_foot_to_world((100, 200, 50, 80), "Ball", H)
    assert (wx, wy) == (125.0, 240.0)


def test_project_foot_to_world_returns_none_when_h_missing():
    assert project_foot_to_world((0, 0, 10, 10), "Player", None) == (None, None)


def test_project_foot_to_world_returns_none_for_unknown_label():
    H = Homography(matrix=np.eye(3))
    assert project_foot_to_world((0, 0, 10, 10), "Referee", H) == (None, None)


def test_homography_projector_implements_protocol_method():
    H = Homography(matrix=np.eye(3))
    p = HomographyProjector(H)
    assert p.project((100, 200, 50, 80), "Team A") == (125.0, 280.0)
    assert p.project((0, 0, 10, 10), "Referee") == (None, None)
