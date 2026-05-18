"""Synthetic camera tests for PinholeGroundProjector.

We construct a known camera (intrinsics + pose), forward-project a known world point
to a pixel, then ask the projector to recover the world point. The recovered point
must match within numerical precision.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from football_log.world.pinhole_ground import (
    GroundPlane,
    PinholeGroundProjector,
)


def _make_synthetic_camera() -> PinholeGroundProjector:
    """Camera 10 m above the ground at (0, 0, 10), looking straight down (-z)."""
    K = np.array(
        [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    dist = np.zeros((5, 1), dtype=np.float64)
    R_wc = np.array(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
    )
    C = np.array([0.0, 0.0, 10.0], dtype=np.float64)
    t_wc = -R_wc @ C
    plane = GroundPlane(normal=np.array([0.0, 0.0, 1.0]), d=0.0)
    return PinholeGroundProjector(K=K, dist=dist, R_wc=R_wc, t_wc=t_wc, plane=plane)


def _project_world_to_pixel(proj: PinholeGroundProjector, X_world: np.ndarray) -> tuple:
    Xc = proj.R_wc @ X_world + proj.t_wc
    rvec, _ = cv2.Rodrigues(np.eye(3))
    pts, _ = cv2.projectPoints(
        Xc.reshape(1, 1, 3),
        rvec,
        np.zeros(3),
        proj.K,
        proj.dist,
    )
    u, v = float(pts[0, 0, 0]), float(pts[0, 0, 1])
    return u, v


def test_optical_axis_hits_ground_origin():
    proj = _make_synthetic_camera()
    x, y, z = proj.pixel_to_ground_xyz(960.0, 540.0)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)
    assert z == pytest.approx(0.0, abs=1e-9)


def test_round_trip_through_synthetic_camera():
    proj = _make_synthetic_camera()
    rng = np.random.default_rng(42)
    for _ in range(20):
        X_truth = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), 0.0])
        u, v = _project_world_to_pixel(proj, X_truth)
        x, y, z = proj.pixel_to_ground_xyz(u, v)
        assert x == pytest.approx(X_truth[0], abs=1e-4)
        assert y == pytest.approx(X_truth[1], abs=1e-4)
        assert z == pytest.approx(0.0, abs=1e-9)


def test_pixel_to_world_xy_m_returns_none_for_pixel_above_horizon():
    """Tilt the camera so half the image looks at the sky — those pixels can't hit ground."""
    K = np.array(
        [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    dist = np.zeros((5, 1), dtype=np.float64)
    R_wc = np.eye(3)
    t_wc = np.array([0.0, 0.0, -2.0], dtype=np.float64)
    plane = GroundPlane(normal=np.array([0.0, 0.0, 1.0]), d=0.0)
    proj = PinholeGroundProjector(K=K, dist=dist, R_wc=R_wc, t_wc=t_wc, plane=plane)

    x, y = proj.pixel_to_world_xy_m(960.0, 0.0)
    assert x is None and y is None


def test_from_dict_with_R_C_extrinsics():
    K = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
    R = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    C = [0.0, 0.0, 10.0]
    proj = PinholeGroundProjector.from_dict(
        {"K": K, "extrinsics": {"R": R, "C": C}, "ground_plane": {"normal": [0, 0, 1], "d": 0}}
    )
    assert proj.camera_center_world == pytest.approx(np.array(C))


def test_from_dict_default_ground_plane_is_z0():
    K = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
    R = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    t = [0.0, 0.0, 10.0]
    proj = PinholeGroundProjector.from_dict({"K": K, "R": R, "t": t})
    assert proj.plane.normal == pytest.approx(np.array([0.0, 0.0, 1.0]))
    assert proj.plane.d == 0.0
