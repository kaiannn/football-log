"""Tests for the auto-calibration components.

We test the pure logic — homography fit, EMA smoother, JSON loader, projector
adapter — without exercising live optical flow (which requires real video).
The optical flow code path is exercised via the existing pipeline integration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from football_log.world.auto_calibration import (
    AutoCalibrationProjector,
    HomographySmoother,
    Keyframe,
    KeyframeOpticalFlowSource,
    fit_homography,
    load_keyframes_json,
)


# ---- fit_homography ------------------------------------------------------------


def test_fit_homography_recovers_identity_from_4_corner_correspondence():
    """4 corners at the same image and world coordinates → H ≈ identity."""
    pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    H = fit_homography(pts, pts)
    assert H is not None
    # Identity within numerical noise (modulo overall scale)
    H = H / H[2, 2]
    assert np.allclose(H, np.eye(3), atol=1e-3)


def test_fit_homography_recovers_known_scale():
    """Image pixels → world meters with a 10x scale-down."""
    image_pts = np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], dtype=np.float32)
    world_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    H = fit_homography(image_pts, world_pts)
    assert H is not None
    # Apply to a probe pixel — image (500, 500) → world (50, 50)
    p = H @ np.array([500.0, 500.0, 1.0])
    p = p / p[2]
    assert p[0] == pytest.approx(50.0, abs=0.5)
    assert p[1] == pytest.approx(50.0, abs=0.5)


def test_fit_homography_returns_none_for_too_few_points():
    pts = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    assert fit_homography(pts, pts) is None


def test_fit_homography_returns_none_for_mismatched_lengths():
    a = np.zeros((4, 2), dtype=np.float32)
    b = np.zeros((5, 2), dtype=np.float32)
    assert fit_homography(a, b) is None


# ---- HomographySmoother --------------------------------------------------------


def test_smoother_first_call_returns_normalized_input():
    s = HomographySmoother(alpha=0.5)
    H = np.array([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 2.0]])
    out = s.smooth(H)
    # H/H[2,2] = identity
    assert np.allclose(out, np.eye(3))


def test_smoother_ema_blends_inputs():
    s = HomographySmoother(alpha=0.5)
    H1 = np.eye(3)
    H2 = np.array([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0]])
    s.smooth(H1)
    out = s.smooth(H2)
    # Halfway between identity and H2.
    expected = 0.5 * H1 + 0.5 * H2
    assert np.allclose(out, expected, atol=1e-6)


def test_smoother_alpha_one_passes_through():
    s = HomographySmoother(alpha=1.0)
    H1 = np.eye(3)
    H2 = np.array([[3.0, 0, 0], [0, 3.0, 0], [0, 0, 1.0]])
    s.smooth(H1)
    out = s.smooth(H2)
    assert np.allclose(out, H2)


def test_smoother_invalid_alpha_raises():
    with pytest.raises(ValueError):
        HomographySmoother(alpha=0.0)
    with pytest.raises(ValueError):
        HomographySmoother(alpha=1.5)


def test_smoother_reset_clears_state():
    s = HomographySmoother(alpha=0.3)
    s.smooth(np.eye(3))
    s.reset()
    H2 = np.array([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 2.0]])
    out = s.smooth(H2)
    # After reset, the second smooth() call should re-initialize from H2.
    assert np.allclose(out, np.eye(3))  # H2 / H2[2,2] = identity


# ---- load_keyframes_json -------------------------------------------------------


def _write_keyframes_file(tmp_path: Path, keyframes: list) -> Path:
    path = tmp_path / "kf.json"
    payload = {"video_path": "x.mp4", "keyframes": keyframes}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_keyframes_json_basic(tmp_path):
    path = _write_keyframes_file(
        tmp_path,
        [
            {
                "frame_idx": 0,
                "points": [
                    {"name": "a", "image_uv": [0, 0], "world_xy_m": [0, 0]},
                    {"name": "b", "image_uv": [100, 0], "world_xy_m": [10, 0]},
                    {"name": "c", "image_uv": [100, 100], "world_xy_m": [10, 10]},
                    {"name": "d", "image_uv": [0, 100], "world_xy_m": [0, 10]},
                ],
            }
        ],
    )
    kfs = load_keyframes_json(path)
    assert len(kfs) == 1
    assert kfs[0].frame_idx == 0
    assert kfs[0].image_points.shape == (4, 2)
    assert kfs[0].world_points.shape == (4, 2)
    assert kfs[0].names == ["a", "b", "c", "d"]


def test_load_keyframes_json_sorts_by_frame_idx(tmp_path):
    pts = [
        {"image_uv": [0, 0], "world_xy_m": [0, 0]},
        {"image_uv": [100, 0], "world_xy_m": [10, 0]},
        {"image_uv": [100, 100], "world_xy_m": [10, 10]},
        {"image_uv": [0, 100], "world_xy_m": [0, 10]},
    ]
    path = _write_keyframes_file(
        tmp_path,
        [
            {"frame_idx": 200, "points": pts},
            {"frame_idx": 50, "points": pts},
            {"frame_idx": 100, "points": pts},
        ],
    )
    kfs = load_keyframes_json(path)
    assert [k.frame_idx for k in kfs] == [50, 100, 200]


def test_load_keyframes_json_rejects_too_few_points(tmp_path):
    path = _write_keyframes_file(
        tmp_path,
        [
            {
                "frame_idx": 0,
                "points": [{"image_uv": [0, 0], "world_xy_m": [0, 0]}],
            }
        ],
    )
    with pytest.raises(ValueError, match="need >= 4"):
        load_keyframes_json(path)


def test_load_keyframes_json_rejects_empty_keyframes(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"keyframes": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty list"):
        load_keyframes_json(path)


# ---- KeyframeOpticalFlowSource: construction + immediate anchor ----------------


def test_optical_flow_source_anchors_at_first_keyframe(tmp_path):
    """Verify that the source picks up the right keyframe's world points."""
    kf = Keyframe(
        frame_idx=0,
        image_points=np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32),
        world_points=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32),
    )
    src = KeyframeOpticalFlowSource([kf])
    # No prepare_for_frame called yet → no homography
    assert src.current_homography() is None


def test_optical_flow_source_finds_active_keyframe():
    """Internal helper: the latest keyframe at-or-before a given frame_idx."""
    kf0 = Keyframe(0, np.zeros((4, 2), dtype=np.float32), np.zeros((4, 2), dtype=np.float32))
    kf100 = Keyframe(100, np.zeros((4, 2), dtype=np.float32), np.zeros((4, 2), dtype=np.float32))
    kf500 = Keyframe(500, np.zeros((4, 2), dtype=np.float32), np.zeros((4, 2), dtype=np.float32))
    src = KeyframeOpticalFlowSource([kf500, kf0, kf100])  # unsorted input
    assert src._find_active_keyframe(0).frame_idx == 0
    assert src._find_active_keyframe(50).frame_idx == 0
    assert src._find_active_keyframe(100).frame_idx == 100
    assert src._find_active_keyframe(150).frame_idx == 100
    assert src._find_active_keyframe(500).frame_idx == 500
    assert src._find_active_keyframe(1000).frame_idx == 500
    # Before the first keyframe → None
    kf10 = Keyframe(10, np.zeros((4, 2), dtype=np.float32), np.zeros((4, 2), dtype=np.float32))
    src2 = KeyframeOpticalFlowSource([kf10])
    assert src2._find_active_keyframe(0) is None
    assert src2._find_active_keyframe(5) is None


def test_optical_flow_source_constructor_validates_keyframes():
    with pytest.raises(ValueError, match="at least one keyframe"):
        KeyframeOpticalFlowSource([])


# ---- AutoCalibrationProjector --------------------------------------------------


class _StubSource:
    """Stand-in for KeyframeOpticalFlowSource — returns a pre-set H."""

    def __init__(self):
        self._H: np.ndarray | None = None
        self._frames_seen = 0

    def set_homography(self, H: np.ndarray | None) -> None:
        self._H = H

    def prepare_for_frame(self, frame_idx, frame_bgr) -> None:
        self._frames_seen += 1

    def current_homography(self):
        return self._H


def test_projector_returns_none_when_no_homography_yet():
    src = _StubSource()
    proj = AutoCalibrationProjector(src)
    proj.prepare_for_frame(0, np.zeros((100, 100, 3), dtype=np.uint8))
    assert proj.project((10, 10, 50, 80), "Team A") == (None, None)


def test_projector_projects_player_via_foot_anchor():
    src = _StubSource()
    src.set_homography(np.eye(3))
    proj = AutoCalibrationProjector(src)
    proj.prepare_for_frame(0, np.zeros((100, 100, 3), dtype=np.uint8))
    # bbox foot anchor: (x + w/2, y + h) = (35, 90)
    wx, wy = proj.project((10, 10, 50, 80), "Team A")
    assert wx == pytest.approx(35.0)
    assert wy == pytest.approx(90.0)


def test_projector_projects_ball_via_bbox_center():
    src = _StubSource()
    src.set_homography(np.eye(3))
    proj = AutoCalibrationProjector(src)
    proj.prepare_for_frame(0, np.zeros((100, 100, 3), dtype=np.uint8))
    # bbox center: (50, 50)
    wx, wy = proj.project((25, 25, 50, 50), "Ball")
    assert wx == pytest.approx(50.0)
    assert wy == pytest.approx(50.0)


def test_projector_returns_none_for_unknown_label():
    src = _StubSource()
    src.set_homography(np.eye(3))
    proj = AutoCalibrationProjector(src)
    proj.prepare_for_frame(0, np.zeros((100, 100, 3), dtype=np.uint8))
    assert proj.project((0, 0, 10, 10), "Referee") == (None, None)


def test_projector_smoother_is_applied():
    """If a smoother is configured, the H actually used should be smoothed."""
    src = _StubSource()
    smoother = HomographySmoother(alpha=0.5)
    proj = AutoCalibrationProjector(src, smoother)

    src.set_homography(np.eye(3))
    proj.prepare_for_frame(0, np.zeros((100, 100, 3), dtype=np.uint8))
    # Switch to a 2x scale homography
    src.set_homography(np.diag([2.0, 2.0, 1.0]))
    proj.prepare_for_frame(1, np.zeros((100, 100, 3), dtype=np.uint8))
    # bbox (10, 10, 50, 80) — foot at (35, 90)
    # Without smoothing, world=(70, 180); with alpha=0.5, after one smoothing step
    # H is halfway between I and 2I → world should be ~(35*1.5, 90*1.5) = (52.5, 135)
    wx, wy = proj.project((10, 10, 50, 80), "Team A")
    assert wx == pytest.approx(52.5, abs=1.0)
    assert wy == pytest.approx(135.0, abs=1.0)


def test_projector_implements_world_projector_protocol():
    from football_log.protocols import WorldProjector

    src = _StubSource()
    src.set_homography(np.eye(3))
    proj = AutoCalibrationProjector(src)
    assert isinstance(proj, WorldProjector)


def test_projector_counts_frames():
    src = _StubSource()
    src.set_homography(np.eye(3))
    proj = AutoCalibrationProjector(src)
    for i in range(5):
        proj.prepare_for_frame(i, np.zeros((100, 100, 3), dtype=np.uint8))
    assert proj.n_frames_seen == 5
    assert proj.n_frames_with_homography == 5
    src.set_homography(None)
    proj.prepare_for_frame(5, np.zeros((100, 100, 3), dtype=np.uint8))
    assert proj.n_frames_seen == 6
    assert proj.n_frames_with_homography == 5
