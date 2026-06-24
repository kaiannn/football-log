"""Tests for pitch/field_estimator.py — grass detection and temporal smoothing."""

from __future__ import annotations

import numpy as np
import pytest

from football_log.pitch.field_estimator import (
    PitchFieldEstimator,
    TemporalPitchSmoother,
    _confidence,
    _quad_from_contour,
    _sort_quad_tl_tr_br_bl,
)
from football_log.pitch.observation import PitchObservation


# ---- _sort_quad_tl_tr_br_bl ---------------------------------------------------


def test_sort_quad_orders_correctly():
    pts = np.array([[90, 90], [10, 10], [90, 10], [10, 90]], dtype=np.float64)
    ordered = _sort_quad_tl_tr_br_bl(pts)
    assert ordered.shape == (4, 2)
    tl, tr, br, bl = ordered
    assert tl[0] < tr[0]
    assert tl[1] < bl[1]


def test_sort_quad_handles_already_ordered():
    pts = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float64)
    ordered = _sort_quad_tl_tr_br_bl(pts)
    assert ordered[0, 0] < ordered[1, 0]  # TL left of TR


# ---- _confidence ---------------------------------------------------------------


def test_confidence_max_when_all_present():
    c = _confidence(0.35, 25, True)
    assert c == pytest.approx(1.0)


def test_confidence_zero_when_no_grass_no_lines_no_quad():
    c = _confidence(0.0, 0, False)
    assert c == pytest.approx(0.0)


def test_confidence_partial():
    c = _confidence(0.175, 12, False)
    assert 0.3 < c < 0.7


# ---- PitchFieldEstimator.estimate ----------------------------------------------


def test_estimate_black_frame_returns_empty():
    est = PitchFieldEstimator()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    obs = est.estimate(frame)
    assert obs.grass_area_ratio == 0.0
    assert obs.confidence == 0.0
    assert len(obs.line_segments) == 0
    assert obs.field_quad_xy is None


def test_estimate_green_frame_detects_grass():
    est = PitchFieldEstimator()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :, 1] = 150  # green channel → HSV green
    obs = est.estimate(frame)
    assert obs.grass_area_ratio > 0.5
    assert obs.confidence > 0.0


def test_estimate_returns_observation_type():
    est = PitchFieldEstimator()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    obs = est.estimate(frame)
    assert isinstance(obs, PitchObservation)


# ---- TemporalPitchSmoother ----------------------------------------------------


def _make_obs(confidence: float = 0.5, quad: bool = True) -> PitchObservation:
    q = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float64) if quad else None
    return PitchObservation(
        grass_mask=np.zeros((100, 100), dtype=np.uint8),
        grass_area_ratio=0.5,
        line_segments=np.zeros((0, 4), dtype=np.float32),
        field_quad_xy=q,
        confidence=confidence,
        meta={},
    )


def test_smoother_first_frame_returns_copy():
    s = TemporalPitchSmoother(alpha=0.5)
    obs = _make_obs()
    out = s.smooth(obs)
    np.testing.assert_array_almost_equal(out.field_quad_xy, obs.field_quad_xy)


def test_smoother_blends_with_previous():
    s = TemporalPitchSmoother(alpha=0.5)
    obs1 = _make_obs()
    s.smooth(obs1)

    obs2 = _make_obs()
    obs2.field_quad_xy = np.array([[10, 10], [110, 10], [110, 60], [10, 60]], dtype=np.float64)
    out = s.smooth(obs2)

    expected = 0.5 * obs2.field_quad_xy + 0.5 * obs1.field_quad_xy
    np.testing.assert_array_almost_equal(out.field_quad_xy, expected)


def test_smoother_skips_low_confidence():
    s = TemporalPitchSmoother(alpha=0.5)
    s.smooth(_make_obs(confidence=0.8))

    low = _make_obs(confidence=0.1)
    out = s.smooth(low)
    # Low confidence → no smoothing, return original
    assert out.meta.get("temporal_smoothed") is not True


def test_smoother_skips_when_no_quad():
    s = TemporalPitchSmoother(alpha=0.5)
    obs = _make_obs(quad=False)
    out = s.smooth(obs)
    assert out.field_quad_xy is None
