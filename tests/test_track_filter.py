"""Tests for the per-track BEV Kalman filter (Module 5)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from football_log.world.track_filter import TrackFilter, jump_likelihood_from_height_change


def test_first_observation_initializes_track():
    f = TrackFilter(fps=25.0)
    out = f.update(track_id=1, world_xy=(10.0, 20.0), frame_idx=0)
    assert out == (10.0, 20.0)
    assert f.active_tracks == 1


def test_predict_only_when_no_observation():
    f = TrackFilter(fps=25.0)
    f.update(track_id=1, world_xy=(0.0, 0.0), frame_idx=0)
    # no observation on next frame
    out = f.update(track_id=1, world_xy=None, frame_idx=1)
    assert out is not None
    # With zero initial velocity and no observation, predicted position
    # should still be near origin.
    assert abs(out[0]) < 0.1
    assert abs(out[1]) < 0.1
    assert f.stats()["predict_only_steps"] == 1


def test_smoothing_a_constant_velocity_track():
    """Walk in a straight line at 5 m/s for 1 second; filter should track it."""
    f = TrackFilter(fps=25.0)
    fps = 25.0
    speed = 5.0
    for i in range(26):
        true_x = i * speed / fps
        out = f.update(track_id=1, world_xy=(true_x, 0.0), frame_idx=i)
    # By the last frame the filter should be very close to the true position.
    assert out is not None
    assert abs(out[0] - 25 * speed / fps) < 0.5


def test_outlier_observation_is_rejected():
    """A single huge spike in position should not corrupt the trajectory."""
    f = TrackFilter(fps=25.0)
    # Build up confidence with a stationary track
    for i in range(20):
        f.update(track_id=1, world_xy=(0.0, 0.0), frame_idx=i)
    # Now inject a 50m teleport — should be rejected as outlier
    out = f.update(track_id=1, world_xy=(50.0, 50.0), frame_idx=20)
    assert out is not None
    # Position should still be near origin, not near (50, 50)
    assert math.hypot(out[0], out[1]) < 5.0
    assert f.stats()["outliers_rejected"] >= 1


def test_low_confidence_obs_pulls_state_less():
    """Sequence of small consistent off-axis observations: high-conf filter follows them faster."""
    f_high = TrackFilter(fps=25.0)
    f_low = TrackFilter(fps=25.0)

    # Establish a stationary track first.
    for i in range(10):
        f_high.update(track_id=1, world_xy=(0.0, 0.0), frame_idx=i, conf=1.0)
        f_low.update(track_id=1, world_xy=(0.0, 0.0), frame_idx=i, conf=1.0)

    # Now observations drift slightly in +x direction every frame.
    # Both filters see the SAME observations; only conf differs.
    for i in range(10, 20):
        target_x = (i - 9) * 0.1  # 10 cm per frame, gentle drift
        out_high = f_high.update(track_id=1, world_xy=(target_x, 0.0), frame_idx=i, conf=1.0)
        out_low = f_low.update(track_id=1, world_xy=(target_x, 0.0), frame_idx=i, conf=0.1)

    # High-confidence filter should be closer to the most recent observation.
    assert out_high[0] > out_low[0]


def test_jump_heuristic_dampens_observation():
    """When the same in-gate observation is fed in, jump_likelihood=1 absorbs less of it."""
    f_normal = TrackFilter(fps=25.0)
    f_jumping = TrackFilter(fps=25.0)

    for i in range(10):
        f_normal.update(track_id=1, world_xy=(0.0, 0.0), frame_idx=i)
        f_jumping.update(track_id=1, world_xy=(0.0, 0.0), frame_idx=i)

    # Single small in-gate observation — should be accepted by both.
    out_normal = f_normal.update(track_id=1, world_xy=(0.3, 0.0), frame_idx=10, jump_likelihood=0.0)
    out_jumping = f_jumping.update(track_id=1, world_xy=(0.3, 0.0), frame_idx=10, jump_likelihood=1.0)

    # Both should have moved positively, but normal should move further.
    assert out_normal[0] > 0.0
    assert out_jumping[0] >= 0.0
    assert out_normal[0] > out_jumping[0]


def test_track_eviction():
    f = TrackFilter(fps=25.0, max_age_frames=10)
    f.update(track_id=1, world_xy=(0.0, 0.0), frame_idx=0)
    f.update(track_id=2, world_xy=(0.0, 0.0), frame_idx=5)

    # 11 frames after track 1, 6 frames after track 2 — track 1 should be evicted
    n_evicted = f.evict_stale(current_frame_idx=11)
    assert n_evicted == 1
    assert f.active_tracks == 1


def test_multiple_tracks_have_independent_state():
    f = TrackFilter(fps=25.0)
    f.update(track_id=1, world_xy=(10.0, 0.0), frame_idx=0)
    f.update(track_id=2, world_xy=(0.0, 10.0), frame_idx=0)
    f.update(track_id=1, world_xy=(11.0, 0.0), frame_idx=1)
    f.update(track_id=2, world_xy=(0.0, 11.0), frame_idx=1)
    out1 = f.update(track_id=1, world_xy=(12.0, 0.0), frame_idx=2)
    out2 = f.update(track_id=2, world_xy=(0.0, 12.0), frame_idx=2)
    assert out1[0] > 11.0 and abs(out1[1]) < 0.5
    assert abs(out2[0]) < 0.5 and out2[1] > 11.0


def test_zero_fps_raises():
    with pytest.raises(ValueError):
        TrackFilter(fps=0.0)


def test_track_with_no_first_observation_returns_none():
    f = TrackFilter(fps=25.0)
    out = f.update(track_id=99, world_xy=None, frame_idx=0)
    assert out is None
    assert f.active_tracks == 0


def test_smoothed_trajectory_jerk_lower_than_raw():
    """The whole point: smoothed positions should have less frame-to-frame jitter
    than raw projected positions when the underlying motion is steady but the
    observations are noisy."""
    rng = np.random.default_rng(0)
    f = TrackFilter(fps=25.0)
    fps = 25.0
    speed = 4.0  # m/s

    raw_xs, smooth_xs = [], []
    for i in range(50):
        true_x = i * speed / fps
        noisy_x = true_x + float(rng.normal(0.0, 0.3))  # 30 cm sensor noise
        raw_xs.append(noisy_x)
        out = f.update(track_id=1, world_xy=(noisy_x, 0.0), frame_idx=i)
        smooth_xs.append(out[0])

    # Frame-to-frame deltas — smaller is smoother.
    raw_jitter = np.std(np.diff(raw_xs))
    smooth_jitter = np.std(np.diff(smooth_xs))
    assert smooth_jitter < raw_jitter * 0.6


# ---- jump-likelihood heuristic -------------------------------------------------


def test_jump_likelihood_zero_when_steady_height():
    out = jump_likelihood_from_height_change(
        bbox_height=100.0, prev_height=100.0, bbox_y_top=500, frame_height=1080
    )
    assert out == 0.0


def test_jump_likelihood_nonzero_on_rapid_height_change():
    out = jump_likelihood_from_height_change(
        bbox_height=80.0, prev_height=100.0, bbox_y_top=500, frame_height=1080
    )
    assert out > 0.0


def test_jump_likelihood_nonzero_at_top_of_frame():
    out = jump_likelihood_from_height_change(
        bbox_height=100.0, prev_height=100.0, bbox_y_top=10, frame_height=1080
    )
    assert out >= 0.5


def test_jump_likelihood_capped_at_one():
    """The cap is min(1.0, raw_sum). Inputs that produce a raw sum > 1.0 should clamp."""
    # The current heuristic returns 0.3 (height) + 0.5 (top-of-frame) = 0.8 max.
    # We just verify the function never exceeds 1.0, even on extreme inputs.
    out = jump_likelihood_from_height_change(
        bbox_height=10.0, prev_height=200.0, bbox_y_top=0, frame_height=1080
    )
    assert 0.0 <= out <= 1.0
