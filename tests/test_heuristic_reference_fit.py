"""Tests for world/heuristic_reference_fit.py — coordinate-descent scale fitting."""

from __future__ import annotations

import numpy as np
import pytest

from football_log.world.heuristic_reference_fit import (
    ReferenceRectangle,
    _loss_from_residual,
    _mean_edge_lengths,
    _order_tl_tr_br_bl,
    apply_scales_to_homography,
    fit_reference_scales,
    fit_reference_scales_multi,
)


# ---- _order_tl_tr_br_bl ------------------------------------------------------


def test_order_quadrilateral_correctly():
    pts = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float64)
    ordered = _order_tl_tr_br_bl(pts)
    tl, tr, br, bl = ordered
    assert tl[0] < tr[0]  # TL left of TR
    assert tl[1] < bl[1]  # TL above BL
    assert br[0] > bl[0]  # BR right of BL


# ---- _mean_edge_lengths -------------------------------------------------------


def test_mean_edge_lengths_square():
    quad = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    width, length = _mean_edge_lengths(quad)
    assert abs(width - 10.0) < 1e-6
    assert abs(length - 10.0) < 1e-6


def test_mean_edge_lengths_rectangle():
    quad = np.array([[0, 0], [20, 0], [20, 10], [0, 10]], dtype=np.float64)
    width, length = _mean_edge_lengths(quad)
    assert abs(width - 20.0) < 1e-6
    assert abs(length - 10.0) < 1e-6


# ---- _loss_from_residual ------------------------------------------------------


def test_loss_none_is_squared():
    assert _loss_from_residual(3.0, robust_loss="none", delta=1.0) == 9.0


def test_loss_huber_quadratic_near_zero():
    assert _loss_from_residual(0.1, robust_loss="huber", delta=1.0) == pytest.approx(0.01)


def test_loss_huber_linear_far_away():
    assert _loss_from_residual(3.0, robust_loss="huber", delta=1.0) == pytest.approx(2.0 * 1.0 * 3.0 - 1.0)


def test_loss_cauchy_bounded():
    val = _loss_from_residual(100.0, robust_loss="cauchy", delta=1.0)
    assert val < 100.0 ** 2  # much less than squared loss


def test_loss_zero_residual():
    assert _loss_from_residual(0.0, robust_loss="huber", delta=1.0) == 0.0
    assert _loss_from_residual(0.0, robust_loss="cauchy", delta=1.0) == 0.0


# ---- fit_reference_scales (single ref) ----------------------------------------


def test_fit_identity_projector_recovers_scale():
    """With an identity projector (pixel == world), scale should converge near 1.0."""
    def identity_projector(u, v):
        return (u, v)

    ref = ReferenceRectangle(
        image_points_xy=np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float64),
        width_m=100.0,
        length_m=50.0,
    )
    result = fit_reference_scales(identity_projector, ref, steps=15)
    assert abs(result.scale_x - 1.0) < 0.1
    assert abs(result.scale_y - 1.0) < 0.1
    assert result.rmse_m < 5.0


def test_fit_with_scaled_projector():
    """If projector returns half the world coords, scale should compensate."""
    def half_projector(u, v):
        return (u * 0.5, v * 0.5)

    ref = ReferenceRectangle(
        image_points_xy=np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float64),
        width_m=100.0,
        length_m=50.0,
    )
    result = fit_reference_scales(half_projector, ref, steps=20)
    assert abs(result.scale_x - 2.0) < 0.3
    assert abs(result.scale_y - 2.0) < 0.3


# ---- fit_reference_scales_multi -----------------------------------------------


def test_multi_fit_empty_refs_raises():
    def proj(u, v):
        return (u, v)
    with pytest.raises(ValueError, match="must not be empty"):
        fit_reference_scales_multi(proj, [])


def test_multi_fit_with_two_refs():
    def identity(u, v):
        return (u, v)

    ref1 = ReferenceRectangle(
        image_points_xy=np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float64),
        width_m=100.0, length_m=50.0,
    )
    ref2 = ReferenceRectangle(
        image_points_xy=np.array([[200, 200], [300, 200], [300, 250], [200, 250]], dtype=np.float64),
        width_m=100.0, length_m=50.0,
    )
    result = fit_reference_scales_multi(identity, [ref1, ref2], steps=15)
    assert result.n_refs == 2
    assert abs(result.scale_x - 1.0) < 0.2
    assert abs(result.scale_y - 1.0) < 0.2


# ---- apply_scales_to_homography -----------------------------------------------


def test_apply_scales_identity():
    H = np.eye(3)
    H2 = apply_scales_to_homography(H, 2.0, 3.0)
    assert H2[0, 0] == pytest.approx(2.0)
    assert H2[1, 1] == pytest.approx(3.0)
    assert H2[2, 2] == pytest.approx(1.0)
