"""标定物启发式尺度拟合（支持多标定物/多帧 + 鲁棒损失）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Literal, Sequence, Tuple

import numpy as np


def _order_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    s = p.sum(axis=1)
    d = np.diff(p, axis=1).reshape(-1)
    tl = p[np.argmin(s)]
    br = p[np.argmax(s)]
    tr = p[np.argmin(d)]
    bl = p[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float64)


@dataclass(frozen=True)
class ReferenceRectangle:
    """在图像中点击的 4 点 + 标定物真实宽长（米）。"""

    image_points_xy: np.ndarray  # (4,2), roughly tl,tr,br,bl
    width_m: float
    length_m: float

    def ordered(self) -> np.ndarray:
        return _order_tl_tr_br_bl(self.image_points_xy)


@dataclass(frozen=True)
class ScaleFitResult:
    scale_x: float
    scale_y: float
    width_err_m: float
    length_err_m: float
    rmse_m: float


@dataclass(frozen=True)
class MultiScaleFitResult:
    scale_x: float
    scale_y: float
    rmse_m: float
    n_refs: int


def _mean_edge_lengths(world_quad: np.ndarray) -> Tuple[float, float]:
    tl, tr, br, bl = world_quad
    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    right = np.linalg.norm(br - tr)
    left = np.linalg.norm(bl - tl)
    width = 0.5 * (top + bottom)
    length = 0.5 * (right + left)
    return float(width), float(length)


def fit_reference_scales(
    projector: Callable[[float, float], Tuple[float, float]],
    ref: ReferenceRectangle,
    *,
    steps: int = 10,
) -> ScaleFitResult:
    """
    用坐标下降拟合各向异性尺度 (sx, sy)，使投影后标定物尺寸贴近真实值。

    projector: 像素 -> 世界 (x, y) 的函数（可来自 Homography 或 Pinhole）。

    单标定物兼容入口；内部复用多参考拟合。
    """
    multi = fit_reference_scales_multi(
        projector,
        [ref],
        steps=steps,
        robust_loss="none",
    )
    q_img = ref.ordered()
    q_w = np.array([projector(float(u), float(v)) for u, v in q_img], dtype=np.float64)
    q_w[:, 0] *= multi.scale_x
    q_w[:, 1] *= multi.scale_y
    w_hat, l_hat = _mean_edge_lengths(q_w)
    return ScaleFitResult(
        scale_x=multi.scale_x,
        scale_y=multi.scale_y,
        width_err_m=float(w_hat - ref.width_m),
        length_err_m=float(l_hat - ref.length_m),
        rmse_m=multi.rmse_m,
    )


def _loss_from_residual(
    r: float,
    *,
    robust_loss: Literal["none", "huber", "cauchy"],
    delta: float,
) -> float:
    a = abs(float(r))
    d = max(1e-9, float(delta))
    if robust_loss == "none":
        return a * a
    if robust_loss == "huber":
        if a <= d:
            return a * a
        return 2.0 * d * a - d * d
    # cauchy
    x = a / d
    return d * d * np.log1p(x * x)


def fit_reference_scales_multi(
    projector: Callable[[float, float], Tuple[float, float]],
    refs: Sequence[ReferenceRectangle],
    *,
    steps: int = 14,
    robust_loss: Literal["none", "huber", "cauchy"] = "huber",
    robust_delta_m: float = 0.2,
) -> MultiScaleFitResult:
    """
    多标定物/多帧联合拟合 (sx, sy)。

    对每个 ref 产生 (宽误差, 长误差) 两个残差，最小化鲁棒损失和。
    """
    if not refs:
        raise ValueError("refs must not be empty")
    valid_refs: List[ReferenceRectangle] = []
    world_quads: List[np.ndarray] = []
    for r in refs:
        if r.width_m <= 0 or r.length_m <= 0:
            continue
        q_img = r.ordered()
        q_w = np.array([projector(float(u), float(v)) for u, v in q_img], dtype=np.float64)
        if not np.isfinite(q_w).all():
            continue
        valid_refs.append(r)
        world_quads.append(q_w)
    if not valid_refs:
        raise ValueError("no valid references for fitting")

    def obj(sx: float, sy: float) -> float:
        total = 0.0
        for q_w, r in zip(world_quads, valid_refs):
            q = q_w.copy()
            q[:, 0] *= sx
            q[:, 1] *= sy
            w_hat, l_hat = _mean_edge_lengths(q)
            e_w = w_hat - r.width_m
            e_l = l_hat - r.length_m
            total += _loss_from_residual(e_w, robust_loss=robust_loss, delta=robust_delta_m)
            total += _loss_from_residual(e_l, robust_loss=robust_loss, delta=robust_delta_m)
        return float(total / (2.0 * len(valid_refs)))

    sx, sy = 1.0, 1.0
    step = 0.35
    for _ in range(max(1, steps)):
        best_val = obj(sx, sy)
        best_sx, best_sy = sx, sy
        candidates = [
            (sx + step, sy),
            (sx - step, sy),
            (sx, sy + step),
            (sx, sy - step),
            (sx + step, sy + step),
            (sx + step, sy - step),
            (sx - step, sy + step),
            (sx - step, sy - step),
        ]
        for cx, cy in candidates:
            if cx <= 1e-6 or cy <= 1e-6:
                continue
            v = obj(cx, cy)
            if v < best_val:
                best_val = v
                best_sx, best_sy = cx, cy
        sx, sy = best_sx, best_sy
        step *= 0.58

    rms_terms: List[float] = []
    for q_w, r in zip(world_quads, valid_refs):
        q = q_w.copy()
        q[:, 0] *= sx
        q[:, 1] *= sy
        w_hat, l_hat = _mean_edge_lengths(q)
        rms_terms.append((w_hat - r.width_m) ** 2)
        rms_terms.append((l_hat - r.length_m) ** 2)
    rmse = float(np.sqrt(np.mean(rms_terms))) if rms_terms else float("nan")
    return MultiScaleFitResult(scale_x=float(sx), scale_y=float(sy), rmse_m=rmse, n_refs=len(valid_refs))


def apply_scales_to_homography(H: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """
    返回尺度修正后的单应矩阵：H' = S @ H，其中 S=diag(scale_x, scale_y, 1)。
    """
    mat = np.asarray(H, dtype=np.float64).reshape(3, 3)
    S = np.array([[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return S @ mat
