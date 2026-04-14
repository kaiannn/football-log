"""完整针孔模型：像素 → 去畸变归一化射线 → 外参到世界系 → 与地面平面求交 → (X, Y) 米。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from football_log.world.homography import bbox_foot_point, is_person_label


def _load_dict(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith((".yaml", ".yml")):
            try:
                import yaml  # type: ignore
            except ImportError as e:
                raise ImportError("YAML 标定文件需要安装 pyyaml: pip install pyyaml") from e
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("标定文件顶层必须是对象")
    return data


@dataclass
class GroundPlane:
    """
    世界系下平面 n·X + d = 0（单位：米）。
    默认 z = 0：n = (0,0,1), d = 0。
    """

    normal: np.ndarray  # shape (3,)
    d: float

    def __post_init__(self) -> None:
        self.normal = np.asarray(self.normal, dtype=np.float64).reshape(3)
        nrm = np.linalg.norm(self.normal)
        if nrm < 1e-12:
            raise ValueError("ground plane normal must be non-zero")
        self.normal = self.normal / nrm


@dataclass
class PinholeGroundProjector:
    """
    OpenCV 约定：X_c = R @ X_w + t（世界点到相机系）。
    相机光心在世界系：C = -R^T @ t。
    像素 (u,v) 经去畸变得归一化方向 d_c ∝ (x_n, y_n, 1)，世界系射线方向 v_w = R^T @ d_c，
    与平面 n·X + d = 0 求交：X = C + λ v_w，λ = -(n·C + d) / (n·v_w)。
    """

    K: np.ndarray
    dist: np.ndarray
    R_wc: np.ndarray  # world → camera
    t_wc: np.ndarray  # world → camera
    plane: GroundPlane

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(self.dist, dtype=np.float64).reshape(-1, 1)
        self.R_wc = np.asarray(self.R_wc, dtype=np.float64).reshape(3, 3)
        self.t_wc = np.asarray(self.t_wc, dtype=np.float64).reshape(3)

    @property
    def camera_center_world(self) -> np.ndarray:
        return (-self.R_wc.T @ self.t_wc).reshape(3)

    def pixel_to_ground_xyz(self, u: float, v: float) -> Tuple[float, float, float]:
        """
        单点 (u, v) → 与地面平面交点 (X, Y, Z)。无效时返回 (nan, nan, nan)。
        """
        pts = np.array([[[u, v]]], dtype=np.float64)
        und = cv2.undistortPoints(pts, self.K, self.dist, P=None)
        x_n, y_n = float(und[0, 0, 0]), float(und[0, 0, 1])
        d_c = np.array([x_n, y_n, 1.0], dtype=np.float64)
        v_w = self.R_wc.T @ d_c
        C = self.camera_center_world
        n = self.plane.normal
        denom = float(n @ v_w)
        if abs(denom) < 1e-12:
            return float("nan"), float("nan"), float("nan")
        lam = -(float(n @ C) + self.plane.d) / denom
        if lam <= 0:
            # 交点在相机后方（沿射线反方向）
            return float("nan"), float("nan"), float("nan")
        Xw = C + lam * v_w
        return float(Xw[0]), float(Xw[1]), float(Xw[2])

    def pixel_to_world_xy_m(self, u: float, v: float) -> Tuple[Optional[float], Optional[float]]:
        x, y, z = self.pixel_to_ground_xyz(u, v)
        if not (np.isfinite(x) and np.isfinite(y)):
            return None, None
        return x, y

    def project_foot_to_world(
        self,
        bbox: Tuple[float, float, float, float],
        label: str,
    ) -> Tuple[Optional[float], Optional[float]]:
        """与 homography.project_foot_to_world 相同约定：球员用底边中点，球用中心。"""
        if not is_person_label(label) and label != "Ball":
            return None, None
        if is_person_label(label):
            u, v = bbox_foot_point(bbox)
        else:
            u = float(bbox[0] + 0.5 * bbox[2])
            v = float(bbox[1] + 0.5 * bbox[3])
        return self.pixel_to_world_xy_m(u, v)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PinholeGroundProjector:
        K = np.asarray(data["K"], dtype=np.float64)
        dist = np.asarray(data.get("dist", data.get("distortion", [])), dtype=np.float64)
        if dist.size == 0:
            dist = np.zeros((5, 1), dtype=np.float64)
        else:
            dist = dist.reshape(-1, 1)

        ext = data.get("extrinsics", data)
        if "R" in ext and "t" in ext:
            R = np.asarray(ext["R"], dtype=np.float64).reshape(3, 3)
            t = np.asarray(ext["t"], dtype=np.float64).reshape(3)
        elif "R" in ext and "C" in ext:
            R = np.asarray(ext["R"], dtype=np.float64).reshape(3, 3)
            C = np.asarray(ext["C"], dtype=np.float64).reshape(3)
            t = (-R @ C).reshape(3)
        else:
            raise ValueError("extrinsics 需要 R+t（OpenCV 世界→相机）或 R+C（相机光心世界坐标）")

        gp = data.get("ground_plane", {})
        if isinstance(gp, dict):
            n = gp.get("normal", [0.0, 0.0, 1.0])
            d = float(gp.get("d", 0.0))
        else:
            n, d = [0.0, 0.0, 1.0], 0.0
        plane = GroundPlane(normal=np.asarray(n, dtype=np.float64), d=d)

        return cls(K=K, dist=dist, R_wc=R, t_wc=t, plane=plane)


def load_pinhole_ground_projector(path: str) -> PinholeGroundProjector:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(path)
    return PinholeGroundProjector.from_dict(_load_dict(path))
