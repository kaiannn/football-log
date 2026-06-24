"""Per-track Kalman smoothing in BEV (bird's-eye-view, world meters) space.

The pixel→world projection assumes the player's foot is on the ground plane.
That assumption breaks on jumps (headers, slide tackles), partial occlusions
(advertising boards covering legs), and ball-in-air. The result is meter-scale
spikes in the per-frame world coordinate.

This filter sits between the projector and the exporter. For each track ID it
runs a constant-velocity Kalman filter; observations are gated by a
Mahalanobis distance test so glaring outliers are rejected rather than baked
into the trajectory.

State: x = [x_m, y_m, vx_m_per_s, vy_m_per_s].T
Transition F: identity on position + velocity·dt; identity on velocity.
Observation H: [[1, 0, 0, 0], [0, 1, 0, 0]] — we only observe position.
Process noise Q: scaled by acceleration std (default 6 m/s² ≈ peak sprint).
Observation noise R: scales with detection-confidence and an optional jump
    heuristic so jump-frames automatically count for less.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


# ---- per-track state -----------------------------------------------------------


@dataclass
class _TrackState:
    """Internal Kalman state for a single track."""

    x: np.ndarray  # shape (4,)
    P: np.ndarray  # shape (4, 4)
    last_frame_idx: int


# ---- filter --------------------------------------------------------------------


class TrackFilter:
    """Manages a constant-velocity Kalman per track ID, in world meters.

    Lifecycle (one call per detection, per frame):

        smoothed_xy = filter.update(
            track_id=t,
            world_xy=(wx, wy),                  # raw projection from foot anchor
            frame_idx=i,
            conf=detection_conf,                # 0..1, scales obs noise
            jump_likelihood=0.0..1.0,           # 0=on ground, 1=likely airborne
        )

    `world_xy=None` means the projector failed for this detection; the filter
    will predict-only and return the predicted position (or None for a
    brand-new track with no observation yet).

    Tracks that go un-updated for too many frames are evicted to keep memory
    bounded.
    """

    def __init__(
        self,
        fps: float,
        accel_std: float = 6.0,
        base_obs_std: float = 0.5,
        max_obs_std: float = 5.0,
        mahalanobis_gate: float = 9.0,  # = 3-sigma squared
        max_age_frames: int = 60,
    ):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.accel_std = float(accel_std)
        self.base_obs_std = float(base_obs_std)
        self.max_obs_std = float(max_obs_std)
        self.mahalanobis_gate = float(mahalanobis_gate)
        self.max_age_frames = int(max_age_frames)
        self._tracks: Dict[int, _TrackState] = {}

        self.n_observations: int = 0
        self.n_outliers_rejected: int = 0
        self.n_predict_only: int = 0

    # ------ matrix helpers ------

    def _F(self, dt: float) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _Q(self, dt: float) -> np.ndarray:
        """Continuous-white-noise acceleration model, state order (x, y, vx, vy)."""
        a2 = self.accel_std ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        return np.array(
            [
                [dt4 / 4.0, 0.0,        dt3 / 2.0, 0.0      ],
                [0.0,       dt4 / 4.0,  0.0,       dt3 / 2.0],
                [dt3 / 2.0, 0.0,        dt2,       0.0      ],
                [0.0,       dt3 / 2.0,  0.0,       dt2      ],
            ],
            dtype=np.float64,
        ) * a2

    @staticmethod
    def _H() -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

    def _R(self, conf: float, jump_likelihood: float) -> np.ndarray:
        """Observation noise covariance — bigger when detection is uncertain or jumping."""
        conf = max(0.05, min(1.0, float(conf)))
        # base × (1/conf) gives 0.5–10 m std for conf 1.0–0.05
        std = self.base_obs_std / conf
        # boost for jumps
        std = std * (1.0 + 4.0 * float(jump_likelihood))
        std = min(std, self.max_obs_std)
        var = std ** 2
        return np.eye(2, dtype=np.float64) * var

    # ------ public API ------

    def update(
        self,
        track_id: int,
        world_xy: Optional[Tuple[float, float]],
        frame_idx: int,
        conf: float = 1.0,
        jump_likelihood: float = 0.0,
    ) -> Optional[Tuple[float, float]]:
        """Update the filter with one observation and return the smoothed (x, y).

        If `world_xy` is None, performs a predict-only step.
        Returns None for a brand-new track that has no observation yet.
        """
        # New track — initialize from observation if we have one, else give up.
        if track_id not in self._tracks:
            if world_xy is None:
                return None
            self._tracks[track_id] = _TrackState(
                x=np.array([world_xy[0], world_xy[1], 0.0, 0.0], dtype=np.float64),
                P=np.diag([1.0, 1.0, 4.0, 4.0]).astype(np.float64),
                last_frame_idx=frame_idx,
            )
            self.n_observations += 1
            return float(world_xy[0]), float(world_xy[1])

        st = self._tracks[track_id]

        # Predict step
        gap = max(1, frame_idx - st.last_frame_idx)
        dt = gap * self.dt
        F = self._F(dt)
        Q = self._Q(dt)
        x_pred = F @ st.x
        P_pred = F @ st.P @ F.T + Q

        if world_xy is None:
            st.x = x_pred
            st.P = P_pred
            st.last_frame_idx = frame_idx
            self.n_predict_only += 1
            return float(x_pred[0]), float(x_pred[1])

        # Update step
        H = self._H()
        z = np.array([world_xy[0], world_xy[1]], dtype=np.float64)
        R = self._R(conf, jump_likelihood)
        y = z - H @ x_pred                          # innovation
        S = H @ P_pred @ H.T + R                    # innovation covariance
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        m_dist = float(y @ S_inv @ y)               # squared Mahalanobis distance

        if m_dist > self.mahalanobis_gate:
            # Outlier — accept the prediction, do not absorb the observation.
            st.x = x_pred
            st.P = P_pred
            st.last_frame_idx = frame_idx
            self.n_outliers_rejected += 1
            return float(x_pred[0]), float(x_pred[1])

        # Normal Kalman gain update
        K = P_pred @ H.T @ S_inv
        x_new = x_pred + K @ y
        P_new = (np.eye(4) - K @ H) @ P_pred
        st.x = x_new
        st.P = P_new
        st.last_frame_idx = frame_idx
        self.n_observations += 1
        return float(x_new[0]), float(x_new[1])

    def evict_stale(self, current_frame_idx: int) -> int:
        """Drop tracks that haven't been updated in `max_age_frames`. Returns count dropped."""
        before = len(self._tracks)
        self._tracks = {
            tid: st
            for tid, st in self._tracks.items()
            if (current_frame_idx - st.last_frame_idx) <= self.max_age_frames
        }
        return before - len(self._tracks)

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    def stats(self) -> Dict[str, int]:
        return {
            "active_tracks": self.active_tracks,
            "observations_absorbed": self.n_observations,
            "outliers_rejected": self.n_outliers_rejected,
            "predict_only_steps": self.n_predict_only,
        }


# ---- jump-likelihood heuristic -------------------------------------------------


def jump_likelihood_from_height_change(
    bbox_height: float,
    prev_height: Optional[float],
    bbox_y_top: float,
    frame_height: int,
) -> float:
    """Cheap heuristic: 1.0 = likely airborne, 0.0 = solidly on ground.

    Two cues combine:
        - bbox shrinks fast → player likely moved away (depth) but could also
          be jumping. Heuristic counts "rapid shrink" as a small signal.
        - bbox top hits frame top → upper body likely cropped, foot anchor
          unreliable.
    """
    out = 0.0
    if prev_height is not None and prev_height > 0:
        ratio = bbox_height / prev_height
        if ratio > 1.15 or ratio < 0.85:
            out += 0.3
    if bbox_y_top < 0.05 * frame_height:
        out += 0.5
    return min(1.0, out)
