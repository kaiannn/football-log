"""BEV radar canvas — draws a top-down pitch with player/ball dots per frame."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Canvas dimensions
_CANVAS_W = 780
_CANVAS_H = 520
_PAD = 50        # padding around pitch lines
_LEGEND_H = 40   # legend strip at bottom inside PAD

# Pitch colours
_GRASS_DARK  = (30, 100, 30)
_GRASS_LIGHT = (38, 120, 38)
_LINE        = (220, 220, 220)
_GOAL_NET    = (180, 180, 180)
_BORDER      = (15, 55, 15)

# Dot colours (BGR)
_DOT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Team A":  (60,  80, 230),   # red/orange
    "Team B":  (220, 160,  30),  # blue/teal
    "Ball":    (255, 255, 255),
    "Referee": (0,  210, 255),   # gold
    "Player":  (160, 160, 160),
}

_DOT_R     = 9
_BALL_R    = 6
_STRIPE_N  = 10   # number of alternating-stripe bands


def _pitch_canvas() -> np.ndarray:
    img = np.zeros((_CANVAS_H, _CANVAS_W, 3), dtype=np.uint8)

    pw = _CANVAS_W - 2 * _PAD
    ph = _CANVAS_H - 2 * _PAD - _LEGEND_H

    # Dark border fill
    img[:] = _BORDER

    # Draw alternating vertical grass stripes
    stripe_w = pw / _STRIPE_N
    for i in range(_STRIPE_N):
        x1 = int(_PAD + i * stripe_w)
        x2 = int(_PAD + (i + 1) * stripe_w)
        col = _GRASS_LIGHT if i % 2 == 0 else _GRASS_DARK
        cv2.rectangle(img, (x1, _PAD), (x2, _PAD + ph), col, -1)

    # Legend strip (below pitch)
    leg_y = _PAD + ph
    cv2.rectangle(img, (_PAD, leg_y), (_PAD + pw, _CANVAS_H - 5), (22, 70, 22), -1)

    def px(nx: float, ny: float) -> Tuple[int, int]:
        return (int(_PAD + nx * pw), int(_PAD + ny * ph))

    lw = 2

    # --- Pitch markings ---
    # Outer boundary
    cv2.rectangle(img, px(0, 0), px(1, 1), _LINE, lw)

    # Halfway line
    cv2.line(img, px(0.5, 0), px(0.5, 1), _LINE, lw)

    # Centre circle (r ≈ 9.15 m / 105 m pitch)
    cx, cy = px(0.5, 0.5)
    r_cc = int(9.15 / 105 * pw)
    cv2.circle(img, (cx, cy), r_cc, _LINE, lw)
    cv2.circle(img, (cx, cy), 3, _LINE, -1)

    # Penalty areas (16.5 m deep × 40.3 m wide on 105×68)
    pa_d = 16.5 / 105
    pa_h = 20.15 / 68
    cv2.rectangle(img, px(0, 0.5 - pa_h), px(pa_d, 0.5 + pa_h), _LINE, lw)
    cv2.rectangle(img, px(1 - pa_d, 0.5 - pa_h), px(1, 0.5 + pa_h), _LINE, lw)

    # Goal areas (5.5 m deep × 18.3 m wide)
    ga_d = 5.5 / 105
    ga_h = 9.15 / 68
    cv2.rectangle(img, px(0, 0.5 - ga_h), px(ga_d, 0.5 + ga_h), _LINE, lw)
    cv2.rectangle(img, px(1 - ga_d, 0.5 - ga_h), px(1, 0.5 + ga_h), _LINE, lw)

    # Penalty arcs (r = 9.15 m centred on penalty spot at 11 m)
    for side_x, angle_offset in [(11 / 105, 0), (1 - 11 / 105, 180)]:
        sp = px(side_x, 0.5)
        cv2.ellipse(img, sp, (r_cc, r_cc), angle_offset, -53, 53, _LINE, lw)

    # Penalty spots
    cv2.circle(img, px(11 / 105, 0.5), 3, _LINE, -1)
    cv2.circle(img, px(1 - 11 / 105, 0.5), 3, _LINE, -1)

    # Goals — small nets behind each goal line
    goal_depth = int(3.5 / 105 * pw)   # ~3.5 m goal depth
    goal_h_half = int(3.66 / 68 * ph)  # ~3.66 m half-width
    gcy = cy  # vertical centre
    for gx, sign in [(px(0, 0.5)[0], -1), (px(1, 0.5)[0], +1)]:
        net_x1 = gx + sign * goal_depth
        pts = np.array([
            [gx,     gcy - goal_h_half],
            [net_x1, gcy - goal_h_half],
            [net_x1, gcy + goal_h_half],
            [gx,     gcy + goal_h_half],
        ], dtype=np.int32)
        cv2.polylines(img, [pts], True, _GOAL_NET, 1)
        # Grid lines inside net
        for step in range(1, 4):
            t = step / 4
            ny = int(gcy - goal_h_half + 2 * goal_h_half * t)
            cv2.line(img, (gx, ny), (net_x1, ny), _GOAL_NET, 1)
        for step in range(1, 3):
            t = step / 3
            nx = int(min(gx, net_x1) + abs(goal_depth) * t)
            cv2.line(img, (nx, gcy - goal_h_half), (nx, gcy + goal_h_half), _GOAL_NET, 1)

    # Corner arcs (r = 1 m)
    r_corner = max(4, int(1.0 / 105 * pw))
    for nx, ny, angle in [(0, 0, 0), (1, 0, 270), (1, 1, 180), (0, 1, 90)]:
        cp = px(nx, ny)
        cv2.ellipse(img, cp, (r_corner, r_corner), angle, 0, 90, _LINE, lw)

    return img


_BLANK_PITCH = _pitch_canvas()


def _dot_color(label: str) -> Tuple[int, int, int]:
    for key, col in _DOT_COLORS.items():
        if key in label:
            return col
    return _DOT_COLORS["Player"]


def _draw_legend(canvas: np.ndarray) -> None:
    pw = _CANVAS_W - 2 * _PAD
    ph = _CANVAS_H - 2 * _PAD - _LEGEND_H
    leg_y = _PAD + ph + _LEGEND_H // 2

    entries = [
        ("Team A", _DOT_COLORS["Team A"]),
        ("Team B", _DOT_COLORS["Team B"]),
        ("Referee", _DOT_COLORS["Referee"]),
        ("Ball", _DOT_COLORS["Ball"]),
    ]
    x = _PAD + 10
    for name, col in entries:
        # Dot icon
        cv2.circle(canvas, (x + 6, leg_y), 6, (0, 0, 0), -1)
        cv2.circle(canvas, (x + 6, leg_y), 5, col, -1)
        # Text
        (tw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(canvas, name, (x + 16, leg_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        x += tw + 34


class RadarRenderer:
    def __init__(self, pitch_length_m: float = 105.0, pitch_width_m: float = 68.0):
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        # H_canvas: pixel → radar canvas coords (set via set_homography or _update_homography)
        self._H_canvas: Optional[np.ndarray] = None

    def set_homography(self, H_pixel_to_world: np.ndarray) -> None:
        """Accept a pixel→world_m homography (from PitchKeypointDetector) and
        derive the pixel→canvas homography by composing with world→canvas."""
        # Build world→canvas homography from 4 known pitch corners.
        pw = _CANVAS_W - 2 * _PAD
        ph = _CANVAS_H - 2 * _PAD - _LEGEND_H
        # World corners in metres (PITCH_KP_WORLD uses [0,105] x [0,68])
        world_corners = np.array([
            [0.0,   0.0 ],   # top-left
            [self.pitch_length_m, 0.0 ],   # top-right
            [self.pitch_length_m, self.pitch_width_m],  # bottom-right
            [0.0,   self.pitch_width_m],   # bottom-left
        ], dtype=np.float32)
        canvas_corners = np.array([
            [_PAD,       _PAD     ],
            [_PAD + pw,  _PAD     ],
            [_PAD + pw,  _PAD + ph],
            [_PAD,       _PAD + ph],
        ], dtype=np.float32)
        H_world_to_canvas, _ = cv2.findHomography(world_corners, canvas_corners)
        if H_world_to_canvas is None:
            return
        # Compose: pixel → world → canvas
        self._H_canvas = H_world_to_canvas @ H_pixel_to_world

    def _update_homography(self, quad: np.ndarray) -> None:
        """Compute pixel→canvas H directly from the 4-corner grass-mask quad."""
        src = quad.astype(np.float32).reshape(4, 2)
        pw = _CANVAS_W - 2 * _PAD
        ph = _CANVAS_H - 2 * _PAD - _LEGEND_H
        dst = np.array([
            [_PAD,       _PAD     ],
            [_PAD + pw,  _PAD     ],
            [_PAD + pw,  _PAD + ph],
            [_PAD,       _PAD + ph],
        ], dtype=np.float32)
        self._H_canvas, _ = cv2.findHomography(src, dst)

    def _pixel_to_radar(self, foot_px: np.ndarray) -> Optional[Tuple[int, int]]:
        if self._H_canvas is None:
            return None
        pt = np.array([[[float(foot_px[0]), float(foot_px[1])]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self._H_canvas)
        return (int(out[0, 0, 0]), int(out[0, 0, 1]))

    def _world_to_radar(self, wx: float, wy: float) -> Tuple[int, int]:
        """Map world coordinates (metres) to radar canvas.

        wx: 0..pitch_length_m  (along pitch length)
        wy: 0..pitch_width_m   (PITCH_KP_WORLD convention — top=0, bottom=68)
        """
        pw = _CANVAS_W - 2 * _PAD
        ph = _CANVAS_H - 2 * _PAD - _LEGEND_H
        nx = float(np.clip(wx / self.pitch_length_m, 0.0, 1.0))
        ny = float(np.clip(wy / self.pitch_width_m, 0.0, 1.0))
        return (int(_PAD + nx * pw), int(_PAD + ny * ph))

    def render(
        self,
        detections: List[Dict],
        *,
        field_quad_xy: Optional[np.ndarray] = None,
        frame_shape: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        canvas = _BLANK_PITCH.copy()

        if field_quad_xy is not None and field_quad_xy.shape == (4, 2):
            self._update_homography(field_quad_xy)

        for obj in detections:
            label = obj.get("label", "Player")
            color = _dot_color(label)
            is_ball = "Ball" in label

            wx = obj.get("world_x_m")
            wy = obj.get("world_y_m")
            if wx is not None and wy is not None:
                rx, ry = self._world_to_radar(float(wx), float(wy))
            elif self._H_canvas is not None:
                bbox = obj.get("bbox")
                if bbox is None:
                    bbox = (int(obj["x"]), int(obj["y"]), int(obj["w"]), int(obj["h"]))
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                foot = np.array([x + w // 2, y + h], dtype=np.float32)
                pt = self._pixel_to_radar(foot)
                if pt is None:
                    continue
                rx, ry = pt
            else:
                continue

            # Clamp to pitch area
            pw = _CANVAS_W - 2 * _PAD
            ph = _CANVAS_H - 2 * _PAD - _LEGEND_H
            if not (_PAD - 15 <= rx <= _PAD + pw + 15 and _PAD - 15 <= ry <= _PAD + ph + 15):
                continue

            r = _BALL_R if is_ball else _DOT_R

            # Shadow
            cv2.circle(canvas, (rx + 2, ry + 2), r + 1, (0, 0, 0), -1, cv2.LINE_AA)
            # Dark outer ring
            cv2.circle(canvas, (rx, ry), r + 2, (20, 20, 20), -1, cv2.LINE_AA)
            # Colored fill
            cv2.circle(canvas, (rx, ry), r, color, -1, cv2.LINE_AA)
            # White inner highlight for depth
            if not is_ball:
                hi_r = max(2, r - 4)
                cv2.circle(canvas, (rx - 2, ry - 2), hi_r, (255, 255, 255), 1, cv2.LINE_AA)

            # ID label above dot
            if not is_ball:
                tid = str(obj.get("id", obj.get("track_id", "")))
                font = cv2.FONT_HERSHEY_SIMPLEX
                fscale, fthick = 0.33, 1
                (tw, th), _ = cv2.getTextSize(tid, font, fscale, fthick)
                tx = rx - tw // 2
                ty = ry - r - 4
                # Small dark backing
                cv2.rectangle(canvas, (tx - 2, ty - th - 1), (tx + tw + 2, ty + 2),
                               (20, 20, 20), -1, cv2.LINE_AA)
                cv2.putText(canvas, tid, (tx, ty), font, fscale,
                            (240, 240, 240), fthick, cv2.LINE_AA)

        _draw_legend(canvas)
        return canvas
