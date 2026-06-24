"""BEV radar — matplotlib vector rendering for crisp output at any resolution."""

from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Circle, Arc

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# Canvas dimensions (output image size)
_CANVAS_W = 1600
_CANVAS_H = 1067

# Pitch layout constants (normalised 0..1)
_PX0, _PY0 = 0.05, 0.04  # left/top margin
_PX1, _PY1 = 0.95, 0.88  # right/bottom of pitch area
_LEGEND_Y0 = 0.90  # legend band start

# Pitch geometry in metres
_PITCH_L = 105.0
_PITCH_W = 68.0
_PA_DEPTH = 16.5
_PA_HALF_W = 20.15
_GA_DEPTH = 5.5
_GA_HALF_W = 9.15
_CC_R = 9.15
_PS_X = 11.0
_CORNER_R = 1.0
_GOAL_DEPTH = 3.5
_GOAL_HALF_W = 3.66

# Colours
_C_GRASS_DARK = "#1a5c1a"
_C_GRASS_LIGHT = "#237823"
_C_LINE = "#d8d8d8"
_C_GOAL = "#aaaaaa"
_C_BORDER = "#0f3710"

_DOT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Team A":  (64,  64, 224),   # BGR
    "Team B":  (224, 144,  48),
    "Ball":    (255, 255, 255),
    "Referee": (0,  200, 255),
    "Player":  (160, 160, 160),
}
_DOT_R = 0.9  # metres, visual radius on pitch
_BALL_R = 0.5


def _world_to_fig(wx: float, wy: float) -> Tuple[float, float]:
    """World metres → figure-normalised coordinates."""
    nx = np.clip(wx / _PITCH_L, 0.0, 1.0)
    ny = np.clip(wy / _PITCH_W, 0.0, 1.0)
    fx = _PX0 + nx * (_PX1 - _PX0)
    fy = _PY0 + ny * (_PY1 - _PY0)
    return fx, fy


def _dot_color(label: str) -> Tuple[int, int, int]:
    for key, col in _DOT_COLORS.items():
        if key in label:
            return col
    return _DOT_COLORS["Player"]


def _draw_pitch(ax: plt.Axes) -> None:
    """Draw pitch markings in world-metre coordinates on the axes."""
    ax.set_xlim(-_PITCH_L * 0.02, _PITCH_L * 1.02)
    ax.set_ylim(-_PITCH_W * 0.02, _PITCH_W * 1.02)
    ax.set_aspect("equal")
    ax.axis("off")

    pw, ph = _PITCH_L, _PITCH_W

    # Grass stripes
    n_stripes = 10
    sw = pw / n_stripes
    for i in range(n_stripes):
        c = _C_GRASS_LIGHT if i % 2 == 0 else _C_GRASS_DARK
        ax.add_patch(mpatches.Rectangle((i * sw, 0), sw, ph, fc=c, ec="none"))

    # Border
    ax.add_patch(mpatches.Rectangle((-1.5, -1.5), pw + 3, ph + 3, fc=_C_BORDER, ec="none", zorder=0))

    # Re-draw grass on top of border
    for i in range(n_stripes):
        c = _C_GRASS_LIGHT if i % 2 == 0 else _C_GRASS_DARK
        ax.add_patch(mpatches.Rectangle((i * sw, 0), sw, ph, fc=c, ec="none", zorder=1))

    lw = 1.2
    lc = _C_LINE

    # Outer boundary
    ax.add_patch(mpatches.Rectangle((0, 0), pw, ph, fc="none", ec=lc, lw=lw, zorder=5))

    # Halfway line
    ax.plot([pw / 2, pw / 2], [0, ph], color=lc, lw=lw, zorder=5)

    # Centre circle + dot
    ax.add_patch(Circle((pw / 2, ph / 2), _CC_R, fc="none", ec=lc, lw=lw, zorder=5))
    ax.plot(pw / 2, ph / 2, "o", color=lc, ms=2.5, zorder=5)

    # Penalty areas
    for sx in [0, pw - _PA_DEPTH]:
        ax.add_patch(mpatches.Rectangle((sx, ph / 2 - _PA_HALF_W), _PA_DEPTH, 2 * _PA_HALF_W,
                                        fc="none", ec=lc, lw=lw, zorder=5))

    # Goal areas
    for sx in [0, pw - _GA_DEPTH]:
        ax.add_patch(mpatches.Rectangle((sx, ph / 2 - _GA_HALF_W), _GA_DEPTH, 2 * _GA_HALF_W,
                                        fc="none", ec=lc, lw=lw, zorder=5))

    # Penalty arcs (only the arc outside the penalty area)
    for px in [_PS_X, pw - _PS_X]:
        ax.add_patch(Arc((px, ph / 2), 2 * _CC_R, 2 * _CC_R,
                         angle=0, theta1=-53, theta2=53,
                         color=lc, lw=lw, zorder=5))

    # Penalty spots
    for px in [_PS_X, pw - _PS_X]:
        ax.plot(px, ph / 2, "o", color=lc, ms=2.5, zorder=5)

    # Goals
    for gx, sign in [(0, -1), (pw, 1)]:
        x1 = gx
        x2 = gx + sign * _GOAL_DEPTH
        y1 = ph / 2 - _GOAL_HALF_W
        y2 = ph / 2 + _GOAL_HALF_W
        ax.add_patch(mpatches.Rectangle((min(x1, x2), y1), abs(x2 - x1), y2 - y1,
                                        fc="none", ec=_C_GOAL, lw=0.8, zorder=5))
        # Net grid
        for t in np.linspace(0, 1, 5)[1:-1]:
            yy = y1 + t * (y2 - y1)
            ax.plot([x1, x2], [yy, yy], color=_C_GOAL, lw=0.3, zorder=4)
        for t in np.linspace(0, 1, 4)[1:-1]:
            xx = x1 + t * (x2 - x1)
            ax.plot([xx, xx], [y1, y2], color=_C_GOAL, lw=0.3, zorder=4)

    # Corner arcs
    for cx, cy, a1 in [(0, 0, 0), (pw, 0, 90), (pw, ph, 180), (0, ph, 270)]:
        ax.add_patch(Arc((cx, cy), 2 * _CORNER_R, 2 * _CORNER_R,
                         angle=a1, theta1=0, theta2=90,
                         color=lc, lw=lw, zorder=5))


def _fig_to_bgr(fig: plt.Figure) -> np.ndarray:
    """Render a matplotlib figure to a BGR numpy array."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi, bbox_inches="tight", pad_inches=0.02,
                facecolor=_C_BORDER, edgecolor="none")
    buf.seek(0)
    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    buf.close()
    if img.shape[1] != _CANVAS_W or img.shape[0] != _CANVAS_H:
        img = cv2.resize(img, (_CANVAS_W, _CANVAS_H), interpolation=cv2.INTER_AREA)
    return img


class RadarRenderer:
    def __init__(self, pitch_length_m: float = 105.0, pitch_width_m: float = 68.0):
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self._H_canvas: Optional[np.ndarray] = None
        self._blank: Optional[np.ndarray] = None
        self._pad_x = int(_PX0 * _CANVAS_W)
        self._pad_y = int(_PY0 * _CANVAS_H)
        self._pw = _CANVAS_W - 2 * self._pad_x
        self._ph = _CANVAS_H - 2 * self._pad_y

    def _get_blank(self) -> np.ndarray:
        if self._blank is None:
            fig, ax = plt.subplots(1, 1, figsize=(_CANVAS_W / 100, _CANVAS_H / 100), dpi=100)
            _draw_pitch(ax)
            self._blank = _fig_to_bgr(fig)
            plt.close(fig)
        return self._blank.copy()

    def set_homography(self, H_pixel_to_world: np.ndarray) -> None:
        src = np.array([
            [0.0, 0.0],
            [self.pitch_length_m, 0.0],
            [self.pitch_length_m, self.pitch_width_m],
            [0.0, self.pitch_width_m],
        ], dtype=np.float32)
        dst = np.array([
            [self._pad_x, self._pad_y],
            [self._pad_x + self._pw, self._pad_y],
            [self._pad_x + self._pw, self._pad_y + self._ph],
            [self._pad_x, self._pad_y + self._ph],
        ], dtype=np.float32)
        H_w2c, _ = cv2.findHomography(src, dst)
        if H_w2c is not None:
            self._H_canvas = H_w2c @ H_pixel_to_world

    def _update_homography(self, quad: np.ndarray) -> None:
        src = quad.astype(np.float32).reshape(4, 2)
        dst = np.array([
            [self._pad_x, self._pad_y],
            [self._pad_x + self._pw, self._pad_y],
            [self._pad_x + self._pw, self._pad_y + self._ph],
            [self._pad_x, self._pad_y + self._ph],
        ], dtype=np.float32)
        self._H_canvas, _ = cv2.findHomography(src, dst)

    def _world_to_canvas(self, wx: float, wy: float) -> Tuple[int, int]:
        nx = float(np.clip(wx / self.pitch_length_m, 0.0, 1.0))
        ny = float(np.clip(wy / self.pitch_width_m, 0.0, 1.0))
        return int(self._pad_x + nx * self._pw), int(self._pad_y + ny * self._ph)

    def _pixel_to_canvas(self, foot_px: np.ndarray) -> Optional[Tuple[int, int]]:
        if self._H_canvas is None:
            return None
        pt = np.array([[[float(foot_px[0]), float(foot_px[1])]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self._H_canvas)
        return int(out[0, 0, 0]), int(out[0, 0, 1])

    def render(
        self,
        detections: List[Dict],
        *,
        field_quad_xy: Optional[np.ndarray] = None,
        frame_shape: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        canvas = self._get_blank()

        if field_quad_xy is not None and field_quad_xy.shape == (4, 2):
            self._update_homography(field_quad_xy)

        pad_x, pad_y, pw, ph = self._pad_x, self._pad_y, self._pw, self._ph

        for obj in detections:
            label = obj.get("label", "Player")
            color_bgr = _dot_color(label)
            is_ball = "Ball" in label

            wx = obj.get("world_x_m")
            wy = obj.get("world_y_m")
            if wx is not None and wy is not None:
                rx, ry = self._world_to_canvas(float(wx), float(wy))
            elif self._H_canvas is not None:
                bbox = obj.get("bbox")
                if bbox is None:
                    bbox = (int(obj["x"]), int(obj["y"]), int(obj["w"]), int(obj["h"]))
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                foot = np.array([x + w // 2, y + h], dtype=np.float32)
                pt = self._pixel_to_canvas(foot)
                if pt is None:
                    continue
                rx, ry = pt
            else:
                continue

            if not (pad_x - 20 <= rx <= pad_x + pw + 20 and pad_y - 20 <= ry <= pad_y + ph + 20):
                continue

            r = _BALL_R if is_ball else _DOT_R
            px_per_m = pw / self.pitch_length_m
            r_px = max(4, int(r * px_per_m))

            # Shadow
            cv2.circle(canvas, (rx + 2, ry + 2), r_px + 2, (0, 0, 0), -1, cv2.LINE_AA)
            # Outer ring
            cv2.circle(canvas, (rx, ry), r_px + 2, (20, 20, 20), -1, cv2.LINE_AA)
            # Fill
            cv2.circle(canvas, (rx, ry), r_px, color_bgr, -1, cv2.LINE_AA)
            # Highlight
            if not is_ball:
                hi_r = max(2, r_px - 4)
                cv2.circle(canvas, (rx - 2, ry - 2), hi_r, (255, 255, 255), 1, cv2.LINE_AA)

            # ID label
            if not is_ball:
                tid = str(obj.get("id", obj.get("track_id", "")))
                font = cv2.FONT_HERSHEY_SIMPLEX
                fscale, fthick = 0.38, 1
                (tw, th), _ = cv2.getTextSize(tid, font, fscale, fthick)
                tx = rx - tw // 2
                ty = ry - r_px - 5
                cv2.rectangle(canvas, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2),
                              (20, 20, 20), -1, cv2.LINE_AA)
                cv2.putText(canvas, tid, (tx, ty), font, fscale,
                            (240, 240, 240), fthick, cv2.LINE_AA)

        # Legend
        leg_y = int(_LEGEND_Y0 * _CANVAS_H) + 10
        entries = [("Team A", _DOT_COLORS["Team A"]), ("Team B", _DOT_COLORS["Team B"]),
                   ("Referee", _DOT_COLORS["Referee"]), ("Ball", _DOT_COLORS["Ball"])]
        lx = pad_x + 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        for name, bgr in entries:
            cv2.circle(canvas, (lx + 7, leg_y + 4), 7, (0, 0, 0), -1)
            cv2.circle(canvas, (lx + 7, leg_y + 4), 6, bgr, -1, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(name, font, 0.42, 1)
            cv2.putText(canvas, name, (lx + 18, leg_y + 8), font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
            lx += tw + 38

        return canvas
