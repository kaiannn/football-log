"""World-projection RMSE on labeled pixel ↔ world correspondences.

The eval set is a JSON of points whose true world coordinates are known
(e.g. corner flags, penalty spots, goal posts):

    {
      "points": [
        {"image_uv": [u, v], "world_xy_m": [x_m, y_m], "name": "corner_flag_NW"},
        ...
      ]
    }

The projector under test is either a 3x3 homography (.npy) or a pinhole
calibration JSON/YAML (same format consumed by --camera-calib in run.py).

Usage:
    python -m football_log.eval.eval_world \\
        --points    data/eval/match_keypoints.json \\
        --homography path/to/homography.npy \\
        --out       runs/baseline/world_metrics.json

    # OR with a pinhole projector:
    python -m football_log.eval.eval_world \\
        --points     data/eval/match_keypoints.json \\
        --camera-calib path/to/camera_calib.json \\
        --out        runs/baseline/world_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from football_log.world.homography import Homography
from football_log.world.pinhole_ground import load_pinhole_ground_projector


def _load_points(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pts = data.get("points", data) if isinstance(data, dict) else data
    if not isinstance(pts, list):
        raise ValueError(f"{path}: expected 'points' to be a list")
    return pts


def _project(
    u: float,
    v: float,
    homography: Homography | None,
    pinhole: Any | None,
) -> Tuple[float, float]:
    if homography is not None:
        return homography.pixel_to_world(u, v)
    wx, wy = pinhole.pixel_to_world_xy_m(u, v)
    return (float("nan"), float("nan")) if wx is None else (wx, wy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--points", required=True, help="JSON of labeled correspondences")
    parser.add_argument("--out", required=True)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--homography", help="3x3 .npy file")
    src.add_argument("--camera-calib", help="Pinhole calib JSON/YAML")
    args = parser.parse_args()

    H: Homography | None = None
    pinhole = None
    if args.homography:
        H = Homography(matrix=np.load(args.homography))
    else:
        pinhole = load_pinhole_ground_projector(args.camera_calib)

    points = _load_points(Path(args.points))
    per_point: List[Dict[str, Any]] = []
    sq_errors: List[float] = []

    for p in points:
        u, v = p["image_uv"]
        gt_x, gt_y = p["world_xy_m"]
        proj_x, proj_y = _project(float(u), float(v), H, pinhole)
        if math.isnan(proj_x) or math.isnan(proj_y):
            per_point.append({"name": p.get("name"), "error_m": None, "reason": "projection failed"})
            continue
        err = math.hypot(proj_x - gt_x, proj_y - gt_y)
        sq_errors.append(err * err)
        per_point.append(
            {
                "name": p.get("name"),
                "image_uv": [u, v],
                "world_xy_truth_m": [gt_x, gt_y],
                "world_xy_pred_m": [proj_x, proj_y],
                "error_m": err,
            }
        )

    n_ok = len(sq_errors)
    rmse = math.sqrt(sum(sq_errors) / n_ok) if n_ok else float("nan")
    valid_errors = [pp["error_m"] for pp in per_point if pp.get("error_m") is not None]
    metrics = {
        "RMSE_m": rmse,
        "MAE_m": (sum(valid_errors) / n_ok) if n_ok else float("nan"),
        "max_error_m": max(valid_errors) if valid_errors else float("nan"),
        "n_points": len(points),
        "n_valid": n_ok,
        "per_point": per_point,
        "_meta": {
            "points_file": str(Path(args.points).resolve()),
            "projector": "homography" if H is not None else "pinhole_ground",
            "projector_path": str(Path(args.homography or args.camera_calib).resolve()),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}  (RMSE={rmse:.3f} m over {n_ok}/{len(points)} valid points)")


if __name__ == "__main__":
    main()
