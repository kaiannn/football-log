"""端到端验证像素→世界坐标投影质量。

用法:
    python3 -m football_log.world.validate_projection --video match.mp4 --homography homography.npy
    python3 -m football_log.world.validate_projection --video match.mp4 --auto-calibration-keyframes keyframes.json

检查项:
    1. 重投影往返误差 (pixel → world → pixel)
    2. 物理约束 (球员速度、出界检测)
    3. 已知标定点验证 (如果提供 --check-points)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from football_log.world.homography import Homography, bbox_foot_point
from football_log.world.auto_calibration import (
    AutoCalibrationProjector,
    HomographySmoother,
    KeyframeOpticalFlowSource,
    load_keyframes_json,
)


def _load_H(path: str) -> Homography:
    arr = np.load(path)
    if arr.shape != (3, 3):
        raise ValueError(f"Homography must be 3x3, got {arr.shape}")
    return Homography(arr)


def check_reprojection(H: Homography, n_samples: int = 200) -> dict:
    """在图像空间均匀采样点，做 pixel → world → pixel 往返，统计误差。"""
    errors = []
    # Sample points across the image (assume 1920x1080, adjust if needed)
    for _ in range(n_samples):
        u = np.random.uniform(100, 1820)
        v = np.random.uniform(100, 980)
        wx, wy = H.pixel_to_world(u, v)
        if np.isnan(wx) or np.isnan(wy):
            continue
        u2, v2 = H.world_to_pixel(wx, wy)
        if np.isnan(u2) or np.isnan(v2):
            continue
        err_px = np.sqrt((u - u2) ** 2 + (v - v2) ** 2)
        errors.append(err_px)
    if not errors:
        return {"error": "no valid points"}
    errors = np.array(errors)
    return {
        "n_points": len(errors),
        "mean_px": float(errors.mean()),
        "median_px": float(np.median(errors)),
        "max_px": float(errors.max()),
        "p95_px": float(np.percentile(errors, 95)),
    }


def check_physics(H: Homography, detections_path: str, fps: float) -> dict:
    """检查轨迹的物理约束：速度、出界。"""
    if not os.path.isfile(detections_path):
        return {"error": f"file not found: {detections_path}"}

    records = []
    with open(detections_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return {"error": "no records"}

    # Group by track_id
    from collections import defaultdict
    tracks = defaultdict(list)
    for r in records:
        tid = r.get("track_id", -1)
        wx = r.get("world_x_m")
        wy = r.get("world_y_m")
        if wx is not None and wy is not None:
            tracks[tid].append((r["frame_idx"], float(wx), float(wy)))

    violations = []
    speeds = []
    out_of_bounds = 0
    total_points = 0

    for tid, points in tracks.items():
        points.sort(key=lambda p: p[0])
        for i in range(len(points)):
            fi, wx, wy = points[i]
            total_points += 1
            # Out of bounds check (with 5m margin for tolerance)
            if wx < -5 or wx > 110 or wy < -5 or wy > 73:
                out_of_bounds += 1
                violations.append(f"  track {tid} frame {fi}: ({wx:.1f}, {wy:.1f}) out of bounds")

            if i > 0:
                fi_prev, wx_prev, wy_prev = points[i - 1]
                dt = (fi - fi_prev) / fps
                if dt > 0:
                    dist = np.sqrt((wx - wx_prev) ** 2 + (wy - wy_prev) ** 2)
                    speed = dist / dt
                    speeds.append(speed)
                    if speed > 15:  # 54 km/h — physically impossible for humans
                        violations.append(
                            f"  track {tid} frame {fi_prev}->{fi}: "
                            f"speed={speed:.1f} m/s ({speed*3.6:.0f} km/h) — impossible"
                        )

    speeds = np.array(speeds) if speeds else np.array([0.0])
    return {
        "total_points": total_points,
        "n_tracks": len(tracks),
        "out_of_bounds": out_of_bounds,
        "speed_mean_mps": float(speeds.mean()),
        "speed_p95_mps": float(np.percentile(speeds, 95)),
        "speed_max_mps": float(speeds.max()),
        "n_speed_violations": sum(1 for s in speeds if s > 15),
        "n_total_violations": len(violations),
        "sample_violations": violations[:10],
    }


def check_known_points(H: Homography, points: List[Tuple[float, float, float, float]]) -> dict:
    """验证已知标定点：(pixel_u, pixel_v, expected_world_x, expected_world_y)。"""
    results = []
    for u, v, exp_wx, exp_wy in points:
        wx, wy = H.pixel_to_world(u, v)
        if np.isnan(wx) or np.isnan(wy):
            results.append({
                "pixel": (u, v),
                "expected": (exp_wx, exp_wy),
                "got": None,
                "error_m": None,
                "status": "FAIL (NaN)",
            })
        else:
            err = np.sqrt((wx - exp_wx) ** 2 + (wy - exp_wy) ** 2)
            results.append({
                "pixel": (u, v),
                "expected": (exp_wx, exp_wy),
                "got": (round(wx, 2), round(wy, 2)),
                "error_m": round(err, 3),
                "status": "OK" if err < 2.0 else "WARN" if err < 5.0 else "FAIL",
            })
    return {"points": results}


def main():
    parser = argparse.ArgumentParser(description="Validate pixel→world projection quality.")
    parser.add_argument("--video", required=True, help="视频路径（用于读取帧尺寸）")
    parser.add_argument("--homography", default=None, help="3×3 单应矩阵 .npy")
    parser.add_argument("--auto-calibration-keyframes", default=None, help="关键帧 JSON")
    parser.add_argument("--detections", default=None, help="JSONL 轨迹文件（用于物理约束检查）")
    parser.add_argument("--fps", type=float, default=25.0, help="视频 FPS")
    parser.add_argument("--check-points", default=None,
                        help='已知标定点 JSON，格式: [[u,v,wx,wy], ...]')
    args = parser.parse_args()

    # Load H
    H = None
    if args.homography:
        H = _load_H(args.homography)
        print(f"[验证] 加载 homography: {args.homography}")
    elif args.auto_calibration_keyframes:
        kfs = load_keyframes_json(Path(args.auto_calibration_keyframes))
        source = KeyframeOpticalFlowSource(kfs)
        smoother = HomographySmoother(alpha=0.3)
        proj = AutoCalibrationProjector(source, smoother)
        # Process first frame to get H
        cap = cv2.VideoCapture(args.video)
        ret, frame = cap.read()
        cap.release()
        if ret:
            proj.prepare_for_frame(0, frame)
            if proj.current_homography is not None:
                H = proj.current_homography
                print(f"[验证] 从关键帧估计 homography (frame 0)")
            else:
                print("[验证] 无法从关键帧估计 homography")
    else:
        print("[验证] 需要 --homography 或 --auto-calibration-keyframes")
        return

    if H is None:
        print("[验证] 无法加载 homography，退出")
        return

    # 1. Reprojection
    print("\n" + "=" * 50)
    print("1. 重投影往返误差 (pixel → world → pixel)")
    print("=" * 50)
    repro = check_reprojection(H)
    for k, v in repro.items():
        print(f"  {k}: {v}")

    # 2. Known points
    if args.check_points:
        print("\n" + "=" * 50)
        print("2. 已知标定点验证")
        print("=" * 50)
        pts = json.loads(args.check_points)
        result = check_known_points(H, [tuple(p) for p in pts])
        for p in result["points"]:
            print(f"  {p['status']}: pixel={p['pixel']} → got={p['got']}, "
                  f"expected={p['expected']}, err={p['error_m']}m")

    # 3. Physics
    if args.detections:
        print("\n" + "=" * 50)
        print("3. 物理约束检查")
        print("=" * 50)
        phys = check_physics(H, args.detections, args.fps)
        for k, v in phys.items():
            if k == "sample_violations":
                if v:
                    print(f"  违规示例:")
                    for line in v:
                        print(f"    {line}")
            else:
                print(f"  {k}: {v}")

    print("\n" + "=" * 50)
    print("诊断建议")
    print("=" * 50)
    if repro.get("mean_px", 0) > 5:
        print("  ⚠ 重投影误差 > 5px — H 矩阵可能不准确")
    if repro.get("mean_px", 0) > 20:
        print("  ✗ 重投影误差 > 20px — H 矩阵严重失准，检查标定点")
    if args.detections and phys.get("n_speed_violations", 0) > 0:
        print(f"  ⚠ {phys['n_speed_violations']} 个速度违规 — 投影或跟踪有误")
    if args.detections and phys.get("out_of_bounds", 0) > 10:
        print(f"  ⚠ {phys['out_of_bounds']} 个出界点 — H 可能方向反了或尺度不对")


if __name__ == "__main__":
    main()
