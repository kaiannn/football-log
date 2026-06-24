"""评估场地关键点模型的重投影精度。

用法:
    python -m football_log.eval.eval_pitch_keypoints \
        --model runs/roboflow/football-pitch-detection.pt \
        --frames "match.mp4:100,200,300,400,500"

在指定帧上运行关键点检测，计算：
1. 检测到的关键点数量和置信度分布
2. 用于拟合单应矩阵的有效关键点数量
3. 单应矩阵的重投影往返误差 (pixel → world → pixel)
4. 关键点空间分布（左右半场覆盖度）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def _parse_frames(spec: str) -> List[Tuple[str, List[int]]]:
    """Parse 'video.mp4:100,200,300' into [(path, [100,200,300])]."""
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected 'video.mp4:100,200,...', got: {spec}")
    path = parts[0]
    frames = [int(x.strip()) for x in parts[1].split(",")]
    return [(path, frames)]


def _extract_frame(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return frame


def evaluate_keypoints(
    model_path: str,
    video_frames: List[Tuple[str, List[int]]],
    imgsz: int = 640,
    conf_threshold: float = 0.3,
) -> Dict:
    """Run keypoint evaluation on specified frames."""
    from football_log.vision.pitch_keypoint_detector import PitchKeypointDetector, PITCH_KP_WORLD

    detector = PitchKeypointDetector(model_path, imgsz=imgsz)

    results = []
    for video_path, frame_idxs in video_frames:
        for fidx in frame_idxs:
            frame = _extract_frame(video_path, fidx)
            H, kps_px, kp_conf = detector.detect(frame)

            # Count valid keypoints (not NaN world coords, not (0,0), conf >= threshold)
            world_pts = PITCH_KP_WORLD
            valid_mask = (
                np.isfinite(world_pts).all(axis=1)
                & (kp_conf >= conf_threshold)
                & ~((kps_px[:, 0] == 0) & (kps_px[:, 1] == 0))
            )
            n_valid = int(valid_mask.sum())
            n_total = len(kps_px)
            n_nan_world = int(np.isnan(world_pts).any(axis=1).sum())

            # Spatial distribution: how many valid points on left vs right half
            valid_world = world_pts[valid_mask]
            if len(valid_world) > 0:
                n_left = int((valid_world[:, 0] < 52.5).sum())
                n_right = int((valid_world[:, 0] >= 52.5).sum())
            else:
                n_left = n_right = 0

            # Reprojection error: pixel → world → pixel round-trip
            repro_errors = []
            if H is not None:
                from football_log.world.homography import Homography
                homo = Homography(H)
                for i in range(n_total):
                    if not valid_mask[i]:
                        continue
                    u, v = float(kps_px[i, 0]), float(kps_px[i, 1])
                    wx, wy = homo.pixel_to_world(u, v)
                    if np.isnan(wx) or np.isnan(wy):
                        continue
                    u2, v2 = homo.world_to_pixel(wx, wy)
                    if np.isnan(u2) or np.isnan(v2):
                        continue
                    err = np.sqrt((u - u2) ** 2 + (v - v2) ** 2)
                    repro_errors.append(float(err))

            result = {
                "video": video_path,
                "frame_idx": fidx,
                "n_keypoints_total": n_total,
                "n_keypoints_valid": n_valid,
                "n_keypoints_nan_world": n_nan_world,
                "n_left_half": n_left,
                "n_right_half": n_right,
                "has_homography": H is not None,
                "mean_conf": float(kp_conf[valid_mask].mean()) if n_valid > 0 else 0.0,
            }
            if repro_errors:
                errs = np.array(repro_errors)
                result["reproj_mean_px"] = float(errs.mean())
                result["reproj_median_px"] = float(np.median(errs))
                result["reproj_max_px"] = float(errs.max())
                result["reproj_p95_px"] = float(np.percentile(errs, 95))

            results.append(result)

    # Aggregate
    n_frames = len(results)
    agg = {
        "n_frames_evaluated": n_frames,
        "mean_valid_keypoints": float(np.mean([r["n_keypoints_valid"] for r in results])),
        "mean_confidence": float(np.mean([r["mean_conf"] for r in results])),
        "frames_with_homography": sum(1 for r in results if r["has_homography"]),
        "spatial_bias": {
            "mean_left": float(np.mean([r["n_left_half"] for r in results])),
            "mean_right": float(np.mean([r["n_right_half"] for r in results])),
        },
    }

    reproj_means = [r.get("reproj_mean_px") for r in results if "reproj_mean_px" in r]
    if reproj_means:
        agg["reprojection"] = {
            "mean_px": float(np.mean(reproj_means)),
            "median_px": float(np.median(reproj_means)),
            "max_px": float(np.max(reproj_means)),
        }

    return {"aggregate": agg, "per_frame": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", required=True, help="Pitch keypoint model weights (.pt)")
    parser.add_argument("--frames", required=True, help="video.mp4:100,200,300,...")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--out", default=None, help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    video_frames = _parse_frames(args.frames)
    result = evaluate_keypoints(args.model, video_frames, imgsz=args.imgsz, conf_threshold=args.conf)

    output = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(output)

    # Print summary
    agg = result["aggregate"]
    print(f"\n{'=' * 50}")
    print(f"关键点模型评估摘要")
    print(f"{'=' * 50}")
    print(f"  评估帧数: {agg['n_frames_evaluated']}")
    print(f"  平均有效关键点: {agg['mean_valid_keypoints']:.1f}")
    print(f"  平均置信度: {agg['mean_confidence']:.3f}")
    print(f"  拟合成功帧数: {agg['frames_with_homography']}/{agg['n_frames_evaluated']}")
    sp = agg["spatial_bias"]
    print(f"  空间分布: 左半场 {sp['mean_left']:.1f} / 右半场 {sp['mean_right']:.1f}")
    if "reprojection" in agg:
        r = agg["reprojection"]
        print(f"  重投影误差: mean={r['mean_px']:.1f}px, median={r['median_px']:.1f}px, max={r['max_px']:.1f}px")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
