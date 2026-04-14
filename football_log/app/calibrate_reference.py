"""交互式/文件式多标定物拟合：点击 4 点或读取 refs-json，拟合并导出修正 homography。"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from football_log.world.heuristic_reference_fit import (
    MultiScaleFitResult,
    ReferenceRectangle,
    apply_scales_to_homography,
    fit_reference_scales_multi,
)
from football_log.world.homography import Homography


def _select_points(frame: np.ndarray) -> np.ndarray:
    pts: List[Tuple[int, int]] = []
    vis = frame.copy()
    win = "Select 4 corners (tl,tr,br,bl-ish), ENTER confirm, r reset, q quit"

    def cb(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, cb)
    while True:
        vis[:] = frame
        for i, p in enumerate(pts):
            cv2.circle(vis, p, 4, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(vis, str(i + 1), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and len(pts) == 4:  # enter
            break
        if k == ord("r"):
            pts.clear()
        if k == ord("q"):
            cv2.destroyWindow(win)
            raise SystemExit("已取消")
    cv2.destroyWindow(win)
    return np.asarray(pts, dtype=np.float64).reshape(4, 2)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fit homography scales from one/multi known reference rectangles.")
    p.add_argument("--video", required=True, help="视频路径")
    p.add_argument("--homography", required=True, help="输入 homography .npy（像素->世界）")
    p.add_argument("--frame-idx", type=int, default=1, help="兼容参数：单帧标定时使用（1-based）")
    p.add_argument(
        "--frame-idxs",
        default="",
        help="多帧交互标定，逗号分隔，如 100,220,360；每帧都点一次 4 点",
    )
    p.add_argument("--ref-width-m", type=float, required=True, help="标定物真实宽（米）")
    p.add_argument("--ref-length-m", type=float, required=True, help="标定物真实长（米）")
    p.add_argument(
        "--refs-json",
        default="",
        help="可选：多参考输入 JSON。格式为列表，每项含 image_points_xy(4x2), width_m, length_m",
    )
    p.add_argument("--robust-loss", default="huber", choices=["none", "huber", "cauchy"], help="鲁棒损失")
    p.add_argument("--robust-delta-m", type=float, default=0.2, help="鲁棒阈值（米）")
    p.add_argument("--fit-steps", type=int, default=14, help="启发式优化迭代步数")
    p.add_argument("--out-homography", required=True, help="输出修正后的 homography .npy")
    return p


def _parse_frame_idxs(frame_idxs: str, fallback_frame_idx: int) -> List[int]:
    txt = (frame_idxs or "").strip()
    if not txt:
        return [max(1, int(fallback_frame_idx))]
    out: List[int] = []
    for part in txt.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(max(1, int(part)))
    return out or [max(1, int(fallback_frame_idx))]


def _load_refs_json(path: str) -> List[ReferenceRectangle]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("refs-json 顶层需为数组")
    refs: List[ReferenceRectangle] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        pts = np.asarray(item.get("image_points_xy"), dtype=np.float64).reshape(4, 2)
        w = float(item.get("width_m"))
        l = float(item.get("length_m"))
        refs.append(ReferenceRectangle(image_points_xy=pts, width_m=w, length_m=l))
    if not refs:
        raise ValueError("refs-json 中没有有效参考")
    return refs


def _collect_refs_interactive(
    video: str,
    frame_idxs: Sequence[int],
    width_m: float,
    length_m: float,
) -> List[ReferenceRectangle]:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit("无法打开视频")
    refs: List[ReferenceRectangle] = []
    try:
        for fi in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(fi) - 1))
            ok, frame = cap.read()
            if not ok:
                raise SystemExit(f"读取第 {fi} 帧失败")
            print(f"[calib] frame={fi}: 请点击4个角点后按 Enter")
            pts = _select_points(frame)
            refs.append(ReferenceRectangle(image_points_xy=pts, width_m=width_m, length_m=length_m))
    finally:
        cap.release()
    return refs


def main() -> None:
    args = build_parser().parse_args()
    if not os.path.isfile(args.video):
        raise SystemExit(f"视频不存在: {args.video}")
    if not os.path.isfile(args.homography):
        raise SystemExit(f"homography 不存在: {args.homography}")

    H = np.load(args.homography)
    Hobj = Homography(H)
    if args.refs_json:
        refs = _load_refs_json(args.refs_json)
    else:
        frame_idxs = _parse_frame_idxs(args.frame_idxs, args.frame_idx)
        refs = _collect_refs_interactive(args.video, frame_idxs, args.ref_width_m, args.ref_length_m)
    fit: MultiScaleFitResult = fit_reference_scales_multi(
        Hobj.pixel_to_world,
        refs,
        robust_loss=args.robust_loss,
        robust_delta_m=args.robust_delta_m,
        steps=args.fit_steps,
    )

    H_new = apply_scales_to_homography(Hobj.matrix, fit.scale_x, fit.scale_y)
    np.save(args.out_homography, H_new)

    print("Reference fit done.")
    print(f"n_refs={fit.n_refs}")
    print(f"scale_x={fit.scale_x:.6f}, scale_y={fit.scale_y:.6f}")
    print(f"rmse_m={fit.rmse_m:.4f}, robust_loss={args.robust_loss}, robust_delta_m={args.robust_delta_m}")
    print(f"Saved: {args.out_homography}")


if __name__ == "__main__":
    main()
