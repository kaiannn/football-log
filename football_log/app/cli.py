"""命令行入口。"""

import argparse
import os

from football_log.app.runner import VideoTrackerPipeline
from football_log.world.pitch_model import PitchSpec


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Football tracking pipeline (YOLO track + structured export).")
    p.add_argument("--video", required=True, help="输入视频路径，或 'cam' / 'cam:0' 使用摄像头")
    p.add_argument("--output-dir", default="outputs", help="输出目录，默认 outputs")
    p.add_argument("--output-format", default="both", choices=["jsonl", "csv", "both"], help="输出格式")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO 模型权重")
    p.add_argument("--tracker", default="bytetrack.yaml", help="跟踪器配置，如 bytetrack.yaml / botsort.yaml")
    p.add_argument("--conf", type=float, default=0.3, help="检测置信度阈值")
    p.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    p.add_argument("--detect-every-n", type=int, default=1, help="每 N 帧更新一次检测/跟踪")
    p.add_argument("--no-ui", action="store_true", help="无界面批处理模式")
    p.add_argument(
        "--homography",
        default=None,
        help="3×3 单应矩阵 .npy 路径（像素→世界米）；未提供则 world_x/y 为空",
    )
    p.add_argument(
        "--camera-calib",
        default=None,
        help="针孔+地面平面标定 JSON/YAML（K、dist、R|t 或 R|C、可选 ground_plane）；"
        "若与 --homography 同时给出，优先使用针孔求交",
    )
    p.add_argument("--pitch-length-m", type=float, default=105.0, help="球场长度（米），写入 meta")
    p.add_argument("--pitch-width-m", type=float, default=68.0, help="球场宽度（米），写入 meta")
    p.add_argument(
        "--pitch-field-detect",
        action="store_true",
        help="启用草地+场线+四边形估计（OpenCV，CPU 可部署）；与 SoccerNet 系场线/标定思路一致",
    )
    p.add_argument(
        "--pitch-field-every-n",
        type=int,
        default=15,
        help="每 N 帧更新一次场地估计（其余帧复用上一结果），默认 15",
    )
    p.add_argument(
        "--pitch-field-filter-tracks",
        action="store_true",
        help="用草地掩膜过滤锚点不在草皮上的目标（球员脚底/球心）",
    )
    p.add_argument(
        "--team-colors",
        default=None,
        help="手动指定两队 BGR 颜色，格式 'B,G,R;B,G,R'，如 '255,255,255;0,255,255'（白 vs 黄）；"
        "未提供则自动 K-Means 聚类",
    )
    return p


def _parse_team_colors(raw: str):
    if not raw:
        return None
    parts = raw.strip().split(";")
    if len(parts) < 2:
        raise SystemExit("--team-colors 需要两组 BGR，用分号分隔，如 '255,255,255;0,255,255'")
    colors = []
    for p in parts[:2]:
        nums = [int(x.strip()) for x in p.split(",")]
        if len(nums) != 3:
            raise SystemExit(f"BGR 颜色需要 3 个整数，得到: {p}")
        colors.append(tuple(nums))
    return colors


def main() -> None:
    args = build_parser().parse_args()
    if not args.video.startswith("cam") and not os.path.exists(args.video):
        raise SystemExit(f"视频文件不存在: {args.video}")

    pitch = PitchSpec(length_m=args.pitch_length_m, width_m=args.pitch_width_m)
    pitch.validate()

    team_colors = _parse_team_colors(args.team_colors)

    pipeline = VideoTrackerPipeline(
        video_path=args.video,
        output_dir=args.output_dir,
        output_format=args.output_format,
        model_name=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        detect_every_n=args.detect_every_n,
        show_ui=not args.no_ui,
        tracker=args.tracker,
        homography_path=args.homography,
        camera_calib_path=args.camera_calib,
        pitch=pitch,
        pitch_field_detect=args.pitch_field_detect,
        pitch_field_every_n=args.pitch_field_every_n,
        pitch_field_filter_tracks=args.pitch_field_filter_tracks,
        team_colors=team_colors,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
