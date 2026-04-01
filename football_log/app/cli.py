"""命令行入口。"""

import argparse
import os

from football_log.app.runner import VideoTrackerPipeline
from football_log.world.pitch_model import PitchSpec


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Football tracking pipeline (YOLO track + structured export).")
    p.add_argument("--video", required=True, help="输入视频路径")
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
    p.add_argument("--pitch-length-m", type=float, default=105.0, help="球场长度（米），写入 meta")
    p.add_argument("--pitch-width-m", type=float, default=68.0, help="球场宽度（米），写入 meta")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not os.path.exists(args.video):
        raise SystemExit(f"视频文件不存在: {args.video}")

    pitch = PitchSpec(length_m=args.pitch_length_m, width_m=args.pitch_width_m)
    pitch.validate()

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
        pitch=pitch,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
