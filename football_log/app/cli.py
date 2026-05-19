"""命令行入口。"""

import argparse
import os
from typing import List, Optional

from football_log.app.runner import VideoTrackerPipeline
from football_log.world.pitch_model import PitchSpec


def _parse_int_list(raw: Optional[str]) -> Optional[List[int]]:
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Football tracking pipeline (YOLO track + structured export).")
    p.add_argument("--video", required=True, help="输入视频路径，或 'cam' / 'cam:0' 使用摄像头")
    p.add_argument("--output-dir", default="outputs", help="输出目录，默认 outputs")
    p.add_argument("--output-format", default="both", choices=["jsonl", "csv", "both"], help="输出格式")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO 模型权重")
    p.add_argument(
        "--tracker",
        default="bytetrack",
        help="跟踪器：bytetrack（默认，最快）/ botsort（更稳的镜头补偿）/ "
        "botsort+reid（BotSORT + Re-ID，遮挡场景下 ID 更稳，比 ByteTrack 慢约 2x）/ "
        "deepsort（独立 DeepSORT：YOLO 只做检测，外部 Kalman + Re-ID 做关联，"
        "遮挡恢复最好，需 pip install deep-sort-realtime）。"
        "也可直接传入自定义 yaml 路径（仅适用于 bytetrack/botsort 系列）。",
    )
    p.add_argument("--conf", type=float, default=0.3, help="检测置信度阈值")
    p.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    p.add_argument("--detect-every-n", type=int, default=1, help="每 N 帧更新一次检测/跟踪")
    p.add_argument("--no-ui", action="store_true", help="无界面批处理模式")

    # ----- Class IDs (defaults match COCO yolov8n.pt; override for fine-tuned weights) -----
    p.add_argument(
        "--player-class-id",
        default=None,
        help="球员类的源 class ID（逗号分隔，多个 ID 视为同一类）。COCO 默认: 0；"
        "Module 1 自训练权重 [player,ball,referee] 应设为 0",
    )
    p.add_argument(
        "--ball-class-id",
        default=None,
        help="球的源 class ID（逗号分隔）。COCO 默认: 32；"
        "Module 1 自训练权重 [player,ball,referee] 应设为 1",
    )
    p.add_argument(
        "--referee-class-id",
        default=None,
        help="裁判的源 class ID（逗号分隔，可选）。未提供则不识别裁判；"
        "Module 1 自训练权重 [player,ball,referee] 应设为 2",
    )

    # ----- World coordinates -----
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
    p.add_argument(
        "--auto-calibration-keyframes",
        default=None,
        help="JSON 路径：手标关键帧（pixel↔world 点对），跨帧用 Lucas–Kanade 光流传播。"
        "比静态 --homography 更适合镜头会移动的转播视频。优先级高于其它标定来源。",
    )
    p.add_argument(
        "--homography-sequence",
        default=None,
        help=".npy 路径：形如 (N, 3, 3) 的逐帧单应矩阵（如 TVCalib 离线输出）。"
        "若与 --auto-calibration-keyframes 同时提供，后者优先。",
    )
    p.add_argument(
        "--homography-smoothing-alpha",
        type=float,
        default=0.3,
        help="单应矩阵平滑 EMA 系数（0,1]，越小越平滑。仅对自动标定生效，默认 0.3。",
    )
    p.add_argument("--pitch-length-m", type=float, default=105.0, help="球场长度（米），写入 meta")
    p.add_argument("--pitch-width-m", type=float, default=68.0, help="球场宽度（米），写入 meta")

    # ----- Pitch field estimation (default ON for spectator filtering) -----
    p.add_argument(
        "--pitch-field-detect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用草地+场线+四边形估计（默认开启）；用 --no-pitch-field-detect 关闭",
    )
    p.add_argument(
        "--pitch-field-every-n",
        type=int,
        default=15,
        help="每 N 帧更新一次场地估计（其余帧复用上一结果），默认 15",
    )
    p.add_argument(
        "--pitch-field-filter-tracks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="用草地掩膜过滤观众席误检（默认开启，依赖 --pitch-field-detect）；"
        "用 --no-pitch-field-filter-tracks 关闭",
    )

    p.add_argument(
        "--bev-smoothing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="启用 BEV (世界坐标) 卡尔曼平滑：每条 track 在米制坐标下做常速度卡尔曼滤波，"
        "降低跳跃/遮挡导致的米级跳变；仅在已配置 --homography 或 --camera-calib 时生效。"
        "输出额外 world_x_m_smoothed / world_y_m_smoothed 列。",
    )

    p.add_argument(
        "--team-classifier",
        default="hsv",
        choices=["hsv", "keypoint"],
        help="分队器：hsv（默认，HSV K-Means，黑白球衣会失效）/ "
        "keypoint（基于姿态关键点 + LAB 颜色投票，更稳但慢约 2x，需 yolov8n-pose.pt）",
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
        auto_calibration_keyframes=args.auto_calibration_keyframes,
        homography_sequence_path=args.homography_sequence,
        homography_smoothing_alpha=args.homography_smoothing_alpha,
        pitch=pitch,
        pitch_field_detect=args.pitch_field_detect,
        pitch_field_every_n=args.pitch_field_every_n,
        pitch_field_filter_tracks=args.pitch_field_filter_tracks,
        team_colors=team_colors,
        team_classifier_kind=args.team_classifier,
        player_class_ids=_parse_int_list(args.player_class_id),
        ball_class_ids=_parse_int_list(args.ball_class_id),
        referee_class_ids=_parse_int_list(args.referee_class_id),
        bev_smoothing=args.bev_smoothing,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
