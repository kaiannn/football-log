"""Gradio Web UI — 视频文件处理 + 实时摄像头两种模式。"""

import os
import tempfile
import threading
from typing import Optional

import cv2
import gradio as gr

from football_log.app.runner import VideoTrackerPipeline
from football_log.vision.label_utils import parse_team_colors
from football_log.world.pitch_model import PitchSpec


_MODULE1_WEIGHTS = "runs/module1_v1/weights/best.pt"
_MODULE1_CLASS_IDS = {"player": [0], "ball": [1], "referee": [2]}
_PITCH_KP_WEIGHTS = "runs/roboflow/football-pitch-detection.pt"
_BALL_WEIGHTS = "runs/roboflow/football-ball-detection.pt"

_COCO_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
_MODELS = (
    [_MODULE1_WEIGHTS] + _COCO_MODELS
    if os.path.isfile(_MODULE1_WEIGHTS)
    else _COCO_MODELS
)
_DEFAULT_MODEL = _MODULE1_WEIGHTS if os.path.isfile(_MODULE1_WEIGHTS) else "yolov8n.pt"
_TRACKERS = ["bytetrack.yaml", "botsort.yaml"]
_FORMATS = ["both", "jsonl", "csv"]

_active_pipeline: Optional[VideoTrackerPipeline] = None
_pipeline_lock = threading.Lock()


def _resolve_file_path(file_obj) -> Optional[str]:
    if file_obj is None:
        return None
    return file_obj.name if hasattr(file_obj, "name") else str(file_obj)


def _build_pipeline(
    video_source: str,
    model: str,
    tracker: str,
    conf: float,
    imgsz: int,
    detect_every_n: int,
    output_format: str,
    pitch_length: float,
    pitch_width: float,
    pitch_field_detect: bool,
    pitch_field_every_n: int,
    pitch_field_filter: bool,
    team_colors_text: str,
    homography_file,
    camera_calib_file,
    pitch_keypoint_model: Optional[str],
    ball_model: Optional[str],
    ball_slicer: bool,
) -> VideoTrackerPipeline:
    pitch = PitchSpec(length_m=pitch_length, width_m=pitch_width)
    pitch.validate()
    output_dir = tempfile.mkdtemp(prefix="football_log_")
    ids = _MODULE1_CLASS_IDS if model == _MODULE1_WEIGHTS else {}
    return VideoTrackerPipeline(
        video_path=video_source,
        output_dir=output_dir,
        output_format=output_format,
        model_name=model,
        conf=conf,
        imgsz=int(imgsz),
        detect_every_n=int(detect_every_n),
        show_ui=False,
        tracker=tracker,
        homography_path=_resolve_file_path(homography_file),
        camera_calib_path=_resolve_file_path(camera_calib_file),
        pitch=pitch,
        pitch_field_detect=pitch_field_detect,
        pitch_field_every_n=int(pitch_field_every_n),
        pitch_field_filter_tracks=pitch_field_filter,
        team_colors=parse_team_colors(team_colors_text),
        player_class_ids=ids.get("player"),
        ball_class_ids=ids.get("ball"),
        referee_class_ids=ids.get("referee"),
        pitch_keypoint_model=pitch_keypoint_model or None,
        ball_model=ball_model or None,
        ball_slicer=ball_slicer,
        save_video=True,
        save_radar=True,
    )


def _collect_output_files(output_dir: str):
    data_files, overlay_path, radar_path = [], None, None
    for f in os.listdir(output_dir):
        full = os.path.join(output_dir, f)
        if not os.path.isfile(full):
            continue
        if f.endswith("_overlay.mp4"):
            overlay_path = full
        elif f.endswith("_radar.mp4"):
            radar_path = full
        else:
            data_files.append(full)
    return data_files, overlay_path, radar_path


def _run_video_streaming(
    video_path, model, tracker, conf, imgsz, detect_every_n,
    output_format, pitch_length, pitch_width,
    pitch_field_detect, pitch_field_every_n, pitch_field_filter,
    team_colors_text, homography_file, camera_calib_file,
    pitch_kp_enable, pitch_keypoint_model, ball_enable, ball_model, ball_slicer,
):
    global _active_pipeline
    if not video_path:
        raise gr.Error("请上传视频文件")
    # Gradio 6 passes gr.Video value as a dict {"video": {"path": ...}, "subtitles": None}
    if isinstance(video_path, dict):
        video_path = (
            video_path.get("video", {}).get("path")
            or video_path.get("path")
            or video_path.get("name")
        )
    elif hasattr(video_path, "name"):
        video_path = video_path.name
    if not video_path:
        raise gr.Error("无法解析视频路径，请重新上传")
    print(f"[web] video_path resolved → {video_path!r}")
    if not os.path.exists(str(video_path)):
        raise gr.Error(f"视频文件不存在: {video_path}")

    pipeline = _build_pipeline(
        video_path, model, tracker, conf, imgsz, detect_every_n,
        output_format, pitch_length, pitch_width,
        pitch_field_detect, pitch_field_every_n, pitch_field_filter,
        team_colors_text, homography_file, camera_calib_file,
        pitch_keypoint_model if pitch_kp_enable else None,
        ball_model if ball_enable else None,
        ball_slicer,
    )
    with _pipeline_lock:
        _active_pipeline = pipeline

    total_frames = int(pipeline.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    rgb = None

    for _frame, display, _current, frame_idx in pipeline._iter_frames():
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        if total_frames > 0:
            pct = int((frame_idx + 1) / total_frames * 100)
            status = f"帧 {frame_idx + 1}/{total_frames} ({pct}%)"
        else:
            status = f"帧 {frame_idx + 1}"
        yield rgb, [], None, None, status

    pipeline._finish()
    with _pipeline_lock:
        _active_pipeline = None

    if rgb is None:
        raise gr.Error("无法读取视频帧，请检查视频格式是否支持（推荐 H.264 MP4）")

    data_files, overlay_path, radar_path = _collect_output_files(pipeline.data_writer.output_dir)
    records = pipeline.data_writer.records_written
    summary = f"处理完成: {records} 条记录, {pipeline.frame_idx} 帧 | 输出: {pipeline.data_writer.output_dir}"
    yield rgb, data_files, overlay_path, radar_path, summary


def _run_camera_streaming(
    cam_device, model, tracker, conf, imgsz, detect_every_n,
    output_format, pitch_length, pitch_width,
    pitch_field_detect, pitch_field_every_n, pitch_field_filter,
    team_colors_text, homography_file, camera_calib_file,
    pitch_kp_enable, pitch_keypoint_model, ball_enable, ball_model, ball_slicer,
):
    global _active_pipeline
    cam_source = f"cam:{int(cam_device)}"

    pipeline = _build_pipeline(
        cam_source, model, tracker, conf, imgsz, detect_every_n,
        output_format, pitch_length, pitch_width,
        pitch_field_detect, pitch_field_every_n, pitch_field_filter,
        team_colors_text, homography_file, camera_calib_file,
        pitch_keypoint_model if pitch_kp_enable else None,
        ball_model if ball_enable else None,
        ball_slicer,
    )
    with _pipeline_lock:
        _active_pipeline = pipeline

    for _frame, display, _current, frame_idx in pipeline._iter_frames():
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        elapsed = frame_idx / pipeline.fps if pipeline.fps > 0 else 0
        mins, secs = divmod(int(elapsed), 60)
        status = f"实时 | 帧 {frame_idx + 1} | {mins:02d}:{secs:02d} | {pipeline.data_writer.records_written} 条记录"
        yield rgb, [], None, None, status

    pipeline._finish()
    with _pipeline_lock:
        _active_pipeline = None

    data_files, overlay_path, radar_path = _collect_output_files(pipeline.data_writer.output_dir)
    records = pipeline.data_writer.records_written
    summary = f"录制结束: {records} 条记录, {pipeline.frame_idx} 帧 | 输出: {pipeline.data_writer.output_dir}"
    yield rgb, data_files, overlay_path, radar_path, summary


def _stop_pipeline():
    global _active_pipeline
    with _pipeline_lock:
        if _active_pipeline is not None:
            _active_pipeline.request_stop()
            return "正在停止..."
    return "当前无运行中的任务"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Football Log") as app:
        gr.Markdown("# Football Log\n视频轨迹提取与分析")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tab("视频文件"):
                    video_input = gr.Video(label="比赛视频", sources=["upload"], format=None, include_audio=False)
                    run_video_btn = gr.Button("开始处理", variant="primary", size="lg")

                with gr.Tab("实时摄像头"):
                    cam_device = gr.Number(value=0, label="摄像头设备 ID", precision=0)
                    run_cam_btn = gr.Button("开始实时识别", variant="primary", size="lg")

                stop_btn = gr.Button("停止", variant="stop", size="lg")

                gr.Markdown("### 检测参数")
                model = gr.Dropdown(_MODELS, value=_DEFAULT_MODEL, label="YOLO 模型")
                tracker = gr.Dropdown(_TRACKERS, value="bytetrack.yaml", label="跟踪器")
                conf = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="置信度阈值")
                imgsz = gr.Slider(320, 1280, value=640, step=32, label="推理尺寸")
                detect_every_n = gr.Slider(1, 10, value=1, step=1, label="每 N 帧检测")

                gr.Markdown("### 输出")
                output_format = gr.Dropdown(_FORMATS, value="both", label="导出格式")

                with gr.Accordion("场地参数", open=False):
                    pitch_length = gr.Number(value=105.0, label="球场长度 (m)")
                    pitch_width = gr.Number(value=68.0, label="球场宽度 (m)")
                    pitch_field_detect = gr.Checkbox(value=False, label="场地检测")
                    pitch_field_every_n = gr.Slider(1, 60, value=15, step=1, label="场地检测间隔帧数")
                    pitch_field_filter = gr.Checkbox(value=False, label="草皮掩膜过滤")

                with gr.Accordion("分队与标定", open=False):
                    team_colors_text = gr.Textbox(
                        label="手动两队 BGR 颜色 (可选)",
                        placeholder="如 255,255,255;0,255,255",
                    )
                    homography_file = gr.File(label="单应矩阵 .npy", file_types=[".npy"])
                    camera_calib_file = gr.File(label="针孔标定 JSON/YAML", file_types=[".json", ".yaml", ".yml"])

                with gr.Accordion("增强模型 (可选)", open=True):
                    _kp_found = os.path.isfile(_PITCH_KP_WEIGHTS)
                    _ball_found = os.path.isfile(_BALL_WEIGHTS)
                    gr.Markdown(
                        "勾选启用对应模型。权重已自动检测，也可手动填写路径。\n"
                        "- **场地关键点**：提升雷达精度和世界坐标准确性\n"
                        "- **专用球检测**：将球的 mAP 从 0.28 提升至 ~0.92"
                    )
                    with gr.Row():
                        pitch_kp_enable = gr.Checkbox(
                            value=_kp_found,
                            label="启用场地关键点模型",
                        )
                        pitch_keypoint_model = gr.Textbox(
                            label="路径",
                            value=_PITCH_KP_WEIGHTS if _kp_found else "",
                            placeholder="runs/roboflow/football-pitch-detection.pt",
                            scale=3,
                        )
                    with gr.Row():
                        ball_enable = gr.Checkbox(
                            value=_ball_found,
                            label="启用专用球检测模型",
                        )
                        ball_model = gr.Textbox(
                            label="路径",
                            value=_BALL_WEIGHTS if _ball_found else "",
                            placeholder="runs/roboflow/football-ball-detection.pt",
                            scale=3,
                        )
                    ball_slicer = gr.Checkbox(
                        value=False,
                        label="启用 InferenceSlicer（精度更高，速度更慢，需安装 supervision）",
                    )

            with gr.Column(scale=2):
                gr.Markdown("### 结果")
                preview = gr.Image(label="预览", height=420)
                status = gr.Textbox(label="状态", interactive=False)
                with gr.Row():
                    overlay_video = gr.Video(label="标注视频 (下载)", interactive=False)
                    radar_video = gr.Video(label="雷达俯视图 (下载)", interactive=False)
                output_files = gr.File(label="数据文件 (JSONL/CSV)", file_count="multiple", interactive=False)

        shared_params = [
            model, tracker, conf, imgsz, detect_every_n,
            output_format, pitch_length, pitch_width,
            pitch_field_detect, pitch_field_every_n, pitch_field_filter,
            team_colors_text, homography_file, camera_calib_file,
            pitch_kp_enable, pitch_keypoint_model, ball_enable, ball_model, ball_slicer,
        ]
        outputs = [preview, output_files, overlay_video, radar_video, status]

        run_video_btn.click(
            fn=_run_video_streaming,
            inputs=[video_input] + shared_params,
            outputs=outputs,
        )

        run_cam_btn.click(
            fn=_run_camera_streaming,
            inputs=[cam_device] + shared_params,
            outputs=outputs,
        )

        stop_btn.click(fn=_stop_pipeline, outputs=[status])

    return app


demo = build_ui()


def main():
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
