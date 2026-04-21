"""视频/摄像头读取 → 跟踪 → 可选世界坐标 → 导出。

Pipeline 支持通过 Protocol 注入自定义组件：
- detector:   protocols.Detector      (默认 YoloByteTrackTracker)
- team_cls:   protocols.TeamClassifierProto (默认 TeamClassifier)
- projector:  protocols.WorldProjector (默认 HomographyProjector / PinholeGroundProjector)
- pitch_est:  protocols.PitchEstimator (默认 PitchFieldEstimator)
- exporter:   protocols.Exporter       (默认 TrackingDataWriter)
"""

import os
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import cv2
import numpy as np

from football_log.io.export import TrackingDataWriter
from football_log.pitch.field_estimator import PitchFieldEstimator, TemporalPitchSmoother
from football_log.pitch.integration import filter_objects_in_grass_mask
from football_log.pitch.observation import PitchObservation
from football_log.protocols import Detection
from football_log.ui.overlay import draw_frame_hud, draw_pitch_observation, draw_tracking_overlay
from football_log.vision.team_classifier import TeamClassifier
from football_log.vision.tracker import YoloByteTrackTracker
from football_log.world.homography import Homography, HomographyProjector
from football_log.world.pitch_model import PitchSpec
from football_log.world.pinhole_ground import PinholeGroundProjector, load_pinhole_ground_projector

if TYPE_CHECKING:
    from football_log.protocols import (
        Detector,
        Exporter,
        PitchEstimator,
        TeamClassifierProto,
        WorldProjector,
    )


def _parse_video_source(source: str):
    if source.startswith("cam"):
        parts = source.split(":")
        device_id = int(parts[1]) if len(parts) > 1 else 0
        return device_id, True
    return source, False


def _load_homography(path: Optional[str]) -> Optional[Homography]:
    if not path or not os.path.isfile(path):
        return None
    arr = np.load(path)
    if arr.shape != (3, 3):
        raise ValueError(f"Homography must be 3x3, got {arr.shape}")
    return Homography(arr)


def _enrich_detections(
    detections: List[Detection],
    projector: Optional["WorldProjector"],
) -> List[Detection]:
    if projector is None:
        return detections
    for det in detections:
        wx, wy = projector.project(det.bbox, det.label)
        det.world_x_m = wx
        det.world_y_m = wy
    return detections


class VideoTrackerPipeline:
    """核心流水线。

    使用方式一（默认，向后兼容）：
        pipeline = VideoTrackerPipeline(video_path="match.mp4")

    使用方式二（注入自定义组件）：
        pipeline = VideoTrackerPipeline(
            video_path="match.mp4",
            detector=MyCustomDetector(),
            projector=MyCustomProjector(),
        )
    """

    def __init__(
        self,
        video_path: str,
        output_dir: str = "outputs",
        output_format: str = "both",
        model_name: str = "yolov8n.pt",
        conf: float = 0.3,
        imgsz: int = 640,
        detect_every_n: int = 1,
        show_ui: bool = True,
        tracker: str = "bytetrack.yaml",
        homography_path: Optional[str] = None,
        camera_calib_path: Optional[str] = None,
        pitch: Optional[PitchSpec] = None,
        pitch_field_detect: bool = False,
        pitch_field_every_n: int = 15,
        pitch_field_temporal_smooth: bool = True,
        pitch_field_filter_tracks: bool = False,
        team_colors: Optional[List[tuple]] = None,
        # ------ 插件注入点 ------
        detector: Optional["Detector"] = None,
        team_cls: Optional["TeamClassifierProto"] = None,
        projector: Optional["WorldProjector"] = None,
        pitch_est: Optional["PitchEstimator"] = None,
        exporter: Optional["Exporter"] = None,
    ):
        self.video_path = video_path
        source, self.is_camera = _parse_video_source(video_path)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            kind = "摄像头" if self.is_camera else "视频文件"
            raise SystemExit(f"无法打开{kind}: {video_path}")

        if self.is_camera:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if not self.fps or self.fps <= 0:
                self.fps = 30.0
        else:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.delay = int(1000 / self.fps)
        self.show_ui = show_ui
        self.detect_every_n = max(1, detect_every_n)
        self.frame_idx = 0
        self._stop_requested = False

        # ------ 组件初始化（注入优先，否则创建默认） ------

        self.team_classifier: Any = team_cls or TeamClassifier(team_colors=team_colors)

        if detector is not None:
            self._detector = detector
        else:
            yolo = YoloByteTrackTracker(
                model_name=model_name,
                conf=conf,
                imgsz=imgsz,
                tracker=tracker,
            )
            yolo.set_team_classifier(self.team_classifier)
            self._detector = yolo
        self.yolo_tracker = self._detector

        if projector is not None:
            self._projector: Optional["WorldProjector"] = projector
        else:
            _pinhole: Optional[PinholeGroundProjector] = None
            if camera_calib_path:
                _pinhole = load_pinhole_ground_projector(camera_calib_path)
            _H = _load_homography(homography_path)
            if _pinhole is not None:
                self._projector = _pinhole
            elif _H is not None:
                self._projector = HomographyProjector(_H)
            else:
                self._projector = None
        self.pitch = pitch or PitchSpec()

        self.pitch_field_detect = bool(pitch_field_detect)
        self.pitch_field_every_n = max(1, int(pitch_field_every_n))
        if pitch_est is not None:
            self._pitch_estimator: Optional[Any] = pitch_est
        else:
            self._pitch_estimator = PitchFieldEstimator() if self.pitch_field_detect else None
        self._pitch_smoother: Optional[TemporalPitchSmoother] = (
            TemporalPitchSmoother() if (self.pitch_field_detect and pitch_field_temporal_smooth) else None
        )
        self._last_pitch_obs: Optional[PitchObservation] = None
        self.pitch_field_filter_tracks = bool(pitch_field_filter_tracks)

        if self.is_camera:
            stem = f"cam_{time.strftime('%Y%m%d_%H%M%S')}"
        else:
            stem = os.path.splitext(os.path.basename(self.video_path))[0]

        world_on = self._projector is not None
        extra_meta = {
            "pitch_length_m": self.pitch.length_m,
            "pitch_width_m": self.pitch.width_m,
            "homography_path": homography_path,
            "camera_calib_path": camera_calib_path,
            "world_coords_enabled": world_on,
            "pitch_field_detect": self.pitch_field_detect,
            "pitch_field_every_n": self.pitch_field_every_n,
            "pitch_field_filter_tracks": self.pitch_field_filter_tracks,
        }

        if exporter is not None:
            self.data_writer: Any = exporter
        else:
            self.data_writer = TrackingDataWriter(
                output_dir=output_dir,
                output_prefix=f"{stem}_tracks",
                output_format=output_format,
                fps=self.fps,
                video_path=self.video_path,
                extra_meta=extra_meta,
            )

        self.last_tracked_detections: List[Detection] = []

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        det = self._detector
        model_info = "custom"
        if isinstance(det, YoloByteTrackTracker):
            m = det.model
            model_info = str(getattr(m, "ckpt_path", None) or getattr(m, "model_name", "yolo"))
        mode = "camera" if self.is_camera else "file"
        print(f"Track pipeline ({mode}): model={model_info}, detect_every_n={self.detect_every_n}")
        print(
            f"Output dir: {getattr(self.data_writer, 'output_dir', '?')}, "
            f"world coords: {self._projector is not None}, "
            f"pitch field: {self.pitch_field_detect}"
        )

        for frame, display, current, frame_idx in self._iter_frames():
            if self.show_ui:
                cv2.imshow("Video Tracker", display)
                key = cv2.waitKey(1 if self.is_camera else self.delay) & 0xFF
                if key == ord("q"):
                    break

        self._finish()

    def _iter_frames(self):
        while not self._stop_requested:
            ret, frame = self.cap.read()
            if not ret:
                break

            should_detect = (self.frame_idx % self.detect_every_n == 0) or not self.last_tracked_detections
            if should_detect:
                self.last_tracked_detections = self._detector.detect(frame)

            detections = list(self.last_tracked_detections)

            if self.pitch_field_detect and self._pitch_estimator is not None:
                if self.frame_idx % self.pitch_field_every_n == 0 or self._last_pitch_obs is None:
                    obs = self._pitch_estimator.estimate(frame)
                    if self._pitch_smoother is not None:
                        obs = self._pitch_smoother.smooth(obs)
                    self._last_pitch_obs = obs
                if self.pitch_field_filter_tracks and self._last_pitch_obs is not None:
                    as_dicts = [d.to_dict() for d in detections]
                    filtered = filter_objects_in_grass_mask(as_dicts, self._last_pitch_obs.grass_mask)
                    filtered_ids = {o["id"] for o in filtered}
                    detections = [d for d in detections if d.track_id in filtered_ids]

            detections = _enrich_detections(detections, self._projector)
            self.data_writer.write_frame(self.frame_idx, detections)

            current_dicts = [d.to_dict() for d in detections]
            display = frame.copy()
            if self.pitch_field_detect and self._last_pitch_obs is not None:
                draw_pitch_observation(display, self._last_pitch_obs)
            draw_tracking_overlay(display, current_dicts)
            draw_frame_hud(display, self.frame_idx, self.detect_every_n)

            yield frame, display, current_dicts, self.frame_idx
            self.frame_idx += 1

    def _finish(self) -> None:
        self.cap.release()
        self.data_writer.close()
        if self.show_ui:
            cv2.destroyAllWindows()
        print(f"轨迹导出完成，记录数: {self.data_writer.records_written}")
