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
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import cv2
import numpy as np

from football_log.io.export import TrackingDataWriter
from football_log.pitch.field_estimator import PitchFieldEstimator, TemporalPitchSmoother
from football_log.pitch.integration import filter_objects_in_grass_mask
from football_log.pitch.observation import PitchObservation
from football_log.protocols import Detection
from football_log.ui.overlay import draw_frame_hud, draw_pitch_observation, draw_tracking_overlay
from football_log.ui.radar import RadarRenderer
from football_log.vision.team_classifier import TeamClassifier
from football_log.vision.team_classifier_keypoint import KeypointTeamClassifier
from football_log.vision.tracker import YoloByteTrackTracker
from football_log.world.auto_calibration import (
    AutoCalibrationProjector,
    HomographySequenceSource,
    HomographySmoother,
    KeyframeOpticalFlowSource,
    load_keyframes_json,
)
from football_log.world.homography import Homography, HomographyProjector
from football_log.world.pitch_model import PitchSpec
from football_log.world.pinhole_ground import PinholeGroundProjector, load_pinhole_ground_projector
from football_log.world.track_filter import TrackFilter, jump_likelihood_from_height_change

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
        auto_calibration_keyframes: Optional[str] = None,
        homography_sequence_path: Optional[str] = None,
        homography_smoothing_alpha: float = 0.3,
        pitch: Optional[PitchSpec] = None,
        pitch_field_detect: bool = True,
        pitch_field_every_n: int = 15,
        pitch_field_temporal_smooth: bool = True,
        pitch_field_filter_tracks: bool = True,
        team_colors: Optional[List[tuple]] = None,
        team_classifier_kind: str = "hsv",
        player_class_ids: Optional[List[int]] = None,
        ball_class_ids: Optional[List[int]] = None,
        referee_class_ids: Optional[List[int]] = None,
        bev_smoothing: bool = False,
        team_class_model: Optional[str] = None,
        save_video: bool = False,
        save_radar: bool = False,
        save_debug_overlay: bool = False,
        pitch_keypoint_model: Optional[str] = None,
        ball_model: Optional[str] = None,
        ball_slicer: bool = False,
        # ------ 插件注入点 ------
        detector: Optional["Detector"] = None,
        team_cls: Optional["TeamClassifierProto"] = None,
        projector: Optional["WorldProjector"] = None,
        pitch_est: Optional["PitchEstimator"] = None,
        exporter: Optional["Exporter"] = None,
    ):
        self.video_path = video_path
        source, self.is_camera = _parse_video_source(video_path)
        # Defer VideoCapture open until after all model loading — on macOS/AVFoundation
        # the file handle can be dropped while PyTorch/Metal initialises CUDA/MPS.
        self._video_source = source

        # Probe FPS before models load (file stays closed after probe).
        _probe = cv2.VideoCapture(source)
        if not _probe.isOpened():
            kind = "摄像头" if self.is_camera else "视频文件"
            raise SystemExit(f"无法打开{kind}: {video_path}")
        if self.is_camera:
            self.fps = _probe.get(cv2.CAP_PROP_FPS)
            if not self.fps or self.fps <= 0:
                self.fps = 30.0
        else:
            self.fps = _probe.get(cv2.CAP_PROP_FPS) or 25
        _probe.release()
        self.delay = int(1000 / self.fps)
        self.show_ui = show_ui
        self.detect_every_n = max(1, detect_every_n)
        self.frame_idx = 0
        self._stop_requested = False

        # ------ 组件初始化（注入优先，否则创建默认） ------

        if team_cls is not None:
            self.team_classifier: Any = team_cls
        elif team_classifier_kind == "keypoint":
            self.team_classifier = KeypointTeamClassifier(team_colors=team_colors)
        else:
            self.team_classifier = TeamClassifier(team_colors=team_colors)

        if detector is not None:
            self._detector = detector
        elif team_class_model:
            # Module 3B: 6-class YOLO encodes team directly — no separate team classifier.
            # Class layout from default_team_class_map():
            #   0=team_a_player  1=team_b_player  2=goalkeeper_a  3=goalkeeper_b
            #   4=referee  5=ball
            yolo = YoloByteTrackTracker(
                model_name=team_class_model,
                conf=conf,
                imgsz=imgsz,
                tracker=tracker,
                player_class_ids=(),
                ball_class_ids=(5,),
                referee_class_ids=(4,),
                team_a_class_ids=(0, 2),
                team_b_class_ids=(1, 3),
            )
            self._detector = yolo
        elif tracker.strip().lower() == "deepsort":
            from football_log.vision.deepsort_tracker import DeepSortTracker
            ds = DeepSortTracker(
                model_name=model_name,
                conf=conf,
                imgsz=imgsz,
                player_class_ids=tuple(player_class_ids) if player_class_ids else (0,),
                ball_class_ids=tuple(ball_class_ids) if ball_class_ids else (32,),
                referee_class_ids=tuple(referee_class_ids) if referee_class_ids else None,
            )
            ds.set_team_classifier(self.team_classifier)
            self._detector = ds
        else:
            yolo = YoloByteTrackTracker(
                model_name=model_name,
                conf=conf,
                imgsz=imgsz,
                tracker=tracker,
                player_class_ids=tuple(player_class_ids) if player_class_ids else (0,),
                ball_class_ids=tuple(ball_class_ids) if ball_class_ids else (32,),
                referee_class_ids=tuple(referee_class_ids) if referee_class_ids else None,
            )
            yolo.set_team_classifier(self.team_classifier)
            self._detector = yolo

        if projector is not None:
            self._projector: Optional["WorldProjector"] = projector
        else:
            _pinhole: Optional[PinholeGroundProjector] = None
            if camera_calib_path:
                _pinhole = load_pinhole_ground_projector(camera_calib_path)
            _H = _load_homography(homography_path)
            _auto: Optional[AutoCalibrationProjector] = None
            if auto_calibration_keyframes:
                kfs = load_keyframes_json(Path(auto_calibration_keyframes))
                _source = KeyframeOpticalFlowSource(kfs)
                _smoother = HomographySmoother(alpha=homography_smoothing_alpha)
                _auto = AutoCalibrationProjector(_source, _smoother)
            elif homography_sequence_path:
                _source = HomographySequenceSource(Path(homography_sequence_path))
                _smoother = HomographySmoother(alpha=homography_smoothing_alpha)
                _auto = AutoCalibrationProjector(_source, _smoother)

            if _auto is not None:
                self._projector = _auto
            elif _pinhole is not None:
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

        # BEV Kalman smoothing — only useful if a projector is set.
        self.bev_smoothing = bool(bev_smoothing) and self._projector is not None
        self._track_filter: Optional[TrackFilter] = (
            TrackFilter(fps=self.fps) if self.bev_smoothing else None
        )
        self._prev_bbox_heights: Dict[int, float] = {}  # track_id → previous bbox height

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
            "auto_calibration_keyframes": auto_calibration_keyframes,
            "homography_sequence_path": homography_sequence_path,
            "world_coords_enabled": world_on,
            "auto_calibration_enabled": isinstance(self._projector, AutoCalibrationProjector),
            "pitch_field_detect": self.pitch_field_detect,
            "pitch_field_every_n": self.pitch_field_every_n,
            "pitch_field_filter_tracks": self.pitch_field_filter_tracks,
            "player_class_ids": list(player_class_ids) if player_class_ids else [0],
            "ball_class_ids": list(ball_class_ids) if ball_class_ids else [32],
            "referee_class_ids": list(referee_class_ids) if referee_class_ids else [],
            "team_classifier_kind": team_classifier_kind if team_cls is None else "custom",
            "bev_smoothing_enabled": self.bev_smoothing,
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
        self._save_video = save_video
        self._save_radar = save_radar
        self._save_debug_overlay = save_debug_overlay
        self._output_dir = output_dir
        self._stem = stem
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._radar_renderer: Optional[RadarRenderer] = None
        self._radar_writer: Optional[cv2.VideoWriter] = None

        # Pitch keypoint detector — replaces grass-mask quad for homography when set.
        self._pitch_kp_detector = None
        if pitch_keypoint_model:
            from football_log.vision.pitch_keypoint_detector import PitchKeypointDetector
            self._pitch_kp_detector = PitchKeypointDetector(pitch_keypoint_model, imgsz=imgsz)
            print(f"Pitch keypoint model → {pitch_keypoint_model}")

        # Dedicated ball detector — overrides main tracker's ball detections when set.
        self._ball_detector = None
        if ball_model:
            from football_log.vision.ball_detector import BallDetector
            self._ball_detector = BallDetector(ball_model, conf=conf, imgsz=imgsz, slicer=ball_slicer)
            print(f"Ball detection model → {ball_model}{' (slicer)' if ball_slicer else ''}")

        # Shared H from keypoint model (smoothed via EMA, normalised H[2,2]=1).
        self._kp_H: Optional[np.ndarray] = None
        self._kp_smoother = HomographySmoother(alpha=0.3)
        self._stop_requested = False

        # Open VideoCapture LAST — after all model init — to avoid AVFoundation
        # dropping the file handle during PyTorch/MPS/Metal initialisation on macOS.
        self.cap = cv2.VideoCapture(self._video_source, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            kind = "摄像头" if self.is_camera else "视频文件"
            raise SystemExit(f"无法打开{kind}: {video_path}")

        # Open video writers NOW that we have cap dimensions.
        if self._save_video:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = str(Path(self._output_dir) / f"{self._stem}_overlay.mp4")
            self._video_writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
            )
            print(f"Recording overlay video → {out_path}")

        if self._save_radar:
            from football_log.ui.radar import _CANVAS_W, _CANVAS_H
            radar_path = str(Path(self._output_dir) / f"{self._stem}_radar.mp4")
            self._radar_renderer = RadarRenderer(
                pitch_length_m=self.pitch.length_m,
                pitch_width_m=self.pitch.width_m,
            )
            self._radar_writer = cv2.VideoWriter(
                radar_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (_CANVAS_W, _CANVAS_H)
            )
            print(f"Recording radar video → {radar_path}")

    def run(self) -> None:
        det = self._detector
        model_info = "custom"
        if hasattr(det, "model"):
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
                # If the team classifier needs per-frame setup (e.g. pose
                # estimation for the keypoint classifier), let it run before
                # detection so its results are ready when instant_label is called.
                prepare = getattr(self.team_classifier, "update_pose_for_frame", None)
                if prepare is not None:
                    prepare(frame, self.frame_idx)
                self.last_tracked_detections = self._detector.detect(frame)

            # Auto-calibration: refresh the per-frame homography before projection.
            calib_prepare = getattr(self._projector, "prepare_for_frame", None)
            if calib_prepare is not None:
                calib_prepare(self.frame_idx, frame)

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
                    # Safety: if the filter would drop every detection (likely a bad
                    # grass-mask frame: replay graphics, weird lighting), fall back
                    # to the unfiltered list rather than throwing the frame away.
                    if filtered or not detections:
                        filtered_ids = {o["id"] for o in filtered}
                        detections = [d for d in detections if d.track_id in filtered_ids]

            # Pitch keypoint model: runs at pitch_field_every_n cadence.
            # Produces a homography that overrides the grass-mask quad in the radar
            # and populates world coords for all detections.
            if self._pitch_kp_detector is not None:
                if self.frame_idx % self.pitch_field_every_n == 0 or self._kp_H is None:
                    H_new, _, _ = self._pitch_kp_detector.detect(frame)
                    if H_new is not None:
                        self._kp_H = self._kp_smoother.smooth(H_new)

            # Dedicated ball detector: runs on detect frames, merges result.
            if self._ball_detector is not None and should_detect:
                ball_dets = self._ball_detector.detect(frame)
                if ball_dets:
                    from football_log.protocols import Detection as _Det
                    # Remove any ball detections from main tracker for this frame.
                    detections = [d for d in detections if "Ball" not in d.label]
                    bd = ball_dets[0]
                    x, y, w, h = bd["bbox"]
                    detections.append(_Det(
                        track_id=-1,  # dedicated ball gets id -1 (no ByteTrack ID)
                        bbox=(x, y, w, h),
                        label="Ball",
                        conf=bd["conf"],
                    ))

            detections = _enrich_detections(detections, self._projector)
            # If no calibrated projector, use keypoint H for world coords.
            if self._projector is None and self._kp_H is not None:
                for det in detections:
                    foot = np.array([[[
                        float(det.bbox[0] + det.bbox[2] / 2),
                        float(det.bbox[1] + det.bbox[3]),
                    ]]], dtype=np.float32)
                    pt = cv2.perspectiveTransform(foot, self._kp_H)
                    det.world_x_m = float(pt[0, 0, 0])
                    det.world_y_m = float(pt[0, 0, 1])
            if self._track_filter is not None:
                frame_h = frame.shape[0]
                for det in detections:
                    # Compute jump likelihood from bbox height change.
                    _, _, _, bh = det.bbox
                    prev_h = self._prev_bbox_heights.get(det.track_id)
                    jl = jump_likelihood_from_height_change(
                        bbox_height=float(bh),
                        prev_height=prev_h,
                        bbox_y_top=float(det.bbox[1]),
                        frame_height=frame_h,
                    )
                    self._prev_bbox_heights[det.track_id] = float(bh)

                    if det.world_x_m is None or det.world_y_m is None:
                        smoothed = self._track_filter.update(
                            track_id=det.track_id,
                            world_xy=None,
                            frame_idx=self.frame_idx,
                            conf=float(det.conf),
                            jump_likelihood=jl,
                        )
                    else:
                        smoothed = self._track_filter.update(
                            track_id=det.track_id,
                            world_xy=(det.world_x_m, det.world_y_m),
                            frame_idx=self.frame_idx,
                            conf=float(det.conf),
                            jump_likelihood=jl,
                        )
                    if smoothed is not None:
                        det.world_x_m_smoothed = smoothed[0]
                        det.world_y_m_smoothed = smoothed[1]
                # Periodic eviction so the filter doesn't grow unbounded.
                if self.frame_idx % 300 == 0:
                    self._track_filter.evict_stale(self.frame_idx)
            self.data_writer.write_frame(self.frame_idx, detections)

            current_dicts = [d.to_dict() for d in detections]
            display = frame.copy()
            if self._save_debug_overlay and self._last_pitch_obs is not None:
                draw_pitch_observation(display, self._last_pitch_obs)
            draw_tracking_overlay(display, current_dicts)
            draw_frame_hud(display, self.frame_idx, self.detect_every_n)

            if self._video_writer is not None:
                self._video_writer.write(display)
            if self._radar_renderer is not None and self._radar_writer is not None:
                if self._kp_H is not None:
                    self._radar_renderer.set_homography(self._kp_H)
                    quad = None
                else:
                    quad = self._last_pitch_obs.field_quad_xy if self._last_pitch_obs is not None else None
                radar_img = self._radar_renderer.render(
                    current_dicts, field_quad_xy=quad, frame_shape=frame.shape
                )
                self._radar_writer.write(radar_img)

            yield frame, display, current_dicts, self.frame_idx
            self.frame_idx += 1

    def _finish(self) -> None:
        self.cap.release()
        if self._video_writer is not None:
            self._video_writer.release()
        if self._radar_writer is not None:
            self._radar_writer.release()
        self.data_writer.close()
        if self.show_ui:
            cv2.destroyAllWindows()
        print(f"轨迹导出完成，记录数: {self.data_writer.records_written}")
