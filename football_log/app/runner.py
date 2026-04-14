"""视频读取 → 跟踪 → 可选世界坐标 → 导出。"""

import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from football_log.io.export import TrackingDataWriter
from football_log.pitch.field_estimator import PitchFieldEstimator, TemporalPitchSmoother
from football_log.pitch.integration import filter_objects_in_grass_mask
from football_log.pitch.observation import PitchObservation
from football_log.ui.overlay import draw_frame_hud, draw_pitch_observation, draw_tracking_overlay
from football_log.vision.team_classifier import TeamClassifier
from football_log.vision.tracker import YoloByteTrackTracker
from football_log.world.homography import Homography, project_foot_to_world
from football_log.world.pitch_model import PitchSpec
from football_log.world.pinhole_ground import PinholeGroundProjector, load_pinhole_ground_projector


def _load_homography(path: Optional[str]) -> Optional[Homography]:
    if not path or not os.path.isfile(path):
        return None
    arr = np.load(path)
    if arr.shape != (3, 3):
        raise ValueError(f"Homography must be 3x3, got {arr.shape}")
    return Homography(arr)


def _enrich_world_coords(
    objs: List[Dict[str, Any]],
    H: Optional[Homography],
    pinhole: Optional[PinholeGroundProjector],
) -> List[Dict[str, Any]]:
    if pinhole is None and H is None:
        return objs
    out: List[Dict[str, Any]] = []
    for o in objs:
        row = dict(o)
        if pinhole is not None:
            wx, wy = pinhole.project_foot_to_world(o["bbox"], o["label"])
        else:
            wx, wy = project_foot_to_world(o["bbox"], o["label"], H)
        row["world_x_m"] = wx
        row["world_y_m"] = wy
        out.append(row)
    return out


class VideoTrackerPipeline:
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
    ):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise SystemExit("无法打开视频文件")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.delay = int(1000 / self.fps)
        self.show_ui = show_ui
        self.detect_every_n = max(1, detect_every_n)
        self.frame_idx = 0
        self.last_tracked_objects: List[Dict[str, Any]] = []

        self.team_classifier = TeamClassifier(team_colors=team_colors)
        self.yolo_tracker = YoloByteTrackTracker(
            model_name=model_name,
            conf=conf,
            imgsz=imgsz,
            tracker=tracker,
        )
        self.H = _load_homography(homography_path)
        self.pinhole: Optional[PinholeGroundProjector] = None
        if camera_calib_path:
            self.pinhole = load_pinhole_ground_projector(camera_calib_path)
        self.pitch = pitch or PitchSpec()

        self.pitch_field_detect = bool(pitch_field_detect)
        self.pitch_field_every_n = max(1, int(pitch_field_every_n))
        self._pitch_estimator: Optional[PitchFieldEstimator] = (
            PitchFieldEstimator() if self.pitch_field_detect else None
        )
        self._pitch_smoother: Optional[TemporalPitchSmoother] = (
            TemporalPitchSmoother() if (self.pitch_field_detect and pitch_field_temporal_smooth) else None
        )
        self._last_pitch_obs: Optional[PitchObservation] = None
        self.pitch_field_filter_tracks = bool(pitch_field_filter_tracks)

        stem = os.path.splitext(os.path.basename(self.video_path))[0]
        world_on = self.pinhole is not None or self.H is not None
        extra_meta = {
            "pitch_length_m": self.pitch.length_m,
            "pitch_width_m": self.pitch.width_m,
            "homography_path": homography_path,
            "camera_calib_path": camera_calib_path,
            "world_coords_method": "pinhole_ground"
            if self.pinhole is not None
            else ("homography" if self.H is not None else None),
            "world_coords_enabled": world_on,
            "pitch_field_detect": self.pitch_field_detect,
            "pitch_field_every_n": self.pitch_field_every_n,
            "pitch_field_filter_tracks": self.pitch_field_filter_tracks,
        }
        self.data_writer = TrackingDataWriter(
            output_dir=output_dir,
            output_prefix=f"{stem}_tracks",
            output_format=output_format,
            fps=self.fps,
            video_path=self.video_path,
            extra_meta=extra_meta,
        )

    def run(self) -> None:
        m = self.yolo_tracker.model
        ckpt = getattr(m, "ckpt_path", None) or getattr(m, "model_name", "yolo")
        print(f"Track pipeline: model={ckpt}, tracker={self.yolo_tracker.tracker}, detect_every_n={self.detect_every_n}")
        print(
            f"Output dir: {self.data_writer.output_dir}, world coords: "
            f"{self.pinhole is not None or self.H is not None} "
            f"({'pinhole' if self.pinhole is not None else 'homography' if self.H is not None else 'off'}), "
            f"pitch field: {self.pitch_field_detect}"
        )

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            should_detect = (self.frame_idx % self.detect_every_n == 0) or not self.last_tracked_objects
            if should_detect:
                self.last_tracked_objects = self.yolo_tracker.track_frame(frame, self.team_classifier)

            tracked = self.last_tracked_objects
            if self.pitch_field_detect and self._pitch_estimator is not None:
                if self.frame_idx % self.pitch_field_every_n == 0 or self._last_pitch_obs is None:
                    obs = self._pitch_estimator.estimate(frame)
                    if self._pitch_smoother is not None:
                        obs = self._pitch_smoother.smooth(obs)
                    self._last_pitch_obs = obs
                if self.pitch_field_filter_tracks and self._last_pitch_obs is not None:
                    tracked = filter_objects_in_grass_mask(tracked, self._last_pitch_obs.grass_mask)

            current = _enrich_world_coords(tracked, self.H, self.pinhole)
            self.data_writer.write_frame(self.frame_idx, current)

            if self.show_ui:
                display = frame.copy()
                if self.pitch_field_detect and self._last_pitch_obs is not None:
                    draw_pitch_observation(display, self._last_pitch_obs)
                draw_tracking_overlay(display, current)
                draw_frame_hud(display, self.frame_idx, self.detect_every_n)
                cv2.imshow("Video Tracker", display)
                key = cv2.waitKey(self.delay) & 0xFF
                if key == ord("q"):
                    break

            self.frame_idx += 1

        self.cap.release()
        self.data_writer.close()
        if self.show_ui:
            cv2.destroyAllWindows()
        print(f"轨迹导出完成，记录数: {self.data_writer.records_written}")
