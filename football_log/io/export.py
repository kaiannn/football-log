"""结构化轨迹 JSONL/CSV 导出。

实现 protocols.Exporter 接口，同时保留旧版 write_frame(frame_idx, List[dict]) 兼容。
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from football_log.protocols import Detection


class TrackingDataWriter:
    def __init__(
        self,
        output_dir: str,
        output_prefix: str,
        output_format: str,
        fps: float,
        video_path: str,
        extra_meta: Optional[dict] = None,
    ):
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self.output_format = output_format
        self.fps = fps if fps > 0 else 25.0
        self.video_path = video_path
        self.extra_meta = extra_meta or {}
        self.started_at = datetime.utcnow().isoformat() + "Z"
        os.makedirs(self.output_dir, exist_ok=True)

        self.jsonl_path = os.path.join(self.output_dir, f"{self.output_prefix}.jsonl")
        self.csv_path = os.path.join(self.output_dir, f"{self.output_prefix}.csv")
        self.meta_path = os.path.join(self.output_dir, f"{self.output_prefix}.meta.json")

        self.jsonl_fp = None
        self.csv_fp = None
        self.csv_writer = None
        self._records_written = 0

        self._fieldnames = [
            "frame_idx",
            "timestamp_sec",
            "track_id",
            "label",
            "x",
            "y",
            "w",
            "h",
            "conf",
            "world_x_m",
            "world_y_m",
        ]
        self._open()
        self._write_meta()

    @property
    def records_written(self) -> int:
        return self._records_written

    @records_written.setter
    def records_written(self, value: int) -> None:
        self._records_written = value

    def _open(self) -> None:
        if self.output_format in ("jsonl", "both"):
            self.jsonl_fp = open(self.jsonl_path, "w", encoding="utf-8")
        if self.output_format in ("csv", "both"):
            self.csv_fp = open(self.csv_path, "w", encoding="utf-8", newline="")
            self.csv_writer = csv.DictWriter(self.csv_fp, fieldnames=self._fieldnames)
            self.csv_writer.writeheader()

    def _write_meta(self) -> None:
        meta = {
            "video_path": self.video_path,
            "fps": self.fps,
            "started_at_utc": self.started_at,
            "output_format": self.output_format,
            **self.extra_meta,
        }
        with open(self.meta_path, "w", encoding="utf-8") as fp:
            json.dump(meta, fp, ensure_ascii=False, indent=2)

    def _write_row(self, frame_idx: int, obj: Dict[str, Any]) -> None:
        timestamp_sec = round(frame_idx / self.fps, 3)
        x, y, w, h = [int(v) for v in obj["bbox"]]
        wx = obj.get("world_x_m")
        wy = obj.get("world_y_m")
        row = {
            "frame_idx": int(frame_idx),
            "timestamp_sec": timestamp_sec,
            "track_id": int(obj.get("id", obj.get("track_id", -1))),
            "label": obj.get("label", ""),
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "conf": round(float(obj.get("conf", 0.0)), 4),
            "world_x_m": "" if wx is None else round(float(wx), 4),
            "world_y_m": "" if wy is None else round(float(wy), 4),
        }
        if self.jsonl_fp:
            json_row = {
                **row,
                "world_x_m": None if wx is None else round(float(wx), 4),
                "world_y_m": None if wy is None else round(float(wy), 4),
            }
            self.jsonl_fp.write(json.dumps(json_row, ensure_ascii=False) + "\n")
        if self.csv_writer:
            self.csv_writer.writerow(row)
        self._records_written += 1

    def write_frame(
        self,
        frame_idx: int,
        tracked_objects: Union[Sequence["Detection"], List[Dict[str, Any]]],
    ) -> None:
        from football_log.protocols import Detection

        for obj in tracked_objects:
            if isinstance(obj, Detection):
                self._write_row(frame_idx, obj.to_dict())
            else:
                self._write_row(frame_idx, obj)

    def close(self) -> None:
        if self.jsonl_fp:
            self.jsonl_fp.close()
        if self.csv_fp:
            self.csv_fp.close()
