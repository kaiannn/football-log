"""Lock the JSONL / CSV / meta schema produced by TrackingDataWriter.

These tests are the contract for downstream consumers (Pandas / DuckDB / BI tools)
and for any future Exporter implementations.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from football_log.io.export import TrackingDataWriter
from football_log.protocols import Detection


EXPECTED_FIELDS = [
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
    "world_x_m_smoothed",
    "world_y_m_smoothed",
]


def _make_writer(tmp_path: Path, output_format: str = "both") -> TrackingDataWriter:
    return TrackingDataWriter(
        output_dir=str(tmp_path),
        output_prefix="test",
        output_format=output_format,
        fps=25.0,
        video_path="dummy.mp4",
    )


def test_csv_header_matches_expected_fields(tmp_path):
    w = _make_writer(tmp_path, "csv")
    w.close()
    with open(tmp_path / "test.csv", encoding="utf-8") as fp:
        header = next(csv.reader(fp))
    assert header == EXPECTED_FIELDS


def test_jsonl_row_has_exactly_expected_keys(tmp_path):
    w = _make_writer(tmp_path, "jsonl")
    det = Detection(
        track_id=3,
        bbox=(342, 188, 48, 112),
        label="Team A",
        conf=0.876543,
        world_x_m=23.4,
        world_y_m=-11.2,
    )
    w.write_frame(120, [det])
    w.close()

    line = (tmp_path / "test.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    row = json.loads(line)
    assert set(row.keys()) == set(EXPECTED_FIELDS)


def test_jsonl_field_types_and_rounding(tmp_path):
    w = _make_writer(tmp_path, "jsonl")
    det = Detection(
        track_id=3,
        bbox=(342, 188, 48, 112),
        label="Team A",
        conf=0.876543,
        world_x_m=23.4567,
        world_y_m=-11.234567,
    )
    w.write_frame(120, [det])
    w.close()

    row = json.loads((tmp_path / "test.jsonl").read_text(encoding="utf-8").strip())

    assert isinstance(row["frame_idx"], int) and row["frame_idx"] == 120
    assert isinstance(row["track_id"], int) and row["track_id"] == 3
    assert isinstance(row["label"], str) and row["label"] == "Team A"
    for k in ("x", "y", "w", "h"):
        assert isinstance(row[k], int)
    assert isinstance(row["timestamp_sec"], float)
    assert row["timestamp_sec"] == pytest.approx(120 / 25.0, abs=1e-6)
    assert row["conf"] == pytest.approx(0.8765, abs=1e-9)
    assert row["world_x_m"] == pytest.approx(23.4567, abs=1e-9)
    assert row["world_y_m"] == pytest.approx(-11.2346, abs=1e-9)


def test_jsonl_world_coords_null_when_missing(tmp_path):
    w = _make_writer(tmp_path, "jsonl")
    det = Detection(track_id=1, bbox=(0, 0, 10, 10), label="Player", conf=0.5)
    w.write_frame(0, [det])
    w.close()

    row = json.loads((tmp_path / "test.jsonl").read_text(encoding="utf-8").strip())
    assert row["world_x_m"] is None
    assert row["world_y_m"] is None


def test_csv_world_coords_blank_when_missing(tmp_path):
    w = _make_writer(tmp_path, "csv")
    det = Detection(track_id=1, bbox=(0, 0, 10, 10), label="Player", conf=0.5)
    w.write_frame(0, [det])
    w.close()

    with open(tmp_path / "test.csv", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    assert rows[0]["world_x_m"] == ""
    assert rows[0]["world_y_m"] == ""
    assert rows[0]["world_x_m_smoothed"] == ""
    assert rows[0]["world_y_m_smoothed"] == ""


def test_smoothed_world_coords_round_trip_when_present(tmp_path):
    w = _make_writer(tmp_path, "both")
    det = Detection(
        track_id=3,
        bbox=(342, 188, 48, 112),
        label="Team A",
        conf=0.9,
        world_x_m=23.4567,
        world_y_m=-11.2345,
        world_x_m_smoothed=23.5,
        world_y_m_smoothed=-11.1,
    )
    w.write_frame(120, [det])
    w.close()

    row = json.loads((tmp_path / "test.jsonl").read_text(encoding="utf-8").strip())
    assert row["world_x_m"] == pytest.approx(23.4567, abs=1e-9)
    assert row["world_x_m_smoothed"] == pytest.approx(23.5, abs=1e-9)
    assert row["world_y_m_smoothed"] == pytest.approx(-11.1, abs=1e-9)


def test_meta_json_contains_required_keys(tmp_path):
    w = _make_writer(tmp_path, "both")
    w.close()
    meta = json.loads((tmp_path / "test.meta.json").read_text(encoding="utf-8"))
    for k in ("video_path", "fps", "started_at_utc", "output_format"):
        assert k in meta
    assert meta["fps"] == 25.0
    assert meta["video_path"] == "dummy.mp4"
    assert meta["output_format"] == "both"
    assert meta["started_at_utc"].endswith("Z")


def test_records_written_counter(tmp_path):
    w = _make_writer(tmp_path, "both")
    dets = [
        Detection(track_id=1, bbox=(0, 0, 10, 10), label="Player", conf=0.5),
        Detection(track_id=2, bbox=(0, 0, 10, 10), label="Player", conf=0.5),
    ]
    w.write_frame(0, dets)
    w.write_frame(1, dets[:1])
    w.close()
    assert w.records_written == 3


def test_dict_input_compatible_with_legacy_callers(tmp_path):
    w = _make_writer(tmp_path, "jsonl")
    legacy_obj = {
        "id": 7,
        "bbox": (10, 20, 30, 40),
        "label": "Ball",
        "conf": 0.42,
    }
    w.write_frame(5, [legacy_obj])
    w.close()
    row = json.loads((tmp_path / "test.jsonl").read_text(encoding="utf-8").strip())
    assert row["track_id"] == 7
    assert row["label"] == "Ball"
    assert (row["x"], row["y"], row["w"], row["h"]) == (10, 20, 30, 40)
