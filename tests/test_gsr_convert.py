"""Tests for the GSR-2025 → YOLO converter.

Builds a tiny synthetic GSR-2025 dataset on disk (one or two sequences with
a Labels-GameState.json + img1/ frames) and exercises the converter end to
end. Mirrors test_yolo_convert.py but for the JSON-based GSR layout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from football_log.data.gsr_convert import (  # noqa: E402
    convert,
    default_class_map,
    detect_format,
    discover_sequences,
    dry_run,
)


# ---------------------------------------------------------------------------
# Synthetic dataset builders
# ---------------------------------------------------------------------------


def _write_labels_json(seq_dir: Path, seq_name: str, frames: list[dict]) -> None:
    """Write a minimal Labels-GameState.json with the supplied per-frame data.

    Each `frames[i]` is:
        {
          "frame_num": int,
          "objects": [{"track_id": int, "category_id": int, "x": float, ...}, ...],
        }
    """
    images = []
    annotations = []
    ann_counter = 1
    for entry in frames:
        frame_num = entry["frame_num"]
        image_id = f"{seq_name}_{frame_num:06d}"
        images.append({
            "is_labeled": True,
            "image_id": image_id,
            "file_name": f"{frame_num:06d}.jpg",
            "height": 1080,
            "width": 1920,
        })
        for obj in entry["objects"]:
            annotations.append({
                "id": str(ann_counter),
                "image_id": image_id,
                "track_id": obj["track_id"],
                "supercategory": obj.get("supercategory", "object"),
                "category_id": obj["category_id"],
                "attributes": obj.get("attributes", {}),
                "bbox_image": {
                    "x": obj["x"],
                    "y": obj["y"],
                    "x_center": obj["x"] + obj["w"] / 2,
                    "y_center": obj["y"] + obj["h"] / 2,
                    "w": obj["w"],
                    "h": obj["h"],
                },
            })
            ann_counter += 1
    payload = {
        "info": {
            "version": "1.3",
            "name": seq_name,
            "frame_rate": 25,
            "seq_length": len(images),
            "im_dir": "img1",
            "im_ext": ".jpg",
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {"supercategory": "object", "id": 1, "name": "player"},
            {"supercategory": "object", "id": 2, "name": "goalkeeper"},
            {"supercategory": "object", "id": 3, "name": "referee"},
            {"supercategory": "object", "id": 4, "name": "ball"},
            {"supercategory": "pitch", "id": 5, "name": "pitch"},
        ],
    }
    (seq_dir / "Labels-GameState.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_image(path: Path, w: int = 1920, h: int = 1080) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), color=(0, 128, 0)).save(path, "JPEG")


def _build_synthetic(root: Path, seq_specs: list[dict]) -> Path:
    """Create a synthetic GSR-2025 source tree under `root`. Return the root."""
    for seq in seq_specs:
        seq_dir = root / seq["name"]
        (seq_dir / "img1").mkdir(parents=True, exist_ok=True)
        for entry in seq["frames"]:
            _write_image(seq_dir / "img1" / f"{entry['frame_num']:06d}.jpg")
        _write_labels_json(seq_dir, seq["name"], seq["frames"])
    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_detect_format_recognizes_gsr(tmp_path):
    _build_synthetic(tmp_path, [{
        "name": "SNGS-001",
        "frames": [{"frame_num": 1, "objects": []}],
    }])
    assert detect_format(tmp_path) == "gsr"


def test_detect_format_rejects_empty(tmp_path):
    assert detect_format(tmp_path) == "unknown"


def test_discover_sequences_finds_seq(tmp_path):
    _build_synthetic(tmp_path, [{
        "name": "SNGS-001",
        "frames": [{
            "frame_num": 1,
            "objects": [{"track_id": 1, "category_id": 1, "x": 100, "y": 200, "w": 50, "h": 100}],
        }],
    }])
    seqs = discover_sequences(tmp_path)
    assert len(seqs) == 1
    assert seqs[0].name == "SNGS-001"
    assert len(seqs[0].images) == 1
    assert len(seqs[0].annotations) == 1


def test_dry_run_reports_class_counts(tmp_path):
    _build_synthetic(tmp_path, [{
        "name": "SNGS-001",
        "frames": [{
            "frame_num": 1,
            "objects": [
                {"track_id": 1, "category_id": 1, "x": 0, "y": 0, "w": 50, "h": 100},
                {"track_id": 2, "category_id": 1, "x": 60, "y": 0, "w": 50, "h": 100},
                {"track_id": 3, "category_id": 4, "x": 200, "y": 200, "w": 10, "h": 10},  # ball
            ],
        }],
    }])
    summary = dry_run(tmp_path)
    assert summary["n_sequences"] == 1
    assert summary["class_id_counts"] == {1: 2, 4: 1}
    assert summary["n_rows_total"] == 3
    assert summary["category_id_to_name"][1] == "player"
    assert summary["category_id_to_name"][4] == "ball"


def test_convert_writes_yolo_labels(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _build_synthetic(src, [{
        "name": "SNGS-001",
        "frames": [{
            "frame_num": 1,
            "objects": [
                {"track_id": 1, "category_id": 1, "x": 100, "y": 200, "w": 100, "h": 200},  # player
                {"track_id": 2, "category_id": 4, "x": 500, "y": 500, "w": 20, "h": 20},   # ball
                {"track_id": 3, "category_id": 3, "x": 800, "y": 100, "w": 60, "h": 150},  # referee
            ],
        }],
    }])

    result = convert(src, out, default_class_map(), split_ratios=(1.0, 0.0, 0.0), seed=0)
    assert result.n_frames == 1
    assert result.n_labels_written == 3
    assert result.rows_per_class == {"player": 1, "ball": 1, "referee": 1}

    label_files = list((out / "labels" / "train").glob("*.txt"))
    assert len(label_files) == 1
    lines = label_files[0].read_text().strip().split("\n")
    assert len(lines) == 3
    # Each line: class cx cy w h, all normalised
    for line in lines:
        parts = line.split()
        assert len(parts) == 5
        cls, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        assert 0 <= cls <= 2
        for v in (cx, cy, w, h):
            assert 0.0 <= v <= 1.0


def test_convert_skips_pitch_and_camera_categories(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _build_synthetic(src, [{
        "name": "SNGS-001",
        "frames": [{
            "frame_num": 1,
            "objects": [
                {"track_id": 1, "category_id": 1, "x": 100, "y": 200, "w": 100, "h": 200},
                {"track_id": 99, "category_id": 5, "supercategory": "pitch",
                 "x": 0, "y": 0, "w": 1920, "h": 1080},
            ],
        }],
    }])

    result = convert(src, out, default_class_map(), split_ratios=(1.0, 0.0, 0.0), seed=0)
    # Only the player (category_id=1) is kept; pitch (5) is skipped at parse time.
    assert result.n_labels_written == 1
    assert result.rows_per_class["player"] == 1


def test_convert_merges_goalkeeper_into_player(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _build_synthetic(src, [{
        "name": "SNGS-001",
        "frames": [{
            "frame_num": 1,
            "objects": [
                {"track_id": 1, "category_id": 1, "x": 100, "y": 200, "w": 100, "h": 200},
                {"track_id": 2, "category_id": 2, "x": 400, "y": 200, "w": 100, "h": 200},  # GK
            ],
        }],
    }])

    result = convert(src, out, default_class_map(), split_ratios=(1.0, 0.0, 0.0), seed=0)
    assert result.rows_per_class["player"] == 2  # both player and GK become 'player'


def test_convert_writes_manifest(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _build_synthetic(src, [{
        "name": "SNGS-001",
        "frames": [{"frame_num": 1, "objects": [
            {"track_id": 1, "category_id": 1, "x": 100, "y": 200, "w": 100, "h": 200},
        ]}],
    }])

    convert(src, out, default_class_map(), split_ratios=(1.0, 0.0, 0.0), seed=42)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["source_format"] == "gsr-2025"
    assert manifest["n_sequences"] == 1
    assert manifest["n_frames"] == 1
    assert manifest["seed"] == 42
    assert manifest["class_map"]["output_classes"] == ["player", "ball", "referee"]


def test_convert_writes_yaml(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _build_synthetic(src, [{
        "name": "SNGS-001",
        "frames": [{"frame_num": 1, "objects": [
            {"track_id": 1, "category_id": 1, "x": 0, "y": 0, "w": 100, "h": 100},
        ]}],
    }])
    convert(src, out, default_class_map(), split_ratios=(1.0, 0.0, 0.0), seed=0)
    yaml_text = (out / "soccernet.yaml").read_text()
    assert "names:" in yaml_text
    assert "0: player" in yaml_text
    assert "1: ball" in yaml_text
    assert "2: referee" in yaml_text


def test_convert_frame_stride(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    frames = [{
        "frame_num": i,
        "objects": [{"track_id": 1, "category_id": 1, "x": 0, "y": 0, "w": 100, "h": 100}],
    } for i in range(1, 11)]  # 10 frames
    _build_synthetic(src, [{"name": "SNGS-001", "frames": frames}])

    result = convert(src, out, default_class_map(),
                     split_ratios=(1.0, 0.0, 0.0), seed=0, frame_stride=3)
    # stride 3 over 10 frames → 4 frames (indices 0, 3, 6, 9)
    assert result.n_frames == 4


def test_convert_max_sequences(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _build_synthetic(src, [
        {
            "name": f"SNGS-{i:03d}",
            "frames": [{"frame_num": 1, "objects": [
                {"track_id": 1, "category_id": 1, "x": 0, "y": 0, "w": 100, "h": 100},
            ]}],
        }
        for i in range(1, 6)  # 5 sequences
    ])

    result = convert(src, out, default_class_map(),
                     split_ratios=(1.0, 0.0, 0.0), seed=0, max_sequences=2)
    assert len(result.sequences) == 2


def test_convert_invalid_bbox_skipped(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _build_synthetic(src, [{
        "name": "SNGS-001",
        "frames": [{
            "frame_num": 1,
            "objects": [
                {"track_id": 1, "category_id": 1, "x": 100, "y": 200, "w": 100, "h": 200},  # OK
                {"track_id": 2, "category_id": 1, "x": 0, "y": 0, "w": 0, "h": 100},        # zero width
                {"track_id": 3, "category_id": 1, "x": 0, "y": 0, "w": 100, "h": 0},        # zero height
            ],
        }],
    }])

    result = convert(src, out, default_class_map(), split_ratios=(1.0, 0.0, 0.0), seed=0)
    assert result.n_labels_written == 1


def test_convert_unknown_category_id_skipped(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _build_synthetic(src, [{
        "name": "SNGS-001",
        "frames": [{
            "frame_num": 1,
            "objects": [
                {"track_id": 1, "category_id": 1, "x": 100, "y": 200, "w": 100, "h": 200},
                {"track_id": 2, "category_id": 99, "x": 0, "y": 0, "w": 50, "h": 50},  # unknown
            ],
        }],
    }])

    result = convert(src, out, default_class_map(), split_ratios=(1.0, 0.0, 0.0), seed=0)
    assert result.n_labels_written == 1
    assert result.n_rows_skipped_unknown_class == 1


def test_convert_split_assignment_deterministic(tmp_path):
    src = tmp_path / "src"
    _build_synthetic(src, [
        {
            "name": f"SNGS-{i:03d}",
            "frames": [{"frame_num": 1, "objects": [
                {"track_id": 1, "category_id": 1, "x": 0, "y": 0, "w": 100, "h": 100},
            ]}],
        }
        for i in range(1, 11)  # 10 sequences
    ])
    r1 = convert(src, tmp_path / "out1", default_class_map(),
                 split_ratios=(0.7, 0.15, 0.15), seed=42)
    r2 = convert(src, tmp_path / "out2", default_class_map(),
                 split_ratios=(0.7, 0.15, 0.15), seed=42)
    assert r1.splits == r2.splits
    # Different seed → different (likely) split
    r3 = convert(src, tmp_path / "out3", default_class_map(),
                 split_ratios=(0.7, 0.15, 0.15), seed=999)
    assert r3.splits != r1.splits or len(r1.sequences) <= 2  # tiny set may collide


def test_convert_with_subfolder_im_ext_alias(tmp_path):
    """Some sequences ship the file_name with leading zeros only — just confirm the parser handles it."""
    src = tmp_path / "src"
    out = tmp_path / "out"
    seq_dir = src / "SNGS-001"
    (seq_dir / "img1").mkdir(parents=True)
    _write_image(seq_dir / "img1" / "000007.jpg")
    payload = {
        "info": {"name": "SNGS-001", "frame_rate": 25, "seq_length": 1, "im_dir": "img1", "im_ext": ".jpg"},
        "images": [{"image_id": "X1", "file_name": "000007.jpg", "height": 1080, "width": 1920}],
        "annotations": [{
            "image_id": "X1", "track_id": 1, "supercategory": "object", "category_id": 1,
            "bbox_image": {"x": 0, "y": 0, "w": 100, "h": 100},
        }],
        "categories": [{"supercategory": "object", "id": 1, "name": "player"}],
    }
    (seq_dir / "Labels-GameState.json").write_text(json.dumps(payload), encoding="utf-8")

    result = convert(src, out, default_class_map(), split_ratios=(1.0, 0.0, 0.0), seed=0)
    assert result.n_frames == 1
    label_files = list((out / "labels" / "train").glob("*.txt"))
    assert any("000007" in p.name for p in label_files)
