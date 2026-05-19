"""Tests for Module 3B: 6-class team-aware GSR-2025 → YOLO conversion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from football_log.data.gsr_convert import (  # noqa: E402
    TeamClassMap,
    convert,
    default_team_class_map,
    dry_run,
)


# ---------------------------------------------------------------------------
# Synthetic dataset helpers (same pattern as test_gsr_convert.py)
# ---------------------------------------------------------------------------


def _write_labels_json(seq_dir: Path, seq_name: str, frames: list[dict]) -> None:
    images, annotations, ann_id = [], [], 1
    for entry in frames:
        fn = entry["frame_num"]
        iid = f"{seq_name}_{fn:06d}"
        images.append({
            "image_id": iid, "file_name": f"{fn:06d}.jpg",
            "height": 1080, "width": 1920,
        })
        for obj in entry["objects"]:
            attrs: dict = {}
            if "team" in obj:
                attrs["team"] = obj["team"]
            if "role" in obj:
                attrs["role"] = obj["role"]
            annotations.append({
                "id": str(ann_id),
                "image_id": iid,
                "track_id": obj["track_id"],
                "supercategory": "object",
                "category_id": obj["category_id"],
                "attributes": attrs,
                "bbox_image": {
                    "x": obj["x"], "y": obj["y"],
                    "x_center": obj["x"] + obj["w"] / 2,
                    "y_center": obj["y"] + obj["h"] / 2,
                    "w": obj["w"], "h": obj["h"],
                },
            })
            ann_id += 1
    payload = {
        "info": {"name": seq_name, "frame_rate": 25, "seq_length": len(images),
                 "im_dir": "img1", "im_ext": ".jpg"},
        "images": images,
        "annotations": annotations,
        "categories": [
            {"supercategory": "object", "id": 1, "name": "player"},
            {"supercategory": "object", "id": 2, "name": "goalkeeper"},
            {"supercategory": "object", "id": 3, "name": "referee"},
            {"supercategory": "object", "id": 4, "name": "ball"},
        ],
    }
    (seq_dir / "Labels-GameState.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1920, 1080), color=(0, 128, 0)).save(path, "JPEG")


def _build(root: Path, frames: list[dict], seq_name: str = "SNGS-001") -> Path:
    seq_dir = root / seq_name
    (seq_dir / "img1").mkdir(parents=True, exist_ok=True)
    for f in frames:
        _write_image(seq_dir / "img1" / f"{f['frame_num']:06d}.jpg")
    _write_labels_json(seq_dir, seq_name, frames)
    return root


# ---------------------------------------------------------------------------
# TeamClassMap unit tests
# ---------------------------------------------------------------------------


def test_team_class_map_player_left():
    m = default_team_class_map()
    assert m.output_index(1, "left") == 0   # team_a_player


def test_team_class_map_player_right():
    m = default_team_class_map()
    assert m.output_index(1, "right") == 1  # team_b_player


def test_team_class_map_goalkeeper_left():
    m = default_team_class_map()
    assert m.output_index(2, "left") == 2   # goalkeeper_a


def test_team_class_map_goalkeeper_right():
    m = default_team_class_map()
    assert m.output_index(2, "right") == 3  # goalkeeper_b


def test_team_class_map_referee_no_team():
    m = default_team_class_map()
    assert m.output_index(3, None) == 4     # referee (wildcard)
    assert m.output_index(3, "left") == 4   # any team → still referee


def test_team_class_map_ball_no_team():
    m = default_team_class_map()
    assert m.output_index(4, None) == 5
    assert m.output_index(4, "right") == 5  # wildcard fallback


def test_team_class_map_unknown_category_returns_none():
    m = default_team_class_map()
    assert m.output_index(99, "left") is None


def test_team_class_map_player_no_team_returns_none():
    """Player with no team label cannot be assigned to either team."""
    m = default_team_class_map()
    assert m.output_index(1, None) is None


def test_team_class_map_output_classes():
    m = default_team_class_map()
    assert m.output_classes == [
        "team_a_player", "team_b_player",
        "goalkeeper_a", "goalkeeper_b",
        "referee", "ball",
    ]


# ---------------------------------------------------------------------------
# End-to-end 6-class conversion tests
# ---------------------------------------------------------------------------


def test_6class_convert_writes_correct_class_indices(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _build(src, [{
        "frame_num": 1,
        "objects": [
            {"track_id": 1, "category_id": 1, "team": "left",  "x": 100, "y": 100, "w": 60, "h": 120},
            {"track_id": 2, "category_id": 1, "team": "right", "x": 200, "y": 100, "w": 60, "h": 120},
            {"track_id": 3, "category_id": 2, "team": "left",  "x": 10,  "y": 10,  "w": 60, "h": 120},
            {"track_id": 4, "category_id": 2, "team": "right", "x": 1800, "y": 10, "w": 60, "h": 120},
            {"track_id": 5, "category_id": 3,                  "x": 500, "y": 300, "w": 40, "h": 80},
            {"track_id": 6, "category_id": 4,                  "x": 960, "y": 540, "w": 15, "h": 15},
        ],
    }])

    result = convert(src, out, default_team_class_map(), split_ratios=(1.0, 0.0, 0.0), seed=0)
    assert result.n_labels_written == 6

    label_files = list((out / "labels" / "train").glob("*.txt"))
    assert len(label_files) == 1
    lines = label_files[0].read_text().strip().split("\n")
    cls_indices = {int(l.split()[0]) for l in lines}
    assert cls_indices == {0, 1, 2, 3, 4, 5}  # all 6 classes present


def test_6class_convert_player_without_team_is_skipped(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _build(src, [{
        "frame_num": 1,
        "objects": [
            {"track_id": 1, "category_id": 1, "team": "left", "x": 100, "y": 100, "w": 60, "h": 120},
            {"track_id": 2, "category_id": 1, "x": 200, "y": 100, "w": 60, "h": 120},  # no team
        ],
    }])
    result = convert(src, out, default_team_class_map(), split_ratios=(1.0, 0.0, 0.0), seed=0)
    assert result.n_labels_written == 1
    assert result.n_rows_skipped_unknown_class == 1


def test_6class_convert_rows_per_class(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _build(src, [{
        "frame_num": 1,
        "objects": [
            {"track_id": 1, "category_id": 1, "team": "left",  "x": 0, "y": 0, "w": 60, "h": 120},
            {"track_id": 2, "category_id": 1, "team": "left",  "x": 100, "y": 0, "w": 60, "h": 120},
            {"track_id": 3, "category_id": 1, "team": "right", "x": 200, "y": 0, "w": 60, "h": 120},
            {"track_id": 4, "category_id": 4, "x": 960, "y": 540, "w": 15, "h": 15},  # ball
        ],
    }])
    result = convert(src, out, default_team_class_map(), split_ratios=(1.0, 0.0, 0.0), seed=0)
    assert result.rows_per_class["team_a_player"] == 2
    assert result.rows_per_class["team_b_player"] == 1
    assert result.rows_per_class["ball"] == 1


def test_6class_convert_writes_yaml_with_6_names(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _build(src, [{
        "frame_num": 1,
        "objects": [
            {"track_id": 1, "category_id": 1, "team": "left", "x": 0, "y": 0, "w": 60, "h": 120},
        ],
    }])
    convert(src, out, default_team_class_map(), split_ratios=(1.0, 0.0, 0.0), seed=0)
    yaml_text = (out / "soccernet.yaml").read_text()
    for name in ["team_a_player", "team_b_player", "goalkeeper_a", "goalkeeper_b", "referee", "ball"]:
        assert name in yaml_text


def test_6class_convert_manifest_records_team_class_map(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _build(src, [{
        "frame_num": 1,
        "objects": [
            {"track_id": 1, "category_id": 1, "team": "left", "x": 0, "y": 0, "w": 60, "h": 120},
        ],
    }])
    convert(src, out, default_team_class_map(), split_ratios=(1.0, 0.0, 0.0), seed=0)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["class_map"]["output_classes"] == [
        "team_a_player", "team_b_player", "goalkeeper_a", "goalkeeper_b", "referee", "ball"
    ]


# ---------------------------------------------------------------------------
# dry_run now reports team/role distributions
# ---------------------------------------------------------------------------


def test_dry_run_reports_team_distribution(tmp_path):
    _build(tmp_path, [{
        "frame_num": 1,
        "objects": [
            {"track_id": 1, "category_id": 1, "team": "left",  "role": "player", "x": 0, "y": 0, "w": 50, "h": 100},
            {"track_id": 2, "category_id": 1, "team": "right", "role": "player", "x": 60, "y": 0, "w": 50, "h": 100},
            {"track_id": 3, "category_id": 4,                                     "x": 200, "y": 200, "w": 10, "h": 10},
        ],
    }])
    summary = dry_run(tmp_path)
    assert summary["team_distribution"] == {"left": 1, "right": 1}
    assert summary["role_distribution"] == {"player": 2}
    assert summary["team_role_distribution"] == {"left:player": 1, "right:player": 1}
