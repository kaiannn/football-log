"""Tests for the SoccerNet → YOLO converter.

We synthesize a tiny MOT-style dataset on the fly (no SoccerNet download needed)
and run the converter against it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_log.data.yolo_convert import (
    ClassMap,
    convert,
    default_class_map,
    discover_sequences,
    dry_run,
)


def _make_image(path: Path, w: int = 1920, h: int = 1080) -> None:
    """Create a tiny valid JPEG. Skips test if Pillow not installed."""
    PIL = pytest.importorskip("PIL.Image")
    path.parent.mkdir(parents=True, exist_ok=True)
    img = PIL.new("RGB", (w, h), color=(0, 128, 0))
    img.save(path, "JPEG", quality=70)


def _make_seqinfo(path: Path, w: int = 1920, h: int = 1080) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"[Sequence]\nname={path.parent.name}\nimWidth={w}\nimHeight={h}\nseqLength=3\n",
        encoding="utf-8",
    )


def _make_gt(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_synthetic_dataset(root: Path, n_sequences: int = 2) -> None:
    """Two sequences, three frames each, mixed classes including unknown ones."""
    for s in range(n_sequences):
        seq = root / f"SNMOT-{s+1:03d}"
        _make_seqinfo(seq / "seqinfo.ini")
        for frame in range(1, 4):
            _make_image(seq / "img1" / f"{frame:06d}.jpg")
        # Class IDs: 1=player_a, 2=player_b, 4=ref, 5=ball, 99=unknown (skip)
        _make_gt(
            seq / "gt" / "gt.txt",
            [
                "1, 1, 100, 200, 50, 80, 1, 1, 1.0",
                "1, 2, 800, 400, 50, 80, 1, 2, 1.0",
                "1, 3, 950, 540, 10, 10, 1, 5, 1.0",
                "2, 1, 110, 210, 50, 80, 1, 1, 1.0",
                "2, 99, 0, 0, 50, 50, 1, 99, 1.0",  # unknown class
                "3, 4, 200, 200, 100, 100, 1, 4, 1.0",
            ],
        )


def test_discover_sequences_finds_mot_layout(tmp_path):
    _build_synthetic_dataset(tmp_path, n_sequences=2)
    seqs = discover_sequences(tmp_path)
    assert [s.name for s in seqs] == ["SNMOT-001", "SNMOT-002"]
    assert seqs[0].width == 1920 and seqs[0].height == 1080


def test_dry_run_reports_class_counts(tmp_path):
    _build_synthetic_dataset(tmp_path, n_sequences=2)
    summary = dry_run(tmp_path)
    assert summary["n_sequences"] == 2
    counts = summary["class_id_counts"]
    assert counts[1] == 2 * 2  # class 1 appears twice per sequence
    assert counts[99] == 2     # unknown class also counted in dry-run


def test_convert_writes_yolo_layout(tmp_path):
    src = tmp_path / "raw"
    out = tmp_path / "yolo"
    _build_synthetic_dataset(src, n_sequences=2)

    result = convert(
        source_dir=src,
        output_dir=out,
        class_map=default_class_map(),
        split_ratios=(1.0, 0.0, 0.0),
        seed=0,
    )

    # Layout
    assert (out / "soccernet.yaml").exists()
    assert (out / "manifest.json").exists()
    assert (out / "images" / "train").is_dir()
    assert (out / "labels" / "train").is_dir()

    # Frames written: 3 per sequence × 2 sequences = 6
    assert result.n_frames == 6
    label_files = list((out / "labels" / "train").glob("*.txt"))
    assert len(label_files) == 6

    # Unknown class (99) was skipped
    assert result.n_rows_skipped_unknown_class == 2  # one per sequence


def test_convert_yolo_label_format(tmp_path):
    src = tmp_path / "raw"
    out = tmp_path / "yolo"
    _build_synthetic_dataset(src, n_sequences=1)

    convert(src, out, default_class_map(), (1.0, 0.0, 0.0), seed=0)

    label = (out / "labels" / "train" / "SNMOT-001_000001.txt").read_text(encoding="utf-8")
    lines = label.strip().splitlines()
    assert len(lines) == 3  # frame 1 has 3 known-class detections (cls 1, 2, 5)

    for line in lines:
        parts = line.split()
        assert len(parts) == 5
        cls = int(parts[0])
        assert 0 <= cls < len(default_class_map().output_classes)
        cx, cy, bw, bh = (float(p) for p in parts[1:])
        for v in (cx, cy, bw, bh):
            assert 0.0 < v <= 1.0


def test_convert_default_classmap_groups_team_a_and_b_as_player(tmp_path):
    src = tmp_path / "raw"
    out = tmp_path / "yolo"
    _build_synthetic_dataset(src, n_sequences=1)

    result = convert(src, out, default_class_map(), (1.0, 0.0, 0.0), seed=0)

    # Default map: 1→player, 2→player, 4→referee, 5→ball, 3→player (unused here)
    # In sequence 1: cls=1 once, cls=2 once, cls=5 once, cls=99 once (skip), cls=1 once, cls=4 once
    # Resulting per-class: player=3, referee=1, ball=1
    assert result.rows_per_class["player"] == 3
    assert result.rows_per_class["ball"] == 1
    assert result.rows_per_class["referee"] == 1


def test_convert_split_ratios_assign_each_sequence_to_one_split(tmp_path):
    src = tmp_path / "raw"
    out = tmp_path / "yolo"
    _build_synthetic_dataset(src, n_sequences=4)

    result = convert(src, out, default_class_map(), (0.5, 0.25, 0.25), seed=0)

    # Each sequence assigned to exactly one split
    all_seqs = sum(result.splits.values(), [])
    assert len(all_seqs) == 4
    assert len(set(all_seqs)) == 4


def test_convert_writes_ultralytics_yaml(tmp_path):
    src = tmp_path / "raw"
    out = tmp_path / "yolo"
    _build_synthetic_dataset(src, n_sequences=1)

    convert(src, out, default_class_map(), (1.0, 0.0, 0.0), seed=0)

    yaml_text = (out / "soccernet.yaml").read_text(encoding="utf-8")
    assert "train: images/train" in yaml_text
    assert "val: images/val" in yaml_text
    assert "test: images/test" in yaml_text
    assert "0: player" in yaml_text
    assert "1: ball" in yaml_text
    assert "2: referee" in yaml_text


def test_manifest_records_provenance(tmp_path):
    src = tmp_path / "raw"
    out = tmp_path / "yolo"
    _build_synthetic_dataset(src, n_sequences=2)

    convert(src, out, default_class_map(), (0.5, 0.5, 0.0), seed=42)

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 42
    assert manifest["split_ratios"] == [0.5, 0.5, 0.0]
    assert manifest["n_sequences"] == 2
    assert "created_at_utc" in manifest
    assert manifest["created_at_utc"].endswith("Z")
    assert "input_to_output" in manifest["class_map"]


def test_custom_class_map_with_only_ball(tmp_path):
    src = tmp_path / "raw"
    out = tmp_path / "yolo"
    _build_synthetic_dataset(src, n_sequences=1)

    ball_only = ClassMap(input_to_output={5: "ball"}, output_classes=["ball"])
    result = convert(src, out, ball_only, (1.0, 0.0, 0.0), seed=0)

    # Per sequence: 1 row of class 5 (in frame 1)
    assert result.rows_per_class["ball"] == 1
    # All non-ball rows skipped
    assert result.n_rows_skipped_unknown_class > 0


def _build_long_sequence(root: Path, name: str, n_frames: int) -> None:
    seq = root / name
    _make_seqinfo(seq / "seqinfo.ini")
    for f in range(1, n_frames + 1):
        _make_image(seq / "img1" / f"{f:06d}.jpg")
    _make_gt(
        seq / "gt" / "gt.txt",
        [f"{f}, 1, 100, 200, 50, 80, 1, 1, 1.0" for f in range(1, n_frames + 1)],
    )


def test_frame_stride_subsamples_frames(tmp_path):
    src = tmp_path / "raw"
    out = tmp_path / "yolo"
    _build_long_sequence(src, "SNMOT-001", n_frames=20)

    result = convert(src, out, default_class_map(), (1.0, 0.0, 0.0), seed=0, frame_stride=5)

    # 20 frames, stride 5 → frames 1, 6, 11, 16 = 4 frames
    assert result.n_frames == 4


def test_max_frames_per_sequence_caps_after_stride(tmp_path):
    src = tmp_path / "raw"
    out = tmp_path / "yolo"
    _build_long_sequence(src, "SNMOT-001", n_frames=20)

    result = convert(
        src, out, default_class_map(), (1.0, 0.0, 0.0),
        seed=0, frame_stride=2, max_frames_per_sequence=5,
    )

    # 20 frames, stride 2 → 10 candidates → cap at 5 = 5 frames written
    assert result.n_frames == 5


def test_max_sequences_limits_dataset(tmp_path):
    src = tmp_path / "raw"
    out = tmp_path / "yolo"
    for i in range(5):
        _build_long_sequence(src, f"SNMOT-{i+1:03d}", n_frames=3)

    result = convert(src, out, default_class_map(), (1.0, 0.0, 0.0), seed=0, max_sequences=2)

    # Only 2 sequences × 3 frames each
    assert len(result.sequences) == 2
    assert result.n_frames == 6


def test_manifest_records_subsample_settings(tmp_path):
    src = tmp_path / "raw"
    out = tmp_path / "yolo"
    _build_long_sequence(src, "SNMOT-001", n_frames=10)

    convert(
        src, out, default_class_map(), (1.0, 0.0, 0.0),
        seed=0, frame_stride=3, max_sequences=1, max_frames_per_sequence=2,
    )

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["frame_stride"] == 3
    assert manifest["max_sequences"] == 1
    assert manifest["max_frames_per_sequence"] == 2
