"""SoccerNet Tracking → YOLO format converter.

SoccerNet Tracking ships annotations in MOT challenge format. This module
converts that layout into the YOLO format that ultralytics expects.

Input layout (one directory per sequence):

    <source_dir>/
        SNMOT-001/
            img1/
                000001.jpg
                000002.jpg
                ...
            gt/
                gt.txt          # MOT CSV: frame, id, x, y, w, h, conf, cls, vis
            seqinfo.ini         # optional, contains imWidth / imHeight
        SNMOT-002/
            ...

Output layout:

    <output_dir>/
        images/{train,val,test}/<seq>_<frame>.jpg
        labels/{train,val,test}/<seq>_<frame>.txt
        soccernet.yaml          # ultralytics dataset config
        manifest.json           # what was converted, when, with which class map

`<seq>_<frame>` is e.g. `SNMOT-001_000001`.

The class IDs in `gt.txt` vary across SoccerNet revisions — always run with
`--dry-run` first to confirm what's actually in your annotations, then write
a class-map YAML accordingly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from configparser import ConfigParser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


# ---------------------------------------------------------------------------
# MOT CSV parsing
# ---------------------------------------------------------------------------


@dataclass
class MotRow:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    cls: int


def _read_mot_csv(path: Path, cls_col: int = 7) -> List[MotRow]:
    rows: List[MotRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) < cls_col + 1:
                continue
            try:
                frame = int(float(parts[0]))
                tid = int(float(parts[1]))
                x, y, w, h = (float(parts[i]) for i in range(2, 6))
                cls = int(float(parts[cls_col]))
            except (ValueError, IndexError):
                continue
            rows.append(MotRow(frame=frame, track_id=tid, x=x, y=y, w=w, h=h, cls=cls))
    return rows


# ---------------------------------------------------------------------------
# Sequence discovery
# ---------------------------------------------------------------------------


@dataclass
class Sequence:
    name: str
    img_dir: Path
    gt_path: Path
    width: Optional[int] = None
    height: Optional[int] = None


def _read_seqinfo(path: Path) -> Tuple[Optional[int], Optional[int]]:
    if not path.exists():
        return None, None
    cp = ConfigParser()
    try:
        cp.read(path)
    except Exception:
        return None, None
    sec = "Sequence" if cp.has_section("Sequence") else (cp.sections()[0] if cp.sections() else None)
    if sec is None:
        return None, None
    w = cp.getint(sec, "imWidth", fallback=0) or None
    h = cp.getint(sec, "imHeight", fallback=0) or None
    return w, h


def discover_sequences(source_dir: Path) -> List[Sequence]:
    seqs: List[Sequence] = []
    for child in sorted(source_dir.iterdir()):
        if not child.is_dir():
            continue
        img_dir = child / "img1"
        gt_path = child / "gt" / "gt.txt"
        if not img_dir.is_dir() or not gt_path.exists():
            continue
        w, h = _read_seqinfo(child / "seqinfo.ini")
        seqs.append(Sequence(name=child.name, img_dir=img_dir, gt_path=gt_path, width=w, height=h))
    return seqs


# ---------------------------------------------------------------------------
# Image size resolution
# ---------------------------------------------------------------------------


def _image_size_for(seq: Sequence, sample_frame: Optional[Path] = None) -> Tuple[int, int]:
    if seq.width and seq.height:
        return seq.width, seq.height
    if sample_frame is None:
        candidates = sorted(seq.img_dir.glob("*"))
        if not candidates:
            raise RuntimeError(f"{seq.name}: no images found")
        sample_frame = candidates[0]
    if Image is None:
        raise RuntimeError(
            "Cannot determine image size: install Pillow (`pip install Pillow`) "
            "or provide imWidth/imHeight in each sequence's seqinfo.ini"
        )
    with Image.open(sample_frame) as im:
        return im.size  # (w, h)


# ---------------------------------------------------------------------------
# Class mapping config
# ---------------------------------------------------------------------------


@dataclass
class ClassMap:
    """Map source class IDs to output class names + ordered output class list."""
    input_to_output: Dict[int, str]
    output_classes: List[str]

    def output_index(self, source_cls: int) -> Optional[int]:
        name = self.input_to_output.get(source_cls)
        if name is None:
            return None
        try:
            return self.output_classes.index(name)
        except ValueError:
            return None


def default_class_map() -> ClassMap:
    """Default mapping for SoccerNet Tracking (subject to verification per release).

    Always run --dry-run first to confirm the source class IDs in your data.
    """
    return ClassMap(
        input_to_output={1: "player", 2: "player", 3: "player", 4: "referee", 5: "ball"},
        output_classes=["player", "ball", "referee"],
    )


def load_class_map(path: Path) -> ClassMap:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise SystemExit("PyYAML required for --class-map: pip install pyyaml") from e
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw_map = data.get("input_to_output", {})
    out_classes = data.get("output_classes", [])
    if not isinstance(raw_map, dict) or not isinstance(out_classes, list):
        raise SystemExit(f"{path}: must define `input_to_output` (mapping) and `output_classes` (list)")
    cmap = {int(k): str(v) for k, v in raw_map.items()}
    return ClassMap(input_to_output=cmap, output_classes=[str(c) for c in out_classes])


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


@dataclass
class ConversionResult:
    sequences: List[str] = field(default_factory=list)
    splits: Dict[str, List[str]] = field(default_factory=dict)
    n_frames: int = 0
    n_labels_written: int = 0
    n_rows_skipped_unknown_class: int = 0
    rows_per_class: Dict[str, int] = field(default_factory=dict)


def _split_assignments(seq_names: List[str], ratios: Tuple[float, float, float], seed: int) -> Dict[str, str]:
    rng = random.Random(seed)
    shuffled = list(seq_names)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    out: Dict[str, str] = {}
    for i, name in enumerate(shuffled):
        if i < n_train:
            out[name] = "train"
        elif i < n_train + n_val:
            out[name] = "val"
        else:
            out[name] = "test"
    return out


def _normalize_bbox(row: MotRow, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    cx = (row.x + 0.5 * row.w) / img_w
    cy = (row.y + 0.5 * row.h) / img_h
    w = row.w / img_w
    h = row.h / img_h
    return cx, cy, w, h


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def _write_yaml(out_dir: Path, class_map: ClassMap) -> None:
    yaml_path = out_dir / "soccernet.yaml"
    lines = [
        f"path: {out_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for i, name in enumerate(class_map.output_classes):
        lines.append(f"  {i}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert(
    source_dir: Path,
    output_dir: Path,
    class_map: ClassMap,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
    copy_images: bool = False,
    frame_stride: int = 1,
    max_sequences: Optional[int] = None,
    max_frames_per_sequence: Optional[int] = None,
) -> ConversionResult:
    seqs = discover_sequences(source_dir)
    if not seqs:
        raise SystemExit(f"No MOT-style sequences found under {source_dir}")
    if max_sequences is not None and max_sequences > 0:
        seqs = seqs[:max_sequences]

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    assignments = _split_assignments([s.name for s in seqs], split_ratios, seed)

    result = ConversionResult(
        sequences=[s.name for s in seqs],
        splits={"train": [], "val": [], "test": []},
        rows_per_class={c: 0 for c in class_map.output_classes},
    )

    for seq in seqs:
        split = assignments[seq.name]
        result.splits[split].append(seq.name)
        rows = _read_mot_csv(seq.gt_path)
        rows_by_frame: Dict[int, List[MotRow]] = defaultdict(list)
        for r in rows:
            rows_by_frame[r.frame].append(r)
        if not rows_by_frame:
            continue
        img_w, img_h = _image_size_for(seq)

        # Apply subsampling: stride first, then optional cap.
        sorted_frames = sorted(rows_by_frame.keys())
        if frame_stride > 1:
            sorted_frames = sorted_frames[::frame_stride]
        if max_frames_per_sequence is not None and max_frames_per_sequence > 0:
            sorted_frames = sorted_frames[:max_frames_per_sequence]

        for frame_num in sorted_frames:
            frame_rows = rows_by_frame[frame_num]
            img_name = f"{frame_num:06d}.jpg"
            src_img = seq.img_dir / img_name
            if not src_img.exists():
                # SoccerNet uses 6-digit padding, but tolerate alternate padding too
                alt = next((p for p in seq.img_dir.glob(f"*{frame_num}*.jpg")), None)
                if alt is None:
                    continue
                src_img = alt

            stem = f"{seq.name}_{frame_num:06d}"
            dst_img = output_dir / "images" / split / f"{stem}.jpg"
            dst_lbl = output_dir / "labels" / split / f"{stem}.txt"
            _link_or_copy(src_img, dst_img, copy=copy_images)

            label_lines: List[str] = []
            for r in frame_rows:
                idx = class_map.output_index(r.cls)
                if idx is None:
                    result.n_rows_skipped_unknown_class += 1
                    continue
                cx, cy, bw, bh = _normalize_bbox(r, img_w, img_h)
                if not (0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                    continue
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                label_lines.append(f"{idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                result.rows_per_class[class_map.output_classes[idx]] = (
                    result.rows_per_class.get(class_map.output_classes[idx], 0) + 1
                )

            dst_lbl.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
            result.n_frames += 1
            result.n_labels_written += len(label_lines)

    _write_yaml(output_dir, class_map)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "class_map": {
            "input_to_output": {str(k): v for k, v in class_map.input_to_output.items()},
            "output_classes": class_map.output_classes,
        },
        "split_ratios": list(split_ratios),
        "seed": seed,
        "image_storage": "copy" if copy_images else "symlink",
        "frame_stride": frame_stride,
        "max_sequences": max_sequences,
        "max_frames_per_sequence": max_frames_per_sequence,
        "n_sequences": len(seqs),
        "n_frames": result.n_frames,
        "n_labels_written": result.n_labels_written,
        "n_rows_skipped_unknown_class": result.n_rows_skipped_unknown_class,
        "rows_per_class": result.rows_per_class,
        "splits": result.splits,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return result


# ---------------------------------------------------------------------------
# Dry-run preview
# ---------------------------------------------------------------------------


def dry_run(source_dir: Path) -> Dict[str, object]:
    seqs = discover_sequences(source_dir)
    summary: Dict[str, object] = {
        "n_sequences": len(seqs),
        "sequences": [s.name for s in seqs[:20]] + (["..."] if len(seqs) > 20 else []),
        "class_id_counts": {},
        "n_rows_total": 0,
    }
    counts: Counter = Counter()
    total = 0
    for seq in seqs:
        for row in _read_mot_csv(seq.gt_path):
            counts[row.cls] += 1
            total += 1
    summary["class_id_counts"] = dict(sorted(counts.items()))
    summary["n_rows_total"] = total
    return summary
