"""SoccerNet Game-State Reconstruction (GSR-2025) → YOLO format converter.

The 2025 release ships annotations as **per-sequence COCO-style JSON** files
(`Labels-GameState.json`) instead of the MOT CSV used by SoccerNet Tracking.
This module converts that layout into the YOLO format that ultralytics
expects, mirroring the public surface of `yolo_convert.py` so callers can
dispatch on dataset format.

Input layout (one directory per sequence):

    <source_dir>/
        SNGS-001/
            img1/
                000001.jpg
                000002.jpg
                ...
            Labels-GameState.json    # COCO-like; see below
        SNGS-002/
            ...

`Labels-GameState.json` schema (relevant fields):

    {
      "info": { "name", "frame_rate", "seq_length", "im_dir", "im_ext" },
      "images": [
        {
          "image_id": "1170000001",
          "file_name": "000001.jpg",
          "height": 1080,
          "width": 1920,
          ...
        },
        ...
      ],
      "annotations": [
        {
          "image_id": "1170000001",
          "track_id": 1,
          "category_id": 1,            # 1=player 2=goalkeeper 3=referee 4=ball 5=pitch 6=camera 7=other
          "supercategory": "object",   # we keep only "object"
          "attributes": { "role": "player", "team": "left", "jersey": "10" },
          "bbox_image": { "x": ..., "y": ..., "w": ..., "h": ... },
          ...
        },
        ...
      ],
      "categories": [ ... ]
    }

Output layout matches `yolo_convert.py`:

    <output_dir>/
        images/{train,val,test}/<seq>_<frame>.jpg
        labels/{train,val,test}/<seq>_<frame>.txt
        soccernet.yaml
        manifest.json
"""

from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

from football_log.data.yolo_convert import ClassMap
from typing import Union as _Union


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


@dataclass
class GsrAnnotation:
    image_id: str
    track_id: int
    category_id: int
    x: float
    y: float
    w: float
    h: float
    team: Optional[str] = None   # "left" | "right" | None
    role: Optional[str] = None   # "player" | "goalkeeper" | None


@dataclass
class GsrImage:
    image_id: str
    file_name: str
    width: int
    height: int


@dataclass
class GsrSequence:
    name: str
    img_dir: Path
    labels_path: Path
    images: List[GsrImage]
    annotations: List[GsrAnnotation]
    categories: List[Dict]


def _parse_labels_json(path: Path) -> Tuple[List[GsrImage], List[GsrAnnotation], List[Dict]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    images = []
    for im in obj.get("images", []):
        try:
            images.append(GsrImage(
                image_id=str(im["image_id"]),
                file_name=str(im["file_name"]),
                width=int(im["width"]),
                height=int(im["height"]),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    anns = []
    for a in obj.get("annotations", []):
        if a.get("supercategory") != "object":
            continue
        bbox = a.get("bbox_image")
        if not isinstance(bbox, dict):
            continue
        try:
            attrs = a.get("attributes") or {}
            anns.append(GsrAnnotation(
                image_id=str(a["image_id"]),
                track_id=int(a.get("track_id", -1)),
                category_id=int(a["category_id"]),
                x=float(bbox["x"]),
                y=float(bbox["y"]),
                w=float(bbox["w"]),
                h=float(bbox["h"]),
                team=str(attrs["team"]).lower() if "team" in attrs else None,
                role=str(attrs["role"]).lower() if "role" in attrs else None,
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return images, anns, obj.get("categories", [])


# ---------------------------------------------------------------------------
# Sequence discovery
# ---------------------------------------------------------------------------


def discover_sequences(source_dir: Path) -> List[GsrSequence]:
    seqs: List[GsrSequence] = []
    for child in sorted(source_dir.iterdir()):
        if not child.is_dir():
            continue
        img_dir = child / "img1"
        labels_path = child / "Labels-GameState.json"
        if not img_dir.is_dir() or not labels_path.exists():
            continue
        images, anns, cats = _parse_labels_json(labels_path)
        if not images:
            continue
        seqs.append(GsrSequence(
            name=child.name,
            img_dir=img_dir,
            labels_path=labels_path,
            images=images,
            annotations=anns,
            categories=cats,
        ))
    return seqs


# ---------------------------------------------------------------------------
# Default class map for GSR-2025
# ---------------------------------------------------------------------------


def default_class_map() -> ClassMap:
    """Default mapping for SoccerNet GSR-2025 categories.

    GSR-2025 source IDs:
      1 = player, 2 = goalkeeper, 3 = referee, 4 = ball
      5 = pitch (skipped), 6 = camera (skipped), 7 = other (skipped)
    """
    return ClassMap(
        input_to_output={1: "player", 2: "player", 3: "referee", 4: "ball"},
        output_classes=["player", "ball", "referee"],
    )


@dataclass
class TeamClassMap:
    """6-class map keyed on (category_id, team) for Module 3B team-as-class detection.

    Source category_id × team → output class:
      (1, "left")  → team_a_player
      (1, "right") → team_b_player
      (2, "left")  → goalkeeper_a
      (2, "right") → goalkeeper_b
      (3, None)    → referee   (no team label)
      (4, None)    → ball      (no team label)

    A (category_id, None) entry acts as a wildcard — used when the annotation
    has no team attribute (referees, ball).
    """

    map_: Dict[Tuple[int, Optional[str]], str]
    output_classes: List[str]

    def output_index(self, source_cls: int, team: Optional[str] = None) -> Optional[int]:
        name = self.map_.get((source_cls, team))
        if name is None:
            name = self.map_.get((source_cls, None))  # wildcard fallback
        if name is None:
            return None
        try:
            return self.output_classes.index(name)
        except ValueError:
            return None

    @property
    def input_to_output(self) -> Dict[str, str]:
        """Serialisable form for manifest.json."""
        return {f"{k[0]}:{k[1] or '*'}": v for k, v in self.map_.items()}


def default_team_class_map() -> TeamClassMap:
    """Default 6-class mapping using GSR-2025 team labels (Module 3B).

    Output class order (YOLO indices):
      0 = team_a_player   (field players, left team)
      1 = team_b_player   (field players, right team)
      2 = goalkeeper_a    (goalkeeper, left team)
      3 = goalkeeper_b    (goalkeeper, right team)
      4 = referee
      5 = ball
    """
    return TeamClassMap(
        map_={
            (1, "left"):  "team_a_player",
            (1, "right"): "team_b_player",
            (2, "left"):  "goalkeeper_a",
            (2, "right"): "goalkeeper_b",
            (3, None):    "referee",
            (4, None):    "ball",
        },
        output_classes=[
            "team_a_player", "team_b_player",
            "goalkeeper_a", "goalkeeper_b",
            "referee", "ball",
        ],
    )


# ---------------------------------------------------------------------------
# Conversion (mirrors yolo_convert.convert signature for drop-in dispatch)
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


def _normalize_bbox(a: GsrAnnotation, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    cx = (a.x + 0.5 * a.w) / img_w
    cy = (a.y + 0.5 * a.h) / img_h
    w = a.w / img_w
    h = a.h / img_h
    return cx, cy, w, h


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def _write_yaml(out_dir: Path, class_map: "_Union[ClassMap, TeamClassMap]") -> None:
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


def _frame_num_from_filename(file_name: str) -> Optional[int]:
    """Extract the integer frame number from "000001.jpg" → 1."""
    stem = Path(file_name).stem
    try:
        return int(stem)
    except ValueError:
        # Tolerate alternate padding like "frame_000001"
        digits = "".join(c for c in stem if c.isdigit())
        return int(digits) if digits else None


def convert(
    source_dir: Path,
    output_dir: Path,
    class_map: "_Union[ClassMap, TeamClassMap]",
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
    copy_images: bool = False,
    frame_stride: int = 1,
    max_sequences: Optional[int] = None,
    max_frames_per_sequence: Optional[int] = None,
) -> ConversionResult:
    seqs = discover_sequences(source_dir)
    if not seqs:
        raise SystemExit(f"No GSR-2025 sequences found under {source_dir}")
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

        # Index annotations by image_id for fast lookup
        anns_by_image: Dict[str, List[GsrAnnotation]] = defaultdict(list)
        for a in seq.annotations:
            anns_by_image[a.image_id].append(a)

        # Order images by parsed frame number so stride is meaningful
        images_with_frame = []
        for im in seq.images:
            frame_num = _frame_num_from_filename(im.file_name)
            if frame_num is None:
                continue
            images_with_frame.append((frame_num, im))
        images_with_frame.sort(key=lambda t: t[0])

        if frame_stride > 1:
            images_with_frame = images_with_frame[::frame_stride]
        if max_frames_per_sequence is not None and max_frames_per_sequence > 0:
            images_with_frame = images_with_frame[:max_frames_per_sequence]

        for frame_num, im in images_with_frame:
            src_img = seq.img_dir / im.file_name
            if not src_img.exists():
                continue

            stem = f"{seq.name}_{frame_num:06d}"
            dst_img = output_dir / "images" / split / f"{stem}.jpg"
            dst_lbl = output_dir / "labels" / split / f"{stem}.txt"
            _link_or_copy(src_img, dst_img, copy=copy_images)

            frame_anns = anns_by_image.get(im.image_id, [])
            label_lines: List[str] = []
            for a in frame_anns:
                idx = class_map.output_index(a.category_id, a.team)
                if idx is None:
                    result.n_rows_skipped_unknown_class += 1
                    continue
                cx, cy, bw, bh = _normalize_bbox(a, im.width, im.height)
                if not (0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                    continue
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                label_lines.append(f"{idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                cls_name = class_map.output_classes[idx]
                result.rows_per_class[cls_name] = result.rows_per_class.get(cls_name, 0) + 1

            dst_lbl.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
            result.n_frames += 1
            result.n_labels_written += len(label_lines)

    _write_yaml(output_dir, class_map)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_format": "gsr-2025",
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
    counts: Counter = Counter()
    team_counts: Counter = Counter()
    role_counts: Counter = Counter()
    team_role_counts: Counter = Counter()
    total = 0
    cat_id_to_name: Dict[int, str] = {}
    for seq in seqs:
        for c in seq.categories:
            try:
                cat_id_to_name[int(c["id"])] = str(c.get("name", "?"))
            except (KeyError, TypeError, ValueError):
                continue
        for a in seq.annotations:
            counts[a.category_id] += 1
            if a.team:
                team_counts[a.team] += 1
            if a.role:
                role_counts[a.role] += 1
            if a.team and a.role:
                team_role_counts[f"{a.team}:{a.role}"] += 1
            total += 1
    summary: Dict[str, object] = {
        "n_sequences": len(seqs),
        "sequences": [s.name for s in seqs[:20]] + (["..."] if len(seqs) > 20 else []),
        "class_id_counts": dict(sorted(counts.items())),
        "category_id_to_name": cat_id_to_name,
        "team_distribution": dict(team_counts),
        "role_distribution": dict(role_counts),
        "team_role_distribution": dict(team_role_counts),
        "n_rows_total": total,
    }
    return summary


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def detect_format(source_dir: Path) -> str:
    """Return 'gsr', 'mot', or 'unknown' based on the first sequence we find.

    Looks at the first sub-directory and decides by the presence of
    Labels-GameState.json (GSR-2025) vs gt/gt.txt (MOT / SoccerNet Tracking).
    """
    if not source_dir.is_dir():
        return "unknown"
    for child in sorted(source_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "Labels-GameState.json").exists():
            return "gsr"
        if (child / "gt" / "gt.txt").exists():
            return "mot"
    return "unknown"
