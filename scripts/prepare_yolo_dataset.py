#!/usr/bin/env python3
"""CLI shim for football_log.data.yolo_convert.

Examples:

    # Preview source class IDs before committing to a class map.
    python scripts/prepare_yolo_dataset.py \\
        --source-dir data/soccernet/raw \\
        --dry-run

    # Convert with default class map (player / ball / referee).
    python scripts/prepare_yolo_dataset.py \\
        --source-dir data/soccernet/raw \\
        --output-dir data/soccernet

    # Convert with a custom class map (recommended).
    python scripts/prepare_yolo_dataset.py \\
        --source-dir data/soccernet/raw \\
        --output-dir data/soccernet \\
        --class-map config/soccernet_classes.yaml \\
        --split-ratios 0.7,0.15,0.15 \\
        --seed 42 \\
        --copy-images
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the football_log package importable when run as a standalone script.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from football_log.data.yolo_convert import (  # noqa: E402
    convert,
    default_class_map,
    dry_run,
    load_class_map,
)


def _parse_split_ratios(s: str) -> tuple[float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("split-ratios must be three comma-separated numbers")
    total = sum(parts)
    if abs(total - 1.0) > 1e-6:
        raise argparse.ArgumentTypeError(f"split-ratios must sum to 1.0 (got {total})")
    return parts[0], parts[1], parts[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--source-dir", required=True, type=Path, help="SoccerNet Tracking raw directory")
    parser.add_argument("--output-dir", type=Path, help="Where to write YOLO dataset (required unless --dry-run)")
    parser.add_argument("--class-map", type=Path, help="YAML: input_to_output (dict) + output_classes (list)")
    parser.add_argument("--split-ratios", type=_parse_split_ratios, default=(0.7, 0.15, 0.15))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--copy-images", action="store_true", help="Copy frames instead of symlinking")
    parser.add_argument("--dry-run", action="store_true", help="Print class-id counts only; no writes")

    # ----- Subsampling knobs (low-memory recipes) -----
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Use every Nth frame per sequence (default 1 = all). E.g. 5 = ~5x less data.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Convert at most N sequences (default: all)",
    )
    parser.add_argument(
        "--max-frames-per-sequence",
        type=int,
        default=None,
        help="Cap frames per sequence after stride (default: all)",
    )
    args = parser.parse_args()

    if args.dry_run:
        summary = dry_run(args.source_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.output_dir is None:
        parser.error("--output-dir is required (or pass --dry-run)")

    class_map = load_class_map(args.class_map) if args.class_map else default_class_map()
    if args.class_map is None:
        print("[warn] using built-in default class map; verify with --dry-run before committing", file=sys.stderr)

    result = convert(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        class_map=class_map,
        split_ratios=args.split_ratios,
        seed=args.seed,
        copy_images=args.copy_images,
        frame_stride=args.frame_stride,
        max_sequences=args.max_sequences,
        max_frames_per_sequence=args.max_frames_per_sequence,
    )

    print(f"Converted {len(result.sequences)} sequences → {args.output_dir}")
    print(f"  frames written: {result.n_frames}")
    print(f"  label rows written: {result.n_labels_written}")
    print(f"  rows skipped (unknown class): {result.n_rows_skipped_unknown_class}")
    print(f"  rows per output class: {result.rows_per_class}")
    print(f"  splits: train={len(result.splits['train'])}, val={len(result.splits['val'])}, test={len(result.splits['test'])}")
    if args.frame_stride > 1 or args.max_sequences or args.max_frames_per_sequence:
        print(
            f"  subsample: stride={args.frame_stride}, max_sequences={args.max_sequences}, "
            f"max_frames_per_sequence={args.max_frames_per_sequence}"
        )
    print(f"  manifest: {args.output_dir / 'manifest.json'}")
    print(f"  ultralytics yaml: {args.output_dir / 'soccernet.yaml'}")


if __name__ == "__main__":
    main()
