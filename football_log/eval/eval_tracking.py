"""Multi-object tracking evaluation (MOTA / IDF1 / ID switches).

Inputs:
    --pred    A JSONL produced by football-log (one record per detection per frame).
    --gt      A ground-truth MOT-style CSV (frame, id, x, y, w, h, conf, cls, vis).
    --out     Where to write metrics JSON.
    --iou     Matching IoU threshold (default 0.5).

Requires `motmetrics` (install with: pip install motmetrics).
The pred and gt files are read independently — the JSONL parser does not need
the original ground-truth column order.

Usage:
    python -m football_log.eval.eval_tracking \\
        --pred outputs/match_tracks.jsonl \\
        --gt   data/soccernet/track_gt/match.txt \\
        --out  runs/baseline/tracking_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


Frame = int
TrackID = int
BBox = Tuple[float, float, float, float]  # x, y, w, h


def _read_jsonl_predictions(path: Path) -> Dict[Frame, List[Tuple[TrackID, BBox]]]:
    out: Dict[Frame, List[Tuple[TrackID, BBox]]] = {}
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            f = int(row["frame_idx"])
            tid = int(row["track_id"])
            if tid < 0:
                continue
            out.setdefault(f, []).append((tid, (float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))))
    return out


def _read_mot_gt(path: Path) -> Dict[Frame, List[Tuple[TrackID, BBox]]]:
    out: Dict[Frame, List[Tuple[TrackID, BBox]]] = {}
    with open(path, "r", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        for row in reader:
            if not row or row[0].lstrip("-").isdigit() is False:
                continue
            f = int(row[0])
            tid = int(row[1])
            x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            if len(row) >= 7:
                try:
                    if float(row[6]) < 0:
                        continue
                except ValueError:
                    pass
            out.setdefault(f, []).append((tid, (x, y, w, h)))
    return out


def _accumulate(
    gt: Dict[Frame, List[Tuple[TrackID, BBox]]],
    pred: Dict[Frame, List[Tuple[TrackID, BBox]]],
    iou_threshold: float,
) -> Any:
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=False)
    all_frames = sorted(set(gt) | set(pred))
    for f in all_frames:
        g = gt.get(f, [])
        p = pred.get(f, [])
        gids = [t for t, _ in g]
        pids = [t for t, _ in p]
        if g and p:
            dist = mm.distances.iou_matrix(
                [b for _, b in g], [b for _, b in p], max_iou=1.0 - iou_threshold
            )
        else:
            dist = [[]]
        acc.update(gids, pids, dist, frameid=f)
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    try:
        import motmetrics as mm
    except ImportError as e:
        raise SystemExit(
            "motmetrics not installed. Install with: pip install motmetrics"
        ) from e

    pred = _read_jsonl_predictions(Path(args.pred))
    gt = _read_mot_gt(Path(args.gt))
    acc = _accumulate(gt, pred, args.iou)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["mota", "idf1", "idp", "idr", "num_switches", "num_fragmentations", "mostly_tracked", "mostly_lost", "num_false_positives", "num_misses"],
        name="overall",
    )
    row = summary.iloc[0].to_dict()

    metrics = {
        "MOTA": float(row.get("mota", float("nan"))),
        "IDF1": float(row.get("idf1", float("nan"))),
        "IDP": float(row.get("idp", float("nan"))),
        "IDR": float(row.get("idr", float("nan"))),
        "ID_switches": int(row.get("num_switches", 0)),
        "fragmentations": int(row.get("num_fragmentations", 0)),
        "mostly_tracked": int(row.get("mostly_tracked", 0)),
        "mostly_lost": int(row.get("mostly_lost", 0)),
        "false_positives": int(row.get("num_false_positives", 0)),
        "misses": int(row.get("num_misses", 0)),
        "_meta": {
            "pred": str(Path(args.pred).resolve()),
            "gt": str(Path(args.gt).resolve()),
            "iou_threshold": args.iou,
            "n_frames_evaluated": len(set(pred) | set(gt)),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
