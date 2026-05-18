"""Detection mAP evaluation.

Wraps `ultralytics.YOLO(weights).val(data=yaml)` and dumps a JSON metrics file.
The JSON is the artifact you compare across experiments — check it into
`runs/<exp_name>/metrics.json` after each training run.

Usage:
    python -m football_log.eval.eval_detection \\
        --weights yolov8n.pt \\
        --data data/soccernet/soccernet.yaml \\
        --out runs/baseline/metrics.json \\
        --imgsz 1280

The --data YAML must follow the ultralytics convention:
    path: data/soccernet
    train: images/train
    val:   images/val
    test:  images/test
    names: [player, ball, referee]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict


def _safe_get(obj: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
        if isinstance(obj, dict) and n in obj:
            return obj[n]
    return default


def _to_metrics_dict(results: Any, class_names: list[str]) -> Dict[str, Any]:
    """Best-effort extraction across ultralytics versions."""
    box = _safe_get(results, "box")
    out: Dict[str, Any] = {}
    if box is not None:
        out["mAP50"] = float(_safe_get(box, "map50", default=float("nan")))
        out["mAP50_95"] = float(_safe_get(box, "map", default=float("nan")))
        out["mean_precision"] = float(_safe_get(box, "mp", default=float("nan")))
        out["mean_recall"] = float(_safe_get(box, "mr", default=float("nan")))
        per_class_ap50 = _safe_get(box, "ap50")
        per_class_ap = _safe_get(box, "ap")
        if per_class_ap50 is not None:
            out["per_class_mAP50"] = {
                name: float(per_class_ap50[i]) if i < len(per_class_ap50) else None
                for i, name in enumerate(class_names)
            }
        if per_class_ap is not None:
            out["per_class_mAP50_95"] = {
                name: float(per_class_ap[i].mean()) if i < len(per_class_ap) else None
                for i, name in enumerate(class_names)
            }
    speed = _safe_get(results, "speed")
    if isinstance(speed, dict):
        out["speed_ms"] = {k: float(v) for k, v in speed.items()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--weights", required=True, help="YOLO weights path (.pt)")
    parser.add_argument("--data", required=True, help="ultralytics data YAML")
    parser.add_argument("--out", required=True, help="Where to write metrics JSON")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001, help="Eval-time confidence threshold (low to recover full PR curve)")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--device", default=None, help="cuda device id, 'cpu', or 'mps'")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(f"ultralytics not installed: {e}")

    model = YOLO(args.weights)
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        split=args.split,
        device=args.device,
        verbose=False,
    )

    class_names = list(getattr(results, "names", {}).values()) if hasattr(results, "names") else []
    metrics = _to_metrics_dict(results, class_names)
    metrics["_meta"] = {
        "weights": os.path.abspath(args.weights),
        "data": os.path.abspath(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
