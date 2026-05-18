"""Walk runs/ and produce a comparison report across experiments.

Outputs:
    runs/REPORT.md   — markdown comparison table + per-experiment details
    runs/REPORT.csv  — flat CSV of headline metrics for spreadsheet import

Usage:
    python -m football_log.eval.report
    python -m football_log.eval.report --runs-root runs --out-dir runs
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


HEADLINE_COLUMNS = [
    "exp_name",
    "started_at_utc",
    "git_sha",
    "duration_sec",
    "mAP50",
    "mAP50_95",
    "mAP50_player",
    "mAP50_ball",
    "mAP50_referee",
    "IDF1",
    "MOTA",
    "ID_switches",
    "world_RMSE_m",
    "world_MAE_m",
    "world_n_valid",
]


def _safe_load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _fmt(v: Any, places: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v:  # NaN
            return "—"
        return f"{v:.{places}f}"
    return str(v)


def _extract_row(exp_dir: Path) -> Dict[str, Any]:
    meta = _safe_load(exp_dir / "meta.json") or {}
    det = _safe_load(exp_dir / "detection.json") or {}
    trk = _safe_load(exp_dir / "tracking.json") or {}
    wrl = _safe_load(exp_dir / "world.json") or {}

    per_class50 = det.get("per_class_mAP50") or {}
    row = {
        "exp_name": meta.get("exp_name", exp_dir.name),
        "started_at_utc": meta.get("started_at_utc"),
        "git_sha": meta.get("git_sha"),
        "duration_sec": meta.get("duration_sec"),
        "mAP50": det.get("mAP50"),
        "mAP50_95": det.get("mAP50_95"),
        "mAP50_player": per_class50.get("player"),
        "mAP50_ball": per_class50.get("ball"),
        "mAP50_referee": per_class50.get("referee"),
        "IDF1": trk.get("IDF1"),
        "MOTA": trk.get("MOTA"),
        "ID_switches": trk.get("ID_switches"),
        "world_RMSE_m": wrl.get("RMSE_m"),
        "world_MAE_m": wrl.get("MAE_m"),
        "world_n_valid": wrl.get("n_valid"),
    }
    return row


def _collect(runs_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not runs_root.is_dir():
        return rows
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "meta.json").exists():
            continue
        rows.append(_extract_row(child))
    rows.sort(key=lambda r: r.get("started_at_utc") or "")
    return rows


def _render_markdown(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "# football-log eval report\n\nNo experiments found in `runs/`.\n"

    lines = ["# football-log eval report\n"]
    lines.append(f"_{len(rows)} experiment(s) tracked. Sorted by start time._\n")

    # Headline comparison table
    headers = [
        ("exp_name", "Experiment"),
        ("started_at_utc", "Started"),
        ("git_sha", "Git"),
        ("mAP50", "mAP@50"),
        ("mAP50_player", "mAP player"),
        ("mAP50_ball", "mAP ball"),
        ("mAP50_referee", "mAP ref"),
        ("IDF1", "IDF1"),
        ("MOTA", "MOTA"),
        ("ID_switches", "ID sw"),
        ("world_RMSE_m", "RMSE m"),
    ]
    lines.append("## Comparison\n")
    lines.append("| " + " | ".join(h[1] for h in headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        cells = []
        for k, _ in headers:
            v = r.get(k)
            if k in ("mAP50", "mAP50_player", "mAP50_ball", "mAP50_referee", "IDF1", "MOTA"):
                cells.append(_fmt(v, 3))
            elif k == "world_RMSE_m":
                cells.append(_fmt(v, 2))
            else:
                cells.append(_fmt(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Per-experiment details
    lines.append("## Per-experiment details\n")
    for r in rows:
        lines.append(f"### {r['exp_name']}\n")
        lines.append(f"- Started: `{r.get('started_at_utc') or '—'}` (git `{r.get('git_sha') or '—'}`)")
        lines.append(f"- Duration: {_fmt(r.get('duration_sec'), 2)} s")
        if r.get("mAP50") is not None:
            lines.append("- **Detection**:")
            lines.append(f"  - mAP@50 = {_fmt(r.get('mAP50'), 3)}, mAP@50-95 = {_fmt(r.get('mAP50_95'), 3)}")
            lines.append(f"  - per-class mAP@50: player={_fmt(r.get('mAP50_player'), 3)}, ball={_fmt(r.get('mAP50_ball'), 3)}, referee={_fmt(r.get('mAP50_referee'), 3)}")
        if r.get("IDF1") is not None:
            lines.append("- **Tracking**:")
            lines.append(f"  - IDF1 = {_fmt(r.get('IDF1'), 3)}, MOTA = {_fmt(r.get('MOTA'), 3)}, ID switches = {_fmt(r.get('ID_switches'))}")
        if r.get("world_RMSE_m") is not None:
            lines.append("- **World**:")
            lines.append(f"  - RMSE = {_fmt(r.get('world_RMSE_m'), 2)} m, MAE = {_fmt(r.get('world_MAE_m'), 2)} m, n_valid = {_fmt(r.get('world_n_valid'))}")
        lines.append("")
    return "\n".join(lines)


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADLINE_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in HEADLINE_COLUMNS})


def generate(runs_root: Path, out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _collect(runs_root)
    md_path = out_dir / "REPORT.md"
    csv_path = out_dir / "REPORT.csv"
    md_path.write_text(_render_markdown(rows), encoding="utf-8")
    _write_csv(rows, csv_path)
    return {"markdown": md_path, "csv": csv_path, "n_experiments": len(rows)}  # type: ignore[dict-item]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--runs-root", default="runs", type=Path)
    parser.add_argument("--out-dir", default="runs", type=Path)
    args = parser.parse_args()
    result = generate(args.runs_root, args.out_dir)
    print(f"Wrote {result['markdown']} and {result['csv']} ({result['n_experiments']} experiments)")


if __name__ == "__main__":
    main()
