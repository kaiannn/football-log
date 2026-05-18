"""End-to-end eval orchestrator.

Reads a YAML config, runs each configured eval (detection / tracking / world)
as a subprocess, and writes everything to `runs/<exp_name>/`:

    runs/<exp_name>/
        meta.json          — git SHA, timestamp, command line, full config used
        detection.json     — eval_detection output (if enabled and inputs exist)
        tracking.json      — eval_tracking output
        world.json         — eval_world output
        run.log            — combined stdout/stderr from each eval

After running, use `python -m football_log.eval.report` to compare across
experiments and produce `runs/REPORT.md`.

Usage:
    python -m football_log.eval --config eval_config.yaml --exp-name baseline
    python -m football_log.eval --config eval_config.yaml --exp-name module1_v1
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or None
    except (FileNotFoundError, OSError):
        return None


def _load_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise SystemExit("PyYAML required for eval config: pip install pyyaml") from e
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"{path}: top-level must be a mapping")
    return cfg


def _exists(p: Optional[str]) -> bool:
    return bool(p) and Path(p).exists()


def _run_subprocess(cmd: List[str], log_fp) -> Tuple[int, str]:
    """Run a subprocess, tee its output to console and log file. Return (rc, last_line)."""
    log_fp.write(f"\n$ {' '.join(cmd)}\n")
    log_fp.flush()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    log_fp.write(proc.stdout)
    log_fp.write(proc.stderr)
    log_fp.flush()
    last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return proc.returncode, last_line


def _build_detection_cmd(cfg: Dict[str, Any], out_path: Path) -> Optional[List[str]]:
    if not cfg.get("enabled", True):
        return None
    if not _exists(cfg.get("weights")) and not str(cfg.get("weights", "")).endswith(".pt"):
        # Allow ultralytics auto-download of stock weights like 'yolov8n.pt'
        pass
    if not _exists(cfg.get("data")):
        return None
    cmd = [
        sys.executable, "-m", "football_log.eval.eval_detection",
        "--weights", str(cfg["weights"]),
        "--data", str(cfg["data"]),
        "--out", str(out_path),
        "--imgsz", str(cfg.get("imgsz", 640)),
        "--conf", str(cfg.get("conf", 0.001)),
        "--iou", str(cfg.get("iou", 0.6)),
        "--split", str(cfg.get("split", "val")),
    ]
    if cfg.get("device") is not None:
        cmd += ["--device", str(cfg["device"])]
    return cmd


def _build_tracking_cmd(cfg: Dict[str, Any], out_path: Path) -> Optional[List[str]]:
    if not cfg.get("enabled", True):
        return None
    if not _exists(cfg.get("pred")) or not _exists(cfg.get("gt")):
        return None
    return [
        sys.executable, "-m", "football_log.eval.eval_tracking",
        "--pred", str(cfg["pred"]),
        "--gt", str(cfg["gt"]),
        "--out", str(out_path),
        "--iou", str(cfg.get("iou_threshold", 0.5)),
    ]


def _build_world_cmd(cfg: Dict[str, Any], out_path: Path) -> Optional[List[str]]:
    if not cfg.get("enabled", True):
        return None
    if not _exists(cfg.get("points")):
        return None
    cmd = [
        sys.executable, "-m", "football_log.eval.eval_world",
        "--points", str(cfg["points"]),
        "--out", str(out_path),
    ]
    if _exists(cfg.get("homography")):
        cmd += ["--homography", str(cfg["homography"])]
    elif _exists(cfg.get("camera_calib")):
        cmd += ["--camera-calib", str(cfg["camera_calib"])]
    else:
        return None
    return cmd


def run(config_path: Path, exp_name: str, runs_root: Path = Path("runs")) -> Dict[str, Any]:
    cfg = _load_config(config_path)
    exp_dir = runs_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    started = _dt.datetime.now(_dt.timezone.utc)
    log_path = exp_dir / "run.log"
    summary: Dict[str, Any] = {"detection": None, "tracking": None, "world": None}
    skipped: List[str] = []

    with open(log_path, "w", encoding="utf-8") as log_fp:
        for kind, builder in (
            ("detection", _build_detection_cmd),
            ("tracking", _build_tracking_cmd),
            ("world", _build_world_cmd),
        ):
            section_cfg = cfg.get(kind, {}) or {}
            out_path = exp_dir / f"{kind}.json"
            cmd = builder(section_cfg, out_path)
            if cmd is None:
                msg = f"[skip] {kind}: disabled or required inputs missing"
                print(msg)
                log_fp.write(msg + "\n")
                skipped.append(kind)
                continue
            rc, _ = _run_subprocess(cmd, log_fp)
            if rc != 0:
                msg = f"[fail] {kind}: exit code {rc}"
                print(msg)
                log_fp.write(msg + "\n")
                summary[kind] = {"status": "failed", "exit_code": rc}
                continue
            try:
                summary[kind] = json.loads(out_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                summary[kind] = {"status": "no_output", "error": str(e)}

    finished = _dt.datetime.now(_dt.timezone.utc)
    meta = {
        "exp_name": exp_name,
        "started_at_utc": started.isoformat().replace("+00:00", "Z"),
        "finished_at_utc": finished.isoformat().replace("+00:00", "Z"),
        "duration_sec": round((finished - started).total_seconds(), 2),
        "git_sha": _git_sha(),
        "command": " ".join([os.path.basename(sys.executable), "-m", "football_log.eval"] + sys.argv[1:]),
        "config_path": str(config_path.resolve()),
        "config": cfg,
        "skipped": skipped,
        "results_present": [k for k, v in summary.items() if v is not None and "status" not in (v if isinstance(v, dict) else {})],
    }
    (exp_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== {exp_name} done in {meta['duration_sec']}s ===")
    print(f"  results in: {exp_dir}")
    print(f"  ran: {meta['results_present']}")
    if skipped:
        print(f"  skipped: {skipped}")

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to eval config YAML")
    parser.add_argument("--exp-name", required=True, help="Experiment name (directory under runs/)")
    parser.add_argument("--runs-root", default="runs", help="Root directory for run outputs (default: runs/)")
    args = parser.parse_args()
    run(Path(args.config), args.exp_name, Path(args.runs_root))


if __name__ == "__main__":
    main()
