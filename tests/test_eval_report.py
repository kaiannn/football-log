"""Tests for the eval report generator (offline, no real eval runs)."""

from __future__ import annotations

import json
from pathlib import Path

from football_log.eval.report import _collect, generate


def _write_experiment(runs_root: Path, name: str, started: str, det=None, trk=None, wrl=None, git="abc1234"):
    d = runs_root / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({
        "exp_name": name,
        "started_at_utc": started,
        "finished_at_utc": started,
        "duration_sec": 1.5,
        "git_sha": git,
    }), encoding="utf-8")
    if det is not None:
        (d / "detection.json").write_text(json.dumps(det), encoding="utf-8")
    if trk is not None:
        (d / "tracking.json").write_text(json.dumps(trk), encoding="utf-8")
    if wrl is not None:
        (d / "world.json").write_text(json.dumps(wrl), encoding="utf-8")


def test_collect_empty_returns_empty_list(tmp_path):
    assert _collect(tmp_path) == []


def test_collect_skips_dirs_without_meta(tmp_path):
    (tmp_path / "stray").mkdir()
    (tmp_path / "stray" / "detection.json").write_text("{}", encoding="utf-8")
    assert _collect(tmp_path) == []


def test_collect_extracts_headline_metrics(tmp_path):
    _write_experiment(
        tmp_path, "baseline", "2026-05-17T10:00:00Z",
        det={"mAP50": 0.65, "mAP50_95": 0.42, "per_class_mAP50": {"player": 0.78, "ball": 0.41}},
        trk={"IDF1": 0.71, "MOTA": 0.62, "ID_switches": 89},
        wrl={"RMSE_m": 2.34, "MAE_m": 1.8, "n_valid": 12},
    )
    rows = _collect(tmp_path)
    assert len(rows) == 1
    r = rows[0]
    assert r["exp_name"] == "baseline"
    assert r["mAP50"] == 0.65
    assert r["mAP50_player"] == 0.78
    assert r["IDF1"] == 0.71
    assert r["ID_switches"] == 89
    assert r["world_RMSE_m"] == 2.34


def test_collect_sorts_experiments_by_start_time(tmp_path):
    _write_experiment(tmp_path, "second", "2026-05-17T11:00:00Z")
    _write_experiment(tmp_path, "first", "2026-05-17T10:00:00Z")
    _write_experiment(tmp_path, "third", "2026-05-17T12:00:00Z")
    names = [r["exp_name"] for r in _collect(tmp_path)]
    assert names == ["first", "second", "third"]


def test_generate_writes_markdown_and_csv(tmp_path):
    _write_experiment(
        tmp_path, "baseline", "2026-05-17T10:00:00Z",
        det={"mAP50": 0.65, "per_class_mAP50": {"player": 0.78, "ball": 0.41}},
    )
    result = generate(tmp_path, tmp_path)
    md = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "baseline" in md
    assert "Comparison" in md
    csv_text = (tmp_path / "REPORT.csv").read_text(encoding="utf-8")
    assert "exp_name" in csv_text
    assert "baseline" in csv_text
    assert result["n_experiments"] == 1


def test_generate_handles_partial_experiments(tmp_path):
    """Detection-only and world-only experiments should both render without crashing."""
    _write_experiment(
        tmp_path, "det_only", "2026-05-17T10:00:00Z",
        det={"mAP50": 0.5, "per_class_mAP50": {"player": 0.5}},
    )
    _write_experiment(
        tmp_path, "world_only", "2026-05-17T11:00:00Z",
        wrl={"RMSE_m": 1.5, "MAE_m": 1.1, "n_valid": 8},
    )
    generate(tmp_path, tmp_path)
    md = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "det_only" in md and "world_only" in md
    # The missing-cell formatting should be em-dash, not blank or "None"
    assert "None" not in md


def test_generate_with_no_experiments_writes_placeholder(tmp_path):
    generate(tmp_path, tmp_path)
    md = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "No experiments" in md
