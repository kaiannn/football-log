"""Tests for tracker logical-name resolution and Re-ID interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from football_log.vision.reid import ReIDExtractor, cosine_similarity
from football_log.vision.tracker_registry import resolve_tracker


# ---- tracker_registry ----------------------------------------------------------


def test_bytetrack_resolves_to_ultralytics_builtin():
    assert resolve_tracker("bytetrack") == "bytetrack.yaml"


def test_botsort_resolves_to_ultralytics_builtin():
    assert resolve_tracker("botsort") == "botsort.yaml"


def test_botsort_reid_resolves_to_packaged_yaml():
    resolved = resolve_tracker("botsort+reid")
    assert resolved.endswith("botsort_reid.yaml")
    assert Path(resolved).is_file()


def test_botsort_reid_yaml_has_reid_enabled():
    resolved = Path(resolve_tracker("botsort+reid"))
    content = resolved.read_text(encoding="utf-8")
    assert "with_reid: True" in content
    assert "tracker_type: botsort" in content


def test_explicit_yaml_path_passes_through():
    assert resolve_tracker("/tmp/custom.yaml") == "/tmp/custom.yaml"
    assert resolve_tracker("custom.yaml") == "custom.yaml"


def test_logical_name_is_case_insensitive():
    assert resolve_tracker("ByteTrack") == "bytetrack.yaml"
    assert resolve_tracker("BOTSORT") == "botsort.yaml"


def test_empty_string_falls_back_to_default():
    assert resolve_tracker("") == "bytetrack.yaml"


# ---- reid Protocol -------------------------------------------------------------


class _DummyExtractor:
    """Minimal class that satisfies the ReIDExtractor Protocol."""

    def extract(self, frame, bbox):
        return np.array([1.0, 2.0, 3.0], dtype=np.float32)


def test_dummy_extractor_satisfies_protocol():
    assert isinstance(_DummyExtractor(), ReIDExtractor)


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_similarity_opposite_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)


def test_cosine_similarity_zero_vector_returns_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 0.0])
    assert cosine_similarity(a, b) == 0.0
