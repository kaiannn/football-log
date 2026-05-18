"""Re-ID extractor Protocol — for plugin-style appearance feature swap-in.

The default Re-ID is provided by ultralytics' BotSORT-with-reid (selected
via `--tracker botsort+reid`); it runs internally and does not surface
through this interface.

This module exists for users who want to inject a custom Re-ID model
(e.g. fine-tuned OSNet on SoccerNet Re-ID) into a custom Detector that
bypasses ultralytics' built-in tracker.

Typical use:

    class FineTunedOSNet:
        def __init__(self, weights_path):
            self.model = load_osnet(weights_path)
        def extract(self, frame, bbox):
            x, y, w, h = bbox
            crop = frame[y:y+h, x:x+w]
            return self.model.embed(crop)  # → np.ndarray of shape (D,)

    extractor: ReIDExtractor = FineTunedOSNet("runs/osnet_soccernet/best.pt")
"""

from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class ReIDExtractor(Protocol):
    """Pluggable appearance-feature extractor.

    Implementations should return a fixed-dimensional embedding vector
    that supports cosine similarity for cross-frame matching.
    """

    def extract(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Return a (D,) float32 embedding for the bbox crop in `frame`.

        - frame: full image, BGR
        - bbox: (x, y, w, h) in pixel coordinates
        """
        ...


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for two embedding vectors. Returns a scalar in [-1, 1]."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
