"""Resolve tracker logical names → ultralytics-compatible YAML paths.

The CLI accepts logical names like `bytetrack`, `botsort`, `botsort+reid`
to insulate users from path lookups. Direct YAML paths still pass through
unchanged.

Logical names:

    bytetrack       → ultralytics built-in (low ID switches, no Re-ID, fastest)
    botsort         → ultralytics built-in (better motion compensation, no Re-ID)
    botsort+reid    → custom config in football_log/configs/ — BotSORT + OSNet Re-ID
                      (the practical DeepSORT-equivalent for Module 2)

Anything else is assumed to be an explicit YAML path and returned as-is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


_LOGICAL_NAMES = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
}


def _packaged_yaml(name: str) -> Optional[Path]:
    candidate = Path(__file__).resolve().parent.parent / "configs" / name
    return candidate if candidate.is_file() else None


def resolve_tracker(name_or_path: str) -> str:
    """Map a logical tracker name (or pass-through path) to what ultralytics expects."""
    if not name_or_path:
        return "bytetrack.yaml"

    key = name_or_path.strip().lower()

    if key == "botsort+reid":
        path = _packaged_yaml("botsort_reid.yaml")
        if path is None:
            raise FileNotFoundError(
                "botsort_reid.yaml not found in football_log/configs/. "
                "Reinstall the package or check the install layout."
            )
        return str(path)

    if key in _LOGICAL_NAMES:
        return _LOGICAL_NAMES[key]

    return name_or_path
