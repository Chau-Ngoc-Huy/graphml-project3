"""Locations for the standalone XAI experiment inside the final repository."""

from __future__ import annotations

import os
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]


def runs_dir() -> Path:
    """Return the local output directory, or an external mounted run directory."""
    override = os.environ.get("XAI_EXPERIMENT_RUNS_DIR")
    return Path(override).expanduser().resolve() if override else EXPERIMENT_ROOT / "runs" / "qa_vg_psg_xai"


def latest_artifact(root: Path, pattern: str) -> Path:
    matches = sorted(Path(root).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern} in {root}; run the preceding notebook or set XAI_EXPERIMENT_RUNS_DIR")
    return matches[-1]
