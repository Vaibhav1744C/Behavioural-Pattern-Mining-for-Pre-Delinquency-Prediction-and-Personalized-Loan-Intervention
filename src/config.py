"""Central repository and data paths used by portable project commands."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("BPM_DATA_ROOT", REPO_ROOT / "data")).expanduser().resolve()


def run_data_dir(run: str) -> Path:
    """Return the shared source data root for a run profile."""
    if run not in {"subsample", "full"}:
        raise ValueError("run must be 'subsample' or 'full'")
    return DATA_ROOT
